"""普通会话请求的取消代次；不管理主控危险事件的序号。"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import queue
import time
import uuid


class TurnCancelled(RuntimeError):
    pass


class CloseDispatcher:
    """有界连接关闭队列；关闭卡住时不增加线程，也不声称后端已经退出。"""
    def __init__(self, workers=2, max_pending=32):
        if type(workers) is not int or workers < 1 or type(max_pending) is not int or max_pending < 1:
            raise ValueError("关闭线程和队列上限必须是正整数")
        self._queue = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stats = {"submitted": 0, "completed": 0, "failed": 0, "rejected": 0, "active": 0}
        self._threads = [threading.Thread(target=self._worker, daemon=True,
                                         name="cancel-connection-worker") for _ in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, close, complete):
        with self._lock:
            if self._stop.is_set():
                self._stats["rejected"] += 1
                return False
            try:
                self._queue.put_nowait((close, complete))
            except queue.Full:
                self._stats["rejected"] += 1
                return False
            self._stats["submitted"] += 1
            return True

    def _worker(self):
        while not self._stop.is_set():
            try:
                close, complete = self._queue.get(timeout=.1)
            except queue.Empty:
                continue
            with self._lock:
                self._stats["active"] += 1
            failed = False
            try:
                close()
            except Exception:
                failed = True
            finally:
                with self._lock:
                    self._stats["active"] -= 1
                    self._stats["completed"] += 1
                    self._stats["failed"] += int(failed)
                complete(failed)
                self._queue.task_done()

    def status(self):
        with self._lock:
            return dict(self._stats, pending=self._queue.qsize(), queue_limit=self._queue.maxsize,
                        workers=sum(t.is_alive() for t in self._threads))

    def shutdown(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=.2)


_CLOSE_DISPATCHER = None
_CLOSE_DISPATCHER_LOCK = threading.Lock()


def _close_dispatcher():
    global _CLOSE_DISPATCHER
    with _CLOSE_DISPATCHER_LOCK:
        if _CLOSE_DISPATCHER is None:
            _CLOSE_DISPATCHER = CloseDispatcher()
        return _CLOSE_DISPATCHER


@dataclass
class Turn:
    session_id: str
    request_id: str
    generation: int
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancelled: threading.Event = field(default_factory=threading.Event)
    compute_finished: threading.Event = field(default_factory=threading.Event)
    cancelled_mono_ms: float | None = None
    compute_finished_mono_ms: float | None = None
    backend_exit_confirmed: bool | None = None
    _closers: list = field(default_factory=list)
    _dispatcher: object | None = None
    _manager_lock: object | None = None
    _current_check: object | None = None
    _close_pending: int = 0
    _close_rejected: int = 0
    _close_failed: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def identity(self):
        return {k: getattr(self, k) for k in ('session_id', 'turn_id', 'request_id', 'generation')}

    def check(self):
        if self.cancelled.is_set():
            raise TurnCancelled('旧回答已取消')

    @contextmanager
    def guard_context(self):
        """原子提交会话事实；外部I/O必须在进入此短临界区之前完成。"""
        lock = self._manager_lock or self._lock
        with lock:
            self.check()
            if self._current_check is not None and not self._current_check():
                raise TurnCancelled('旧回答不再属于当前代次')
            yield self

    def register_close(self, close):
        with self._lock:
            if not self.cancelled.is_set():
                if close in self._closers:
                    return True
                if len(self._closers) >= 8:
                    self._close_rejected += 1
                    return False
                self._closers.append(close)
                return True
        return self._close_async(close)

    def unregister_close(self, close):
        with self._lock:
            if close in self._closers:
                self._closers.remove(close)

    def _close_async(self, close):
        dispatcher = self._dispatcher or _close_dispatcher()
        with self._lock:
            self._close_pending += 1
        def complete(failed):
            with self._lock:
                self._close_pending -= 1
                self._close_failed += int(failed)
        if dispatcher.submit(close, complete):
            return True
        with self._lock:
            self._close_pending -= 1
            self._close_rejected += 1
        return False

    def close_status(self):
        with self._lock:
            return {"connection_close_pending": self._close_pending,
                    "connection_close_rejected": self._close_rejected,
                    "connection_close_failed": self._close_failed}

    def cancel(self):
        lock = self._manager_lock or self._lock
        with lock:
            with self._lock:
                if self.cancelled.is_set():
                    return
                self.cancelled_mono_ms = time.monotonic_ns() / 1e6
                self.cancelled.set()
                closers, self._closers = self._closers, []
            for close in closers:
                self._close_async(close)

    def finish_compute(self):
        with self._lock:
            self._closers.clear()
        self.compute_finished_mono_ms = time.monotonic_ns() / 1e6
        self.compute_finished.set()


class TurnManager:
    def __init__(self, max_sessions=32, close_dispatcher=None):
        self.max_sessions = max_sessions
        self._dispatcher = close_dispatcher
        self._lock = threading.RLock()
        self._active = OrderedDict()
        self._generation = 0

    def begin(self, session_id='', request_id=''):
        sid = session_id or 'default'
        with self._lock:
            old = self._active.get(sid)
            if old and request_id and old.request_id == request_id and not old.cancelled.is_set():
                return old
            self._generation += 1
            token = Turn(sid, request_id or uuid.uuid4().hex, self._generation,
                         _dispatcher=self._dispatcher, _manager_lock=self._lock)
            token._current_check = lambda t=token: self._active.get(t.session_id) is t and not t.cancelled.is_set()
            if old:
                old.cancel()
            self._active[sid] = token
            self._active.move_to_end(sid)
            while len(self._active) > self.max_sessions:
                _, evicted = self._active.popitem(last=False)
                evicted.cancel()
            return token

    def close_status(self):
        dispatcher = self._dispatcher or _CLOSE_DISPATCHER
        if dispatcher is None:
            return {"workers": 0, "active": 0, "pending": 0, "queue_limit": 32,
                    "submitted": 0, "completed": 0, "failed": 0, "rejected": 0}
        return dispatcher.status()

    def is_current(self, identity):
        if not isinstance(identity, dict):
            return False
        with self._lock:
            token = self._active.get(identity.get('session_id'))
            return bool(token and not token.cancelled.is_set() and token.identity() == identity)

    def cancel(self, session_id=None, request_id=None):
        with self._lock:
            tokens = [t for t in self._active.values()
                      if (session_id is None or t.session_id == (session_id or 'default'))
                      and (request_id is None or t.request_id == request_id)]
            for token in tokens:
                token.cancel()
        return [{'identity': t.identity(), 'cancelled_mono_ms': t.cancelled_mono_ms,
                 'compute_finished': t.compute_finished.is_set(),
                 'compute_finished_mono_ms': t.compute_finished_mono_ms, 'backend_exit_confirmed': t.backend_exit_confirmed, **t.close_status()} for t in tokens]
