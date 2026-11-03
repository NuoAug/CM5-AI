"""普通会话请求的取消代次；不管理主控危险事件的序号。"""
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time
import uuid


class TurnCancelled(RuntimeError):
    pass


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
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def identity(self):
        return {k: getattr(self, k) for k in ('session_id', 'turn_id', 'request_id', 'generation')}

    def check(self):
        if self.cancelled.is_set():
            raise TurnCancelled('旧回答已取消')

    def register_close(self, close):
        with self._lock:
            if not self.cancelled.is_set():
                self._closers.append(close)
                return
        self._close_async(close)

    def unregister_close(self, close):
        with self._lock:
            if close in self._closers:
                self._closers.remove(close)

    @staticmethod
    def _close_async(close):
        # HTTP close 可能等待正在读取的套接字；不能阻塞停声或危险入口。
        def run():
            try:
                close()
            except Exception:
                pass
        threading.Thread(target=run, daemon=True, name='cancel-connection').start()

    def cancel(self):
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
    def __init__(self, max_sessions=32):
        self.max_sessions = max_sessions
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
            token = Turn(sid, request_id or uuid.uuid4().hex, self._generation)
            if old:
                old.cancel()
            self._active[sid] = token
            self._active.move_to_end(sid)
            while len(self._active) > self.max_sessions:
                _, evicted = self._active.popitem(last=False)
                evicted.cancel()
            return token

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
                 'compute_finished_mono_ms': t.compute_finished_mono_ms, 'backend_exit_confirmed': t.backend_exit_confirmed} for t in tokens]
