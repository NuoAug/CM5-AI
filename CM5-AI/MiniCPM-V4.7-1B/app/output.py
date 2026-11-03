# -*- coding: utf-8 -*-
"""统一语音输出仲裁：危险提示可抢占普通播报，队列有界，过期事件丢弃。

文本和语音由同一个事件生成，避免“文字说安全、语音说危险”之类的分叉。
浏览器通过 SSE 订阅；CM5 上可替换为 ALSA 播放 sink。
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from turns import TurnManager

PRIORITY_DANGER = 0
PRIORITY_INTERRUPT = 1
PRIORITY_CONFIRM = 2
PRIORITY_ANSWER = 3

MAX_TEXT = 260
MAX_QUEUE = 16


@dataclass
class SpeechEvent:
    id: int
    text: str
    priority: int
    source: str
    reason: str = ""
    ttl: float = 3.0
    preempt: bool = False
    dedupe_key: str = ""
    epoch: int = 0
    created_mono: float = field(default_factory=time.monotonic)
    meta: dict = field(default_factory=dict)
    utterance_id: str = ""
    segment_id: str = ""
    sequence: int = 0
    is_final: bool = True
    deadline_s: Optional[float] = None
    playback: str = "browser"

    def to_dict(self) -> dict:
        return {
            "type": "speech",
            "id": self.id,
            "utterance_id": self.utterance_id,
            "segment_id": self.segment_id,
            "sequence": self.sequence,
            "is_final": self.is_final,
            "deadline_s": self.deadline_s,
            "text": self.text,
            "priority": self.priority,
            "source": self.source,
            "reason": self.reason,
            "ttl": self.ttl,
            "preempt": self.preempt,
            "epoch": self.epoch,
            "age_s": round(time.monotonic() - self.created_mono, 3),
            "playback": self.playback,
            "meta": self.meta,
        }


class LiveCancel(threading.Event):
    def __init__(self, valid):
        super().__init__()
        self.valid = valid

    def is_set(self):
        return super().is_set() or not self.valid()


class OutputBus:
    def __init__(self):
        self._lock = threading.RLock()
        self.turns = TurnManager()
        self._subs: set[queue.Queue] = set()
        self._ids = itertools.count(1)
        self._epoch = 0
        self._last: dict[str, float] = {}
        self._stats = {
            "pushed": 0,
            "deduped": 0,
            "dropped": 0,
            "cancelled": 0,
            "subscribers": 0,
            "sink": "none",
            "sink_played": 0,
            "last_event": None,
        }
        self._sink = None
        self._sink_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=32)
        self._sink_thread: Optional[threading.Thread] = None
        self._sink_stop = threading.Event()
        self._history: deque = deque(maxlen=128)
        self._prepare_queues = {False: queue.Queue(maxsize=8), True: queue.Queue(maxsize=4)}
        self._prepare_threads = []
        self._active_jobs = {}
        self._playing_payload = None
        self._sink_queue = queue.PriorityQueue(maxsize=4)

    def begin_turn(self, session_id='', request_id=''):
        token = self.turns.begin(session_id, request_id)
        with self._lock:
            playing = self._playing_payload
            sink = self._sink
            need_interrupt = False
            for _, event, payload in self._active_jobs.values():
                identity = payload.get('turn') or {}
                if identity.get('session_id') == token.session_id and identity != token.identity():
                    event.set()
                    need_interrupt = True
        if sink is not None and need_interrupt and callable(getattr(sink, "cancel_synthesis", None)):
            sink.cancel_synthesis()
        if sink is not None and playing and playing.get('turn', {}).get('session_id') == token.session_id and playing.get('turn') != token.identity():
            sink.stop()
        with self._lock:
            self._fanout({'type': 'turn_started', 'turn': token.identity()})
        return token

    def new_utterance_id(self, prefix: str = "u") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def history_since(self, last_id: Optional[int]) -> list[dict]:
        with self._lock:
            if last_id is None:
                return []
            return [item for item in list(self._history) if item.get("id", -1) > int(last_id) and (item.get("type") not in ("speech", "speech_end") or self._payload_live(item))]

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        with self._lock:
            self._subs.add(q)
            self._stats["subscribers"] = len(self._subs)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)
            self._stats["subscribers"] = len(self._subs)

    def attach_sink(self, sink) -> None:
        """设备端 ALSA sink；浏览器调试时不挂载。"""
        with self._lock:
            self._sink = sink
            self._sink_stop.clear()
            self._stats["sink"] = type(sink).__name__
        if callable(getattr(sink, 'prepare', None)):
            for urgent in (False, True):
                thread = threading.Thread(target=self._prepare_worker, args=(urgent,),
                                          name='urgent-tts' if urgent else 'ordinary-tts', daemon=True)
                self._prepare_threads.append(thread)
                thread.start()
        if self._sink_thread is None or not self._sink_thread.is_alive():
            self._sink_thread = threading.Thread(target=self._sink_worker, name="output-sink", daemon=True)
            self._sink_thread.start()

    def detach_sink(self) -> None:
        with self._lock:
            sink = self._sink
            self._sink = None
            self._stats["sink"] = "none"
        self._sink_stop.set()
        if sink is not None:
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                pass

    def _payload_live(self, payload):
        if time.monotonic() * 1000 >= payload.get('expires_mono_ms', float('inf')):
            return False
        if payload.get('priority', PRIORITY_ANSWER) <= PRIORITY_DANGER:
            return True
        if payload.get('turn'):
            return self.turns.is_current(payload['turn'])
        return payload.get('epoch') == self._epoch

    def _prepare_worker(self, urgent):
        while not self._sink_stop.is_set():
            try:
                priority, event_id, payload = self._prepare_queues[urgent].get(timeout=.1)
            except queue.Empty:
                continue
            with self._lock:
                sink = self._sink
                cancel = LiveCancel(lambda p=payload: self._payload_live(p))
                self._active_jobs[event_id] = (priority, cancel, payload)
            try:
                if sink is None or not self._payload_live(payload):
                    continue
                wav = b"" if payload.get("type") == "speech_end" else sink.prepare(payload, cancel)
                if len(wav) > 16 * 1024 * 1024:
                    raise ValueError('单段PCM超过16MiB限制')
                prepared = dict(payload, _prepared_wav=wav)
                while not cancel.is_set() and self._payload_live(payload):
                    try:
                        self._sink_queue.put((priority, event_id, prepared), timeout=.05)
                        break
                    except queue.Full:
                        continue
            except Exception:
                with self._lock:
                    self._stats['prepare_failed'] = self._stats.get('prepare_failed', 0) + 1
            finally:
                with self._lock:
                    self._active_jobs.pop(event_id, None)

    def _sink_worker(self) -> None:
        while not self._sink_stop.is_set():
            try:
                _priority, _event_id, payload = self._sink_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                sink = self._sink
            if sink is None or not self._payload_live(payload):
                continue
            cancel = LiveCancel(lambda p=payload: self._payload_live(p))
            with self._lock:
                self._active_jobs[_event_id] = (_priority, cancel, payload)
                self._playing_payload = payload
            try:
                if not self._payload_live(payload):
                    continue
                sink.speak(payload, cancel)
                with self._lock:
                    self._stats["sink_played"] += 1
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._stats["sink_failed"] = self._stats.get("sink_failed", 0) + 1
                continue
            finally:
                with self._lock:
                    self._active_jobs.pop(_event_id, None)
                    self._playing_payload = None

    def _drain_sink_queue(self, min_priority: int) -> None:
        kept = []
        while True:
            try:
                item = self._sink_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] < min_priority:
                kept.append(item)
        for item in kept:
            try:
                self._sink_queue.put_nowait(item)
            except queue.Full:
                self._stats["dropped"] += 1

    def _cancel_jobs(self, min_priority):
        with self._lock:
            for priority, event, payload in list(self._active_jobs.values()):
                if priority >= min_priority:
                    event.set()
            for pending in self._prepare_queues.values():
                keep = []
                while True:
                    try:
                        item = pending.get_nowait()
                    except queue.Empty:
                        break
                    if item[0] < min_priority:
                        keep.append(item)
                for item in keep:
                    try:
                        pending.put_nowait(item)
                    except queue.Full:
                        self._stats['dropped'] += 1

    def _fanout(self, payload: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    self._stats["dropped"] += 1

    def push_segment(
        self,
        text: str,
        *,
        utterance_id: str = "",
        segment_id: str = "",
        sequence: int = 0,
        is_final: bool = True,
        priority: int = PRIORITY_ANSWER,
        source: str = "app",
        reason: str = "",
        ttl: float = 3.0,
        dedupe_key: Optional[str] = None,
        preempt: Optional[bool] = None,
        deadline_s: Optional[float] = None,
        meta: Optional[dict] = None,
        turn=None,
    ) -> Optional[dict]:
        text = (text or "").strip()
        if not text:
            return None
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT]
        uid = utterance_id or self.new_utterance_id("u")
        sid = segment_id or f"{uid}-{int(sequence)}"
        now = time.monotonic()
        key = dedupe_key or f"{uid}|{sid}|{int(sequence)}|{text}"
        window = 3.0 if priority <= PRIORITY_INTERRUPT else 2.0
        if turn is not None and not self.turns.is_current(turn.identity()):
            self._stats['dropped'] += 1
            return None
        with self._lock:
            if turn is not None and not self.turns.is_current(turn.identity()):
                self._stats['dropped'] += 1
                return None
            last = self._last.get(key)
            if last is not None and now - last < window:
                self._stats["deduped"] += 1
                return None
            self._last[key] = now
            if len(self._last) > 128:
                cutoff = now - 30.0
                self._last = {k: v for k, v in self._last.items() if v >= cutoff}
                while len(self._last) > 128:
                    self._last.pop(next(iter(self._last)))

            if priority <= PRIORITY_DANGER:
                self.turns.cancel()
                self._cancel_jobs(PRIORITY_INTERRUPT)
                # 危险提示：清掉尚未播出的普通排队项与历史，避免重连后重播旧回答。
                for q in list(self._subs):
                    kept = []
                    while True:
                        try:
                            item = q.get_nowait()
                        except queue.Empty:
                            break
                        if item.get("priority", 99) <= PRIORITY_DANGER:
                            kept.append(item)
                    for item in kept:
                        try:
                            q.put_nowait(item)
                        except queue.Full:
                            self._stats["dropped"] += 1
                self._history.clear()

            event = SpeechEvent(
                id=next(self._ids),
                text=text,
                priority=int(priority),
                source=source,
                reason=reason,
                ttl=float(ttl),
                preempt=(priority <= PRIORITY_INTERRUPT if preempt is None else bool(preempt)),
                dedupe_key=key,
                epoch=self._epoch,
                meta=meta or {},
                utterance_id=uid,
                segment_id=sid,
                sequence=int(sequence),
                is_final=bool(is_final),
                deadline_s=deadline_s,
                playback=("device" if self._sink is not None else "browser"),
            )
            payload = event.to_dict()
            payload['expires_mono_ms'] = (event.created_mono + event.ttl) * 1000
            if turn is not None:
                payload['turn'] = turn.identity()
            self._fanout(payload)
            self._history.append(payload)
            self._stats["pushed"] += 1
            self._stats["last_event"] = {
                "id": event.id,
                "utterance_id": uid,
                "segment_id": sid,
                "sequence": int(sequence),
                "text": text,
                "priority": event.priority,
                "source": source,
                "reason": reason,
                "at_mono": now,
            }
            sink = self._sink
        if sink is not None:
            if priority <= PRIORITY_INTERRUPT or event.preempt:
                try:
                    sink.stop()
                except Exception:  # noqa: BLE001
                    pass
            if priority <= PRIORITY_DANGER:
                self._drain_sink_queue(PRIORITY_INTERRUPT)
            elif event.preempt:
                # 新回答的第一段抢占旧回答，丢弃尚未播出的旧片段。
                self._drain_sink_queue(int(priority))
            destination = self._prepare_queues[priority <= PRIORITY_DANGER] if callable(getattr(sink, 'prepare', None)) else self._sink_queue
            try:
                destination.put_nowait((int(priority), event.id, payload))
            except queue.Full:
                try:
                    destination.get_nowait()
                except queue.Empty:
                    pass
                try:
                    destination.put_nowait((int(priority), event.id, payload))
                except queue.Full:
                    self._stats["dropped"] += 1
        return payload

    def finish_utterance(self, utterance_id, turn=None):
        if turn and not self.turns.is_current(turn.identity()):
            return
        payload = {'type': 'speech_end', 'id': next(self._ids), 'text': '',
                   'utterance_id': utterance_id, 'priority': PRIORITY_ANSWER,
                   'epoch': self._epoch, 'expires_mono_ms': (time.monotonic() + 15) * 1000}
        if turn:
            payload['turn'] = turn.identity()
        with self._lock:
            self._fanout(payload)
            sink = self._sink
        if sink is not None:
            destination = self._prepare_queues[False] if callable(getattr(sink, 'prepare', None)) else self._sink_queue
            try:
                destination.put_nowait((PRIORITY_ANSWER, payload['id'], payload))
            except queue.Full:
                self._stats['dropped'] += 1

    def push(
        self,
        text: str,
        priority: int = PRIORITY_ANSWER,
        source: str = "app",
        reason: str = "",
        ttl: float = 3.0,
        dedupe_key: Optional[str] = None,
        preempt: Optional[bool] = None,
        meta: Optional[dict] = None,
    ) -> Optional[dict]:
        """兼容单段调用：生成新 utterance，只有一个 final 片段。"""
        return self.push_segment(
            text,
            priority=priority,
            source=source,
            reason=reason,
            ttl=ttl,
            dedupe_key=dedupe_key,
            preempt=preempt,
            meta=meta,
        )

    def cancel(self, min_priority: int = PRIORITY_INTERRUPT, source: str = "voice") -> dict:
        """停止普通播报，保留尚未播出的高优先级提示。"""
        turns = self.turns.cancel()
        self._cancel_jobs(min_priority)
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            for q in list(self._subs):
                kept = []
                while True:
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        break
                    if item.get("priority", 99) < min_priority:
                        kept.append(item)
                for item in kept:
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        self._stats["dropped"] += 1
            self._history.clear()
            payload = {
                "type": "cancel",
                "id": next(self._ids),
                "source": source,
                "min_priority": int(min_priority),
                "epoch": epoch,
                "ts": time.monotonic(),
                "turns": turns,
                "generation_workers_finished": all(t['compute_finished'] for t in turns),
                "tts_jobs_active": sum(1 for p, _, _ in self._active_jobs.values() if p >= min_priority),
                "compute_finished": False if any(not t['compute_finished'] for t in turns) or any(p >= min_priority for p, _, _ in self._active_jobs.values()) else (True if all(t.get('backend_exit_confirmed') is True for t in turns) else None),
            }
            self._fanout(payload)
            self._history.append(payload)
            self._stats["cancelled"] += 1
            sink = self._sink
        if sink is not None and (self._playing_payload is None or self._playing_payload.get("priority", 3) >= min_priority):
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                pass
            self._drain_sink_queue(min_priority)
        return payload

    def stats(self) -> dict:
        with self._lock:
            data = dict(self._stats)
            data["queue_limit"] = MAX_QUEUE
            data["epoch"] = self._epoch
            data['active_jobs'] = len(self._active_jobs)
            data['prepare_queues'] = {str(k): q.qsize() for k, q in self._prepare_queues.items()}
            last = data.get("last_event")
            if last:
                data["last_event"] = {k: v for k, v in last.items() if k != "at_mono"}
                data["last_event_age_s"] = round(time.monotonic() - last["at_mono"], 2)
            return data


OUTPUT = OutputBus()
