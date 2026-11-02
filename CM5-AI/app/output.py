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

PRIORITY_DANGER = 0
PRIORITY_GUIDE = 1
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


class OutputBus:
    def __init__(self):
        self._lock = threading.RLock()
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
            "stale_generation": 0,
            "last_event": None,
        }
        self._sink = None
        self._sink_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=32)
        self._sink_thread: Optional[threading.Thread] = None
        self._sink_stop = threading.Event()
        self._sink_current: Optional[dict] = None
        self._history: deque = deque(maxlen=128)
        self._status_hook = None

    def new_utterance_id(self, prefix: str = "u") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def current_epoch(self) -> int:
        with self._lock:
            return self._epoch

    def is_current(self, epoch: int) -> bool:
        with self._lock:
            return int(epoch) == self._epoch

    def history_since(self, last_id: Optional[int]) -> list[dict]:
        with self._lock:
            if last_id is None:
                return []
            return [item for item in list(self._history) if item.get("id", -1) > int(last_id)]

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

    def set_status_hook(self, hook) -> None:
        """播放状态回调；只报告真实观察到的阶段，不把浏览器排队说成已出声。"""
        self._status_hook = hook

    def _emit_status(self, payload: Optional[dict], state: str,
                     basis: Optional[str] = None, reason: str = "") -> None:
        hook = self._status_hook
        if hook is None or not payload:
            return
        try:
            hook(payload, state, basis, reason)
        except Exception:  # noqa: BLE001
            pass

    def attach_sink(self, sink) -> None:
        """设备端 ALSA sink；浏览器调试时不挂载。"""
        with self._lock:
            self._sink = sink
            self._sink_stop.clear()
            self._stats["sink"] = type(sink).__name__
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

    def _sink_worker(self) -> None:
        while not self._sink_stop.is_set():
            try:
                _priority, _event_id, payload = self._sink_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                sink = self._sink
            if sink is None:
                continue
            cancel = threading.Event()
            with self._lock:
                self._sink_current = payload
            try:
                sink.speak(payload, cancel)
                with self._lock:
                    self._stats["sink_played"] += 1
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._stats["sink_failed"] = self._stats.get("sink_failed", 0) + 1
                self._emit_status(payload, "failed", reason="sink_error")
            finally:
                with self._lock:
                    self._sink_current = None

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
        generation_epoch: Optional[int] = None,
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
        window = 3.0 if priority <= PRIORITY_GUIDE else 2.0
        with self._lock:
            if generation_epoch is not None and int(generation_epoch) != self._epoch:
                self._stats["stale_generation"] += 1
                return None
            last = self._last.get(key)
            if last is not None and now - last < window:
                self._stats["deduped"] += 1
                return None
            self._last[key] = now
            if len(self._last) > 128:
                cutoff = now - 30.0
                self._last = {k: v for k, v in self._last.items() if v >= cutoff}

            if priority <= PRIORITY_DANGER:
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
                preempt=(priority <= PRIORITY_GUIDE if preempt is None else bool(preempt)),
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
            if priority <= PRIORITY_GUIDE or event.preempt:
                try:
                    sink.stop()
                except Exception:  # noqa: BLE001
                    pass
            if priority <= PRIORITY_DANGER:
                self._drain_sink_queue(PRIORITY_GUIDE)
            elif event.preempt:
                # 新回答的第一段抢占旧回答，丢弃尚未播出的旧片段。
                self._drain_sink_queue(int(priority))
            try:
                self._sink_queue.put_nowait((int(priority), event.id, payload))
            except queue.Full:
                try:
                    self._sink_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._sink_queue.put_nowait((int(priority), event.id, payload))
                except queue.Full:
                    self._stats["dropped"] += 1
        self._emit_status(payload, "queued")
        return payload

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

    def cancel(self, min_priority: int = PRIORITY_GUIDE, source: str = "voice") -> dict:
        """停止普通播报，但保留尚未播出的危险提示；不关闭引导检测。"""
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
            }
            self._fanout(payload)
            self._history.append(payload)
            self._stats["cancelled"] += 1
            sink = self._sink
        if sink is not None:
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                pass
            self._drain_sink_queue(min_priority)
        return payload

    def drop_guidance(self, event_id: Optional[str] = None) -> dict:
        """只撤销主控引导事件，保留危险事件与普通回答；主控重启/撤销时使用。"""
        removed = 0
        current_hit = False

        def match(item: dict) -> bool:
            if item.get("priority") != PRIORITY_GUIDE:
                return False
            if event_id is None:
                return True
            return (item.get("meta") or {}).get("event_id") == event_id

        with self._lock:
            for q in list(self._subs):
                kept = []
                while True:
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        break
                    if match(item):
                        removed += 1
                    else:
                        kept.append(item)
                for item in kept:
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        self._stats["dropped"] += 1
            kept = []
            while True:
                try:
                    item = self._sink_queue.get_nowait()
                except queue.Empty:
                    break
                payload = item[2]
                if match(payload):
                    removed += 1
                else:
                    kept.append(item)
            for item in kept:
                try:
                    self._sink_queue.put_nowait(item)
                except queue.Full:
                    self._stats["dropped"] += 1
            self._history = deque(
                (item for item in self._history if not match(item)), maxlen=128
            )
            current = self._sink_current
            current_hit = bool(current and match(current))
            sink = self._sink
        if current_hit and sink is not None:
            try:
                sink.stop()
            except Exception:  # noqa: BLE001
                pass
        return {"removed": removed, "stopped_current": current_hit, "event_id": event_id}

    def stats(self) -> dict:
        with self._lock:
            data = dict(self._stats)
            data["queue_limit"] = MAX_QUEUE
            data["pending"] = self._sink_queue.qsize()
            data["epoch"] = self._epoch
            last = data.get("last_event")
            if last:
                data["last_event"] = {k: v for k, v in last.items() if k != "at_mono"}
                data["last_event_age_s"] = round(time.monotonic() - last["at_mono"], 2)
            return data


OUTPUT = OutputBus()
