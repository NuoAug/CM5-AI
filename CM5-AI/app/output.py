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

    def to_dict(self) -> dict:
        return {
            "type": "speech",
            "id": self.id,
            "text": self.text,
            "priority": self.priority,
            "source": self.source,
            "reason": self.reason,
            "ttl": self.ttl,
            "preempt": self.preempt,
            "epoch": self.epoch,
            "age_s": round(time.monotonic() - self.created_mono, 3),
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
            "last_event": None,
        }
        self._sink = None
        self._sink_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=32)
        self._sink_thread: Optional[threading.Thread] = None
        self._sink_stop = threading.Event()

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
                priority, event_id, text = self._sink_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                sink = self._sink
            if sink is None:
                continue
            cancel = threading.Event()
            try:
                sink.speak(text, cancel)
                with self._lock:
                    self._stats["sink_played"] += 1
            except Exception:  # noqa: BLE001
                continue

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
        text = (text or "").strip()
        if not text:
            return None
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT]
        now = time.monotonic()
        key = dedupe_key or f"{source}|{priority}|{text}"
        window = 3.0 if priority <= PRIORITY_GUIDE else 2.0
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < window:
                self._stats["deduped"] += 1
                return None
            self._last[key] = now
            if len(self._last) > 64:
                cutoff = now - 30.0
                self._last = {k: v for k, v in self._last.items() if v >= cutoff}

            if priority <= PRIORITY_DANGER:
                # 危险提示：清掉尚未播出的普通排队项，避免“安全”话术压住危险提示。
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
            )
            payload = event.to_dict()
            self._fanout(payload)
            self._stats["pushed"] += 1
            self._stats["last_event"] = {
                "id": event.id,
                "text": text,
                "priority": event.priority,
                "source": source,
                "reason": reason,
                "at_mono": now,
            }
            sink = self._sink
        if sink is not None:
            if priority <= PRIORITY_GUIDE:
                try:
                    sink.stop()
                except Exception:  # noqa: BLE001
                    pass
            if priority <= PRIORITY_DANGER:
                self._drain_sink_queue(PRIORITY_GUIDE)
            try:
                self._sink_queue.put_nowait((int(priority), event.id, text))
            except queue.Full:
                try:
                    self._sink_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._sink_queue.put_nowait((int(priority), event.id, text))
                except queue.Full:
                    self._stats["dropped"] += 1
        return payload

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
            payload = {
                "type": "cancel",
                "source": source,
                "min_priority": int(min_priority),
                "epoch": epoch,
                "ts": time.monotonic(),
            }
            self._fanout(payload)
            self._stats["cancelled"] += 1
            sink = self._sink
        if sink is not None:
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
            last = data.get("last_event")
            if last:
                data["last_event"] = {k: v for k, v in last.items() if k != "at_mono"}
                data["last_event_age_s"] = round(time.monotonic() - last["at_mono"], 2)
            return data


OUTPUT = OutputBus()
