# -*- coding: utf-8 -*-
"""请求级时间线记录：区分说话结束、ASR、意图、视觉、首段、TTS、播放和完成。"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional


class RequestTrace:
    def __init__(self, trace_id: str, kind: str, **meta):
        self.trace_id = trace_id
        self.kind = kind
        self.meta = meta
        self.t0_mono = time.monotonic()
        self.t0_wall_ms = int(time.time() * 1000)
        self.marks: list[dict] = []
        self.status = "running"
        self.error = ""

    def mark(self, stage: str, **data) -> None:
        now_wall = int(time.time() * 1000)
        self.marks.append({
            "stage": stage,
            "wall_ms": now_wall,
            "mono_ms": int((time.monotonic() - self.t0_mono) * 1000),
            **data,
        })

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.status = status
        self.error = error
        self.mark("finish", status=status)

    def to_dict(self) -> dict:
        marks = {m["stage"]: m for m in self.marks}
        metrics = {}

        def diff(stage, base):
            a = marks.get(stage)
            b = marks.get(base)
            if a and b:
                return int(a.get("wall_ms", 0) - b.get("wall_ms", 0))
            return None

        def first_diff(stage, base):
            """同一请求有多个片段时，取第一次到达的 stage。"""
            base_mark = marks.get(base)
            if not base_mark:
                return None
            vals = [m.get("wall_ms") for m in self.marks if m.get("stage") == stage]
            if not vals:
                return None
            return int(min(vals) - base_mark.get("wall_ms", 0))

        speech = marks.get("speech_end")
        if speech:
            metrics["vad_tail_ms"] = diff("vad_end", "speech_end")
            metrics["speech_to_asr_ms"] = diff("asr_end", "speech_end")
            metrics["speech_to_intent_ms"] = diff("intent_end", "speech_end")
            metrics["speech_to_first_token_ms"] = diff("first_token", "speech_end")
            metrics["speech_to_first_segment_ms"] = diff("first_segment", "speech_end")
            metrics["speech_to_tts_ready_ms"] = first_diff("tts_ready", "speech_end")
            metrics["speech_to_play_start_ms"] = first_diff("play_start", "speech_end")
        answer = marks.get("answer")
        req = marks.get("request_start")
        if answer:
            base_wall = req.get("wall_ms") if req else self.t0_wall_ms
            metrics["request_to_answer_ms"] = int(answer.get("wall_ms", 0) - base_wall)
        ready = marks.get("tts_ready")
        if ready and isinstance(ready.get("value_ms"), (int, float)):
            metrics["tts_gen_ms"] = ready["value_ms"]
        for name in ("prefill_ms", "decode_ms", "tts_gen_ms"):
            for mark in self.marks:
                if mark["stage"] == name:
                    metrics[name] = mark.get("value_ms")
                    break
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "start_wall_ms": self.t0_wall_ms,
            "duration_ms": int((time.monotonic() - self.t0_mono) * 1000),
            "meta": self.meta,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "marks": self.marks,
        }


def _pct(values, p):
    if not values:
        return None
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


class TraceStore:
    def __init__(self, path: Path, max_records: int = 500):
        self.path = Path(path)
        self.max_records = max_records
        self._records: deque = deque(maxlen=max_records)
        self._lock = threading.RLock()
        self._active: dict[str, RequestTrace] = {}
        self._recent: dict[str, tuple[RequestTrace, float]] = {}
        self._listeners = []

    def start(self, trace_id: str, kind: str, **meta) -> RequestTrace:
        trace = RequestTrace(trace_id, kind, **meta)
        with self._lock:
            self._active[trace_id] = trace
        return trace

    def mark(self, trace_id: str, stage: str, **data) -> None:
        now = time.monotonic()
        with self._lock:
            trace = self._active.get(trace_id)
            finished = False
            if trace is None:
                item = self._recent.get(trace_id)
                if item is None or now - item[1] > 120:
                    self._recent.pop(trace_id, None)
                    return
                trace = item[0]
                finished = True
        if trace is not None:
            trace.mark(stage, **data)
            if finished:
                snapshot = trace.to_dict()
                with self._lock:
                    for idx, row in enumerate(self._records):
                        if row.get("trace_id") == trace_id:
                            self._records[idx] = snapshot
                            break
                self._write(snapshot)

    def finish(self, trace_id: str, status: str = "ok", error: str = "") -> Optional[dict]:
        with self._lock:
            trace = self._active.pop(trace_id, None)
            if trace is not None:
                self._recent[trace_id] = (trace, time.monotonic())
        if trace is None:
            return None
        trace.finish(status, error)
        data = trace.to_dict()
        with self._lock:
            self._records.append(data)
            # 清理过期 recent，避免无限增长
            if len(self._recent) > 200:
                now = time.monotonic()
                self._recent = {k: v for k, v in self._recent.items() if now - v[1] <= 120}
        self._write(data)
        return data

    def subscribe(self, callback):
        with self._lock:
            if callback not in self._listeners:
                if len(self._listeners) >= 8:
                    raise ValueError("trace listener limit")
                self._listeners.append(callback)

    def unsubscribe(self, callback):
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > 4 * 1024 * 1024:
                backup = self.path.with_suffix(".jsonl.1")
                self.path.replace(backup)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except OSError:
            pass
        with self._lock:
            callbacks = list(self._listeners)
        for callback in callbacks:
            try:
                callback(data)
            except Exception:
                # Observers must not interrupt generation/playback; batch logger records its own failures.
                import logging
                logging.getLogger("cm5").exception("trace observer failed")

    def records(self, limit: int = 100, kind: Optional[str] = None) -> list[dict]:
        with self._lock:
            rows = list(self._records)
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        return rows[-limit:]

    def summary(self) -> dict:
        with self._lock:
            rows = list(self._records)
        by_kind: dict[str, list[dict]] = {}
        for row in rows:
            by_kind.setdefault(row.get("kind", "unknown"), []).append(row)
        out = {}
        for kind, items in by_kind.items():
            metrics = {}
            for name in (
                "speech_to_asr_ms", "speech_to_intent_ms", "speech_to_first_token_ms",
                "speech_to_first_segment_ms", "speech_to_tts_ready_ms", "speech_to_play_start_ms",
                "request_to_answer_ms", "prefill_ms", "decode_ms", "tts_gen_ms", "vad_tail_ms",
            ):
                vals = [i.get("metrics", {}).get(name) for i in items]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    metrics[name] = {
                        "n": len(vals), "p50": _pct(vals, 0.50),
                        "p95": _pct(vals, 0.95), "p99": _pct(vals, 0.99),
                        "min": min(vals), "max": max(vals),
                    }
            out[kind] = {
                "n": len(items),
                "ok": sum(1 for i in items if i.get("status") == "ok"),
                "failed": sum(1 for i in items if i.get("status") != "ok"),
                "metrics": metrics,
            }
        return out

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)
