# -*- coding: utf-8 -*-
"""服务端持续引导循环：只保留最新帧，独立于浏览器轮询运行。

浏览器只负责送帧和显示；CM5 真机由 Picamera2 采集线程直接 submit。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

import guide
from output import OUTPUT, PRIORITY_CONFIRM, PRIORITY_DANGER, PRIORITY_GUIDE

log = logging.getLogger("cm5")


@dataclass
class FramePacket:
    image: Image.Image
    frame_id: str
    received_mono: float
    captured_mono: float
    source: str
    seq: int
    captured_at: str = ""
    meta: dict = field(default_factory=dict)


class GuideRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._latest: Optional[FramePacket] = None
        self._seq = 0
        self._mode = "browser"
        self._needs_frames = True
        self._sm = guide.GuideStateMachine()
        self._decision: dict = {
            "state": "unavailable",
            "direction": "unavailable",
            "reason": "no_frame",
            "raw_direction": "unavailable",
            "raw_reason": "no_frame",
            "speech": "",
            "seq": 0,
        }
        self._observation: dict = {}
        self._last_frame: Optional[FramePacket] = None
        self._last_frame_mono = 0.0
        self._heartbeat = time.monotonic()
        self._stuck = False
        self._stuck_since = 0.0
        self._had_frame = False
        self._last_stale_publish = 0.0
        self._stats = {
            "processed": 0,
            "dropped": 0,
            "stale": 0,
            "errors": 0,
            "started_at": None,
            "last_error": None,
            "last_inference_ms": None,
            "inference_samples": [],
            "age_samples": [],
        }

    # ---------- 生命周期 ----------
    def start(self, source: str = "browser") -> dict:
        if not guide.available():
            raise RuntimeError("引导模型不可用：" + (guide.load_error() or str(guide.MODEL_PATH)))
        with self._lock:
            if self._running:
                return self.status()
            self._mode = source
            self._needs_frames = source != "device"
            self._running = True
            self._stop_event.clear()
            self._sm.reset()
            self._decision = {
                "state": "unavailable",
                "direction": "unavailable",
                "reason": "no_frame",
                "raw_direction": "unavailable",
                "raw_reason": "no_frame",
                "speech": "",
                "seq": 0,
            }
            self._observation = {}
            self._last_frame = None
            self._last_frame_mono = 0.0
            self._had_frame = False
            self._heartbeat = time.monotonic()
            self._stuck = False
            self._stats.update(
                processed=0,
                dropped=0,
                stale=0,
                errors=0,
                started_at=time.monotonic(),
                last_error=None,
                inference_samples=[],
                age_samples=[],
            )
            self._thread = threading.Thread(target=self._loop, name="guide-loop", daemon=True)
            self._watchdog = threading.Thread(target=self._watchdog_loop, name="guide-watchdog", daemon=True)
            self._thread.start()
            self._watchdog.start()
        return self.status()

    def stop(self, reason: str = "manual", announce: bool = True) -> dict:
        with self._lock:
            if not self._running:
                return self.status()
            self._running = False
            self._stop_event.set()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        if thread is None or not thread.is_alive():
            guide.unload()
        with self._lock:
            self._decision = {
                "state": "unavailable",
                "direction": "unavailable",
                "reason": "stopped",
                "raw_direction": "unavailable",
                "raw_reason": "stopped",
                "speech": "",
                "seq": self._sm.seq,
            }
            self._observation = {}
            self._thread = None
            self._watchdog = None
        if announce:
            OUTPUT.push(
                "引导已停止", priority=PRIORITY_CONFIRM,
                source="guide", reason="stopped", ttl=2.0,
                dedupe_key="guide-stopped",
            )
        return self.status()

    def submit_frame(
        self,
        image: Image.Image,
        frame_id: str = "",
        captured_at: str = "",
        source: str = "browser",
        captured_mono: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> dict:
        now = time.monotonic()
        fid = (frame_id or "").strip() or f"auto-{time.time_ns()}"
        with self._cond:
            self._seq += 1
            if self._latest is not None:
                self._stats["dropped"] += 1
            packet = FramePacket(
                image=image,
                frame_id=fid,
                received_mono=now,
                captured_mono=captured_mono if captured_mono is not None else now,
                source=source,
                seq=self._seq,
                captured_at=captured_at,
                meta=meta or {},
            )
            self._latest = packet
            self._cond.notify()
        return {
            "accepted": True,
            "seq": self._seq,
            "frame_id": fid,
            "running": self._running,
        }

    # ---------- 主循环 ----------
    def _loop(self) -> None:
        period = 1.0 / max(1.0, guide.TARGET_FPS)
        while not self._stop_event.is_set():
            t_loop = time.monotonic()
            with self._lock:
                self._heartbeat = time.monotonic()
                packet = self._latest
                self._latest = None
            now = time.monotonic()
            if packet is None:
                if self._had_frame and now - self._last_frame_mono > guide.FRAME_MAX_AGE:
                    if now - self._last_stale_publish >= 1.0:
                        self._publish_unavailable("stale", now)
                        self._last_stale_publish = now
                time.sleep(min(0.05, max(0.01, guide.FRAME_MAX_AGE / 4)))
                continue

            age = now - packet.received_mono
            with self._lock:
                samples = self._stats["age_samples"]
                samples.append(int(age * 1000))
                if len(samples) > 200:
                    del samples[:-200]
            if age > guide.FRAME_MAX_AGE:
                self._stats["stale"] += 1
                if now - self._last_stale_publish >= 1.0:
                    self._publish_unavailable("stale", now)
                    self._last_stale_publish = now
                continue

            if packet is self._last_frame or packet.seq == (self._last_frame.seq if self._last_frame else -1):
                continue
            self._last_frame = packet
            self._last_frame_mono = packet.received_mono
            self._had_frame = True
            try:
                t_work = time.monotonic()
                obs = guide.analyze_image(
                    packet.image,
                    frame={
                        "frame_id": packet.frame_id,
                        "source": packet.source,
                        "captured_at": packet.captured_at,
                        "age_ms": int(age * 1000),
                        "seq": packet.seq,
                    },
                )
                decision = self._sm.update(obs, time.monotonic())
                self._publish(obs, decision, packet)
            except Exception as exc:  # noqa: BLE001
                self._stats["errors"] += 1
                self._stats["last_error"] = str(exc)
                self._publish_unavailable("error", time.monotonic())
            work_ms = int((time.monotonic() - t_work) * 1000)
            self._stats["last_work_ms"] = work_ms
            elapsed = time.monotonic() - t_loop
            delay = period - elapsed
            if delay > 0:
                time.sleep(min(delay, period))
            self._stats["last_iter_ms"] = int((time.monotonic() - t_loop) * 1000)

    def _publish(self, obs: dict, decision: dict, packet: FramePacket) -> None:
        now = time.monotonic()
        if not self._running or self._stop_event.is_set():
            return
        inference_ms = (obs.get("timing") or {}).get("inference_ms")
        with self._lock:
            self._heartbeat = now
            self._stuck = False
            self._stats["processed"] += 1
            self._stats["last_inference_ms"] = inference_ms
            if isinstance(inference_ms, (int, float)):
                samples = self._stats["inference_samples"]
                samples.append(int(inference_ms))
                if len(samples) > 120:
                    del samples[:-120]
            self._observation = obs
            decision = dict(decision)
            decision["frame"] = obs.get("frame", {})
            decision["quality"] = obs.get("quality", {})
            decision["sensor"] = obs.get("sensor", {})
            decision["inference_ms"] = inference_ms
            decision["updated_at"] = time.time()
            self._decision = decision

        if decision.get("changed") or self._stats["processed"] % 10 == 1:
            log.info(
                "guide state=%s reason=%s raw=%s/%s frame=%s age=%sms infer=%sms quality=%s",
                decision.get("state"), decision.get("reason"),
                decision.get("raw_direction"), decision.get("raw_reason"),
                packet.frame_id, obs.get("frame", {}).get("age_ms"), inference_ms,
                (obs.get("quality") or {}).get("reasons"),
            )

        speech = decision.get("speech") or ""
        # 只在状态变化时播报：重复播报同一 no_seed/危险状态会让 Piper+aplay
        # 持续抢占 CPU，实测把引导吞吐从 3.6 fps 压到 1.4 fps。
        if speech and decision.get("changed"):
            reason = decision.get("reason") or ""
            priority = PRIORITY_DANGER if (decision.get("urgent") and reason in guide.DANGER_REASONS | {"stale", "model_stuck", "error"}) else PRIORITY_GUIDE
            OUTPUT.push(
                speech,
                priority=priority,
                source="guide",
                reason=reason,
                ttl=5.0 if priority == PRIORITY_DANGER else 2.5,
                dedupe_key=f"guide:{decision.get('state')}:{reason}",
                meta={
                    "direction": decision.get("state"),
                    "raw_direction": decision.get("raw_direction"),
                    "frame_id": packet.frame_id,
                },
            )

    def _publish_unavailable(self, reason: str, now: float) -> None:
        obs = {
            "direction": "unavailable",
            "reason": reason,
            "raw_direction": "unavailable",
            "raw_reason": reason,
            "metrics": {},
            "quality": {"ok": False, "reasons": [reason]},
            "sensor": {"depth": {"available": False}, "imu": {"available": False}},
            "timing": {},
            "frame": {},
        }
        decision = self._sm.update(obs, now)
        with self._lock:
            self._observation = obs
            decision = dict(decision)
            decision["updated_at"] = time.time()
            self._decision = decision
        if reason in ("stale", "model_stuck", "error") and self._had_frame:
            OUTPUT.push(
                guide.REASON_TEXT.get(reason, "引导异常，请停下"),
                priority=PRIORITY_DANGER,
                source="guide",
                reason=reason,
                ttl=5.0,
                dedupe_key=f"guide:{reason}",
            )

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.5)
            now = time.monotonic()
            with self._lock:
                if not self._running:
                    continue
                lag = now - self._heartbeat
                if lag > guide.WATCHDOG_SECONDS and not self._stuck:
                    self._stuck = True
                    self._stuck_since = now
            if lag > guide.WATCHDOG_SECONDS and self._stuck and now - self._stuck_since < 0.6:
                self._publish_unavailable("model_stuck", now)

    # ---------- 查询 ----------
    def status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            decision = dict(self._decision)
            obs = dict(self._observation)
            last = self._last_frame
            frame_age_ms = int((now - self._last_frame_mono) * 1000) if self._last_frame_mono else None
            samples = list(self._stats["inference_samples"])
            p50 = p95 = None
            if samples:
                arr = sorted(samples)
                p50 = arr[int(len(arr) * 0.5) - 1] if len(arr) > 1 else arr[0]
                p95 = arr[min(len(arr) - 1, int(len(arr) * 0.95))]
            age_samples = list(self._stats.get("age_samples") or [])
            age_p50 = age_p95 = None
            if age_samples:
                arr = sorted(age_samples)
                age_p50 = arr[int(len(arr) * 0.5) - 1] if len(arr) > 1 else arr[0]
                age_p95 = arr[min(len(arr) - 1, int(len(arr) * 0.95))]
            stats = {
                "processed": self._stats["processed"],
                "dropped": self._stats["dropped"],
                "stale": self._stats["stale"],
                "errors": self._stats["errors"],
                "last_error": self._stats["last_error"],
                "last_inference_ms": self._stats["last_inference_ms"],
                "last_work_ms": self._stats.get("last_work_ms"),
                "last_iter_ms": self._stats.get("last_iter_ms"),
                "inference_p50_ms": p50,
                "inference_p95_ms": p95,
                "age_p50_ms": age_p50,
                "age_p95_ms": age_p95,
                "target_fps": guide.TARGET_FPS,
                "frame_max_age_s": guide.FRAME_MAX_AGE,
                "watchdog_s": guide.WATCHDOG_SECONDS,
                "uptime_s": int(now - self._stats["started_at"]) if self._stats["started_at"] else 0,
            }
            return {
                "running": self._running,
                "mode": self._mode,
                "needs_frames": self._needs_frames,
                "frame_id": last.frame_id if last else "",
                "frame_source": last.source if last else "",
                "frame_age_ms": frame_age_ms,
                "stuck": self._stuck,
                "seq": self._sm.seq,
                "decision": decision,
                "metrics": obs.get("metrics", {}),
                "quality": obs.get("quality", {}),
                "sensor": obs.get("sensor", {}),
                "model": guide.status(),
                "stats": stats,
                "speech": OUTPUT.stats(),
            }


GUIDE = GuideRuntime()


def status_json() -> str:
    return json.dumps(GUIDE.status(), ensure_ascii=False)
