# -*- coding: utf-8 -*-
"""interaction.v0.1 主控适配器：唯一接入口，管理握手、时钟、时效、取消与事件。

本模块只负责协议与事实校验，不计算通行、距离或空间判断。
没有配置 MAIN_CONTROL_URL 时明确返回 unavailable，不假装已经接入真实主控。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import requests
from PIL import Image

SCHEMA_VERSION = "interaction.v0.1"
ALLOWED_ACTIONS = {
    "start_guidance", "pause_guidance", "resume_guidance",
    "end_guidance", "repeat_guidance",
}
PRIORITY_NAMES = {0: "emergency", 1: "navigation", 2: "information", 3: "information"}
ERROR_CODES = {
    "DEPENDENCY_UNAVAILABLE", "NO_FRAME", "STALE_FRAME", "INVALID_FRAME_REF",
    "UNSUPPORTED_VERSION", "CLOCK_UNVERIFIED", "STALE_STATE", "INVALID_TARGET",
    "INVALID_PAYLOAD", "BUSY", "DEADLINE_EXCEEDED", "CANCELLED",
    "OUTPUT_UNAVAILABLE",
}


def now_mono_ms() -> int:
    return int(time.monotonic() * 1000)


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _error(code: str, message: str, retryable: bool = False) -> dict:
    if code not in ERROR_CODES:
        code = "INVALID_PAYLOAD"
    return {"code": code, "message": message, "retryable": bool(retryable)}


class MainControlAdapter:
    def __init__(
        self,
        url: str = "",
        *,
        rpc_path: str = "/interaction/rpc",
        events_path: str = "/interaction/events",
        allow_fixture: Optional[bool] = None,
        asset_root: str = "",
        instance_id: str = "",
    ):
        self.url = (url or os.environ.get("MAIN_CONTROL_URL", "")).strip().rstrip("/")
        self.rpc_path = os.environ.get("MAIN_CONTROL_RPC_PATH", rpc_path)
        self.events_path = os.environ.get("MAIN_CONTROL_EVENTS_PATH", events_path)
        self.allow_fixture = (
            os.environ.get("MAIN_CONTROL_ALLOW_FIXTURE", "0").strip().lower() in ("1", "true", "yes")
            if allow_fixture is None else bool(allow_fixture)
        )
        self.asset_root = Path(asset_root or os.environ.get("MAIN_CONTROL_ASSET_ROOT", "") or ".").resolve()
        self.instance_id = instance_id or f"interaction-{uuid.uuid4().hex[:10]}"
        self.clock_id: Optional[str] = None
        self.master_instance_id: Optional[str] = None
        self.evidence_mode: Optional[str] = None
        self.capabilities: Optional[dict] = None
        self.connected = False
        self.last_error = ""
        self.last_handshake_ms: Optional[int] = None
        self._lock = threading.RLock()
        self._scene: Optional[dict] = None
        self._scene_seq = -1
        self._scene_at_ms: Optional[int] = None
        self._frame_sources: dict[str, str] = {}
        self._seen_events: "OrderedDict[str, float]" = OrderedDict()
        self._event_seq = -1
        self._stop = threading.Event()
        self._event_thread: Optional[threading.Thread] = None
        self._on_event: Optional[Callable[[dict], None]] = None
        self._on_cancel: Optional[Callable[[dict], None]] = None
        self._on_state: Optional[Callable[[str, str], None]] = None
        self._auth = os.environ.get("MAIN_CONTROL_TOKEN", "").strip()

    # ---------- 公共消息 ----------
    def _producer(self) -> dict:
        return {"role": "interaction", "instance_id": self.instance_id}

    def _envelope(self, msg_type: str, data: dict, *, request_id: Optional[str] = None,
                  session_id: Optional[str] = None, deadline_ms: Optional[int] = None) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": msg_type,
            "message_id": f"msg-{uuid.uuid4().hex[:12]}",
            "reply_to": None,
            "request_id": request_id,
            "session_id": session_id,
            "producer": self._producer(),
            "clock_id": self.clock_id,
            "timestamp_mono_ms": now_mono_ms(),
            "deadline_mono_ms": deadline_ms,
            "evidence_mode": "live",
            "data": data,
        }

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._auth}"} if self._auth else {}

    def _post(self, message: dict, timeout: float = 5.0) -> dict:
        """返回真实响应或本地构造的 unavailable 错误；不伪造主控成功。"""
        if not self.url:
            return {"status": "unavailable", "error": _error("DEPENDENCY_UNAVAILABLE", "MAIN_CONTROL_URL 未配置")}
        try:
            resp = requests.post(self.url + self.rpc_path, json=message, headers=self._headers(), timeout=timeout)
            if resp.status_code != 200:
                return {"status": "unavailable", "error": _error(
                    "DEPENDENCY_UNAVAILABLE", f"主控 HTTP {resp.status_code}", retryable=True)}
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": _error(
                "DEPENDENCY_UNAVAILABLE", f"主控请求失败：{exc}", retryable=True)}
        return self._validate_response(payload, message)

    def _validate_response(self, payload: dict, request: dict) -> dict:
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            return {"status": "error", "error": _error("UNSUPPORTED_VERSION", "主控响应合同版本不匹配")}
        if payload.get("reply_to") != request.get("message_id") or payload.get("request_id") != request.get("request_id"):
            return {"status": "error", "error": _error("INVALID_PAYLOAD", "主控响应缺少对应 reply_to/request_id")}
        producer = payload.get("producer") or {}
        role = producer.get("role")
        allowed_roles = {"main_control"} | ({"test_fixture"} if self.allow_fixture else set())
        if role not in allowed_roles:
            return {"status": "error", "error": _error("INVALID_TARGET", f"不接受的生产者角色：{role}")}
        mode = payload.get("evidence_mode")
        if mode not in ("live", "replay", "synthetic"):
            return {"status": "error", "error": _error("INVALID_PAYLOAD", "evidence_mode 非法")}
        if role == "test_fixture" and mode == "live":
            return {"status": "error", "error": _error("INVALID_PAYLOAD", "测试替身不得声明 live")}
        if role == "main_control" and mode != "live":
            return {"status": "error", "error": _error("INVALID_PAYLOAD", "真实主控响应对应 live")}
        deadline = request.get("deadline_mono_ms")
        if deadline is not None and now_mono_ms() > int(deadline):
            return {"status": "error", "error": _error("DEADLINE_EXCEEDED", "响应已超过请求期限", retryable=True)}
        if payload.get("status") != "ok" and not payload.get("error"):
            return {"status": "error", "error": _error("INVALID_PAYLOAD", "失败响应缺少 error")}
        return payload

    def _mark(self, error: dict) -> None:
        self.last_error = f"{error.get('code')}: {error.get('message')}"

    # ---------- 握手 ----------
    def handshake(self) -> dict:
        req_id = f"req-cap-{uuid.uuid4().hex[:10]}"
        message = self._envelope(
            "capabilities.get", {"want": ["frame_provider", "scene_state", "control", "target", "guidance"]},
            request_id=req_id, deadline_ms=now_mono_ms() + 5000,
        )
        result = self._post(message, timeout=5.0)
        if result.get("status") != "ok":
            self.connected = False
            self._mark(result.get("error") or {})
            self._notify_state("disconnected", self.last_error)
            return result
        data = result.get("data") or {}
        producer = result.get("producer") or {}
        new_instance = producer.get("instance_id") or ""
        new_clock = result.get("clock_id")
        if not new_clock:
            self.connected = False
            return {"status": "error", "error": _error("CLOCK_UNVERIFIED", "握手未返回 clock_id")}
        old_instance, old_clock = self.master_instance_id, self.clock_id
        if old_instance and (old_instance != new_instance or old_clock != new_clock):
            self._invalidate("master_restart")
        self.master_instance_id = new_instance
        self.clock_id = new_clock
        self.evidence_mode = result.get("evidence_mode")
        self.capabilities = data
        self.connected = True
        self.last_error = ""
        self.last_handshake_ms = now_mono_ms()
        self._notify_state("connected", "")
        return result

    def _invalidate(self, reason: str) -> None:
        with self._lock:
            had = self._scene is not None
            self._scene = None
            self._scene_seq = -1
            self._scene_at_ms = None
        if had:
            self._notify_state("invalidated", reason)

    def invalidate(self, reason: str) -> None:
        self._invalidate(reason)

    # ---------- FrameProvider ----------
    def acquire_frame(self, *, mode: str = "latest", purpose: str = "vqa",
                      frame_id: Optional[str] = None, max_age_ms: int = 800,
                      max_side_px: int = 512, roi_xyxy_px: Optional[list] = None,
                      session_id: Optional[str] = None, timeout: float = 8.0) -> dict:
        if mode not in ("latest", "pinned"):
            raise ValueError("frame mode 必须是 latest 或 pinned")
        if purpose not in ("vqa", "ocr", "compare"):
            raise ValueError("purpose 必须是 vqa/ocr/compare")
        if mode == "pinned" and not frame_id:
            raise ValueError("pinned 取图缺少 frame_id")
        req_id = f"req-frame-{uuid.uuid4().hex[:10]}"
        message = self._envelope(
            "frame.acquire",
            {"mode": mode, "purpose": purpose, "frame_id": frame_id if mode == "pinned" else None,
             "max_age_ms": int(max_age_ms), "max_side_px": int(max_side_px),
             "roi_xyxy_px": roi_xyxy_px},
            request_id=req_id, session_id=session_id,
            deadline_ms=now_mono_ms() + int(timeout * 1000),
        )
        result = self._post(message, timeout=timeout)
        if result.get("status") != "ok":
            raise RuntimeError((result.get("error") or {}).get("message") or "主控取图不可用")
        data = result.get("data") or {}
        fid = str(data.get("frame_id") or "")
        if not fid:
            raise RuntimeError("主控取图响应缺少 frame_id")
        if data.get("source_kind") not in ("camera", "upload", "fixture"):
            raise RuntimeError("主控取图 source_kind 非法")
        if data.get("color_space") not in ("RGB", None):
            raise RuntimeError("主控取图 color_space 非 RGB；本适配器不做静默通道交换")
        asset = data.get("asset") or {}
        if asset.get("transport") != "file" or not asset.get("path"):
            raise RuntimeError("当前只支持 file 传输的帧资产")
        path = Path(str(asset["path"]))
        if not path.is_absolute():
            path = (self.asset_root / path).resolve()
        until = asset.get("available_until_mono_ms")
        if until is not None and now_mono_ms() > int(until):
            raise RuntimeError("帧资产租期已过")
        sha = str(asset.get("sha256") or "")
        if not _hex64(sha):
            raise RuntimeError("帧资产缺少有效 sha256")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != sha.lower():
            raise RuntimeError("帧资产 sha256 不匹配")
        with self._lock:
            old = self._frame_sources.get(fid)
            if old and old != actual:
                raise RuntimeError(f"同一 frame_id 对应了不同内容：{fid}")
            self._frame_sources[fid] = actual
            if len(self._frame_sources) > 64:
                self._frame_sources.pop(next(iter(self._frame_sources)))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        captured_mono = data.get("captured_mono_ms")
        captured_at = ""
        if captured_mono is not None and self.clock_id:
            age_ms = max(0, now_mono_ms() - int(captured_mono))
            captured_at = (datetime.now(timezone.utc) - timedelta(milliseconds=age_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "image": img,
            "frame_id": fid,
            "captured_mono_ms": captured_mono,
            "captured_at": captured_at,
            "is_historical": bool(data.get("is_historical")),
            "source_kind": data.get("source_kind"),
            "quality": data.get("quality") or {},
            "evidence_mode": result.get("evidence_mode"),
            "width_px": data.get("width_px"),
            "height_px": data.get("height_px"),
            "asset": asset,
        }

    # ---------- SceneState ----------
    def query_scene(self, *, fields: Optional[list] = None, track_id: Optional[str] = None,
                    session_id: Optional[str] = None, timeout: float = 4.0) -> dict:
        req_id = f"req-scene-{uuid.uuid4().hex[:10]}"
        message = self._envelope(
            "scene.query", {"fields": fields, "track_id": track_id},
            request_id=req_id, session_id=session_id,
            deadline_ms=now_mono_ms() + int(timeout * 1000),
        )
        result = self._post(message, timeout=timeout)
        if result.get("status") != "ok":
            raise RuntimeError((result.get("error") or {}).get("message") or "主控场景状态不可用")
        data = result.get("data") or {}
        for key in ("state_id", "sequence", "observed_mono_ms", "valid_until_mono_ms",
                    "coordinate_frame", "navigation", "sensor_status"):
            if key not in data:
                raise RuntimeError(f"场景状态缺少字段：{key}")
        seq = int(data.get("sequence"))
        with self._lock:
            if self._scene is not None and seq <= self._scene_seq:
                if seq < self._scene_seq:
                    raise RuntimeError("主控序号倒退，拒绝旧状态")
            self._scene_seq = max(self._scene_seq, seq)
            self._scene = data
            self._scene_at_ms = now_mono_ms()
        stale = now_mono_ms() > int(data["valid_until_mono_ms"])
        return {**data, "_stale": stale, "_sequence": seq,
                "_producer": result.get("producer"), "_evidence_mode": result.get("evidence_mode")}

    def last_scene(self) -> Optional[dict]:
        with self._lock:
            if self._scene is None:
                return None
            stale = self._scene_at_ms is None or now_mono_ms() > int(self._scene["valid_until_mono_ms"])
            return {**self._scene, "_stale": stale,
                    "_age_ms": now_mono_ms() - (self._scene_at_ms or now_mono_ms())}

    def guidance_active(self) -> bool:
        scene = self.last_scene()
        if not scene or scene.get("_stale"):
            return False
        nav = scene.get("navigation") or {}
        return str(nav.get("mode") or "") not in ("", "idle", "stopped", "unknown", "paused")

    def guidance_paused(self) -> bool:
        scene = self.last_scene()
        if not scene or scene.get("_stale"):
            return False
        return str((scene.get("navigation") or {}).get("mode") or "") == "paused"

    # ---------- ControlIntent ----------
    def control(self, action: str, *, utterance_id: str, final_transcript: str,
                target_ref: Optional[str] = None, user_confirmed: bool = False,
                session_id: Optional[str] = None, timeout: float = 6.0) -> dict:
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"不允许的控制动作：{action}")
        if not utterance_id or not final_transcript.strip():
            raise ValueError("控制意图只能基于最终有效话语")
        req_id = f"req-control-{uuid.uuid4().hex[:10]}"
        message = self._envelope(
            "control.intent",
            {"action": action, "utterance_id": utterance_id,
             "final_transcript": final_transcript.strip(), "target_ref": target_ref,
             "user_confirmed": bool(user_confirmed)},
            request_id=req_id, session_id=session_id,
            deadline_ms=now_mono_ms() + int(timeout * 1000),
        )
        result = self._post(message, timeout=timeout)
        if result.get("status") != "ok":
            raise RuntimeError((result.get("error") or {}).get("message") or "主控控制接口不可用")
        data = result.get("data") or {}
        for key in ("accepted", "operation_state", "actual_mode", "reason_code", "state_id"):
            if key not in data:
                raise RuntimeError(f"控制结果缺少字段：{key}")
        if data.get("operation_state") not in ("pending", "completed", "rejected"):
            raise RuntimeError("控制结果 operation_state 非法")
        return {**data, "_request_id": req_id, "_evidence_mode": result.get("evidence_mode")}

    def target_observation(self, *, visual_target_id: str, frame_id: str, roi_xyxy_px: list,
                           label: str, text: str, source: str, user_selected: bool,
                           confidence: Optional[float] = None,
                           session_id: Optional[str] = None) -> dict:
        req_id = f"req-target-{uuid.uuid4().hex[:10]}"
        message = self._envelope(
            "target.observation",
            {"visual_target_id": visual_target_id, "frame_id": frame_id,
             "roi_xyxy_px": roi_xyxy_px, "label": label, "text": text,
             "source": source, "user_selected": bool(user_selected),
             "confidence": confidence, "confidence_calibrated": False},
            request_id=req_id, session_id=session_id,
            deadline_ms=now_mono_ms() + 5000,
        )
        result = self._post(message, timeout=5.0)
        if result.get("status") != "ok":
            raise RuntimeError((result.get("error") or {}).get("message") or "主控目标观测接口不可用")
        return {**(result.get("data") or {}), "_request_id": req_id}

    # ---------- PlaybackStatus ----------
    def send_playback_status(self, payload: dict, state: str, *, start_basis: Optional[str] = None,
                             reason_code: str = "") -> None:
        meta = (payload or {}).get("meta") or {}
        source_event = meta.get("event_id")
        source_request = meta.get("request_id")
        generation = meta.get("generation")
        data = {
            "source_event_id": source_event,
            "source_request_id": source_request,
            "utterance_id": (payload or {}).get("utterance_id") or "",
            "segment_index": max(0, int((payload or {}).get("sequence") or 1) - 1),
            "generation": generation,
            "priority": PRIORITY_NAMES.get(int((payload or {}).get("priority", 3)), "information"),
            "state": state,
            "occurred_mono_ms": now_mono_ms(),
            "start_basis": start_basis,
            "reason_code": reason_code or None,
        }
        message = self._envelope("playback.status", data, request_id=f"req-play-{uuid.uuid4().hex[:10]}")
        threading.Thread(target=lambda: self._post(message, timeout=2.0), daemon=True).start()

    # ---------- GuidanceEvent 事件通道 ----------
    def start_events(self, on_event: Callable[[dict], None],
                     on_cancel: Callable[[dict], None],
                     on_state: Callable[[str, str], None] | None = None) -> bool:
        if not self.url:
            return False
        if self._event_thread and self._event_thread.is_alive():
            return True
        self._on_event = on_event
        self._on_cancel = on_cancel
        self._on_state = on_state
        self._stop.clear()
        self._event_thread = threading.Thread(target=self._event_loop, name="main-control-events", daemon=True)
        self._event_thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=2.0)

    def _notify_state(self, state: str, detail: str) -> None:
        cb = self._on_state
        if cb:
            try:
                cb(state, detail)
            except Exception:  # noqa: BLE001
                pass

    def _event_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                handshake = self.handshake()
                if handshake.get("status") != "ok":
                    raise RuntimeError((handshake.get("error") or {}).get("message") or "握手失败")
                resp = requests.get(self.url + self.events_path, headers=self._headers(),
                                    stream=True, timeout=(5, 90))
                if resp.status_code != 200:
                    raise RuntimeError(f"事件通道 HTTP {resp.status_code}")
                backoff = 1.0
                for line in resp.iter_lines(decode_unicode=True):
                    if self._stop.is_set():
                        break
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk:
                        continue
                    try:
                        event = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    self._handle_event(event)
                resp.close()
            except Exception as exc:  # noqa: BLE001
                self.connected = False
                self.last_error = f"事件通道中断：{exc}"
                self._notify_state("disconnected", self.last_error)
            if self._stop.wait(backoff):
                break
            backoff = min(5.0, backoff * 1.6)

    def _handle_event(self, event: dict) -> None:
        if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
            return
        producer = event.get("producer") or {}
        role = producer.get("role")
        allowed_roles = {"main_control"} | ({"test_fixture"} if self.allow_fixture else set())
        if role not in allowed_roles:
            return
        if event.get("type") == "output.cancel":
            if self._on_cancel:
                self._on_cancel(event)
            return
        if event.get("type") != "guidance.event":
            return
        data = event.get("data") or {}
        eid = str(data.get("event_id") or "")
        if not eid or data.get("priority") not in ("emergency", "navigation", "information"):
            return
        text = (data.get("text") or "").strip()
        if not text:
            return
        valid_until = data.get("valid_until_mono_ms")
        if valid_until is None or now_mono_ms() > int(valid_until):
            return
        with self._lock:
            if eid in self._seen_events:
                return
            self._seen_events[eid] = time.monotonic()
            if len(self._seen_events) > 256:
                self._seen_events.popitem(last=False)
            seq = data.get("sequence")
            if isinstance(seq, int):
                if seq < self._event_seq:
                    return
                self._event_seq = seq
        if self._on_event:
            self._on_event(event)

    # ---------- 状态 ----------
    def status(self) -> dict:
        scene = self.last_scene()
        return {
            "configured": bool(self.url),
            "connected": self.connected,
            "url": self.url or None,
            "instance_id": self.instance_id,
            "master_instance_id": self.master_instance_id,
            "clock_id": self.clock_id,
            "evidence_mode": self.evidence_mode,
            "fixture_allowed": self.allow_fixture,
            "last_handshake_ms": self.last_handshake_ms,
            "last_error": self.last_error or None,
            "scene": {
                "state_id": scene.get("state_id"),
                "sequence": scene.get("sequence"),
                "stale": scene.get("_stale"),
                "valid_until_mono_ms": scene.get("valid_until_mono_ms"),
                "mode": (scene.get("navigation") or {}).get("mode"),
            } if scene else None,
        }


def format_scene_answer(scene: Optional[dict]) -> str:
    """只复述主控事实：不补距离、不把 invalid 写成 0，不做通行授权。"""
    if not scene:
        return "还没有拿到主控场景状态，不能判断能不能走。"
    if scene.get("_stale"):
        return "主控结果已过期，不能判断，请先停下。"
    nav = scene.get("navigation") or {}
    fact = (nav.get("speech_fact") or "").strip()
    parts = []
    if fact:
        parts.append(fact.rstrip("。"))
    else:
        mode = nav.get("mode") or "unknown"
        parts.append({
            "walking": "主控正在提供行走引导",
            "paused": "主控当前建议暂停",
            "stopped": "主控引导已停止",
            "idle": "主控当前没有行走任务",
        }.get(mode, "主控当前模式未知"))
        reason = nav.get("reason_code")
        if reason:
            parts.append(f"原因码 {reason}")
    measurement = scene.get("measurement") or {}
    if not measurement:
        objects = scene.get("objects") or []
        if objects:
            measurement = objects[0].get("measurement") or {}
    if measurement.get("valid") is True and measurement.get("distance_m") is not None:
        try:
            dist = float(measurement["distance_m"])
            if dist >= 0:
                parts.append("距离约不到 0.1 米" if dist < 0.05 else f"主控距离约 {dist:.1f} 米")
        except (TypeError, ValueError):
            parts.append("主控距离字段无法解析，不能使用")
    else:
        code = measurement.get("reason_code") or "UNKNOWN"
        parts.append(f"主控距离暂时测不准（{code}）")
    sensor = scene.get("sensor_status") or {}
    if sensor.get("depth") == "invalid":
        parts.append("深度当前无效")
    if sensor.get("rgb") == "invalid":
        parts.append("RGB 当前无效")
    return "；".join(parts) + "。"
