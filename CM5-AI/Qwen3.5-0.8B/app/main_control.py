# -*- coding: utf-8 -*-
"""interaction.v0.1 的 M0 请求入口；不含导航算法、默认地址或设备采集。

transport 是显式注入的同步 callable(request) -> response。它负责有界 I/O 超时；
适配器仍在返回后复查期限。正式传输、心跳、紧急事件与播放回执在 M1 接入。
"""

from collections import OrderedDict, deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

SCHEMA_VERSION = "interaction.v0.1"
ACTIONS = frozenset(("start_guidance", "pause_guidance", "resume_guidance", "end_guidance", "repeat_guidance"))
RESULT_TYPES = {"capabilities.get": "capabilities.result", "frame.acquire": "frame.result", "scene.query": "scene.result", "control.intent": "control.result", "target.observation": "target.result"}
CAPABILITIES = {"frame.acquire": "frame_provider", "scene.query": "scene_state", "control.intent": "control_intent", "target.observation": "target_observation"}


class ProtocolError(Exception):
    def __init__(self, code, message, status="rejected"):
        super().__init__(message)
        self.code, self.status = code, status


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _require(condition, message, code="INVALID_PAYLOAD"):
    if not condition:
        raise ProtocolError(code, message)


class MainControlAdapter:
    def __init__(self, transport=None, *, clock_id=None, host_boot_id=None,
                 evidence_mode="live", asset_roots=(), now_ms=None,
                 max_asset_bytes=32 * 1024 * 1024):
        if evidence_mode not in ("live", "synthetic", "replay"):
            raise ValueError("未知 evidence_mode")
        self.transport = transport
        self.clock_id = clock_id
        self.host_boot_id = host_boot_id
        self.evidence_mode = evidence_mode
        self.instance_id = "interaction-" + uuid.uuid4().hex
        self.asset_roots = tuple(Path(p).resolve() for p in asset_roots)
        self.max_asset_bytes = max_asset_bytes
        self.now_ms = now_ms or (lambda: time.monotonic_ns() / 1_000_000)
        self._lock = threading.RLock()
        self._peer = None
        self._capabilities = {}
        self._scene = None
        self._frames = OrderedDict()
        self._control_ids = OrderedDict()
        self._last_error = None
        self._epoch = 0

    def invalidate(self, reason="主控连接失效"):
        with self._lock:
            self._peer = None
            self._capabilities = {}
            self._scene = None
            self._frames.clear()
            self._epoch += 1
            self._last_error = {"code": "DEPENDENCY_UNAVAILABLE", "message": reason, "retryable": True}

    def status(self):
        """只读本地快照，不等待传输锁，不触发握手、设备或模型加载。"""
        peer, capabilities = self._peer, self._capabilities
        return {"schema_version": SCHEMA_VERSION, "connected": peer is not None,
                "transport_configured": self.transport is not None,
                "evidence_mode": self.evidence_mode,
                "main_instance_id": peer, "clock_id": self.clock_id,
                "capabilities": deepcopy(capabilities), "connection_epoch": self._epoch,
                "not_implemented": ["heartbeat", "target.observation", "guidance.event", "output.cancel", "playback.status"],
                "last_error": deepcopy(self._last_error)}

    def _request(self, kind, data, session_id, request_id, timeout_ms):
        _require(kind in RESULT_TYPES, "未知请求类型")
        now = self.now_ms()
        return {"schema_version": SCHEMA_VERSION, "type": kind, "message_id": uuid.uuid4().hex,
                "reply_to": None, "request_id": request_id if _text(request_id) else uuid.uuid4().hex,
                "session_id": session_id if _text(session_id) else None, "producer": {"role": "interaction", "instance_id": self.instance_id},
                "clock_id": self.clock_id, "timestamp_mono_ms": now,
                "deadline_mono_ms": now + timeout_ms if _number(timeout_ms) and timeout_ms > 0 else now, "evidence_mode": self.evidence_mode, "data": deepcopy(data)}

    def _failure(self, req, exc):
        error = {"code": exc.code, "message": str(exc), "retryable": exc.code in ("DEPENDENCY_UNAVAILABLE", "BUSY", "DEADLINE_EXCEEDED")}
        self._last_error = error
        return {"schema_version": SCHEMA_VERSION, "type": RESULT_TYPES[req["type"]],
                "message_id": uuid.uuid4().hex, "reply_to": req["message_id"],
                "request_id": req["request_id"], "session_id": req["session_id"],
                "producer": {"role": "interaction", "instance_id": self.instance_id},
                "clock_id": self.clock_id, "timestamp_mono_ms": self.now_ms(),
                "evidence_mode": self.evidence_mode, "status": exc.status, "error": error, "data": {}}

    def _exchange(self, req, handshake=False):
        if self.transport is None:
            raise ProtocolError("DEPENDENCY_UNAVAILABLE", "主控传输尚未配置；未启用独立引导或相机回退", "unavailable")
        try:
            result = deepcopy(self.transport(deepcopy(req)))
        except (TimeoutError, ConnectionError, OSError) as exc:
            self.invalidate(str(exc))
            raise ProtocolError("DEPENDENCY_UNAVAILABLE", "主控传输不可用", "unavailable") from exc
        _require(self.now_ms() <= req["deadline_mono_ms"], "响应超过请求期限", "DEADLINE_EXCEEDED")
        _require(isinstance(result, dict), "响应不是对象")
        _require(result.get("schema_version") == SCHEMA_VERSION, "不支持的合同版本", "UNSUPPORTED_VERSION")
        for key in ("message_id", "reply_to", "request_id", "session_id", "producer", "clock_id", "timestamp_mono_ms", "evidence_mode", "status", "error", "data"):
            _require(key in result, "响应缺少字段: " + key)
        _require(_text(result["message_id"]) and result.get("type") == RESULT_TYPES[req["type"]], "响应类型或身份无效")
        _require(all(result[k] == req[k] for k in ("request_id", "session_id")) and result["reply_to"] == req["message_id"], "响应与请求不匹配")
        producer = result["producer"]
        role = "main_control" if self.evidence_mode == "live" else "test_fixture"
        _require(isinstance(producer, dict) and producer.get("role") == role and _text(producer.get("instance_id")), "响应来源无效")
        _require(result["evidence_mode"] == self.evidence_mode, "禁止混用现场与测试证据")
        _require(_number(result["timestamp_mono_ms"]), "响应时间无效")
        if not handshake:
            if producer["instance_id"] != self._peer or result["clock_id"] != self.clock_id:
                self.invalidate("主控实例或时钟变化，必须重新握手")
                raise ProtocolError("CLOCK_UNVERIFIED", "主控实例或时钟变化，旧状态已失效")
            _require(req["timestamp_mono_ms"] <= result["timestamp_mono_ms"] <= self.now_ms(), "响应时间不在本次请求范围")
        _require(result["status"] in ("ok", "unavailable", "rejected", "error"), "响应状态无效")
        _require(isinstance(result["data"], dict), "响应data不是对象")
        if result["status"] == "ok":
            _require(result["error"] is None, "成功响应不得带错误")
        else:
            error = result["error"]
            _require(isinstance(error, dict) and _text(error.get("code")) and _text(error.get("message")) and isinstance(error.get("retryable"), bool), "失败响应缺少错误信息")
        return result

    def handshake(self, *, timeout_ms=500):
        return self._call("capabilities.get", {"supported_versions": [SCHEMA_VERSION],
                          "host_boot_id": self.host_boot_id, "clock_source": "monotonic",
                          "clock_id": self.clock_id}, timeout_ms=timeout_ms)

    def _accept_handshake(self, result):
        data = result["data"]
        versions = data.get("supported_versions")
        _require(isinstance(versions, list) and SCHEMA_VERSION in versions, "主控未声明支持合同版本", "UNSUPPORTED_VERSION")
        _require(_text(self.clock_id) and _text(self.host_boot_id) and
                 result["clock_id"] == self.clock_id and data.get("clock_id") == self.clock_id and
                 data.get("host_boot_id") == self.host_boot_id and data.get("clock_source") == "monotonic",
                 "未验证同主机单调时钟和启动标识", "CLOCK_UNVERIFIED")
        caps = data.get("capabilities")
        _require(isinstance(caps, dict) and all(isinstance(v, bool) for v in caps.values()), "能力声明无效")
        peer = result["producer"]["instance_id"]
        if peer != self._peer:
            self.invalidate("重新握手")
        self._peer, self._capabilities = peer, deepcopy(caps)
        self._last_error = None

    def _call(self, kind, data, *, session_id=None, request_id=None, timeout_ms=500):
        with self._lock:
            req = self._request(kind, data, session_id, request_id, timeout_ms)
            try:
                _require(_number(timeout_ms) and timeout_ms > 0, "timeout_ms 必须是正数")
                _require(session_id is None or _text(session_id), "session_id 无效")
                _require(request_id is None or _text(request_id), "request_id 无效")
                if kind == "scene.query":
                    _require(data["track_id"] is None or _text(data["track_id"]), "track_id 无效")
                    _require(data["fields"] is None or (isinstance(data["fields"], list) and all(_text(field) for field in data["fields"])), "fields 必须是字段名列表")
                if kind != "capabilities.get":
                    if self._peer is None:
                        hello = self.handshake(timeout_ms=timeout_ms)
                        if hello["status"] != "ok":
                            raise ProtocolError(hello["error"]["code"], hello["error"]["message"], hello["status"])
                    _require(self.now_ms() <= req["deadline_mono_ms"], "握手后请求已过期", "DEADLINE_EXCEEDED")
                    _require(self._capabilities.get(CAPABILITIES[kind]) is True, "主控未提供请求能力", "DEPENDENCY_UNAVAILABLE")
                result = self._exchange(req, handshake=kind == "capabilities.get")
                if result["status"] == "ok":
                    if kind == "capabilities.get":
                        self._accept_handshake(result)
                    elif kind == "scene.query":
                        self._validate_scene(result)
                    elif kind == "frame.acquire":
                        self._validate_frame(result, req)
                    elif kind == "control.intent":
                        state = result["data"]
                        _require(isinstance(state.get("accepted"), bool) and state.get("operation_state") in ("pending", "completed", "rejected") and all(k in state for k in ("actual_mode", "reason_code", "state_id")), "控制结果缺少实际执行状态")
                if result["status"] != "ok":
                    self._last_error = deepcopy(result["error"])
                return result
            except (TypeError, ValueError, KeyError) as exc:
                if kind == "capabilities.get":
                    self.invalidate("握手数据格式错误")
                return self._failure(req, ProtocolError("INVALID_PAYLOAD", "主控响应格式错误"))
            except ProtocolError as exc:
                if kind == "capabilities.get":
                    self.invalidate(str(exc))
                return self._failure(req, exc)

    def _validate_scene(self, result):
        data, now = result["data"], self.now_ms()
        _require(_text(data.get("state_id")) and type(data.get("sequence")) is int and data["sequence"] >= 0, "场景身份无效")
        _require(_number(data.get("observed_mono_ms")) and _number(data.get("valid_until_mono_ms")) and data["observed_mono_ms"] <= now < data["valid_until_mono_ms"], "场景已过期或观测时间无效", "STALE_STATE")
        _require(_text(data.get("coordinate_frame")) and isinstance(data.get("navigation"), dict) and isinstance(data.get("objects"), list) and isinstance(data.get("sensor_status"), dict), "场景字段不完整")
        nav = data["navigation"]
        _require(all(k in nav for k in ("mode", "decision", "reason_code", "speech_fact")), "主控导航事实不完整")
        if self._scene is not None:
            old = self._scene["data"]
            if data["sequence"] < old["sequence"] or (data["sequence"] == old["sequence"] and data != old):
                self.invalidate("场景序号倒退或同序号内容变化")
                raise ProtocolError("STALE_STATE", "场景序号无效，旧状态已撤销")
        for obj in data["objects"]:
            _require(isinstance(obj, dict) and _text(obj.get("track_id")), "空间对象缺少主控track_id")
            m = obj.get("measurement")
            if m is None:
                continue
            _require(isinstance(m, dict) and isinstance(m.get("valid"), bool) and _number(m.get("valid_until_mono_ms")), "测量有效性无效")
            if not m["valid"]:
                _require(m.get("distance_m") is None, "无效测量必须用null表示距离")
            else:
                _require(_number(m.get("distance_m")) and m["distance_m"] >= 0, "距离必须是非负米数")
                _require(now < m["valid_until_mono_ms"], "关联测量已过期", "STALE_STATE")
        self._scene = deepcopy(result)

    def scene_is_current(self, result):
        """消费场景/准备播报前复查，不延长主控有效期。"""
        with self._lock:
            return bool(self._scene and result == self._scene and self._peer == result["producer"]["instance_id"]
                        and result["clock_id"] == self.clock_id
                        and self.now_ms() < result["data"]["valid_until_mono_ms"]
                        and all(not (obj.get("measurement") or {}).get("valid") or self.now_ms() < obj["measurement"]["valid_until_mono_ms"] for obj in result["data"]["objects"]))

    def scene_query(self, *, session_id=None, request_id=None, track_id=None, fields=None, timeout_ms=500):
        return self._call("scene.query", {"track_id": track_id, "fields": fields}, session_id=session_id, request_id=request_id, timeout_ms=timeout_ms)

    def control_intent(self, action, final_transcript, *, utterance_id, is_final,
                       user_confirmed=False, target_ref=None, session_id=None,
                       request_id=None, timeout_ms=500):
        data = {"action": action, "utterance_id": utterance_id, "final_transcript": final_transcript,
                "target_ref": target_ref, "user_confirmed": user_confirmed}
        with self._lock:
            if (not _text(action) or action not in ACTIONS or is_final is not True
                    or not _text(final_transcript) or not _text(utterance_id)
                    or type(user_confirmed) is not bool
                    or (session_id is not None and not _text(session_id))
                    or (request_id is not None and not _text(request_id))):
                req = self._request("control.intent", data, session_id, request_id, timeout_ms)
                return self._failure(req, ProtocolError("INVALID_PAYLOAD", "控制意图必须来自最终话语及允许动作"))
            if re.search(r"不要|别|不是|不用|不想|不许|禁止|不准|没有让|没让|不(?:再)?(?:开始|暂停|恢复|结束|停止|重复)", final_transcript):
                req = self._request("control.intent", data, session_id, request_id, timeout_ms)
                return self._failure(req, ProtocolError("INVALID_PAYLOAD", "否定或改口话语须先澄清，不能按关键词提交控制"))
            if request_id:
                key = (session_id, request_id)
                fingerprint = json.dumps(data, sort_keys=True, ensure_ascii=False)
                if key in self._control_ids and self._control_ids[key] != fingerprint:
                    req = self._request("control.intent", data, session_id, request_id, timeout_ms)
                    return self._failure(req, ProtocolError("INVALID_PAYLOAD", "相同请求ID不得改变控制内容"))
                self._control_ids[key] = fingerprint
                if len(self._control_ids) > 128:
                    self._control_ids.popitem(last=False)
            return self._call("control.intent", data, session_id=session_id, request_id=request_id, timeout_ms=timeout_ms)

    def acquire_frame(self, *, mode="latest", purpose="vqa", frame_id=None, max_age_ms=300,
                      max_side_px=1600, roi_xyxy_px=None, session_id=None, request_id=None, timeout_ms=500):
        data = {"mode": mode, "purpose": purpose, "frame_id": frame_id, "max_age_ms": max_age_ms,
                "max_side_px": max_side_px, "roi_xyxy_px": roi_xyxy_px}
        req = self._request("frame.acquire", data, session_id, request_id, timeout_ms)
        try:
            _require(mode in ("latest", "pinned") and purpose in ("vqa", "ocr", "compare"), "取图参数无效")
            _require((mode == "pinned" and _text(frame_id)) or (mode == "latest" and frame_id is None), "帧身份与请求模式不符", "INVALID_FRAME_REF")
            _require(_number(max_age_ms) and max_age_ms >= 0 and type(max_side_px) is int and max_side_px > 0, "图像尺寸或龄期无效")
            _require(roi_xyxy_px is None or (mode == "pinned" and isinstance(roi_xyxy_px, (list, tuple)) and len(roi_xyxy_px) == 4 and all(_number(v) for v in roi_xyxy_px) and 0 <= roi_xyxy_px[0] < roi_xyxy_px[2] and 0 <= roi_xyxy_px[1] < roi_xyxy_px[3]), "ROI必须绑定依据帧并具有有效范围", "INVALID_FRAME_REF")
            return self._call("frame.acquire", data, session_id=session_id, request_id=request_id, timeout_ms=timeout_ms)
        except ProtocolError as exc:
            return self._failure(req, exc)

    frame_acquire = acquire_frame

    def _validate_frame(self, result, req):
        d, now = result["data"], self.now_ms()
        _require(all(k in d for k in ("frame_id", "captured_mono_ms", "source_kind", "is_historical", "width_px", "height_px", "color_space", "quality", "asset")), "帧字段不完整", "INVALID_FRAME_REF")
        _require(_text(d["frame_id"]) and d["source_kind"] in ("camera", "upload", "fixture") and type(d["is_historical"]) is bool, "帧身份或来源无效", "INVALID_FRAME_REF")
        _require(all(type(d[k]) is int and d[k] > 0 for k in ("width_px", "height_px")) and _text(d["color_space"]) and isinstance(d["quality"], dict), "帧元数据无效", "INVALID_FRAME_REF")
        _require((d["captured_mono_ms"] is None and d["is_historical"]) or (_number(d["captured_mono_ms"]) and d["captured_mono_ms"] <= now), "不能用接收时刻替代未知拍摄时间", "INVALID_FRAME_REF")
        if req["data"]["mode"] == "latest":
            _require(not d["is_historical"] and _number(d["captured_mono_ms"]) and now - d["captured_mono_ms"] <= req["data"]["max_age_ms"], "当前画面过期或仅有历史照片", "STALE_FRAME")
        else:
            _require(d["frame_id"] == req["data"]["frame_id"], "固定帧请求被替换", "INVALID_FRAME_REF")
        a = d["asset"]
        _require(isinstance(a, dict) and a.get("transport") == "file" and _text(a.get("asset_id")) and _text(a.get("path")) and _text(a.get("encoding")) and re.fullmatch(r"[0-9a-fA-F]{64}", str(a.get("sha256", ""))) and _number(a.get("available_until_mono_ms")), "文件资产描述无效", "INVALID_FRAME_REF")
        _require(now < a["available_until_mono_ms"], "文件资产租期已过", "STALE_FRAME")
        if d.get("parent_frame_id") is not None:
            transform = d.get("transform_to_parent")
            _require(_text(d["parent_frame_id"]) and isinstance(transform, list) and len(transform) == 9 and all(_number(v) for v in transform), "派生资产缺少3x3原图变换", "INVALID_FRAME_REF")
        identity = (d["captured_mono_ms"], d["source_kind"], d["is_historical"], d["width_px"], d["height_px"], a["sha256"].lower(), d.get("parent_frame_id"), tuple(d["transform_to_parent"]) if d.get("transform_to_parent") is not None else None)
        if d["frame_id"] in self._frames:
            _require(self._frames[d["frame_id"]] == identity, "同一frame_id的内容或采集身份变化", "INVALID_FRAME_REF")
        self._frames[d["frame_id"]] = identity
        if len(self._frames) > 64:
            self._frames.popitem(last=False)

    def read_frame_asset(self, frame_result):
        """只读显式授权根目录内文件，并校验实际字节；不删除原资产。"""
        with self._lock:
            _require(frame_result.get("status") == "ok" and frame_result.get("type") == "frame.result" and frame_result["producer"]["instance_id"] == self._peer and frame_result.get("evidence_mode") == self.evidence_mode, "帧不属于当前主控实例", "INVALID_FRAME_REF")
            d = frame_result["data"]
            a = d["asset"]
            _require(d["frame_id"] in self._frames and self._frames[d["frame_id"]][5] == a["sha256"].lower(), "未登记帧或内容身份不符", "INVALID_FRAME_REF")
            _require(self.now_ms() < a["available_until_mono_ms"], "文件资产租期已过", "STALE_FRAME")
            path = Path(a["path"]).resolve()
            _require(path.is_absolute() and any(path.is_relative_to(root) for root in self.asset_roots), "资产路径不在显式授权根目录中", "INVALID_FRAME_REF")
            try:
                with path.open("rb") as stream:
                    content = stream.read(self.max_asset_bytes + 1)
            except OSError as exc:
                raise ProtocolError("INVALID_FRAME_REF", "资产文件不可读") from exc
            _require(len(content) <= self.max_asset_bytes, "资产超出有界读取大小", "INVALID_FRAME_REF")
            _require(hashlib.sha256(content).hexdigest() == a["sha256"].lower(), "资产摘要与实际字节不符", "INVALID_FRAME_REF")
            _require(self.now_ms() < a["available_until_mono_ms"], "读取期间资产租期已过", "STALE_FRAME")
            return content

    def target_observation(self, data, *, session_id=None, request_id=None, timeout_ms=500):
        """保留入口；M1 对接目标观测字段后启用，不发送未校验的目标。"""
        req = self._request("target.observation", data, session_id, request_id, timeout_ms)
        return self._failure(req, ProtocolError("DEPENDENCY_UNAVAILABLE", "目标观测尚未实现，待M1对接", "unavailable"))


class FixtureTransport:
    """显式测试替身，只回放调用方给出的响应内容，不推断空间或控制结果。"""
    def __init__(self, responses=None, *, clock_id="fixture-clock", host_boot_id="fixture-boot",
                 instance_id="fixture-main", evidence_mode="synthetic", now_ms=None):
        if evidence_mode not in ("synthetic", "replay"):
            raise ValueError("替身禁止使用live证据模式")
        self.responses = responses or {}
        self.clock_id, self.host_boot_id = clock_id, host_boot_id
        self.instance_id, self.evidence_mode = instance_id, evidence_mode
        self.now_ms = now_ms or (lambda: time.monotonic_ns() / 1_000_000)
        self.requests = deque(maxlen=128)

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        if request["type"] == "capabilities.get":
            payload = {"status": "ok", "error": None, "data": {
                "supported_versions": [SCHEMA_VERSION], "host_boot_id": self.host_boot_id,
                "clock_source": "monotonic", "clock_id": self.clock_id,
                "capabilities": {cap: kind in self.responses for kind, cap in CAPABILITIES.items()}}}
        else:
            payload = self.responses.get(request["type"])
            if payload is None:
                payload = {"status": "unavailable", "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "替身未配置此样例", "retryable": False}, "data": {}}
            elif callable(payload):
                payload = payload(deepcopy(request))
        result = {"schema_version": SCHEMA_VERSION, "type": RESULT_TYPES[request["type"]],
                  "message_id": uuid.uuid4().hex, "reply_to": request["message_id"],
                  "request_id": request["request_id"], "session_id": request["session_id"],
                  "producer": {"role": "test_fixture", "instance_id": self.instance_id},
                  "clock_id": self.clock_id, "timestamp_mono_ms": self.now_ms(),
                  "evidence_mode": self.evidence_mode, "status": "ok", "error": None, "data": {}}
        result.update(deepcopy(payload))
        return result
