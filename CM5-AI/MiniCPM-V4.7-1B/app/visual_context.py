"""有界图片、对象与 OCR 上下文；图片内容不能授予空间判断权限。

FrameProvider 结果应先经 MainControlAdapter 校验并读取资产字节，再登记。
本模块只保存帧元数据；显式 retain_copy 或上传图可保留受总字节预算限制的副本。
"""
from collections import OrderedDict
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import re
import threading
import time
import uuid


class VisualContextError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _check(ok, message, code="INVALID_PAYLOAD"):
    if not ok:
        raise VisualContextError(code, message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _roi(value, frame, *, optional=False):
    if value is None and optional:
        return None
    _check(isinstance(value, (list, tuple)) and len(value) == 4 and all(_finite(x) for x in value), "ROI 必须为四个有限像素坐标", "INVALID_FRAME_REF")
    x1, y1, x2, y2 = value
    _check(0 <= x1 < x2 <= frame["width_px"] and 0 <= y1 < y2 <= frame["height_px"], "ROI 超出其依据原图", "INVALID_FRAME_REF")
    return list(value)


class VisualContextStore:
    """会话彼此隔离；同一帧不可刷新内容/时间，纠正对象只移动引用。"""
    def __init__(self, *, max_sessions=16, max_frames_per_session=8,
                 max_objects_per_session=16, max_ocr_per_frame=128,
                 max_frame_bytes=8 * 1024 * 1024, max_cached_bytes=16 * 1024 * 1024,
                 session_ttl_ms=900_000, now_ms=None):
        for value in (max_sessions, max_frames_per_session, max_objects_per_session,
                      max_ocr_per_frame, max_frame_bytes, max_cached_bytes, session_ttl_ms):
            _check(type(value) is int and value > 0, "容量和期限必须是正整数")
        self.max_sessions = max_sessions
        self.max_frames = max_frames_per_session
        self.max_objects = max_objects_per_session
        self.max_ocr = max_ocr_per_frame
        self.max_frame_bytes = max_frame_bytes
        self.max_cached_bytes = max_cached_bytes
        self.session_ttl_ms = session_ttl_ms
        self.now_ms = now_ms or (lambda: time.monotonic_ns() / 1_000_000)
        self._sessions = OrderedDict()
        self._cached_bytes = 0
        self._lock = threading.RLock()

    def _drop_session(self, sid):
        session = self._sessions.pop(sid)
        self._cached_bytes -= sum(len(f["copy"]) for f in session["frames"].values() if f["copy"] is not None)

    def _session(self, sid):
        _check(_text(sid) and len(sid) <= 128, "图片上下文必须有明确且有界的 session_id")
        now = self.now_ms()
        for key in list(self._sessions):
            if now - self._sessions[key]["updated_ms"] >= self.session_ttl_ms:
                self._drop_session(key)
        if sid not in self._sessions:
            while len(self._sessions) >= self.max_sessions:
                self._drop_session(next(iter(self._sessions)))
            self._sessions[sid] = {"updated_ms": now, "frames": OrderedDict(),
                                   "objects": OrderedDict(), "slots": {},
                                   "current_frame_id": None, "current_object_id": None}
        session = self._sessions[sid]
        session["updated_ms"] = now
        self._sessions.move_to_end(sid)
        return session

    @staticmethod
    def _frame(session, frame_id):
        _check(_text(frame_id) and frame_id in session["frames"], "帧未登记或已从有界上下文释放", "NO_FRAME")
        return session["frames"][frame_id]

    def _available(self, record):
        return record["copy"] is not None or self.now_ms() < record["data"]["asset"]["available_until_mono_ms"]

    def _view(self, record, *, require_asset=False):
        available = self._available(record)
        _check(not require_asset or available, "图片资产租期已过，需重新申请同帧资产", "STALE_FRAME")
        result = deepcopy(record["data"])
        result.update(evidence_mode=record["evidence_mode"], producer=deepcopy(record["producer"]),
                      clock_id=record["clock_id"], asset_available=available,
                      historical_reference=record["data"]["is_historical"] or self.now_ms() >= record["data"]["asset"]["available_until_mono_ms"],
                      retained_copy=record["copy"] is not None)
        return result

    def remember_frame(self, session_id, frame_result, *, asset_bytes, relation="new",
                       object_id=None, slot=None, retain_copy=False):
        """登记已读取的主控帧；显式副本不沿用外部租期，拍摄时间始终不变。"""
        _check(isinstance(frame_result, dict), "frame.result 必须是对象")
        _check(frame_result.get("schema_version") == "interaction.v0.1", "不支持的帧合同版本", "UNSUPPORTED_VERSION")
        _check(frame_result.get("type") == "frame.result" and frame_result.get("status") == "ok", "FrameProvider 未返回可用帧", "NO_FRAME")
        producer = frame_result.get("producer")
        evidence = frame_result.get("evidence_mode")
        _check(isinstance(producer, dict) and _text(producer.get("instance_id")) and len(producer["instance_id"]) <= 256, "帧缺少有界来源实例")
        _check((producer.get("role") == "main_control" and evidence == "live") or
               (producer.get("role") == "test_fixture" and evidence in ("synthetic", "replay")), "帧来源与证据模式不符")
        _check(_text(frame_result.get("clock_id")) and len(frame_result["clock_id"]) <= 256, "FrameProvider 帧缺少已验证时钟", "CLOCK_UNVERIFIED")
        return self._remember(session_id, frame_result.get("data"), asset_bytes=asset_bytes,
                              evidence_mode=evidence, producer=producer,
                              clock_id=frame_result.get("clock_id"), relation=relation,
                              object_id=object_id, slot=slot, retain_copy=retain_copy)

    def remember_upload(self, session_id, frame_id, asset_bytes, *, width_px, height_px,
                        captured_mono_ms=None, relation="new", object_id=None,
                        slot=None, quality=None):
        """客户端上传只作为历史照片；实际字节保存在有界副本缓存。"""
        _check(isinstance(asset_bytes, bytes), "图片内容必须是 bytes")
        now = self.now_ms()
        data = {"frame_id": frame_id, "captured_mono_ms": captured_mono_ms,
                "source_kind": "upload", "is_historical": True, "width_px": width_px,
                "height_px": height_px, "color_space": "RGB", "parent_frame_id": None,
                "transform_to_parent": None, "quality": quality or {"sharpness": "unknown", "glare": "unknown", "exposure": "unknown"},
                "asset": {"asset_id": "upload-" + hashlib.sha256(asset_bytes).hexdigest(),
                          "transport": "owned_copy", "encoding": "caller_supplied",
                          "sha256": hashlib.sha256(asset_bytes).hexdigest(),
                          "available_until_mono_ms": now + self.session_ttl_ms}}
        return self._remember(session_id, data, asset_bytes=asset_bytes, evidence_mode="live",
                              producer={"role": "interaction", "instance_id": "uploaded-photo"},
                              clock_id=None, relation=relation, object_id=object_id,
                              slot=slot, retain_copy=True)

    def _remember(self, sid, data, *, asset_bytes, evidence_mode, producer, clock_id,
                  relation, object_id, slot, retain_copy):
        _check(isinstance(data, dict) and isinstance(asset_bytes, bytes), "帧描述或字节格式错误")
        _check(0 < len(asset_bytes) <= self.max_frame_bytes, "图片超过单帧预算", "INVALID_FRAME_REF")
        _check(relation in ("new", "flip", "refresh"), "未知帧关系")
        _check(slot is None or slot in ("A", "B"), "比较槽位只能是 A 或 B")
        _check(type(retain_copy) is bool, "retain_copy 必须为布尔值")
        d = deepcopy(data)
        _check(_text(d.get("frame_id")) and d.get("source_kind") in ("camera", "upload", "fixture"), "帧身份或来源无效", "INVALID_FRAME_REF")
        _check(type(d.get("is_historical")) is bool and all(type(d.get(k)) is int and d[k] > 0 for k in ("width_px", "height_px")), "帧尺寸或历史标记无效", "INVALID_FRAME_REF")
        captured = d.get("captured_mono_ms")
        _check((captured is None and d["is_historical"]) or (_finite(captured) and captured <= self.now_ms()), "帧采集时间未知或来自未来", "INVALID_FRAME_REF")
        _check(d["source_kind"] != "upload" or d["is_historical"], "上传图片不能冒充当前采集")
        _check(_text(d.get("color_space")) and isinstance(d.get("quality"), dict), "帧缺少颜色/质量元数据", "INVALID_FRAME_REF")
        a = d.get("asset")
        _check(isinstance(a, dict) and _text(a.get("asset_id")) and _finite(a.get("available_until_mono_ms")), "帧资产字段不完整", "INVALID_FRAME_REF")
        _check(self.now_ms() < a["available_until_mono_ms"], "帧资产租期已过", "STALE_FRAME")
        digest = hashlib.sha256(asset_bytes).hexdigest()
        _check(isinstance(a.get("sha256"), str) and digest == a["sha256"].lower(), "帧实际内容与摘要不符", "INVALID_FRAME_REF")
        _check(d.get("parent_frame_id") is None or (_text(d["parent_frame_id"]) and isinstance(d.get("transform_to_parent"), list) and len(d["transform_to_parent"]) == 9 and all(_finite(x) for x in d["transform_to_parent"])), "派生帧缺少原图坐标变换", "INVALID_FRAME_REF")
        try:
            metadata_bytes = len(json.dumps(d, ensure_ascii=False, allow_nan=False).encode())
        except (TypeError, ValueError) as exc:
            raise VisualContextError("INVALID_FRAME_REF", "帧元数据不是有效 JSON") from exc
        _check(metadata_bytes <= 32768, "帧元数据超过预算")
        identity = (digest, captured, d["source_kind"], d["is_historical"], d["width_px"], d["height_px"], d["color_space"], d.get("parent_frame_id"), d.get("transform_to_parent"), deepcopy(producer), clock_id)
        with self._lock:
            session = self._session(sid)
            frames = session["frames"]
            existing = frames.get(d["frame_id"])
            if existing:
                _check(existing["identity"] == identity, "同一 frame_id 不得更换内容、来源或采集时间", "INVALID_FRAME_REF")
            if object_id is not None:
                _check(_text(object_id) and len(object_id) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
            if relation != "new":
                _check(_text(object_id) and object_id in session["objects"], "翻面/刷新必须明确已有对象", "INVALID_TARGET")
            elif object_id is not None:
                _check(_text(object_id) and len(object_id) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
                _check(object_id in session["objects"], "尚未登记的对象不能直接绑定帧", "INVALID_TARGET")
            if existing and relation in ("flip", "refresh"):
                _check(not existing, "翻面/刷新必须取得新帧；历史图请显式 select_frame", "INVALID_FRAME_REF")
            extra = len(asset_bytes) if retain_copy and (not existing or existing["copy"] is None) else 0
            _check(self._cached_bytes + extra <= self.max_cached_bytes, "图片副本总字节预算不足，请释放旧帧", "BUSY")
            if not existing and len(frames) >= self.max_frames:
                protected = {session["current_frame_id"], *(ref["frame_id"] for ref in session["slots"].values())}
                victim = next((fid for fid in frames if fid not in protected), None)
                _check(victim is not None, "当前帧与比较槽位占满容量，请先释放引用", "BUSY")
                self._drop_frame(session, victim)
            if existing:
                # 资产可重新申请租期，但身份与原始拍摄时刻不能变化。
                existing["data"]["asset"] = deepcopy(a)
                if extra:
                    existing["copy"] = asset_bytes
            else:
                frames[d["frame_id"]] = {"data": d, "identity": identity,
                    "producer": deepcopy(producer), "clock_id": clock_id,
                    "evidence_mode": evidence_mode, "copy": asset_bytes if retain_copy else None,
                    "ocr": []}
            self._cached_bytes += extra
            session["current_frame_id"] = d["frame_id"]
            session["current_object_id"] = object_id
            if object_id is not None:
                views = session["objects"][object_id]["views"]
                if not any(v["frame_id"] == d["frame_id"] for v in views):
                    views.append({"frame_id": d["frame_id"], "roi_xyxy_px": None, "relation": relation})
            if slot:
                session["slots"][slot] = {"object_id": object_id, "frame_id": d["frame_id"]}
            return self._view(frames[d["frame_id"]])

    def _drop_frame(self, session, frame_id):
        frame = session["frames"].pop(frame_id)
        if frame["copy"] is not None:
            self._cached_bytes -= len(frame["copy"])
        for key in list(session["slots"]):
            if session["slots"][key]["frame_id"] == frame_id:
                del session["slots"][key]
        for oid in list(session["objects"]):
            obj = session["objects"][oid]
            obj["views"] = [v for v in obj["views"] if v["frame_id"] != frame_id]
            if not obj["views"]:
                del session["objects"][oid]
                if session["current_object_id"] == oid:
                    session["current_object_id"] = None
        if session["current_frame_id"] == frame_id:
            session["current_frame_id"] = None

    def release_frame(self, session_id, frame_id):
        with self._lock:
            session = self._session(session_id)
            self._frame(session, frame_id)
            self._drop_frame(session, frame_id)

    def select_frame(self, session_id, frame_id, *, require_asset=True):
        with self._lock:
            session = self._session(session_id)
            frame = self._frame(session, frame_id)
            result = self._view(frame, require_asset=require_asset)
            session["current_frame_id"] = frame_id
            session["current_object_id"] = None
            return result

    def current_frame(self, session_id, *, require_asset=True):
        with self._lock:
            session = self._session(session_id)
            return self._view(self._frame(session, session["current_frame_id"]), require_asset=require_asset)

    def frame(self, session_id, frame_id, *, require_asset=True):
        with self._lock:
            return self._view(self._frame(self._session(session_id), frame_id), require_asset=require_asset)

    def image_bytes(self, session_id, frame_id):
        with self._lock:
            record = self._frame(self._session(session_id), frame_id)
            _check(record["copy"] is not None, "没有本地副本，需经 FrameProvider 重新读取有效资产", "NO_FRAME")
            return record["copy"]

    def bind_object(self, session_id, frame_id, roi_xyxy_px, *, object_id=None,
                    name="", source="user_selection", slot=None):
        with self._lock:
            session = self._session(session_id)
            frame = self._frame(session, frame_id)["data"]
            roi = _roi(roi_xyxy_px, frame)
            _check(isinstance(name, str) and len(name) <= 256 and _text(source) and len(source) <= 64, "对象名称或来源格式错误", "INVALID_TARGET")
            _check(slot is None or slot in ("A", "B"), "比较槽位只能是 A 或 B")
            if object_id is not None:
                _check(_text(object_id) and len(object_id) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
            oid = object_id or "visual-" + uuid.uuid4().hex
            _check(_text(oid) and len(oid) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
            if oid not in session["objects"]:
                _check(len(session["objects"]) < self.max_objects, "视觉对象容量不足，请释放旧帧", "BUSY")
                session["objects"][oid] = {"visual_target_id": oid, "name": name,
                                           "source": source, "views": []}
            obj = session["objects"][oid]
            existing = next((v for v in obj["views"] if v["frame_id"] == frame_id), None)
            if existing is not None:
                _check(existing["roi_xyxy_px"] is None or existing["roi_xyxy_px"] == roi, "同一对象/帧的区域不可静默替换，请建立新对象并纠正引用", "INVALID_TARGET")
                existing["roi_xyxy_px"] = roi
            else:
                obj["views"].append({"frame_id": frame_id, "roi_xyxy_px": roi, "relation": "observed"})
            session["current_object_id"], session["current_frame_id"] = oid, frame_id
            if slot:
                session["slots"][slot] = {"object_id": oid, "frame_id": frame_id}
            return deepcopy(obj)

    def correct_object(self, session_id, object_id, *, frame_id=None):
        """“不是左边，是右边”只更换选择，不把左侧 OCR 挪给右侧。"""
        with self._lock:
            session = self._session(session_id)
            _check(_text(object_id) and object_id in session["objects"], "纠正目标未登记", "INVALID_TARGET")
            obj = session["objects"][object_id]
            fid = frame_id or obj["views"][-1]["frame_id"]
            _check(any(v["frame_id"] == fid for v in obj["views"]), "纠正对象不属于所选图片", "INVALID_TARGET")
            self._view(self._frame(session, fid), require_asset=True)
            session["current_object_id"], session["current_frame_id"] = object_id, fid
            return deepcopy(obj)

    def assign_slot(self, session_id, slot, frame_id, *, object_id=None):
        _check(slot in ("A", "B"), "比较槽位只能是 A 或 B")
        with self._lock:
            session = self._session(session_id)
            self._frame(session, frame_id)
            if object_id is not None:
                _check(_text(object_id) and len(object_id) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
                _check(object_id in session["objects"] and any(v["frame_id"] == frame_id for v in session["objects"][object_id]["views"]), "比较槽位的对象与图片不匹配", "INVALID_TARGET")
            session["slots"][slot] = {"frame_id": frame_id, "object_id": object_id}

    def frame_for_slot(self, session_id, slot, *, require_asset=True):
        _check(slot in ("A", "B"), "比较槽位只能是 A 或 B")
        with self._lock:
            session = self._session(session_id)
            _check(slot in session["slots"], "比较槽位尚无图片", "NO_FRAME")
            ref = deepcopy(session["slots"][slot])
            ref["frame"] = self._view(self._frame(session, ref["frame_id"]), require_asset=require_asset)
            return ref

    def swap_slots(self, session_id):
        with self._lock:
            session = self._session(session_id)
            _check(all(k in session["slots"] for k in ("A", "B")), "比较需要先指定 A/B", "NO_FRAME")
            session["slots"]["A"], session["slots"]["B"] = session["slots"]["B"], session["slots"]["A"]

    def add_ocr(self, session_id, frame_id, rows, *, object_id=None, recognizer="unknown", replace=False):
        _check(isinstance(rows, list) and len(rows) <= self.max_ocr, "OCR 行数超过预算")
        _check(type(replace) is bool, "replace 必须是布尔值")
        _check(_text(recognizer) and len(recognizer) <= 128, "OCR 来源格式错误")
        with self._lock:
            session = self._session(session_id)
            frame = self._frame(session, frame_id)
            if object_id is not None:
                _check(_text(object_id) and len(object_id) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
                _check(object_id in session["objects"] and any(v["frame_id"] == frame_id for v in session["objects"][object_id]["views"]), "OCR 对象与图片不匹配", "INVALID_TARGET")
            object_roi = None
            if object_id is not None:
                object_roi = next(v["roi_xyxy_px"] for v in session["objects"][object_id]["views"] if v["frame_id"] == frame_id)
                _check(object_roi is not None, "对象的新视图尚无明确 ROI，OCR 先保持未关联", "INVALID_TARGET")
            added = []
            for index, row in enumerate(rows):
                _check(isinstance(row, dict) and isinstance(row.get("text"), str) and len(row["text"]) <= 512, "OCR 原文格式错误")
                _check(row.get("frame_id", frame_id) == frame_id and row.get("object_id", object_id) == object_id, "OCR 不得迁移来源帧或对象", "INVALID_FRAME_REF")
                score = row.get("score")
                _check(score is None or (_finite(score) and 0 <= score <= 1), "OCR 分数无效")
                row_roi = _roi(row.get("roi_xyxy_px"), frame["data"])
                if object_roi is not None:
                    _check(object_roi[0] <= row_roi[0] and object_roi[1] <= row_roi[1] and row_roi[2] <= object_roi[2] and row_roi[3] <= object_roi[3], "OCR 区域不在所绑定对象内，必须保持未关联", "INVALID_TARGET")
                added.append({"row_id": "ocr-" + uuid.uuid4().hex, "frame_id": frame_id,
                              "object_id": object_id, "raw_text": row["text"],
                              "roi_xyxy_px": row_roi,
                              "score": score, "recognizer": recognizer,
                              "evidence_mode": frame["evidence_mode"]})
            previous = [r for r in frame["ocr"] if not (replace and r["recognizer"] == recognizer and r["object_id"] == object_id)]
            combined = previous + added
            _check(len(combined) <= self.max_ocr, "该帧 OCR 累计行数超过预算", "BUSY")
            frame["ocr"] = combined
            return deepcopy(added)

    def associate_ocr(self, session_id, frame_id, object_id, *, row_ids=None):
        """明确确认本帧对象后，才将其 ROI 内尚未关联的文字绑定到该对象。"""
        _check(_text(object_id) and len(object_id) <= 128, "视觉对象 ID 无效", "INVALID_TARGET")
        if row_ids is not None:
            _check(isinstance(row_ids, list) and len(row_ids) <= self.max_ocr and all(_text(x) for x in row_ids), "row_ids 必须是有界 OCR 行 ID 列表")
        with self._lock:
            session = self._session(session_id)
            record = self._frame(session, frame_id)
            _check(object_id in session["objects"], "对象尚未登记", "INVALID_TARGET")
            view = next((v for v in session["objects"][object_id]["views"] if v["frame_id"] == frame_id), None)
            _check(view is not None and view["roi_xyxy_px"] is not None, "对象不属于该帧或缺少已确认 ROI", "INVALID_TARGET")
            bounds = view["roi_xyxy_px"]
            by_id = {r["row_id"]: r for r in record["ocr"]}
            if row_ids is not None:
                _check(all(rid in by_id for rid in row_ids), "指定 OCR 行不属于本帧", "INVALID_FRAME_REF")
                candidates = [by_id[rid] for rid in dict.fromkeys(row_ids)]
            else:
                candidates = [r for r in record["ocr"] if r["object_id"] is None]
            selected = []
            for row in candidates:
                if row["object_id"] == object_id:
                    continue  # 明确重试同一绑定是幂等的。
                if row["object_id"] is not None:
                    _check(row_ids is None, "已绑定其他对象的文字不能改归属", "INVALID_TARGET")
                    continue
                roi = row["roi_xyxy_px"]
                inside = bounds[0] <= roi[0] and bounds[1] <= roi[1] and roi[2] <= bounds[2] and roi[3] <= bounds[3]
                if row_ids is not None:
                    _check(inside, "指定文字不在所确认对象 ROI 内", "INVALID_TARGET")
                if inside:
                    selected.append(row)
            # 全部检验完成才修改，防止显式列表中的错误行造成部分错绑。
            for row in selected:
                row["object_id"] = object_id
            evidence_rows = [r for r in record["ocr"] if r["object_id"] == object_id]
            return {"frame_id": frame_id, "object_id": object_id,
                    "associated_count": len(selected),
                    "associated_row_ids": [r["row_id"] for r in selected],
                    "evidence": extract_fields(deepcopy(evidence_rows), object_id=object_id)}

    def ocr_for(self, session_id, frame_id, *, object_id=None):
        with self._lock:
            rows = self._frame(self._session(session_id), frame_id)["ocr"]
            return deepcopy([r for r in rows if r["object_id"] == object_id])

    def structured_fields(self, session_id, frame_id, *, object_id=None, min_score=0.6):
        _check(_finite(min_score) and 0 <= min_score <= 1, "OCR 阈值无效")
        rows = self.ocr_for(session_id, frame_id, object_id=object_id)
        return extract_fields(rows, object_id=object_id, min_score=min_score)

    def snapshot(self, session_id):
        with self._lock:
            session = self._session(session_id)
            return {"current_frame_id": session["current_frame_id"],
                    "current_object_id": session["current_object_id"],
                    "frames": [self._view(f) for f in session["frames"].values()],
                    "objects": deepcopy(list(session["objects"].values())),
                    "slots": deepcopy(session["slots"]),
                    "cached_bytes": self._cached_bytes,
                    "limits": {"frames_per_session": self.max_frames, "objects_per_session": self.max_objects,
                               "ocr_per_frame": self.max_ocr, "cached_bytes_total": self.max_cached_bytes}}


_DATE = re.compile(r"(20\d{2})\s*[年./-]\s*(\d{1,2})\s*[月./-]\s*(\d{1,2})\s*日?")
_DATE_LABELS = (("production_date", r"生产日期|制造日期|\bMFG\b|\bPROD\b"),
                ("expiry_date", r"到期日|有效期至|保质期至|\bEXP\b"))
_PRICE = re.compile(r"(?:[￥¥]\s*(\d+(?:\.\d{1,2})?))|(?:(\d+(?:\.\d{1,2})?)\s*元)|(?:(?:价格|售价|单价|现价|促销价|会员价)\s*[:：]?\s*(\d+(?:\.\d{1,2})?))")
_UNIT = re.compile(r"(?:每|/)\s*(公斤|千克|kg|斤|克|g|盒|袋|瓶|件|个)", re.I)


def extract_fields(rows, *, object_id=None, min_score=0.6):
    """只归一化带明确标签的同一 OCR 行；不跨行猜价签、日期或单位。"""
    fields = {key: [] for key in ("production_date", "expiry_date", "shelf_life", "batch_id", "price")}
    candidates = []
    for row in rows:
        text = row["raw_text"]
        source = {key: deepcopy(row[key]) for key in ("row_id", "frame_id", "object_id", "roi_xyxy_px", "score")}
        confident = row["score"] is not None and row["score"] >= min_score
        found = False
        for field, label in _DATE_LABELS:
            match = re.search(label, text, re.I)
            if not match:
                continue
            next_labels = [m.start() for _, other in _DATE_LABELS for m in re.finditer(other, text, re.I) if m.start() > match.start()]
            segment_end = min(next_labels) if next_labels else len(text)
            value = _DATE.search(text, match.end(), segment_end)
            if value:
                try:
                    normalized = date(*map(int, value.groups())).isoformat()
                except ValueError:
                    normalized = None
                if normalized and confident:
                    fields[field].append({"value": normalized, "raw_value": value.group(), "source": source})
                    found = True
        duration = re.search(r"保质期\s*[:：]?\s*(\d+)\s*(个月|月|年|天)", text)
        if duration and confident:
            fields["shelf_life"].append({"value": int(duration.group(1)), "unit": duration.group(2), "raw_value": duration.group(), "source": source})
            found = True
        batch = re.search(r"(?:批号|批次|\bLOT\b|\bBATCH\b)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9._/-]*)", text, re.I)
        if batch and confident:
            fields["batch_id"].append({"value": batch.group(1), "raw_value": batch.group(), "source": source})
            found = True
        prices = list(_PRICE.finditer(text))
        for price in prices:
            if confident:
                unit = _UNIT.search(text) if len(prices) == 1 else None
                conditions = [word for word in ("会员", "促销", "限购", "第二件", "满减") if word in text]
                fields["price"].append({"value": next(g for g in price.groups() if g is not None),
                                        "raw_value": price.group(), "unit": unit.group(1) if unit else None,
                                        "currency": "CNY" if any(ch in price.group() for ch in ("元", "￥", "¥")) else None,
                                        "conditions": conditions, "conditions_raw": text if conditions else None,
                                        "unit_ambiguous": len(prices) > 1, "source": source})
                found = True
        if not found and re.search(r"\d", text):
            candidates.append({"raw_text": text, "reason": "LOW_CONFIDENCE" if not confident else "UNLABELLED_OR_INVALID_NUMBER", "source": source})
    return {"association": "confirmed" if object_id is not None else "ambiguous",
            "object_id": object_id, "fields": fields, "numeric_candidates": candidates,
            "raw_observations": deepcopy(rows)}
