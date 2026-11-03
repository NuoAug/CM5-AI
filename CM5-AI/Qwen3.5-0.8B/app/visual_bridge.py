"""将 HTTP 图片输入与有界图片上下文连接；不加载模型或采集设备。"""
import base64
from contextlib import nullcontext
import binascii
import io
import re
from PIL import Image

from visual_context import VisualContextError


class VisualBridge:
    def __init__(self, store, main_control_getter, *, max_image_bytes=8 * 1024 * 1024,
                 max_pixels=16_000_000):
        self.store = store
        self.main_control_getter = main_control_getter
        self.max_image_bytes = min(max_image_bytes, store.max_frame_bytes)
        self.max_pixels = max_pixels
        if type(max_image_bytes) is not int or max_image_bytes <= 0 or type(max_pixels) is not int or max_pixels <= 0:
            raise ValueError("图片容量必须为正整数")

    @staticmethod
    def _error(code, message):
        raise VisualContextError(code, message)

    @staticmethod
    def _sid(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise VisualContextError("INVALID_PAYLOAD", "必须提供非空且不超过128字符的 session_id")
        return value

    def _inspect(self, content):
        if not isinstance(content, bytes) or not 0 < len(content) <= self.max_image_bytes:
            self._error("INVALID_FRAME_REF", "图片为空或超过压缩字节预算")
        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > self.max_pixels:
                    self._error("INVALID_FRAME_REF", "图片像素超过预算")
                fmt = image.format
                if fmt not in ("JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"):
                    self._error("INVALID_FRAME_REF", "不支持的图片格式")
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            if isinstance(exc, VisualContextError):
                raise
            raise VisualContextError("INVALID_FRAME_REF", "图片字节无法解析") from exc
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
                "BMP": "image/bmp", "GIF": "image/gif", "TIFF": "image/tiff"}[fmt]
        return width, height, mime

    def _decode(self, value):
        if not isinstance(value, str) or len(value) > ((self.max_image_bytes + 2) // 3) * 4 + 2048:
            self._error("INVALID_FRAME_REF", "图片编码格式错误或超过预算")
        encoded = value
        if value.startswith("data:"):
            header, sep, encoded = value.partition(",")
            if not sep or not re.fullmatch(r"data:image/[A-Za-z0-9.+-]+;base64", header):
                self._error("INVALID_FRAME_REF", "图片 data URL 必须使用 base64 图像内容")
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise VisualContextError("INVALID_FRAME_REF", "图片不是有效 base64") from exc
        width, height, mime = self._inspect(raw)
        return raw, width, height, mime

    @staticmethod
    def _mutate(commit_guard, operation, *args, **kwargs):
        # guard_context 工厂先获取轮次管理锁，store 方法再取图片锁，保持固定锁序。
        # 普通检查回调可用于兼容；需要原子代次保护时必须传可持锁的上下文工厂。
        if commit_guard is not None and not callable(commit_guard):
            raise VisualContextError("INVALID_PAYLOAD", "commit_guard 必须是检查或上下文工厂")
        context = commit_guard() if commit_guard is not None else None
        with context if context is not None else nullcontext():
            return operation(*args, **kwargs)

    def _context(self, sid, frame, *, requested_relation="new"):
        snapshot = self.store.snapshot(sid)
        selected_object = snapshot["current_object_id"]
        if selected_object is not None:
            obj = next((o for o in snapshot["objects"] if o["visual_target_id"] == selected_object), None)
            if obj is None or not any(v["frame_id"] == frame["frame_id"] for v in obj["views"]):
                selected_object = None
        return {"cache_enabled": True, "session_id": sid, "used_frame_id": frame["frame_id"],
                "used_frame": frame, "selected_object_id": selected_object,
                "requested_relation": requested_relation, "snapshot": snapshot}

    def _result(self, sid, frame, content, *, requested_relation="new", object_id=None, commit_guard=None):
        _, _, mime = self._inspect(content)
        if object_id is not None:
            self._mutate(commit_guard, self.store.correct_object, sid, object_id, frame_id=frame["frame_id"])
        return {"image": "data:" + mime + ";base64," + base64.b64encode(content).decode("ascii"),
                # 使用同一原图让现有推理入口生成缩图，不信任可能指向别物的 image_vlm。
                "image_vlm": "", "frame_id": frame["frame_id"], "captured_at": "",
                "frame_context": self._context(sid, frame, requested_relation=requested_relation)}

    def _provider(self, sid, *, purpose, frame_id=None, relation="new", object_id=None, slot=None, commit_guard=None):
        try:
            adapter = self.main_control_getter()
            if adapter is None:
                self._error("DEPENDENCY_UNAVAILABLE", "主控帧服务尚未配置")
            response = adapter.acquire_frame(mode="pinned" if frame_id else "latest", purpose=purpose,
                                             frame_id=frame_id, session_id=sid)
            if not isinstance(response, dict) or response.get("status") != "ok":
                error = response.get("error") or {} if isinstance(response, dict) else {}
                self._error(error.get("code") or "DEPENDENCY_UNAVAILABLE", error.get("message") or "主控暂时无法提供画面")
            content = adapter.read_frame_asset(response)
            width, height, _ = self._inspect(content)
            data = response.get("data") or {}
            if data.get("width_px") != width or data.get("height_px") != height:
                self._error("INVALID_FRAME_REF", "实际图片尺寸与主控帧描述不一致")
            if object_id is not None and relation == "new":
                snap = self.store.snapshot(sid)
                known = next((o for o in snap["objects"] if o["visual_target_id"] == object_id), None)
                if known is None or not any(v["frame_id"] == data.get("frame_id") for v in known["views"]):
                    self._error("INVALID_TARGET", "所选对象不属于主控返回图片，需明确新视图关系或重新确认")
            stored = self._mutate(commit_guard, self.store.remember_frame, sid, response, asset_bytes=content, retain_copy=True,
                                                relation=relation, object_id=object_id, slot=slot)
            return self._result(sid, stored, content, requested_relation=relation, object_id=object_id, commit_guard=commit_guard)
        except VisualContextError:
            raise
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise VisualContextError("DEPENDENCY_UNAVAILABLE", "主控帧服务连接失败") from exc
        except Exception as exc:
            # MainControlAdapter 的协议异常保留真实 code；未知错误不伪装成成功。
            if hasattr(exc, "code"):
                raise VisualContextError(exc.code, str(exc)) from exc
            raise

    def resolve(self, session_id, body, purpose="vqa", reuse=True, commit_guard=None):
        if not isinstance(body, dict):
            self._error("INVALID_PAYLOAD", "图片请求必须是对象")
        if session_id is None or session_id == "":
            # 兼容未建立会话的旧调试入口；不在不同匿名请求之间保留图片。
            return {key: body.get(key, "") for key in ("image", "image_vlm", "frame_id", "captured_at")} | {"frame_context": {"cache_enabled": False, "session_id": None}}
        sid = self._sid(session_id)
        if purpose not in ("vqa", "ocr", "compare") or type(reuse) is not bool:
            self._error("INVALID_PAYLOAD", "图片用途或复用参数无效")
        if "new_frame" in body and type(body["new_frame"]) is not bool:
            self._error("INVALID_PAYLOAD", "new_frame 必须为布尔值")
        frame_id, slot = body.get("frame_id"), body.get("slot")
        if frame_id is not None and (not isinstance(frame_id, str) or len(frame_id) > 256):
            self._error("INVALID_FRAME_REF", "frame_id 格式无效")
        if slot is not None and slot not in ("A", "B"):
            self._error("INVALID_PAYLOAD", "比较槽位只能是 A 或 B")
        relation = body.get("frame_relation", "new")
        if relation not in ("new", "flip", "refresh"):
            self._error("INVALID_PAYLOAD", "未知帧关系")
        question = " ".join(str(body.get(k) or "") for k in ("text", "question", "prompt"))
        if "frame_relation" not in body:
            if "翻面" in question:
                relation = "flip"
            elif body.get("new_frame", False) or re.search(r"现在呢|重新看|重新拍|再拍一张|换一张", question):
                relation = "refresh"
        fresh = body.get("new_frame", False) or relation in ("flip", "refresh") or bool(re.search(r"现在呢|翻面|重新看|重新拍|再拍一张|换一张", question))
        snapshot = self.store.snapshot(sid)
        object_id = body.get("object_id")
        if object_id is not None:
            if not isinstance(object_id, str) or not object_id.strip() or len(object_id) > 128:
                self._error("INVALID_TARGET", "object_id 格式无效")
            if not any(o["visual_target_id"] == object_id for o in snapshot["objects"]):
                self._error("INVALID_TARGET", "所选对象尚未登记")
        if object_id is None and relation in ("flip", "refresh"):
            object_id = snapshot["current_object_id"]
        # 无明确对象时只保存新帧，不将前一物品的OCR自动归给它。
        stored_relation = relation if relation == "new" or object_id is not None else "new"
        supplied = body.get("image") or body.get("image_vlm")
        if supplied:
            content, width, height, _ = self._decode(supplied)
            if not frame_id:
                import hashlib
                frame_id = "upload-" + hashlib.sha256(content).hexdigest()[:24]
            if object_id is not None and stored_relation == "new":
                known = next(o for o in snapshot["objects"] if o["visual_target_id"] == object_id)
                if not any(v["frame_id"] == frame_id for v in known["views"]):
                    self._error("INVALID_TARGET", "所选对象不属于上传图片，需明确翻面/刷新或重新确认")
            frame = self._mutate(commit_guard, self.store.remember_upload, sid, frame_id, content, width_px=width, height_px=height,
                                               captured_mono_ms=None, relation=stored_relation,
                                               object_id=object_id, slot=slot)
            result = self._result(sid, frame, content, requested_relation=relation, object_id=object_id, commit_guard=commit_guard)
            return result
        if fresh or not reuse:
            result = self._provider(sid, purpose=purpose, relation=stored_relation, object_id=object_id, slot=slot, commit_guard=commit_guard)
            result["frame_context"]["requested_relation"] = relation
            return result
        if slot:
            slot_ref = snapshot["slots"].get(slot)
            if not slot_ref:
                self._error("NO_FRAME", "比较槽位尚无图片，不能用当前画面替换")
            if frame_id and frame_id != slot_ref["frame_id"]:
                self._error("INVALID_FRAME_REF", "显式帧号与比较槽位不一致")
            if object_id is not None and slot_ref["object_id"] not in (None, object_id):
                self._error("INVALID_TARGET", "显式对象与比较槽位对象不一致")
            try:
                ref = self.store.frame_for_slot(sid, slot)
                result = self._result(sid, ref["frame"], self.store.image_bytes(sid, ref["frame_id"]), object_id=object_id or ref["object_id"], commit_guard=commit_guard)
                result["frame_context"].update(used_slot=slot, selected_object_id=object_id or ref["object_id"])
                return result
            except VisualContextError as exc:
                if exc.code not in ("NO_FRAME", "STALE_FRAME"):
                    raise
                result = self._provider(sid, purpose=purpose, frame_id=slot_ref["frame_id"], object_id=object_id or slot_ref["object_id"], commit_guard=commit_guard)
                result["frame_context"].update(used_slot=slot, selected_object_id=object_id or slot_ref["object_id"])
                return result
        if frame_id:
            if object_id is not None:
                known = next(o for o in snapshot["objects"] if o["visual_target_id"] == object_id)
                if not any(v["frame_id"] == frame_id for v in known["views"]):
                    self._error("INVALID_TARGET", "所选对象不属于显式指定图片")
            try:
                frame = self._mutate(commit_guard, self.store.select_frame, sid, frame_id)
                return self._result(sid, frame, self.store.image_bytes(sid, frame_id), object_id=object_id, commit_guard=commit_guard)
            except VisualContextError as exc:
                if exc.code not in ("NO_FRAME", "STALE_FRAME"):
                    raise
                return self._provider(sid, purpose=purpose, frame_id=frame_id, object_id=object_id, commit_guard=commit_guard)
        current_id = snapshot["current_frame_id"]
        if current_id:
            try:
                frame = self.store.current_frame(sid)
                return self._result(sid, frame, self.store.image_bytes(sid, current_id), object_id=object_id, commit_guard=commit_guard)
            except VisualContextError as exc:
                if exc.code not in ("NO_FRAME", "STALE_FRAME"):
                    raise
                return self._provider(sid, purpose=purpose, frame_id=current_id, object_id=object_id, commit_guard=commit_guard)
        return self._provider(sid, purpose=purpose, commit_guard=commit_guard)

    def register_routes(self, app):
        from flask import jsonify, request
        statuses = {"NO_FRAME": 404, "STALE_FRAME": 409, "CLOCK_UNVERIFIED": 409,
                    "BUSY": 429, "DEPENDENCY_UNAVAILABLE": 503}

        def failed(exc):
            return jsonify({"ok": False, "code": exc.code, "error": str(exc)}), statuses.get(exc.code, 400)

        @app.get("/api/session/visual")
        def visual_context_status():
            try:
                return jsonify({"ok": True, "context": self.store.snapshot(self._sid(request.args.get("session_id")))})
            except VisualContextError as exc:
                return failed(exc)

        @app.post("/api/session/visual/action")
        def visual_context_action():
            try:
                body = request.get_json(silent=True)
                if not isinstance(body, dict):
                    self._error("INVALID_PAYLOAD", "操作请求必须是 JSON 对象")
                sid = self._sid(body.get("session_id"))
                action = body.get("action")
                if action == "bind":
                    result = self.store.bind_object(sid, body.get("frame_id"), body.get("roi_xyxy_px"),
                                                    object_id=body.get("object_id"), name=body.get("name", ""),
                                                    source="user_selection", slot=body.get("slot"))
                elif action == "correct":
                    result = self.store.correct_object(sid, body.get("object_id"), frame_id=body.get("frame_id"))
                elif action == "associate_ocr":
                    result = self.store.associate_ocr(sid, body.get("frame_id"), body.get("object_id"), row_ids=body.get("row_ids"))
                elif action == "assign_slot":
                    result = self.store.assign_slot(sid, body.get("slot"), body.get("frame_id"), object_id=body.get("object_id"))
                elif action in ("swap", "swap_slots"):
                    result = self.store.swap_slots(sid)
                elif action == "select":
                    result = self.store.select_frame(sid, body.get("frame_id"))
                elif action == "release":
                    result = self.store.release_frame(sid, body.get("frame_id"))
                else:
                    self._error("INVALID_PAYLOAD", "未知图片会话操作")
                return jsonify({"ok": True, "action": action, "result": result, "context": self.store.snapshot(sid)})
            except VisualContextError as exc:
                return failed(exc)
