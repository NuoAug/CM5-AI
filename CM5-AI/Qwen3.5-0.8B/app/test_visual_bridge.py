"""真实小 PNG + 构造主控协议验证桥接，不加载模型或真实设备。"""
import base64
import hashlib
import io
import json
import threading
import unittest
from unittest import mock

from flask import Flask
from PIL import Image
from visual_bridge import VisualBridge
from visual_context import VisualContextError, VisualContextStore
from turns import TurnCancelled, TurnManager


def picture(color="red"):
    out = io.BytesIO()
    Image.new("RGB", (20, 10), color).save(out, format="PNG")
    return out.getvalue()


def url(content):
    return "data:image/png;base64," + base64.b64encode(content).decode()


def envelope(frame_id, content):
    return {"schema_version": "interaction.v0.1", "type": "frame.result", "status": "ok",
            "producer": {"role": "test_fixture", "instance_id": "fixture-main"},
            "clock_id": "fixture-clock", "evidence_mode": "synthetic",
            "data": {"frame_id": frame_id, "captured_mono_ms": 900, "source_kind": "fixture",
                     "is_historical": False, "width_px": 20, "height_px": 10,
                     "color_space": "RGB", "parent_frame_id": None, "transform_to_parent": None,
                     "quality": {"sharpness": "unknown"}, "asset": {
                         "asset_id": "asset-" + frame_id, "transport": "file", "path": "synthetic.png",
                         "encoding": "png", "sha256": hashlib.sha256(content).hexdigest(),
                         "available_until_mono_ms": 2000}}}


class FixtureProvider:
    def __init__(self):
        self.responses = []
        self.contents = {}
        self.requests = []

    def add(self, frame_id, content):
        result = envelope(frame_id, content)
        self.responses.append(result)
        self.contents[frame_id] = content
        return result

    def acquire_frame(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0) if self.responses else {"status": "unavailable", "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "fixture unavailable"}}

    def read_frame_asset(self, response):
        return self.contents[response["data"]["frame_id"]]


class VisualBridgeTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000
        self.store = VisualContextStore(now_ms=lambda: self.now)
        self.provider = FixtureProvider()
        self.getter = mock.Mock(return_value=self.provider)
        self.bridge = VisualBridge(self.store, self.getter)
        self.red = picture("red")
        self.blue = picture("blue")

    def upload(self, frame_id="one", content=None, **extra):
        return self.bridge.resolve("s", {"frame_id": frame_id, "image": url(content or self.red), **extra})

    def assert_code(self, code, func, *args, **kwargs):
        with self.assertRaises(VisualContextError) as caught:
            func(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_upload_preserves_original_compressed_bytes_and_unknown_time(self):
        result = self.upload(captured_at="2026-11-03T12:00:00Z", captured_mono_ms=1000)
        self.assertEqual(result["image"], url(self.red))
        self.assertEqual(result["captured_at"], "")
        self.assertEqual(result["image_vlm"], "")
        self.assertEqual(self.store.image_bytes("s", "one"), self.red)
        self.assertIsNone(result["frame_context"]["used_frame"]["captured_mono_ms"])
        self.assertTrue(result["frame_context"]["used_frame"]["is_historical"])
        self.getter.assert_not_called()

    def test_vlm_image_cannot_replace_the_original_photo(self):
        result = self.upload(image_vlm=url(self.blue))
        self.assertEqual(result["image"], url(self.red))
        self.assertEqual(result["image_vlm"], "")

    def test_visual_only_input_and_raw_base64_are_supported(self):
        result = self.bridge.resolve("s", {"image_vlm": base64.b64encode(self.red).decode()})
        self.assertEqual(result["image"], url(self.red))
        self.assertEqual(self.store.image_bytes("s", result["frame_id"]), self.red)

    def test_followup_reuses_same_bytes_identity_and_capture_time(self):
        first = self.upload()
        self.now = 1500
        second = self.bridge.resolve("s", {"question": "它是什么颜色"})
        self.assertEqual(first["image"], second["image"])
        self.assertEqual(second["frame_id"], "one")
        self.assertIsNone(second["frame_context"]["used_frame"]["captured_mono_ms"])
        self.getter.assert_not_called()

    def test_same_id_new_content_rejected(self):
        self.upload()
        self.assert_code("INVALID_FRAME_REF", self.upload, content=self.blue)
        self.assertEqual(self.store.image_bytes("s", "one"), self.red)

    def test_fresh_request_failure_does_not_fall_back_to_old_photo(self):
        self.upload()
        for body in ({"new_frame": True}, {"question": "现在呢"}, {"text": "翻面再看"}):
            self.assert_code("DEPENDENCY_UNAVAILABLE", self.bridge.resolve, "s", body)
        self.assertEqual(len(self.provider.requests), 3)
        self.assertTrue(all(r["mode"] == "latest" for r in self.provider.requests))
        self.assertEqual(self.store.current_frame("s")["frame_id"], "one")

    def test_explicit_new_frame_ignores_old_pinned_id(self):
        self.upload()
        self.provider.add("new-camera", self.blue)
        result = self.bridge.resolve("s", {"new_frame": True, "frame_id": "one"})
        self.assertEqual(result["frame_id"], "new-camera")
        self.assertEqual(self.provider.requests[0]["mode"], "latest")
        self.assertIsNone(self.provider.requests[0]["frame_id"])
        self.assertEqual(result["frame_context"]["used_frame"]["evidence_mode"], "synthetic")
        self.assertEqual(result["frame_context"]["used_frame"]["captured_mono_ms"], 900)

    def test_flip_links_selected_object_but_does_not_copy_ocr(self):
        self.upload()
        self.store.bind_object("s", "one", [0, 0, 20, 10], object_id="package")
        self.store.add_ocr("s", "one", [{"text": "生产日期2026-11-02", "score": .9, "roi_xyxy_px": [0, 0, 20, 2]}], object_id="package")
        self.provider.add("back-camera", self.blue)
        result = self.bridge.resolve("s", {"text": "翻面现在呢"})
        self.assertEqual(result["frame_context"]["selected_object_id"], "package")
        self.assertEqual(result["frame_context"]["requested_relation"], "flip")
        self.assertEqual(self.store.ocr_for("s", "back-camera", object_id="package"), [])
        self.assertEqual(len(self.store.ocr_for("s", "one", object_id="package")), 1)

    def test_explicit_frame_pin_uses_existing_historical_photo(self):
        self.upload()
        self.upload("two", self.blue)
        result = self.bridge.resolve("s", {"frame_id": "one"})
        self.assertEqual(result["image"], url(self.red))
        self.getter.assert_not_called()

    def test_missing_explicit_frame_requests_pinned_not_latest(self):
        self.provider.add("requested-old", self.red)
        result = self.bridge.resolve("s", {"frame_id": "requested-old"})
        self.assertEqual(result["frame_id"], "requested-old")
        self.assertEqual(self.provider.requests[0]["mode"], "pinned")
        self.assertEqual(self.provider.requests[0]["frame_id"], "requested-old")

    def test_empty_session_preserves_legacy_input_without_cache_or_provider(self):
        body = {"image": "legacy-not-validated-here", "image_vlm": "legacy-low", "frame_id": "legacy", "captured_at": "wall-time"}
        result = self.bridge.resolve("", body)
        for key in body:
            self.assertEqual(result[key], body[key])
        self.assertFalse(result["frame_context"]["cache_enabled"])
        self.getter.assert_not_called()

    def test_missing_cache_acquires_provider_and_verifies_actual_dimensions(self):
        frame = self.provider.add("p", self.red)
        frame["data"]["width_px"] = 21
        self.assert_code("INVALID_FRAME_REF", self.bridge.resolve, "s", {})
        self.assertEqual(self.store.snapshot("s")["frames"], [])

    def test_bad_image_payloads_are_rejected_without_model_or_camera_calls(self):
        for value in ("not-base64!", "data:text/html;base64,PHNjcmlwdD4=", url(b"bad-png")):
            self.assert_code("INVALID_FRAME_REF", self.bridge.resolve, "s", {"image": value})
        self.getter.assert_not_called()

    def test_image_byte_and_pixel_limits(self):
        tiny = VisualBridge(self.store, self.getter, max_image_bytes=2)
        self.assert_code("INVALID_FRAME_REF", tiny.resolve, "s", {"image": url(self.red)})
        small = VisualBridge(self.store, self.getter, max_pixels=199)
        self.assert_code("INVALID_FRAME_REF", small.resolve, "s", {"image": url(self.red)})
        self.assertEqual(self.store.snapshot("s")["frames"], [])

    def test_compare_slot_never_substitutes_current_photo(self):
        self.upload()
        self.assert_code("NO_FRAME", self.bridge.resolve, "s", {"slot": "B"})
        self.getter.assert_not_called()
        self.store.assign_slot("s", "A", "one")
        self.upload("two", self.blue)
        self.store.assign_slot("s", "B", "two")
        self.store.swap_slots("s")
        self.assertEqual(self.bridge.resolve("s", {"slot": "A"})["image"], url(self.blue))
        self.assertEqual(self.bridge.resolve("s", {"slot": "B"})["image"], url(self.red))
        self.assert_code("INVALID_FRAME_REF", self.bridge.resolve, "s", {"slot": "A", "frame_id": "one"})

    def test_provider_metadata_without_copy_is_reacquired_as_same_frame(self):
        frame = envelope("metadata-only", self.red)
        self.store.remember_frame("s", frame, asset_bytes=self.red)
        self.provider.add("metadata-only", self.red)
        result = self.bridge.resolve("s", {})
        self.assertEqual(result["frame_id"], "metadata-only")
        self.assertEqual(self.provider.requests[0]["mode"], "pinned")
        self.assertEqual(self.provider.requests[0]["frame_id"], "metadata-only")

    def test_explicit_object_on_cached_frame_actually_changes_selection(self):
        self.upload()
        self.store.bind_object("s", "one", [0, 0, 10, 10], object_id="left")
        self.store.bind_object("s", "one", [10, 0, 20, 10], object_id="right")
        self.store.correct_object("s", "left")
        result = self.bridge.resolve("s", {"object_id": "right", "question": "它的颜色"})
        self.assertEqual(result["frame_context"]["selected_object_id"], "right")
        self.assertEqual(self.store.snapshot("s")["current_object_id"], "right")
        self.getter.assert_not_called()

    def test_explicit_object_must_belong_to_pinned_frame_without_partial_selection(self):
        self.upload()
        self.store.bind_object("s", "one", [0, 0, 10, 10], object_id="first")
        self.upload("two", self.blue)
        self.store.bind_object("s", "two", [0, 0, 10, 10], object_id="second")
        self.assert_code("INVALID_TARGET", self.bridge.resolve, "s", {"frame_id": "one", "object_id": "second"})
        self.assertEqual(self.store.current_frame("s")["frame_id"], "two")
        self.assertEqual(self.store.snapshot("s")["current_object_id"], "second")
        self.getter.assert_not_called()

    def test_object_cannot_silently_move_to_a_new_upload(self):
        self.upload()
        self.store.bind_object("s", "one", [0, 0, 10, 10], object_id="first")
        self.assert_code("INVALID_TARGET", self.bridge.resolve, "s", {"frame_id": "different", "image": url(self.blue), "object_id": "first"})
        self.assertEqual(len(self.store.snapshot("s")["frames"]), 1)

    def test_cancelled_turn_cannot_upload_select_or_correct_cached_object(self):
        self.upload()
        self.store.bind_object("s", "one", [0, 0, 10, 10], object_id="first")
        self.store.bind_object("s", "one", [10, 0, 20, 10], object_id="second")
        self.store.correct_object("s", "first")
        self.upload("two", self.blue)
        manager = TurnManager()
        old = manager.begin("s", "old")
        manager.begin("s", "new")
        before = self.store.snapshot("s")
        for body in ({"frame_id": "cancelled-upload", "image": url(self.red)},
                     {"frame_id": "one"},
                     {"frame_id": "two", "object_id": "first"}):
            # Invalid object membership is validated before any state mutation.
            expected = VisualContextError if body.get("object_id") else TurnCancelled
            with self.subTest(body=body), self.assertRaises(expected):
                self.bridge.resolve("s", body, commit_guard=old.guard_context)
            self.assertEqual(self.store.snapshot("s"), before)
        self.bridge.resolve("s", {"frame_id": "one"})
        before = self.store.snapshot("s")
        with self.assertRaises(TurnCancelled):
            self.bridge.resolve("s", {"object_id": "second"}, commit_guard=old.guard_context)
        self.assertEqual(self.store.snapshot("s"), before)

    def test_delayed_provider_old_turn_cannot_replace_newer_uploaded_frame(self):
        manager = TurnManager()
        old = manager.begin("s", "old")
        self.provider.add("delayed-camera", self.red)
        entered, release, new_done = threading.Event(), threading.Event(), threading.Event()
        outcomes, new_tokens = [], []
        original_read = self.provider.read_frame_asset
        def blocked_read(response):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test release did not arrive")
            return original_read(response)
        self.provider.read_frame_asset = blocked_read
        def resolve_old():
            try:
                outcomes.append(self.bridge.resolve("s", {"new_frame": True}, commit_guard=old.guard_context))
            except Exception as exc:
                outcomes.append(exc)
        def begin_new():
            new_tokens.append(manager.begin("s", "new"))
            new_done.set()
        worker = threading.Thread(target=resolve_old)
        newcomer = threading.Thread(target=begin_new)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            newcomer.start()
            # Provider I/O must not hold the turn-manager commit lock.
            self.assertTrue(new_done.wait(1))
            self.bridge.resolve("s", {"frame_id": "new-upload", "image": url(self.blue)},
                                commit_guard=new_tokens[0].guard_context)
        finally:
            release.set()
            worker.join(2)
            if newcomer.ident is not None:
                newcomer.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], TurnCancelled)
        self.assertEqual(self.store.current_frame("s")["frame_id"], "new-upload")
        self.assertEqual([f["frame_id"] for f in self.store.snapshot("s")["frames"]], ["new-upload"])

    def test_guard_lock_remains_held_until_store_mutation_finishes(self):
        manager = TurnManager()
        old = manager.begin("s", "old")
        entered, release, started, new_done = (threading.Event() for _ in range(4))
        errors = []
        original_remember = self.store.remember_upload
        def blocked_remember(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test release did not arrive")
            self.assertFalse(old.cancelled.is_set())
            return original_remember(*args, **kwargs)
        def resolve_old():
            try:
                self.bridge.resolve("s", {"frame_id": "one", "image": url(self.red)},
                                    commit_guard=old.guard_context)
            except Exception as exc:
                errors.append(exc)
        def begin_new():
            started.set()
            manager.begin("s", "new")
            new_done.set()
        with mock.patch.object(self.store, "remember_upload", side_effect=blocked_remember):
            worker = threading.Thread(target=resolve_old)
            newcomer = threading.Thread(target=begin_new)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                newcomer.start()
                self.assertTrue(started.wait(1))
                self.assertFalse(new_done.wait(.05))
            finally:
                release.set()
                worker.join(2)
                if newcomer.ident is not None:
                    newcomer.join(2)
        self.assertEqual(errors, [])
        self.assertTrue(new_done.is_set())
        self.assertTrue(old.cancelled.is_set())

    def test_guard_factory_is_never_included_in_json_context(self):
        self.upload()
        self.store.bind_object("s", "one", [0, 0, 20, 10], object_id="item", slot="A")
        token = TurnManager().begin("s", "one")
        result = self.bridge.resolve("s", {"slot": "A"}, commit_guard=token.guard_context)
        self.assertEqual(result["frame_context"]["selected_object_id"], "item")
        self.assertNotIn("commit_guard", result["frame_context"])
        json.dumps(result)

    def test_invalid_json_parameters_have_typed_errors(self):
        for body in ({"new_frame": "yes"}, {"slot": "C"}, {"frame_id": []}, {"frame_relation": []}):
            self.assert_code("INVALID_PAYLOAD" if "frame_id" not in body else "INVALID_FRAME_REF", self.bridge.resolve, "s", body)
        self.assert_code("INVALID_PAYLOAD", self.bridge.resolve, [], {})


class VisualRouteTests(unittest.TestCase):
    def setUp(self):
        self.store = VisualContextStore()
        self.getter = mock.Mock(side_effect=AssertionError("诊断不能加载主控或模型"))
        self.bridge = VisualBridge(self.store, self.getter)
        self.bridge.resolve("s", {"frame_id": "f", "image": url(picture())})
        app = Flask(__name__)
        self.bridge.register_routes(app)
        self.client = app.test_client()

    def test_status_only_returns_metadata_and_does_not_call_dependencies(self):
        response = self.client.get("/api/session/visual?session_id=s")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["context"]["current_frame_id"], "f")
        self.assertNotIn("data:image", json.dumps(response.json))
        self.getter.assert_not_called()

    def test_routes_require_explicit_bounded_session_id(self):
        self.assertEqual(self.client.get("/api/session/visual").status_code, 400)
        for sid in ("", None, [], "x" * 129):
            response = self.client.post("/api/session/visual/action", json={"session_id": sid, "action": "swap"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["code"], "INVALID_PAYLOAD")

    def test_bind_correct_select_slots_and_release(self):
        def act(action, **data):
            response = self.client.post("/api/session/visual/action", json={"session_id": "s", "action": action, **data})
            self.assertEqual(response.status_code, 200, response.json)
            return response.json
        result = act("bind", frame_id="f", object_id="right", roi_xyxy_px=[10, 0, 20, 10], source="main_control")
        self.assertEqual(result["result"]["source"], "user_selection")
        act("correct", object_id="right")
        act("assign_slot", frame_id="f", slot="A", object_id="right")
        act("assign_slot", frame_id="f", slot="B")
        act("swap")
        act("select", frame_id="f")
        result = act("release", frame_id="f")
        self.assertEqual(result["context"]["frames"], [])
        self.assertEqual(result["context"]["cached_bytes"], 0)

    def test_associate_ocr_action_returns_source_bound_structured_evidence(self):
        self.store.bind_object("s", "f", [0, 0, 20, 10], object_id="item")
        added = self.store.add_ocr("s", "f", [{"text": "会员价9元/袋", "roi_xyxy_px": [0, 0, 15, 5], "score": .9}], recognizer="RapidOCR")
        response = self.client.post("/api/session/visual/action", json={"session_id": "s", "action": "associate_ocr", "frame_id": "f", "object_id": "item", "row_ids": [added[0]["row_id"]]})
        self.assertEqual(response.status_code, 200)
        result = response.json["result"]
        self.assertEqual(result["associated_count"], 1)
        self.assertEqual(result["evidence"]["association"], "confirmed")
        self.assertEqual(result["evidence"]["fields"]["price"][0]["source"]["frame_id"], "f")
        self.getter.assert_not_called()

    def test_invalid_actions_and_unknown_frames_return_error_json(self):
        response = self.client.post("/api/session/visual/action", json={"session_id": "s", "action": "select", "frame_id": "unknown"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json["code"], "NO_FRAME")
        self.assertEqual(self.client.post("/api/session/visual/action", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/api/session/visual/action", json={"session_id": "s", "action": "invent_distance"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
