# -*- coding: utf-8 -*-
"""M0 协议边界回归。全部为 synthetic 构造，不代表真实主控或硬件通过。"""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest

from main_control import FixtureTransport, MainControlAdapter, ProtocolError


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000
        self.scene = {"state_id": "state-1", "sequence": 1, "observed_mono_ms": 990,
                      "valid_until_mono_ms": 1500, "coordinate_frame": "body_forward",
                      "navigation": {"mode": "paused", "decision": "pause",
                                     "reason_code": "SPACE_UNCONFIRMED", "speech_fact": "两侧空间尚未确认。"},
                      "objects": [{"track_id": "obstacle-1", "class_name": "静态实体", "measurement": {
                          "valid": False, "distance_m": None, "valid_until_mono_ms": 1500,
                          "reason_code": "DEPTH_INVALID"}}],
                      "sensor_status": {"rgb": "valid", "depth": "invalid", "heading": "valid"}}

    def clock(self):
        return self.now

    def adapter(self, responses=None, **kwargs):
        fixture = FixtureTransport(responses, now_ms=self.clock)
        adapter = MainControlAdapter(fixture, clock_id=fixture.clock_id,
                                     host_boot_id=fixture.host_boot_id,
                                     evidence_mode="synthetic", now_ms=self.clock, **kwargs)
        return adapter, fixture

    def assert_error(self, result, code):
        self.assertNotEqual(result["status"], "ok")
        self.assertEqual(result["error"]["code"], code)

    def test_invalid_request_parameters_return_errors_not_exceptions(self):
        adapter, fixture = self.adapter()
        for kwargs in ({"session_id": 3}, {"request_id": []}, {"timeout_ms": float("nan")}, {"timeout_ms": -1}, {"track_id": {}}, {"fields": "navigation"}):
            with self.subTest(kwargs=kwargs):
                self.assert_error(adapter.scene_query(**kwargs), "INVALID_PAYLOAD")
        self.assert_error(adapter.control_intent({}, "开始引导", utterance_id="u", is_final=True), "INVALID_PAYLOAD")
        self.assert_error(adapter.control_intent("start_guidance", "开始引导", utterance_id="u", is_final=True, request_id=["x"]), "INVALID_PAYLOAD")
        self.assert_error(adapter.acquire_frame(session_id=[]), "INVALID_PAYLOAD")
        self.assertEqual(len(fixture.requests), 0)

    def test_default_has_no_address_no_fallback(self):
        adapter = MainControlAdapter(now_ms=self.clock)
        self.assertFalse(adapter.status()["connected"])
        for result in (adapter.scene_query(), adapter.acquire_frame(), adapter.control_intent(
                "start_guidance", "开始引导", utterance_id="u", is_final=True)):
            self.assert_error(result, "DEPENDENCY_UNAVAILABLE")
            self.assertEqual(result["producer"]["role"], "interaction")
        self.assertFalse(adapter.status()["transport_configured"])

    def test_status_does_not_call_transport(self):
        def forbidden(_):
            self.fail("健康查询不应触发传输")
        self.assertFalse(MainControlAdapter(forbidden).status()["connected"])

    def test_status_does_not_wait_for_blocked_request(self):
        entered, release, status_done = threading.Event(), threading.Event(), threading.Event()
        def blocked(_):
            entered.set()
            release.wait(2)
            return {"data": self.scene}
        adapter, _ = self.adapter({"scene.query": blocked})
        worker = threading.Thread(target=adapter.scene_query)
        worker.start()
        self.assertTrue(entered.wait(1))
        reader = threading.Thread(target=lambda: (adapter.status(), status_done.set()))
        reader.start()
        try:
            self.assertTrue(status_done.wait(0.2), "诊断不应等待主控请求锁")
        finally:
            release.set()
            worker.join(2)
            reader.join(2)

    def test_fixture_must_be_explicit(self):
        with self.assertRaises(ValueError):
            FixtureTransport(evidence_mode="live")
        fixture = FixtureTransport(now_ms=self.clock)
        live = MainControlAdapter(fixture, now_ms=self.clock)
        self.assert_error(live.handshake(), "INVALID_PAYLOAD")

    def test_clock_boot_identity_must_match(self):
        adapter, fixture = self.adapter()
        fixture.host_boot_id = "another-boot"
        self.assert_error(adapter.handshake(), "CLOCK_UNVERIFIED")
        self.assertFalse(adapter.status()["connected"])

    def test_missing_capability_not_treated_as_ready(self):
        adapter, fixture = self.adapter()
        self.assert_error(adapter.scene_query(), "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(len(fixture.requests), 1)
        self.assertEqual(fixture.requests[0]["type"], "capabilities.get")

    def test_scene_keeps_invalid_distance_null_and_source(self):
        adapter, fixture = self.adapter({"scene.query": {"data": self.scene}})
        result = adapter.scene_query(session_id="s", request_id="q")
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["data"]["objects"][0]["measurement"]["distance_m"])
        self.assertEqual(result["evidence_mode"], "synthetic")
        self.assertEqual(result["request_id"], "q")
        self.assertTrue(adapter.scene_is_current(result))
        self.now = 1500
        self.assertFalse(adapter.scene_is_current(result))

    def test_expired_state_never_becomes_current(self):
        self.scene["valid_until_mono_ms"] = self.now
        adapter, _ = self.adapter({"scene.query": {"data": self.scene}})
        self.assert_error(adapter.scene_query(), "STALE_STATE")

    def test_invalid_distance_cannot_be_zero(self):
        self.scene["objects"][0]["measurement"]["distance_m"] = 0
        adapter, _ = self.adapter({"scene.query": {"data": self.scene}})
        self.assert_error(adapter.scene_query(), "INVALID_PAYLOAD")

    def test_shorter_measurement_expiry_checked(self):
        measurement = self.scene["objects"][0]["measurement"]
        measurement.update(valid=True, distance_m=1.2, valid_until_mono_ms=1100)
        adapter, _ = self.adapter({"scene.query": {"data": self.scene}})
        result = adapter.scene_query()
        self.assertTrue(adapter.scene_is_current(result))
        self.now = 1100
        self.assertFalse(adapter.scene_is_current(result))
        self.assert_error(adapter.scene_query(), "STALE_STATE")

    def test_restart_invalidates_previous_state_requires_handshake(self):
        adapter, fixture = self.adapter({"scene.query": {"data": self.scene}})
        first = adapter.scene_query()
        fixture.instance_id = "restarted-main"
        self.assert_error(adapter.scene_query(), "CLOCK_UNVERIFIED")
        self.assertFalse(adapter.scene_is_current(first))
        self.assertFalse(adapter.status()["connected"])
        self.assertEqual(adapter.scene_query()["status"], "ok")
        self.assertEqual(sum(req["type"] == "capabilities.get" for req in fixture.requests), 2)

    def test_sequence_rollback_invalidates(self):
        adapter, fixture = self.adapter({"scene.query": {"data": self.scene}})
        first = adapter.scene_query()
        lower = deepcopy(self.scene)
        lower.update(sequence=0, state_id="old")
        fixture.responses["scene.query"] = {"data": lower}
        self.assert_error(adapter.scene_query(), "STALE_STATE")
        self.assertFalse(adapter.scene_is_current(first))
        self.assertFalse(adapter.status()["connected"])

    def test_protocol_version_and_correlation_rejected(self):
        for override, expected in (({"schema_version": "interaction.v9"}, "UNSUPPORTED_VERSION"),
                                   ({"reply_to": "other-request"}, "INVALID_PAYLOAD"),
                                   ({"session_id": "another-user"}, "INVALID_PAYLOAD")):
            with self.subTest(override=override):
                adapter, fixture = self.adapter({"scene.query": {"data": self.scene, **override}})
                self.assert_error(adapter.scene_query(), expected)

    def test_late_result_discarded(self):
        def late(_):
            self.now += 501
            return {"data": self.scene}
        adapter, _ = self.adapter({"scene.query": late})
        self.assert_error(adapter.scene_query(timeout_ms=500), "DEADLINE_EXCEEDED")

    def test_transport_disconnect_invalidates(self):
        adapter, fixture = self.adapter({"scene.query": {"data": self.scene}})
        first = adapter.scene_query()
        def disconnect(_):
            raise ConnectionError("断开")
        adapter.transport = disconnect
        self.assert_error(adapter.scene_query(), "DEPENDENCY_UNAVAILABLE")
        self.assertFalse(adapter.scene_is_current(first))

    def test_partial_asr_cannot_submit_control(self):
        adapter, fixture = self.adapter({"control.intent": {"data": {}}})
        self.assert_error(adapter.control_intent("start_guidance", "开始", utterance_id="u", is_final=False), "INVALID_PAYLOAD")
        self.assertEqual(len(fixture.requests), 0)

    def test_final_negation_does_not_submit_keyword_control(self):
        adapter, fixture = self.adapter({"control.intent": {"data": {}}})
        for transcript in ("不要开始引导", "不是让你结束引导", "别暂停"):
            self.assert_error(adapter.control_intent("start_guidance", transcript, utterance_id="u", is_final=True), "INVALID_PAYLOAD")
        self.assertEqual(len(fixture.requests), 0)

    def test_malformed_response_returns_error_instead_of_exception(self):
        adapter, fixture = self.adapter()
        base = fixture.__call__
        def malformed(request):
            result = base(request)
            result["data"]["supported_versions"] = None
            return result
        adapter.transport = malformed
        self.assert_error(adapter.handshake(), "UNSUPPORTED_VERSION")
        self.assertFalse(adapter.status()["connected"])

    def test_scene_without_measurement_has_no_distance_claim(self):
        self.scene["objects"][0]["measurement"] = None
        adapter, _ = self.adapter({"scene.query": {"data": self.scene}})
        result = adapter.scene_query()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(adapter.scene_is_current(result))

    def test_control_retry_keeps_business_id_and_pending_is_not_completed(self):
        state = {"accepted": True, "operation_state": "pending", "actual_mode": "idle",
                 "reason_code": "WAITING", "state_id": None}
        adapter, fixture = self.adapter({"control.intent": {"data": state}})
        for _ in range(2):
            result = adapter.control_intent("start_guidance", "开始引导", utterance_id="u", is_final=True, request_id="same", session_id="s")
            self.assertEqual(result["data"]["operation_state"], "pending")
        requests = [r for r in fixture.requests if r["type"] == "control.intent"]
        self.assertEqual([r["request_id"] for r in requests], ["same", "same"])
        self.assertNotEqual(requests[0]["message_id"], requests[1]["message_id"])
        self.assert_error(adapter.control_intent("end_guidance", "结束引导", utterance_id="u2", is_final=True, request_id="same", session_id="s"), "INVALID_PAYLOAD")

    def test_control_requires_actual_result_fields(self):
        adapter, _ = self.adapter({"control.intent": {"data": {"accepted": True}}})
        self.assert_error(adapter.control_intent("start_guidance", "开始引导", utterance_id="u", is_final=True), "INVALID_PAYLOAD")

    def frame(self, path, content):
        return {"frame_id": "frame-1", "captured_mono_ms": 950, "source_kind": "fixture", "is_historical": False,
                "width_px": 32, "height_px": 32, "color_space": "RGB", "quality": {"sharpness": "unknown"},
                "parent_frame_id": None, "transform_to_parent": None,
                "asset": {"asset_id": "asset-1", "transport": "file", "path": str(path), "encoding": "jpeg",
                          "sha256": hashlib.sha256(content).hexdigest(), "available_until_mono_ms": 1400}}

    def test_file_asset_bytes_hash_lease_and_authorized_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            path.write_bytes(b"fixed-test-bytes")
            frame = self.frame(path, b"fixed-test-bytes")
            adapter, _ = self.adapter({"frame.acquire": {"data": frame}}, asset_roots=[directory])
            result = adapter.acquire_frame()
            self.assertEqual(result["status"], "ok")
            self.assertEqual(adapter.read_frame_asset(result), b"fixed-test-bytes")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ProtocolError, "摘要"):
                adapter.read_frame_asset(result)
            self.now = 1400
            with self.assertRaisesRegex(ProtocolError, "租期"):
                adapter.read_frame_asset(result)

    def test_asset_outside_authorized_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            path.write_bytes(b"sample")
            adapter, _ = self.adapter({"frame.acquire": {"data": self.frame(path, b"sample")}})
            result = adapter.acquire_frame()
            with self.assertRaisesRegex(ProtocolError, "授权根目录"):
                adapter.read_frame_asset(result)

    def test_frame_identity_cannot_silently_change(self):
        frame = self.frame("does-not-open.jpg", b"first")
        adapter, fixture = self.adapter({"frame.acquire": {"data": frame}})
        self.assertEqual(adapter.acquire_frame()["status"], "ok")
        changed = deepcopy(frame)
        changed["asset"]["sha256"] = hashlib.sha256(b"other").hexdigest()
        fixture.responses["frame.acquire"] = {"data": changed}
        self.assert_error(adapter.acquire_frame(), "INVALID_FRAME_REF")

    def test_unknown_time_upload_only_allowed_as_history(self):
        frame = self.frame("unused.jpg", b"sample")
        frame.update(captured_mono_ms=None, source_kind="upload", is_historical=True)
        adapter, _ = self.adapter({"frame.acquire": {"data": frame}})
        self.assert_error(adapter.acquire_frame(), "STALE_FRAME")
        result = adapter.acquire_frame(mode="pinned", frame_id="frame-1")
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["data"]["captured_mono_ms"])

    def test_roi_must_be_bound_to_pinned_frame(self):
        adapter, fixture = self.adapter()
        self.assert_error(adapter.acquire_frame(roi_xyxy_px=[0, 0, 20, 20]), "INVALID_FRAME_REF")
        self.assertEqual(len(fixture.requests), 0)

    def test_target_entry_explicitly_unavailable_never_forwards(self):
        adapter, fixture = self.adapter({"target.observation": {"data": {"track_id": "forged"}}})
        for observation in ({"visual_target_id": "v", "frame_id": "unknown", "roi_xyxy_px": ["x", None, -10, float("nan")]},
                            {"visual_target_id": "v", "frame_id": "frame-1", "roi_xyxy_px": [0, 0, 10, 10]}):
            result = adapter.target_observation(observation)
            self.assert_error(result, "DEPENDENCY_UNAVAILABLE")
            self.assertEqual(result["status"], "unavailable")
        self.assertEqual(len(fixture.requests), 0)
        self.assertIn("target.observation", adapter.status()["not_implemented"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
