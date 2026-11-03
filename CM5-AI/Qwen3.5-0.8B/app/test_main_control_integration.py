"""M0主控入口回归：只验证交互适配，不代表整机或真人音频通过。"""
import json
import unittest
from unittest import mock

import server
from main_control import MainControlAdapter


class MainControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_default_unavailable_does_not_load_models(self):
        with mock.patch.object(server, "ensure_asr") as asr, \
                mock.patch.object(server, "init_ocr") as ocr, \
                mock.patch.object(server, "vlm_health") as vlm:
            data = self.client.get("/api/main-control/status").json
        self.assertFalse(data["connected"])
        asr.assert_not_called()
        ocr.assert_not_called()
        vlm.assert_not_called()

    def test_partial_and_negated_ask_never_submit_control(self):
        for text, final in (("开始引导", False), ("不要开始引导", True),
                            ("不是让你结束引导", True), ("开始引导改成暂停引导", True)):
            with self.subTest(text=text), mock.patch.object(server.MAIN_CONTROL, "control_intent") as control, \
                    mock.patch.object(server, "run_eyes") as model:
                response = self.client.post("/api/ask", json={"text": text, "is_final": final})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["intent"], "clarify")
            control.assert_not_called()
            model.assert_not_called()

    def test_explicit_control_preserves_request_identity(self):
        with mock.patch.object(server.MAIN_CONTROL, "control_intent", wraps=server.MAIN_CONTROL.control_intent) as control:
            data = self.client.post("/api/ask", json={
                "text": "暂停引导", "request_id": "control-001", "session_id": "s1",
                "utterance_id": "speech-001"}).json
        self.assertEqual(data["intent"], "control_intent")
        self.assertEqual(control.call_args.args, ("pause_guidance", "暂停引导"))
        self.assertEqual(control.call_args.kwargs["request_id"], "control-001")
        self.assertEqual(control.call_args.kwargs["utterance_id"], "speech-001")
        self.assertEqual(data["main_control"]["schema_version"], "interaction.v0.1")

    def test_debug_routes_cannot_bypass_spatial_boundary(self):
        for path, field in (("/api/talk", "question"), ("/api/eyes", "question"),
                            ("/api/vlm", "prompt")):
            with self.subTest(path=path), mock.patch.object(server, "run_eyes") as run, \
                    mock.patch.object(server, "stream_vlm") as stream, \
                    mock.patch.object(server, "call_vlm") as call:
                response = self.client.post(path, json={"task": "chat", field: "前面距离多少米"})
            self.assertEqual(response.status_code, 200)
            if path == "/api/talk":
                self.assertEqual(response.mimetype, "text/event-stream")
                data = json.loads(response.get_data(as_text=True).strip()[5:])
                self.assertTrue(data["done"])
            else:
                data = response.json
            self.assertEqual(data["intent"], "scene_query")
            self.assertEqual(data["main_control"]["status"], "unavailable")
            run.assert_not_called()
            stream.assert_not_called()
            call.assert_not_called()

    def test_memory_budget_does_not_block_main_control_intents(self):
        with mock.patch.object(server, "budget_guard", return_value=False), \
                mock.patch.object(server.MAIN_CONTROL, "control_intent", wraps=server.MAIN_CONTROL.control_intent) as control:
            response = self.client.post("/api/ask", json={"text": "暂停引导"})
        self.assertEqual(response.status_code, 200)
        control.assert_called_once()

    def test_vlm_prompt_cannot_be_hidden_by_unused_question(self):
        with mock.patch.object(server, "call_vlm") as model:
            response = self.client.post("/api/vlm", json={
                "question": "你好", "prompt": "前面距离多少米", "image": "ignored"})
        self.assertEqual(response.json["intent"], "scene_query")
        model.assert_not_called()

    def test_mixed_definition_and_walking_question_does_not_bypass_guard(self):
        for text in ("这个标牌是什么意思，我现在能走吗", "我现在能走吗，这个标牌是什么意思",
                     "距离是什么意思，它离我多远"):
            with self.subTest(text=text), mock.patch.object(server, "run_eyes") as model:
                data = self.client.post("/api/ask", json={"text": text}).json
            self.assertEqual(data["intent"], "scene_query")
            model.assert_not_called()

    def test_control_endpoint_requires_final_matching_text(self):
        bad_bodies = (
            {"action": "start_guidance", "final_transcript": "开始引导", "is_final": False},
            {"action": "start_guidance", "final_transcript": "不要开始引导", "is_final": True},
            {"action": "end_guidance", "final_transcript": "开始引导", "is_final": True},
        )
        with mock.patch.object(server.MAIN_CONTROL, "control_intent") as control:
            for body in bad_bodies:
                self.assertEqual(self.client.post("/api/main-control/control", json=body).status_code, 400)
        control.assert_not_called()

    def test_frame_endpoint_has_no_camera_fallback(self):
        with mock.patch.object(server.DEVICE_CAMERA, "start") as start, \
                mock.patch.object(server.DEVICE_CAMERA, "capture_still") as capture:
            response = self.client.post("/api/main-control/frame", json={"purpose": "ocr"})
        self.assertEqual(response.json["error"]["code"], "DEPENDENCY_UNAVAILABLE")
        start.assert_not_called()
        capture.assert_not_called()

    def test_main_control_mode_refuses_local_camera_controls(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "main-control"), \
                mock.patch.object(server.DEVICE_CAMERA, "start") as start:
            response = self.client.post("/api/device/camera", json={"action": "start"})
        self.assertEqual(response.status_code, 409)
        start.assert_not_called()

    def test_main_control_device_voice_does_not_receive_local_camera(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "main-control"), \
                mock.patch.object(server, "VOICE_SOURCE", "device"), \
                mock.patch.object(server, "OUTPUT_SINK", "browser"), \
                mock.patch.object(server, "DEVICE_VOICE", None), \
                mock.patch.object(server.DEVICE_CAMERA, "start") as start, \
                mock.patch.object(server.device_voice, "DeviceVoice") as voice:
            voice.return_value.start.return_value = True
            server.start_device_services()
        voice.assert_called_once_with(camera=None)
        start.assert_not_called()

    def test_m0_scene_facts_are_not_queued_as_ordinary_speech(self):
        response = {"status": "ok", "evidence_mode": "live", "data": {
            "navigation": {"speech_fact": "可以走"}}}
        with mock.patch.object(server.MAIN_CONTROL, "scene_query", return_value=response), \
                mock.patch.object(server.OUTPUT, "push") as push:
            data = self.client.post("/api/ask", json={"text": "现在能走吗"}).json
        self.assertFalse(data["spoken"])
        self.assertEqual(data["main_control"], response)
        self.assertIn("live_fact_speech", data["unimplemented"])
        push.assert_not_called()

    def test_control_accepted_is_not_reported_completed(self):
        response = {"status": "ok", "evidence_mode": "live", "data": {
            "accepted": True, "operation_state": "pending", "actual_mode": "idle"}}
        with mock.patch.object(server.MAIN_CONTROL, "control_intent", return_value=response):
            data = self.client.post("/api/ask", json={"text": "开始引导"}).json
        self.assertNotIn("已经开始", data["answer"])
        self.assertFalse(data["spoken"])


if __name__ == "__main__":
    unittest.main()
