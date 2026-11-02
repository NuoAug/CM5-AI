# -*- coding: utf-8 -*-
"""语音控制动作与 API 回归：控制意图必须交给主控，停止播报不得关闭引导。"""

import base64
import io
import unittest
import wave
from unittest import mock

import numpy as np

import server


def voice_audio():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.tile(np.array([10000, -10000], dtype=np.int16), 8000).tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def control_result(mode="walking", state="completed", accepted=True):
    return {
        "accepted": accepted, "operation_state": state, "actual_mode": mode,
        "reason_code": "FIXTURE_OK", "state_id": "state-1",
    }


def scene_result(mode="walking", distance=1.2, valid=True, stale=False):
    return {
        "state_id": "state-1", "sequence": 1,
        "observed_mono_ms": 100, "valid_until_mono_ms": -1 if stale else 10 ** 12,
        "coordinate_frame": "body_forward",
        "navigation": {"mode": mode, "decision": "straight",
                       "reason_code": "CLEAR_AHEAD",
                       "speech_fact": "前方通道还没有确认，请用盲杖确认。"},
        "objects": [],
        "measurement": {"distance_m": distance, "valid": valid,
                        "reason_code": "DEPTH_INVALID" if not valid else None},
        "sensor_status": {"rgb": "valid", "depth": "valid" if valid else "invalid",
                          "heading": "valid"},
        "_stale": stale,
    }


class VoiceActionTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        server.SESSIONS.clear()

    def send(self, text, **extra):
        body = {"audio": voice_audio(), "session_id": "action-test"}
        body.update(extra)
        spoken = text if server.WAKE_PATTERN.search(text) else f"小冰小冰，{text}"
        with mock.patch.object(server, "asr_transcribe", lambda samples: spoken):
            return self.client.post("/api/voice", json=body).get_json()

    def test_speech_stop_does_not_call_control(self):
        with mock.patch.object(server.MAIN_CONTROL, "control") as control:
            result = self.send("停止播报")
        self.assertEqual(result["intent"], "speech_stop")
        self.assertTrue(result["cancelled"])
        control.assert_not_called()

    def test_guide_stop_sends_end_intent(self):
        with mock.patch.object(server.MAIN_CONTROL, "control", return_value=control_result("idle")) as control:
            result = self.send("关闭引导")
        self.assertEqual(result["intent"], "guide_stop")
        control.assert_called_once()
        self.assertEqual(control.call_args.args[0], "end_guidance")
        self.assertIn("已结束引导", result["answer"])

    def test_guide_start_sends_start_intent(self):
        with mock.patch.object(server.MAIN_CONTROL, "control", return_value=control_result("walking")) as control:
            result = self.send("开始引导")
        self.assertEqual(result["intent"], "guide_start")
        control.assert_called_once()
        self.assertEqual(control.call_args.args[0], "start_guidance")
        self.assertIn("已开始引导", result["answer"])

    def test_pause_and_resume_map_to_distinct_actions(self):
        with mock.patch.object(server.MAIN_CONTROL, "control", return_value=control_result("paused")) as control:
            result = self.send("暂停引导")
        self.assertEqual(result["intent"], "guide_pause")
        self.assertEqual(control.call_args.args[0], "pause_guidance")
        self.assertIn("已暂停引导", result["answer"])

        with mock.patch.object(server.MAIN_CONTROL, "control", return_value=control_result("walking")) as control:
            result = self.send("恢复引导")
        self.assertEqual(result["intent"], "guide_resume")
        self.assertEqual(control.call_args.args[0], "resume_guidance")

    def test_negated_start_is_not_sent(self):
        server.SESSIONS.clear()
        with mock.patch.object(server.MAIN_CONTROL, "control") as control:
            result = self.send("不要开始引导")
        self.assertIn(result["intent"], ("ack", "unknown", "clarify"))
        control.assert_not_called()

    def test_unavailable_control_is_explicit(self):
        with mock.patch.object(server.MAIN_CONTROL, "control", side_effect=RuntimeError("MAIN_CONTROL_URL 未配置")):
            result = self.send("开始引导")
        self.assertIn("主控不可用", result["answer"])

    def test_camera_off_requests_client_close(self):
        with mock.patch.object(server.MAIN_CONTROL, "guidance_active", return_value=False):
            result = self.send("关闭摄像头")
        self.assertEqual(result["intent"], "camera_off")
        self.assertTrue(result["close_camera"])

    def test_unknown_does_not_call_model(self):
        with mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("那个东西呢")
        self.assertEqual(result["intent"], "unknown")
        self.assertIn("没听清", result["answer"])
        run_eyes.assert_not_called()

    def test_missing_provider_frame_does_not_call_model(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "main_control"), \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("前面有什么")
        self.assertIn("主控", result["answer"])
        run_eyes.assert_not_called()

    def test_guide_status_reads_scene_without_model(self):
        with mock.patch.object(server.MAIN_CONTROL, "query_scene", return_value=scene_result()), \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("现在能走吗")
        self.assertEqual(result["intent"], "guide_status")
        self.assertIn("距离约 1.2 米", result["answer"])
        run_eyes.assert_not_called()

    def test_invalid_distance_is_not_zero(self):
        with mock.patch.object(server.MAIN_CONTROL, "query_scene",
                               return_value=scene_result(distance=None, valid=False)):
            result = self.send("现在能走吗")
        self.assertIn("暂时测不准", result["answer"])
        self.assertNotIn("0 米", result["answer"])

    def test_clarify_does_not_call_model(self):
        with mock.patch.object(server.MAIN_CONTROL, "guidance_active", return_value=True), \
                mock.patch.object(server.MAIN_CONTROL, "guidance_paused", return_value=False), \
                mock.patch.object(server, "recent_speech_active", lambda *a, **k: True), \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("停一下")
        self.assertEqual(result["intent"], "clarify")
        run_eyes.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
