# -*- coding: utf-8 -*-
"""语音控制动作与 API 回归：停止播报不得关闭引导。"""

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


class VoiceActionTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        server.SESSIONS.clear()
        if server.GUIDE.status().get("running"):
            server.GUIDE.stop(announce=False)

    def tearDown(self):
        if server.GUIDE.status().get("running"):
            server.GUIDE.stop(announce=False)

    def send(self, text, **extra):
        body = {"audio": voice_audio(), "session_id": "action-test"}
        body.update(extra)
        spoken = text if server.WAKE_PATTERN.search(text) else f"小冰小冰，{text}"
        with mock.patch.object(server, "asr_transcribe", lambda samples: spoken):
            return self.client.post("/api/voice", json=body).get_json()

    def test_speech_stop_does_not_stop_guide(self):
        server.GUIDE.start("browser")
        result = self.send("停止播报")
        self.assertEqual(result["intent"], "speech_stop")
        self.assertTrue(result["cancelled"])
        self.assertTrue(server.GUIDE.status()["running"])

    def test_guide_stop_stops_guide(self):
        server.GUIDE.start("browser")
        result = self.send("关闭引导")
        self.assertEqual(result["intent"], "guide_stop")
        self.assertFalse(server.GUIDE.status()["running"])

    def test_guide_start_starts_guide(self):
        result = self.send("开始引导")
        self.assertEqual(result["intent"], "guide_start")
        self.assertTrue(server.GUIDE.status()["running"])
        self.assertTrue(result["guide"]["needs_frames"])

    def test_camera_off_requests_client_close(self):
        result = self.send("关闭摄像头")
        self.assertEqual(result["intent"], "camera_off")
        self.assertTrue(result["close_camera"])

    def test_unknown_does_not_call_model(self):
        with mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("那个东西呢")
        self.assertEqual(result["intent"], "unknown")
        self.assertIn("没听清", result["answer"])
        run_eyes.assert_not_called()

    def test_missing_image_does_not_call_model(self):
        with mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("前面有什么")
        self.assertIn("还没有画面", result["answer"])
        run_eyes.assert_not_called()

    def test_speech_stop_then_guide_still_detects(self):
        server.GUIDE.start("browser")
        self.send("停止播报")
        self.assertTrue(server.GUIDE.status()["running"])
        server.GUIDE.stop(announce=False)

    def test_ambiguous_stop_with_guide_only_stops_guide(self):
        server.GUIDE.start("browser")
        result = self.send("停一下")
        self.assertEqual(result["intent"], "guide_stop")
        self.assertFalse(server.GUIDE.status()["running"])

    def test_guide_status_fast_path_does_not_call_model(self):
        status = {
            "running": True,
            "frame_age_ms": 120,
            "frame_id": "g1",
            "decision": {"state": "straight", "reason": "clear"},
        }
        with mock.patch.object(server.GUIDE, "status", return_value=status), \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("现在能走吗")
        self.assertEqual(result["intent"], "guide_status")
        self.assertIn("直行通道可用", result["answer"])
        run_eyes.assert_not_called()

    def test_clarify_does_not_call_model(self):
        server.GUIDE.start("browser")
        with mock.patch.object(server, "recent_speech_active", lambda *a, **k: True), \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("停一下", image="data:image/jpeg;base64," + base64.b64encode(b"x").decode())
        self.assertEqual(result["intent"], "clarify")
        run_eyes.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
