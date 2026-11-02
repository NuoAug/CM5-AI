# -*- coding: utf-8 -*-
"""语音控制动作与已移除导航功能的 API 回归。"""

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

    def send(self, text, **extra):
        body = {"audio": voice_audio(), "session_id": "action-test"}
        body.update(extra)
        spoken = text if server.WAKE_PATTERN.search(text) else f"小冰小冰，{text}"
        with mock.patch.object(server, "asr_transcribe", lambda samples: spoken):
            return self.client.post("/api/voice", json=body).get_json()

    def test_speech_stop_cancels_output_without_stopping_camera(self):
        with mock.patch.object(server.OUTPUT, "cancel") as cancel, \
                mock.patch.object(server.DEVICE_CAMERA, "stop") as camera_stop:
            result = self.send("停止播报")
        self.assertEqual(result["intent"], "speech_stop")
        self.assertTrue(result["cancelled"])
        cancel.assert_called_once()
        camera_stop.assert_not_called()

    def test_removed_navigation_requests_do_not_call_model_or_camera(self):
        for text in ("开始引导", "关闭引导", "引导状态", "现在能走吗", "前面能不能走楼梯"):
            with self.subTest(text=text):
                server.SESSIONS.clear()
                with mock.patch.object(server, "run_eyes") as run_eyes, \
                        mock.patch.object(server.DEVICE_CAMERA, "start") as camera_start, \
                        mock.patch.object(server.DEVICE_CAMERA, "stop") as camera_stop:
                    result = self.send(text)
                self.assertEqual(result["intent"], "unsupported")
                self.assertIn("不提供引导或通行判断", result["answer"])
                self.assertEqual(result["task"]["actions"], [])
                self.assertFalse(result["task"]["needs_image"])
                self.assertTrue(result["spoken"])
                run_eyes.assert_not_called()
                camera_start.assert_not_called()
                camera_stop.assert_not_called()

    def test_camera_off_requests_client_close(self):
        with mock.patch.object(server.DEVICE_CAMERA, "stop") as camera_stop:
            result = self.send("关闭摄像头")
        self.assertEqual(result["intent"], "camera_off")
        self.assertTrue(result["close_camera"])
        camera_stop.assert_called_once()
        self.assertNotIn("引导", result["answer"])

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

    def test_ambiguous_stop_with_speech_cancels_output(self):
        with mock.patch.object(server, "recent_speech_active", return_value=True), \
                mock.patch.object(server.OUTPUT, "cancel") as cancel, \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("停一下")
        self.assertEqual(result["intent"], "speech_stop")
        self.assertTrue(result["cancelled"])
        cancel.assert_called_once()
        run_eyes.assert_not_called()

    def test_ambiguous_stop_without_speech_does_not_call_model(self):
        with mock.patch.object(server, "recent_speech_active", return_value=False), \
                mock.patch.object(server, "run_eyes") as run_eyes:
            result = self.send("停一下", image="data:image/jpeg;base64," + base64.b64encode(b"x").decode())
        self.assertEqual(result["intent"], "clarify")
        self.assertIn("没有正在进行的播报", result["answer"])
        run_eyes.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
