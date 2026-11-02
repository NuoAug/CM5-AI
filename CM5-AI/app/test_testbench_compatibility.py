"""v1.0.0 测试台与设备相机的兼容性。"""
import unittest
from unittest import mock
import server

class TestbenchCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_optional_guide_missing_does_not_break_idle_web_health(self):
        state = {"running": False, "model": {"ready": False}}
        with mock.patch.object(server.GUIDE, "status", return_value=state), mock.patch.object(server, "piper_error", return_value=None), mock.patch.object(server, "over_budget", return_value=False):
            self.assertEqual(self.client.get("/api/health").status_code, 200)
            state["running"] = True
            self.assertEqual(self.client.get("/api/health").status_code, 503)

    def test_main_control_tab_reports_unavailable(self):
        status = self.client.get("/api/main-control/status").json["status"]
        self.assertFalse(status["connected"])
        self.assertEqual(self.client.get("/api/main-control/scene").status_code, 409)
        self.assertEqual(self.client.post("/api/main-control/control", json={"action": "start_guidance"}).status_code, 409)

    def test_web_preview_preserves_guide_frame_callback(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "standalone-device-test"), mock.patch.object(server.DEVICE_CAMERA, "start", return_value=True) as start:
            self.assertEqual(self.client.post("/api/device/camera", json={"action": "start"}).status_code, 200)
            start.assert_called_once_with(server._on_device_frame)

    def test_stop_guide_keeps_camera_for_device_voice_and_preview(self):
        with mock.patch.object(server, "VOICE_SOURCE", "device"), mock.patch.object(server.GUIDE, "stop", return_value={"mode": "device"}), mock.patch.object(server.DEVICE_CAMERA, "stop") as stop:
            self.assertEqual(self.client.post("/api/guide/stop", json={}).status_code, 200)
            stop.assert_not_called()

if __name__ == "__main__": unittest.main()
