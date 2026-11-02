"""v1.0.0 测试台与设备相机的兼容性。"""
import base64
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

    def test_status_exposes_model_and_machine_metrics(self):
        data = self.client.get("/api/status").json
        self.assertIn("mem_used_gb", data["system"])
        self.assertIn("system_percent", data["cpu"])

    def test_ask_visual_without_image_does_not_call_model(self):
        with mock.patch.object(server, "run_eyes") as run_eyes:
            data = self.client.post("/api/ask", json={"text": "找卫生间"}).json
        self.assertIn("还没有画面", data["answer"])
        run_eyes.assert_not_called()

    def test_ask_chat_ignores_available_image(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
        with mock.patch.object(server, "run_eyes", return_value={
                "task": "chat", "source": "vlm", "answer": "我在听", "latency_ms": 1}) as run_eyes:
            data = self.client.post("/api/ask", json={"text": "今天有点烦", "image": image}).json
        self.assertEqual(data["intent"], "chat")
        self.assertEqual(run_eyes.call_args.args[0], "chat")
        self.assertNotIn("image_b64", run_eyes.call_args.kwargs)

    def test_ask_toilet_with_image_routes_toilet_task(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
        with mock.patch.object(server, "run_eyes", return_value={
                "task": "toilet", "source": "vlm", "answer": "画面里没看到卫生间标识", "latency_ms": 1}) as run_eyes:
            data = self.client.post("/api/ask", json={"text": "找卫生间", "image": image}).json
        self.assertEqual(data["intent"], "find_object")
        self.assertEqual(run_eyes.call_args.args[0], "toilet")
        self.assertEqual(run_eyes.call_args.kwargs["image_b64"], image)

    def test_ask_unknown_with_image_routes_followup(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
        with mock.patch.object(server, "run_eyes", return_value={
                "task": "followup", "source": "vlm", "answer": "左边是门", "latency_ms": 1}) as run_eyes:
            data = self.client.post("/api/ask", json={"text": "左边那是什么", "image": image}).json
        self.assertEqual(data["intent"], "followup")
        self.assertEqual(run_eyes.call_args.args[0], "followup")
        self.assertEqual(run_eyes.call_args.kwargs["image_b64"], image)

if __name__ == "__main__": unittest.main()
