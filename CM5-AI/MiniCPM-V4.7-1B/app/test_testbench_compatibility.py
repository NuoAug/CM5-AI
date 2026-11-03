"""测试台、设备相机和已移除导航功能的兼容性。"""
import base64
import unittest
from unittest import mock
import server

class TestbenchCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_health_uses_current_models_without_navigation_fields(self):
        with mock.patch.object(server, "piper_error", return_value=None), mock.patch.object(server, "over_budget", return_value=False):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertFalse(any(key.startswith("guide") for key in response.json))
        with mock.patch.object(server, "piper_error", return_value="missing Piper model"):
            self.assertEqual(self.client.get("/api/health").status_code, 503)

    def test_main_control_diagnostics_available_without_restoring_old_panel(self):
        self.assertEqual(self.client.get("/api/main-control/status").status_code, 200)
        response = self.client.get("/api/main-control/scene")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["error"]["code"], "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(self.client.post("/api/main-control/control", json={}).status_code, 400)
        with self.client.get("/") as response:
            page = response.get_data(as_text=True)
        self.assertNotIn("refreshMainControl", page)
        self.assertNotIn("btn-mc-", page)

    def test_web_preview_starts_camera_without_navigation_callback(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "standalone-device-test"), mock.patch.object(server.DEVICE_CAMERA, "start", return_value=True) as start:
            self.assertEqual(self.client.post("/api/device/camera", json={"action": "start"}).status_code, 200)
            start.assert_called_once_with()

    def test_removed_navigation_routes_are_not_available(self):
        routes = (("post", "/api/guide"), ("post", "/api/guide/start"),
                  ("post", "/api/guide/stop"), ("post", "/api/guide/frame"),
                  ("get", "/api/guide/status"))
        with mock.patch.object(server.DEVICE_CAMERA, "start") as start, mock.patch.object(server.DEVICE_CAMERA, "stop") as stop:
            for method, route in routes:
                with self.subTest(route=route):
                    self.assertEqual(getattr(self.client, method)(route).status_code, 404)
            start.assert_not_called()
            stop.assert_not_called()

    def test_status_exposes_model_and_machine_metrics(self):
        data = self.client.get("/api/status").json
        self.assertIn("mem_used_gb", data["system"])
        self.assertIn("system_percent", data["cpu"])
        self.assertNotIn("guide", data)
        self.assertNotIn("guide_model", data)
        self.assertNotIn("guide_running", data["active"])
        self.assertNotIn("guide_source", data["device"])

    def test_ask_main_control_never_falls_through_to_model(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
        for text in ("开始引导", "关闭引导", "引导状态", "现在能走吗", "前面能不能走楼梯"):
            for picture in ("", image):
                with self.subTest(text=text, has_image=bool(picture)), mock.patch.object(server, "run_eyes") as run_eyes:
                    response = self.client.post("/api/ask", json={"text": text, "image": picture})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(response.json["intent"], ("control_intent", "scene_query"))
                    self.assertEqual(response.json["source"], "main_control_adapter")
                    self.assertIn("主控当前不可用", response.json["answer"])
                    self.assertEqual(response.json["task"]["actions"], [])
                    self.assertFalse(response.json["task"]["needs_image"])
                    run_eyes.assert_not_called()

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

    def test_ask_text_item_routes_item_task(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
        with mock.patch.object(server, "run_eyes", return_value={
                "task": "item", "source": "vlm", "answer": "水杯在右边", "latency_ms": 1}) as run_eyes:
            data = self.client.post("/api/ask", json={"text": "帮我找一下水杯", "image": image}).json
        self.assertEqual(data["intent"], "find_object")
        self.assertEqual(run_eyes.call_args.args[0], "item")
        self.assertEqual(run_eyes.call_args.kwargs["item"], "水杯")

    def test_ask_unknown_with_image_routes_followup(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
        with mock.patch.object(server, "run_eyes", return_value={
                "task": "followup", "source": "vlm", "answer": "左边是门", "latency_ms": 1}) as run_eyes:
            data = self.client.post("/api/ask", json={"text": "左边那是什么", "image": image}).json
        self.assertEqual(data["intent"], "followup")
        self.assertEqual(run_eyes.call_args.args[0], "followup")
        self.assertEqual(run_eyes.call_args.kwargs["image_b64"], image)

if __name__ == "__main__": unittest.main()
