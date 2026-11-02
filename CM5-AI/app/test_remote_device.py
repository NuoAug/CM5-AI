"""远程测试台外设路由：只操作设备，不绕过正式主控相机策略。"""
import time
import unittest
from unittest import mock
from PIL import Image
import server
from device_io import DeviceCamera


class RemoteDeviceTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_status_exposes_explicit_device_policy(self):
        with mock.patch.object(server, "VOICE_SOURCE", "device"):
            self.assertTrue(self.client.get("/api/device/status").json["remote_device_mode"])
        with mock.patch.object(server, "VOICE_SOURCE", "browser"), mock.patch.object(server, "OUTPUT_SINK", "browser"):
            self.assertFalse(self.client.get("/api/device/status").json["remote_device_mode"])

    def test_camera_controls_cannot_bypass_main_control(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "main_control"), mock.patch.object(server, "DEVICE_CAMERA") as camera:
            self.assertEqual(self.client.post("/api/device/camera", json={"action": "start"}).status_code, 409)
            self.assertEqual(self.client.get("/api/camera/preview").status_code, 409)
            camera.start.assert_not_called()

    def test_camera_start_and_preview_reuse_one_device(self):
        camera = mock.Mock()
        camera.start.return_value = True
        camera.status.return_value = {"running": True}
        camera.preview_frame.return_value = (Image.new("RGB", (64, 48)), "cam-4", "2026-11-02T12:00:00Z", 1)
        with mock.patch.object(server, "IMAGE_SOURCE", "standalone-device-test"), mock.patch.object(server, "DEVICE_CAMERA", camera):
            self.assertEqual(self.client.post("/api/device/camera", json={"action": "start"}).status_code, 200)
            response = self.client.get("/api/camera/preview")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["X-Frame-Id"], "cam-4")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            camera.capture_still.assert_not_called()

    def test_missing_preview_is_failure_not_placeholder(self):
        with mock.patch.object(server, "IMAGE_SOURCE", "standalone-device-test"), mock.patch.object(server.DEVICE_CAMERA, "preview_frame", return_value=None):
            self.assertEqual(self.client.get("/api/camera/preview").status_code, 503)

    def test_voice_controls_device_and_reports_hardware_failure(self):
        voice = mock.Mock()
        voice.status.return_value = {"running": False, "error": "mic missing"}
        voice.start.return_value = False
        with mock.patch.object(server, "VOICE_SOURCE", "device"), mock.patch.object(server, "DEVICE_VOICE", voice):
            self.assertEqual(self.client.post("/api/device/voice", json={"action": "start"}).status_code, 503)
            self.assertEqual(self.client.post("/api/device/voice", json={"action": "stop"}).status_code, 200)
            voice.stop.assert_called_once()
        with mock.patch.object(server, "VOICE_SOURCE", "browser"):
            self.assertEqual(self.client.post("/api/device/voice", json={"action": "start"}).status_code, 409)

    def test_speech_uses_output_arbiter_and_checks_device(self):
        sink = mock.Mock()
        sink.player.available.return_value = True
        with mock.patch.object(server, "OUTPUT_SINK", "alsa"), mock.patch.object(server, "DEVICE_SINK", sink), mock.patch.object(server.OUTPUT, "push_segment", return_value={}) as push:
            response = self.client.post("/api/device/speak", json={"text": "树莓派试听"})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json["playback"], "device")
            self.assertEqual(push.call_args.kwargs["source"], "remote_testbench")
            sink.player.available.return_value = False
            self.assertEqual(self.client.post("/api/device/speak", json={"text": "试听"}).status_code, 503)
            self.assertEqual(push.call_count, 1)

    def test_invalid_commands_do_not_touch_devices(self):
        for path in ("/api/device/voice", "/api/device/camera"):
            self.assertEqual(self.client.post(path, json={"action": "invalid"}).status_code, 400)
        self.assertEqual(self.client.post("/api/device/speak", json={"text": ""}).status_code, 400)

    def test_cached_preview_rejects_stopped_or_stale_camera(self):
        camera = DeviceCamera()
        camera._running = True
        camera._latest = (Image.new("RGB", (8, 8)), "cam-1", "original-time", time.monotonic())
        self.assertEqual(camera.preview_frame()[1:3], ("cam-1", "original-time"))
        camera._latest = (*camera._latest[:3], time.monotonic() - 2)
        self.assertIsNone(camera.preview_frame())
        camera._running = False
        self.assertIsNone(camera.preview_frame())


if __name__ == "__main__":
    unittest.main()
