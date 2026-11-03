"""音量调节只写入配置的播放控制，非法值不能到达 ALSA。"""
import unittest
from unittest import mock
import device_volume
import server

SAMPLE = """Simple mixer control 'PCM',0
  Capabilities: pvolume pswitch
  Limits: Playback 0 - 60
  Front Left: Playback 37 [62%] [-23.00dB] [on]
  Front Right: Playback 37 [62%] [-23.00dB] [on]
"""

class DeviceVolumeTests(unittest.TestCase):
    def test_read_stereo_playback(self):
        with mock.patch.object(device_volume, "_mixer", return_value=(["amixer", "-c", "Array"], "Array", "PCM,0")), mock.patch.object(device_volume, "_run", return_value=SAMPLE):
            state = device_volume.get_volume()
            self.assertEqual(state["volume_percent"], 62)
            self.assertFalse(state["muted"])
            self.assertTrue(state["mute_supported"])
            with mock.patch.object(device_volume, "_run", return_value=SAMPLE.replace("[on]", "[off]")):
                self.assertTrue(device_volume.get_volume()["muted"])

    def test_set_uses_output_control(self):
        with mock.patch.object(device_volume, "_mixer", return_value=(["amixer", "-c", "Array"], "Array", "PCM,0")), mock.patch.object(device_volume, "_run", return_value=SAMPLE) as run:
            device_volume.set_volume(55, True)
            self.assertIn(mock.call(["amixer", "-c", "Array", "sset", "PCM,0", "55%", "mute"]), run.call_args_list)

    def test_reject_invalid_without_touching_mixer(self):
        with mock.patch.object(device_volume, "_mixer") as mixer:
            for value in [-1, 101, True, "50", 1.5]:
                with self.assertRaises(ValueError): device_volume.set_volume(value)
            with self.assertRaises(ValueError): device_volume.set_volume(muted="false")
            with self.assertRaises(ValueError): device_volume.set_volume()
            mixer.assert_not_called()

    def test_missing_mixer_is_explicit_failure(self):
        with mock.patch.object(device_volume.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError): device_volume.get_volume()

    def test_capture_control_is_not_accepted(self):
        with mock.patch.object(device_volume, "_mixer", return_value=(["amixer"], "Array", "PCM,1")), mock.patch.object(device_volume, "_run", return_value="Capabilities: cvolume\nCapture 20 [60%]"):
            with self.assertRaises(RuntimeError): device_volume.get_volume()

    def test_api_mode_errors_and_set(self):
        client = server.app.test_client()
        with mock.patch.object(server, "OUTPUT_SINK", "browser"):
            self.assertEqual(client.get("/api/device/volume").status_code, 409)
        with mock.patch.object(server, "OUTPUT_SINK", "alsa"), mock.patch.object(device_volume, "get_volume", side_effect=RuntimeError("missing")):
            self.assertEqual(client.get("/api/device/volume").status_code, 503)
        with mock.patch.object(server, "OUTPUT_SINK", "alsa"), mock.patch.object(device_volume, "set_volume", return_value={"volume_percent": 60}) as setter:
            response = client.post("/api/device/volume", json={"volume_percent": 60})
            self.assertEqual(response.json["volume_percent"], 60)
            setter.assert_called_once_with(60, None)
            self.assertEqual(client.post("/api/device/volume", json=[]).status_code, 400)

if __name__ == "__main__": unittest.main()
