# -*- coding: utf-8 -*-
"""帧标识与 VLM 计时字段回归测试。"""

import base64
import io
import unittest
from unittest import mock

from PIL import Image

import server


def b64_image(color):
    img = Image.new("RGB", (32, 32), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class FrameIdentityTests(unittest.TestCase):
    def setUp(self):
        with server.FRAME_LOCK:
            server.FRAME_REGISTRY.clear()

    def test_same_id_different_image_gets_new_id(self):
        first = server.canonical_frame_id("f1", b64_image("red"))
        second = server.canonical_frame_id("f1", b64_image("blue"))
        self.assertEqual(first, "f1")
        self.assertNotEqual(second, "f1")
        self.assertTrue(second.startswith("f1-"))

    def test_same_id_same_image_kept(self):
        img = b64_image("green")
        self.assertEqual(server.canonical_frame_id("f2", img), "f2")
        self.assertEqual(server.canonical_frame_id("f2", img), "f2")

    def test_missing_id_generated(self):
        fid = server.canonical_frame_id("", b64_image("white"))
        self.assertTrue(fid.startswith("f-"))

    def test_registry_bounded(self):
        for i in range(80):
            server.canonical_frame_id(f"id{i}", b64_image((i * 3) % 255))
        self.assertLessEqual(len(server.FRAME_REGISTRY), 64)


class VlmTimingTests(unittest.TestCase):
    def test_prefill_and_decode_named_separately(self):
        payload = {
            "choices": [{"message": {"content": "测试"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 223, "completion_tokens": 18},
            "timings": {"prompt_ms": 2461.4, "predicted_ms": 443.1},
        }

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return payload

        with mock.patch.object(server, "VLM_URL", "http://127.0.0.1:18101"), mock.patch.object(server.ModelRunner, "probe", return_value={"ready": True}), mock.patch.object(server.requests, "post", return_value=Resp()):
            result = server.call_vlm("", "测试", max_tokens=8)
        self.assertEqual(result["prefill_ms"], 2461.4)
        self.assertEqual(result["decode_ms"], 443.1)
        self.assertNotIn("total_ms", result)
        self.assertEqual(result["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
