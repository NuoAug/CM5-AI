# -*- coding: utf-8 -*-
"""服务端引导循环测试：最新帧、过期、看门狗、停止。"""

import threading
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import guide
import guide_service
from output import OUTPUT, PRIORITY_DANGER


def fake_obs(direction="straight", reason="clear", sleep=0.0):
    if sleep:
        time.sleep(sleep)
    return {
        "raw_direction": direction,
        "raw_reason": reason,
        "metrics": {},
        "quality": {"ok": True, "reasons": []},
        "sensor": {"depth": {"available": False}, "imu": {"available": False}},
        "timing": {"inference_ms": 5},
        "frame": {},
    }


def make_image():
    return Image.fromarray(np.full((72, 96, 3), 128, dtype=np.uint8))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = guide_service.GuideRuntime()
        self.runtime._mode = "browser"

    def tearDown(self):
        self.runtime.stop(announce=False)
        time.sleep(0.1)

    def test_continuous_frames_reach_straight(self):
        with mock.patch.object(guide, "analyze_image", lambda img, frame=None: fake_obs()):
            self.runtime.start("browser")
            for i in range(6):
                self.runtime.submit_frame(make_image(), frame_id=f"f{i}")
                time.sleep(0.18)
            time.sleep(0.2)
            status = self.runtime.status()
            self.assertEqual(status["decision"]["state"], "straight")
            self.assertGreaterEqual(status["stats"]["processed"], 2)

    def test_latest_frame_replaces_older(self):
        with mock.patch.object(guide, "analyze_image", lambda img, frame=None: fake_obs(sleep=0.05)):
            self.runtime.start("browser")
            for i in range(10):
                self.runtime.submit_frame(make_image(), frame_id=f"q{i}")
            time.sleep(0.3)
            status = self.runtime.status()
            self.assertGreaterEqual(status["stats"]["dropped"], 1)
            self.assertLessEqual(status["stats"]["processed"], 10)

    def test_stale_frame_becomes_unavailable(self):
        with mock.patch.object(guide, "analyze_image", lambda img, frame=None: fake_obs()):
            self.runtime.start("browser")
            self.runtime.submit_frame(make_image(), frame_id="one")
            time.sleep(guide.FRAME_MAX_AGE + 0.4)
            status = self.runtime.status()
            self.assertEqual(status["decision"]["state"], "unavailable")
            self.assertEqual(status["decision"]["reason"], "stale")

    def test_watchdog_marks_stuck(self):
        old = guide.WATCHDOG_SECONDS
        guide.WATCHDOG_SECONDS = 0.4
        try:
            with mock.patch.object(guide, "analyze_image", lambda img, frame=None: fake_obs(sleep=1.2)):
                self.runtime.start("browser")
                self.runtime.submit_frame(make_image(), frame_id="slow")
                time.sleep(0.8)
                status = self.runtime.status()
                self.assertTrue(status["stuck"])
                self.assertEqual(status["decision"]["reason"], "model_stuck")
        finally:
            guide.WATCHDOG_SECONDS = old

    def test_danger_pushes_priority_zero_output(self):
        q = OUTPUT.subscribe()
        try:
            with mock.patch.object(guide, "analyze_image", lambda img, frame=None: fake_obs("stop", "stairs")):
                self.runtime.start("browser")
                self.runtime.submit_frame(make_image(), frame_id="stairs")
                time.sleep(0.4)
                items = []
                while not q.empty():
                    items.append(q.get_nowait())
                self.assertTrue(any(i.get("priority") == PRIORITY_DANGER for i in items))
        finally:
            OUTPUT.unsubscribe(q)

    def test_stop_sets_unavailable_stopped(self):
        with mock.patch.object(guide, "analyze_image", lambda img, frame=None: fake_obs()):
            self.runtime.start("browser")
            time.sleep(0.1)
            self.runtime.stop(announce=False)
            status = self.runtime.status()
            self.assertFalse(status["running"])
            self.assertEqual(status["decision"]["reason"], "stopped")

    def test_needs_frames_for_browser(self):
        status = self.runtime.start("browser")
        self.assertTrue(status["needs_frames"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
