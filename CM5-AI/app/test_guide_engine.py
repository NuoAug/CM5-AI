# -*- coding: utf-8 -*-
"""引导几何、质量和状态机单元测试。使用确定性合成掩码，不加载模型。"""

import unittest

import numpy as np
from PIL import Image

import guide

H, W = 72, 96


def pred_base(value=11):
    return np.full((H, W), value, dtype=np.int64)


def conf_base(value=0.9):
    return np.full((H, W), value, dtype=np.float32)


def ent_base(value=0.0):
    return np.zeros((H, W), dtype=np.float32)


def fill(pred, cls, y0, y1, x0, x1):
    pred[int(H * y0):int(H * y1), int(W * x0):int(W * x1)] = cls
    return pred


def analyze(pred, conf=None, entropy=None, **kwargs):
    return guide.analyze_pred(
        pred,
        conf if conf is not None else conf_base(),
        entropy if entropy is not None else ent_base(),
        **kwargs,
    )


class GeometryTests(unittest.TestCase):
    def test_clear_corridor_straight(self):
        result = analyze(pred_base())
        self.assertEqual(result["raw_direction"], "straight")
        self.assertEqual(result["raw_reason"], "clear")

    def test_no_seed_is_unavailable(self):
        result = analyze(pred_base(value=0))
        self.assertEqual(result["raw_direction"], "unavailable")
        self.assertEqual(result["raw_reason"], "no_seed")

    def test_stairs_stop(self):
        p = fill(pred_base(), 53, 0.6, 0.9, 0.2, 0.8)
        self.assertEqual(analyze(p)["raw_reason"], "stairs")

    def test_road_stop(self):
        p = fill(pred_base(), 6, 0.55, 0.95, 0.1, 0.9)
        self.assertEqual(analyze(p)["raw_reason"], "road")

    def test_person_stop(self):
        p = fill(pred_base(), 12, 0.55, 0.95, 0.3, 0.7)
        self.assertEqual(analyze(p)["raw_reason"], "person")

    def test_obstacle_ahead_blocks_center(self):
        # 障碍物在近处地面上方，但仍在向前看的范围内；仅靠近处连通性会漏掉。
        p = fill(pred_base(), 127, 0.40, 0.60, 0.31, 0.69)
        p = fill(p, 127, 0.40, 0.60, 0.05, 0.95)
        result = analyze(p)
        self.assertNotEqual(result["raw_direction"], "straight")
        self.assertEqual(result["raw_reason"], "obstacle")

    def test_left_channel_when_center_narrow(self):
        p = pred_base()
        fill(p, 0, 0.60, 0.82, 0.40, 0.60)
        fill(p, 0, 0.60, 0.82, 0.80, 1.0)
        result = analyze(p)
        self.assertIn(result["raw_direction"], ("left", "right"))
        self.assertEqual(result["raw_reason"], "narrow")

    def test_unknown_high_entropy(self):
        result = analyze(pred_base(), conf=conf_base(0.4), entropy=np.full((H, W), 0.9, dtype=np.float32))
        self.assertEqual(result["raw_direction"], "unavailable")
        self.assertEqual(result["raw_reason"], "uncertain")

    def test_low_quality_short_circuits(self):
        result = analyze(pred_base(), quality={"ok": False, "reasons": ["too_dark"]})
        self.assertEqual(result["raw_reason"], "low_quality")

    def test_imu_tilt(self):
        imu = guide.ImuFrame(pitch_deg=20.0, roll_deg=0.0, source="replay")
        result = analyze(pred_base(), imu=imu)
        self.assertEqual(result["raw_reason"], "tilted")

    def test_calibrated_depth_stops(self):
        depth = guide.DepthFrame(
            depth_m=np.full((48, 64), 0.3, dtype=np.float32),
            valid=True,
            calibrated=True,
            source="replay",
        )
        result = analyze(pred_base(), depth=depth)
        self.assertEqual(result["raw_reason"], "depth_obstacle")
        self.assertAlmostEqual(result["metrics"]["depth_near_m"], 0.3, places=2)

    def test_uncalibrated_depth_does_not_stop(self):
        depth = guide.DepthFrame(
            depth_m=np.full((48, 64), 0.3, dtype=np.float32),
            valid=True,
            calibrated=False,
            source="replay",
        )
        result = analyze(pred_base(), depth=depth)
        self.assertEqual(result["raw_direction"], "straight")

    def test_pixel_width_not_calibrated(self):
        result = analyze(pred_base())
        self.assertFalse(result["metrics"]["calibrated"])
        self.assertEqual(result["metrics"]["width_unit"], "pixel")


class QualityTests(unittest.TestCase):
    def test_dark_image(self):
        img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        q = guide.image_quality(img)
        self.assertFalse(q["ok"])
        self.assertIn("too_dark", q["reasons"])

    def test_white_image_overexposed(self):
        img = Image.fromarray(np.full((H, W, 3), 255, dtype=np.uint8))
        q = guide.image_quality(img)
        self.assertFalse(q["ok"])
        self.assertIn("overexposed", q["reasons"])

    def test_textured_image_ok(self):
        rng = np.random.default_rng(7)
        arr = rng.integers(60, 200, size=(H, W, 3), dtype=np.uint8)
        q = guide.image_quality(Image.fromarray(arr))
        self.assertTrue(q["ok"], q)


class StateMachineTests(unittest.TestCase):
    def obs(self, direction, reason="clear"):
        return {"raw_direction": direction, "raw_reason": reason}

    def test_straight_needs_confirmation(self):
        sm = guide.GuideStateMachine()
        self.assertEqual(sm.update(self.obs("straight"), now=0.0)["state"], "unavailable")
        d = sm.update(self.obs("straight"), now=0.7)
        self.assertEqual(d["state"], "straight")
        self.assertTrue(d["changed"])

    def test_danger_stop_is_immediate(self):
        sm = guide.GuideStateMachine()
        sm.update(self.obs("straight"), now=0.0)
        sm.update(self.obs("straight"), now=0.7)
        d = sm.update(self.obs("stop", "stairs"), now=1.0)
        self.assertEqual(d["state"], "stop")
        self.assertEqual(d["reason"], "stairs")
        self.assertTrue(d["urgent"])
        self.assertTrue(d["changed"])

    def test_stop_holds_before_recovery(self):
        sm = guide.GuideStateMachine()
        sm.update(self.obs("stop", "stairs"), now=1.0)
        d = sm.update(self.obs("straight"), now=2.0)
        self.assertEqual(d["state"], "stop")
        self.assertTrue(d["holding"])

    def test_recovery_needs_frames_and_time(self):
        sm = guide.GuideStateMachine()
        sm.update(self.obs("stop", "stairs"), now=1.0)
        # 保持 1.5s，且恢复需要连续多帧、至少 0.8s
        for t in (2.6, 2.8, 3.0, 3.2, 3.4, 3.6):
            d = sm.update(self.obs("straight"), now=t)
            self.assertEqual(d["state"], "stop")
        d = sm.update(self.obs("straight"), now=3.8)
        self.assertEqual(d["state"], "straight")

    def test_left_switch_hysteresis(self):
        sm = guide.GuideStateMachine()
        sm.update(self.obs("straight"), now=0.0)
        sm.update(self.obs("straight"), now=0.7)
        d = sm.update(self.obs("left", "narrow"), now=1.0)
        self.assertEqual(d["state"], "straight")
        d = sm.update(self.obs("left", "narrow"), now=1.3)
        self.assertEqual(d["state"], "straight")
        d = sm.update(self.obs("left", "narrow"), now=1.7)
        self.assertEqual(d["state"], "left")

    def test_straight_repeat_keeps_reason(self):
        sm = guide.GuideStateMachine()
        sm.update(self.obs("straight"), now=0.0)
        sm.update(self.obs("straight"), now=0.7)
        d = sm.update(self.obs("straight", "narrow"), now=1.4)
        self.assertEqual(d["state"], "straight")
        self.assertEqual(d["reason"], "narrow")

    def test_unavailable_resets_state(self):
        sm = guide.GuideStateMachine()
        sm.update(self.obs("straight"), now=0.0)
        sm.update(self.obs("straight"), now=0.7)
        d = sm.update(self.obs("unavailable", "stale"), now=1.4)
        self.assertEqual(d["state"], "unavailable")
        self.assertEqual(d["reason"], "stale")
        self.assertTrue(d["urgent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
