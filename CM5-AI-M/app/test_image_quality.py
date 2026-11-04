"""Synthetic regressions for refusing an extremely dark VLM frame."""
import json
import unittest

from PIL import Image, ImageDraw, ImageFilter

from image_quality import assess_image_quality


class ImageQualityTests(unittest.TestCase):
    def test_near_black_dark_noise_returns_action_and_exact_evidence(self):
        img = Image.new("RGB", (128, 72), (7, 8, 9))
        img.putpixel((31, 15), (12, 14, 20))
        before = img.tobytes()
        quality = assess_image_quality(img)
        self.assertTrue(quality["blocked"])
        self.assertEqual(quality["reason"], "near_black")
        self.assertIn("再拍一张", quality["message"])
        self.assertEqual(quality["evidence"]["image_size_px"], [128, 72])
        self.assertEqual(quality["evidence"]["min_rgb_value"], 7)
        self.assertEqual(quality["evidence"]["max_rgb_value"], 20)
        self.assertEqual(quality["evidence"]["rgb_value_range"], 13)
        self.assertEqual(img.tobytes(), before)
        json.dumps(quality)

    def test_dark_image_with_small_bright_mark_remains_available(self):
        img = Image.new("RGB", (1280, 720), (0, 0, 0))
        img.putpixel((1278, 716), (2, 120, 5))
        quality = assess_image_quality(img)
        self.assertFalse(quality["blocked"])
        self.assertIsNone(quality["message"])
        self.assertEqual(quality["evidence"]["max_rgb_value"], 120)

    def test_dark_label_with_low_but_real_contrast_remains_available(self):
        img = Image.new("L", (100, 50), 1)
        ImageDraw.Draw(img).text((10, 10), "12.50", fill=23)
        quality = assess_image_quality(img)
        self.assertFalse(quality["blocked"])
        self.assertLessEqual(quality["evidence"]["max_rgb_value"], 24)

    def test_dark_saturated_color_remains_available(self):
        quality = assess_image_quality(Image.new("RGB", (25, 25), (23, 0, 0)))
        self.assertFalse(quality["blocked"])
        self.assertEqual(quality["evidence"]["max_rgb_value"], 23)

    def test_sharpness_remains_unknown_for_smooth_and_blurry_photos(self):
        smooth = Image.new("RGB", (80, 80), (97, 97, 97))
        detailed = smooth.copy()
        ImageDraw.Draw(detailed).rectangle((10, 10, 40, 60), fill=(185, 64, 32))
        for img in (smooth, detailed.filter(ImageFilter.GaussianBlur(radius=12))):
            with self.subTest(mode=img.mode):
                quality = assess_image_quality(img)
                self.assertFalse(quality["blocked"])
                self.assertEqual(quality["evidence"]["sharpness"], "unknown")

    def test_zero_size_does_not_become_near_black(self):
        with self.assertRaises(ValueError):
            assess_image_quality(Image.new("RGB", (0, 20)))


if __name__ == "__main__":
    unittest.main()
