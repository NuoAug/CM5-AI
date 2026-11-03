"""模拟Picamera2按官方格式返回的数组，验证preview/still颜色；不是实机。

官方来源：Picamera2手册附录A；raspberrypi/picamera2 request.py的
_get_pil_mode映射中，BGR888为RGB字节，RGB888为BGR字节。
https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py
"""
import base64
import io
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import device_io


class FormatAwarePicamera:
    """场景真值左红右蓝；根据请求的libcamera format编码通道。"""
    def __init__(self, *, lores_supported=True, probe_valid=True):
        self.lores_supported = lores_supported
        self.probe_valid = probe_valid
        self.configuration = None
        self.attempted = []
        self.started = False
        self.closed = False
        self.array_streams = []

    def create_video_configuration(self, **kwargs):
        return kwargs

    def configure(self, configuration):
        self.attempted.append(configuration)
        if 'lores' in configuration and not self.lores_supported:
            raise RuntimeError('fixture lores format unsupported')
        self.configuration = configuration

    def start(self): self.started = True
    def stop(self): self.started = False
    def close(self): self.closed = True

    def capture_array(self, stream):
        self.array_streams.append(stream)
        if stream == 'lores' and not self.probe_valid:
            return None
        config = self.configuration[stream]
        width, height = config['size']
        rgb = np.zeros((height,width,3), dtype=np.uint8)
        rgb[:, :width//2] = (240,10,5)
        rgb[:, width//2:] = (5,10,240)
        if config['format'] == 'BGR888':
            return rgb
        if config['format'] == 'RGB888':
            return rgb[:,:,::-1].copy()
        raise AssertionError('fixture only implements real 24bit camera formats')


class DeviceCameraColorTests(unittest.TestCase):
    def camera(self, **driver_kwargs):
        driver = FormatAwarePicamera(**driver_kwargs)
        module = types.SimpleNamespace(Picamera2=lambda:driver)
        patches = [mock.patch.dict(sys.modules, {'picamera2':module}),
                   mock.patch.object(device_io, 'picamera_available', return_value=True)]
        for patch in patches:
            patch.start();self.addCleanup(patch.stop)
        camera = device_io.DeviceCamera(high_size=(32,24),low_size=(16,12),fps=100)
        ready = threading.Event()
        self.assertTrue(camera.start(callback=lambda *args: ready.set()), camera.status())
        self.addCleanup(camera.stop)
        self.assertTrue(ready.wait(1), 'fixture preview did not arrive')
        return camera, driver

    def assert_red_blue(self, image):
        self.assertEqual(image.mode, 'RGB')
        width,height=image.size
        left=image.getpixel((width//4,height//2))
        right=image.getpixel((3*width//4,height//2))
        self.assertGreater(left[0],200, 'red target must remain red in PIL')
        self.assertLess(left[2],40)
        self.assertGreater(right[2],200, 'blue target must remain blue in PIL')
        self.assertLess(right[0],40)

    def test_lores_preview_keeps_red_and_blue_in_pil_order(self):
        camera,driver=self.camera()
        self.assertEqual(driver.configuration['main']['format'],'BGR888')
        self.assertEqual(driver.configuration['lores']['format'],'BGR888')
        preview=camera.preview_frame()
        self.assertIsNotNone(preview)
        self.assertEqual(preview[0].size,(16,12))
        self.assert_red_blue(preview[0])

    def test_still_keeps_original_rgb_for_ocr_vlm(self):
        camera,driver=self.camera()
        image=camera.capture_still()
        self.assertIsNotNone(image)
        self.assertEqual(image.size,(32,24))
        self.assert_red_blue(image)
        # Parent downstream paths consume encoded PIL images. PNG must not swap again.
        encoded=io.BytesIO();image.save(encoded,format='PNG')
        with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
            self.assert_red_blue(decoded)

    def test_main_only_fallback_preserves_preview_and_still_colors(self):
        camera,driver=self.camera(lores_supported=False)
        self.assertEqual(len(driver.attempted),2)
        self.assertNotIn('lores',driver.configuration)
        self.assertEqual(driver.configuration['main']['format'],'BGR888')
        self.assertFalse(camera.status()['use_lores'])
        preview=camera.preview_frame()
        self.assertEqual(preview[0].size,(16,12))
        self.assert_red_blue(preview[0])
        self.assert_red_blue(camera.capture_still())

    def test_invalid_lores_probe_falls_back_to_rgb_main(self):
        camera,driver=self.camera(probe_valid=False)
        self.assertFalse(camera.status()['use_lores'])
        self.assertIn('main',driver.array_streams)
        self.assert_red_blue(camera.preview_frame()[0])
        self.assert_red_blue(camera.capture_still())

    def test_driver_fixture_demonstrates_old_rgb888_array_would_swap(self):
        driver=FormatAwarePicamera()
        driver.configure({'main':{'size':(32,24),'format':'RGB888'}})
        wrong=Image.fromarray(driver.capture_array('main'))
        self.assertGreater(wrong.getpixel((8,12))[2],200)
        self.assertLess(wrong.getpixel((8,12))[0],40)
        driver.configure({'main':{'size':(32,24),'format':'BGR888'}})
        self.assert_red_blue(Image.fromarray(driver.capture_array('main')))


if __name__=='__main__':
    unittest.main()
