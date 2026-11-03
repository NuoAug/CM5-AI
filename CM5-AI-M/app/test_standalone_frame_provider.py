"""构造FakeCamera返回实际PIL字节，验证独立桥接；不是实机或视觉质量通过。"""
from copy import deepcopy
import hashlib
import io
import threading
import unittest
from unittest import mock

from PIL import Image
from standalone_frame_provider import (
    DeviceTestVisualContextStore, StandaloneFrameProvider, configure_standalone_visual,
    SCHEMA, SOURCE, ROLE, EVIDENCE,
)
from visual_context import VisualContextError, VisualContextStore
from visual_bridge import VisualBridge


class Clock:
    def __init__(self): self.value = 1000
    def __call__(self): return self.value


class FakeCamera:
    def __init__(self, clock):
        self.clock = clock
        self.running = True
        self.available = True
        self.calls = 0
        self.start = mock.Mock()
        self.image = Image.new('RGB', (20, 10), (10, 20, 30))
    def status(self): return {'running': self.running, 'available': self.available}
    def capture_still(self):
        self.calls += 1
        self.clock.value += 4
        return self.image.copy() if isinstance(self.image, Image.Image) else self.image


class StandaloneFrameProviderTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.camera = FakeCamera(self.clock)
        self.provider = StandaloneFrameProvider(self.camera, now_ms=self.clock)
        self.store = DeviceTestVisualContextStore(allow_standalone_device=True, now_ms=self.clock)

    def frame(self, **extra):
        return self.provider.acquire_frame(session_id='s', **extra)

    def assert_code(self, code, func, *args, **kwargs):
        with self.assertRaises(VisualContextError) as caught:
            func(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_actual_pil_is_encoded_and_described_without_files(self):
        result = self.frame()
        raw = self.provider.read_frame_asset(result)
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.format, 'PNG')
            self.assertEqual(image.size, (20,10))
            self.assertEqual(image.getpixel((0,0)), (10,20,30))
        self.assertEqual(result['data']['width_px'], 20)
        self.assertEqual(result['data']['asset']['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['data']['asset']['size_bytes'], len(raw))
        self.assertEqual(result['data']['asset']['transport'], 'standalone_memory')
        self.assertNotIn('path', result['data']['asset'])
        self.camera.start.assert_not_called()

    def test_source_is_explicit_development_not_main_control(self):
        result = self.frame()
        self.assertEqual(result['schema_version'], SCHEMA)
        self.assertEqual(result['producer']['role'], ROLE)
        self.assertEqual(result['evidence_mode'], EVIDENCE)
        self.assertEqual(result['image_source'], SOURCE)
        self.assertTrue(result['producer']['instance_id'].startswith('standalone-device-'))
        self.assertNotEqual(result['producer']['role'], 'main_control')

    def test_time_is_capture_return_not_sensor_exposure(self):
        result = self.frame()
        data = result['data']
        self.assertEqual(data['captured_mono_ms'], 1004)
        self.assertEqual(data['capture_time_basis'], 'host_capture_return')
        self.assertIsNone(data['sensor_capture_mono_ms'])
        self.assertEqual(data['host_capture_window'], {'call_start_mono_ms':1000, 'return_mono_ms':1004})
        self.assertFalse(self.provider.stats()['sensor_capture_time_known'])

    def test_camera_not_running_returns_no_frame_without_start_or_capture(self):
        self.camera.running = False
        result = self.frame()
        self.assertEqual(result['error']['code'], 'NO_FRAME')
        self.assertEqual(self.camera.calls, 0)
        self.camera.start.assert_not_called()

    def test_no_camera_or_unavailable_returns_no_frame(self):
        self.assertEqual(StandaloneFrameProvider(None).acquire_frame(session_id='s')['error']['code'], 'NO_FRAME')
        self.camera.available = False
        self.assertEqual(self.frame()['error']['code'], 'NO_FRAME')
        self.assertEqual(self.camera.calls, 0)

    def test_none_or_wrong_capture_type_is_not_a_frame(self):
        self.camera.image = None
        self.assertEqual(self.frame()['error']['code'], 'NO_FRAME')
        self.camera.image = b'not PIL'
        self.assertEqual(self.frame()['error']['code'], 'INVALID_FRAME_REF')
        self.assertEqual(self.provider.stats()['frames'], 0)

    def test_capture_exception_returns_recoverable_no_frame(self):
        self.camera.capture_still = mock.Mock(side_effect=OSError('fixture driver failure'))
        result = self.frame()
        self.assertEqual(result['error']['code'], 'NO_FRAME')
        self.assertTrue(result['error']['retryable'])

    def test_query_validation_does_not_capture(self):
        for args in ({'mode':'unknown'}, {'purpose':'walk'}, {'mode':'pinned'}, {'frame_id':'old'}, {'max_age_ms':100}):
            with self.subTest(args=args):
                self.assertNotEqual(self.frame(**args)['status'], 'ok')
        for sid in (None, '', 'x'*129, 4):
            self.assertEqual(self.provider.acquire_frame(session_id=sid)['error']['code'], 'INVALID_PAYLOAD')
        self.assertEqual(self.camera.calls, 0)

    def test_latest_always_has_new_identity_even_same_pixels(self):
        first, second = self.frame(), self.frame()
        self.assertNotEqual(first['data']['frame_id'], second['data']['frame_id'])
        self.assertEqual(first['data']['asset']['sha256'], second['data']['asset']['sha256'])
        self.assertLess(first['data']['captured_mono_ms'], second['data']['captured_mono_ms'])

    def test_pinned_reuses_original_identity_time_and_bytes_even_camera_off(self):
        first = self.frame()
        self.camera.running = False
        self.clock.value += 20
        second = self.frame(mode='pinned', frame_id=first['data']['frame_id'])
        self.assertEqual(second, first)
        self.assertEqual(self.provider.read_frame_asset(second), self.provider.read_frame_asset(first))
        self.assertEqual(self.camera.calls, 1)

    def test_pinned_is_session_scoped(self):
        first = self.frame()
        result = self.provider.acquire_frame(session_id='other', mode='pinned', frame_id=first['data']['frame_id'])
        self.assertEqual(result['error']['code'], 'NO_FRAME')
        self.assertEqual(self.camera.calls, 1)

    def test_expired_pinned_is_stale_and_never_captures_replacement(self):
        first = self.frame()
        self.clock.value = first['data']['asset']['available_until_mono_ms']
        result = self.frame(mode='pinned', frame_id=first['data']['frame_id'])
        self.assertEqual(result['error']['code'], 'STALE_FRAME')
        self.assertEqual(self.camera.calls, 1)
        self.assert_code('STALE_FRAME', self.provider.read_frame_asset, first)
        self.assertEqual(self.provider.stats()['cached_bytes'], 0)

    def test_returned_metadata_cannot_mutate_provider_copy(self):
        first = self.frame()
        changed = deepcopy(first)
        changed['data']['asset']['sha256'] = '0'*64
        self.assert_code('INVALID_FRAME_REF', self.provider.read_frame_asset, changed)
        changed = deepcopy(first)
        changed['producer']['role'] = 'main_control'
        self.assert_code('INVALID_FRAME_REF', self.provider.read_frame_asset, changed)
        self.assertEqual(self.frame(mode='pinned', frame_id=first['data']['frame_id']), first)

    def test_live_asset_capacity_returns_busy_and_expiry_releases(self):
        self.provider = StandaloneFrameProvider(self.camera, now_ms=self.clock, max_frames=1, lease_ms=10)
        first = self.frame()
        self.assertEqual(self.frame()['error']['code'], 'BUSY')
        self.assertEqual(self.camera.calls, 1)
        self.clock.value += 11
        self.assertEqual(self.frame()['status'], 'ok')
        self.assertEqual(self.provider.stats()['frames'], 1)
        self.assert_code('STALE_FRAME', self.provider.read_frame_asset, first)

    def test_asset_byte_budget_is_bounded(self):
        self.provider = StandaloneFrameProvider(self.camera, now_ms=self.clock, max_cached_bytes=1)
        self.assertEqual(self.frame()['error']['code'], 'BUSY')
        self.assertEqual(self.provider.stats()['frames'], 0)
        self.assertEqual(self.provider.stats()['cached_bytes'], 0)

    def test_per_frame_and_pixel_limits(self):
        self.provider = StandaloneFrameProvider(self.camera, now_ms=self.clock, max_frame_bytes=1)
        self.assertEqual(self.frame()['error']['code'], 'INVALID_FRAME_REF')
        self.provider = StandaloneFrameProvider(self.camera, now_ms=self.clock, max_pixels=10)
        self.assertEqual(self.frame()['error']['code'], 'INVALID_FRAME_REF')
        self.assertEqual(self.provider.stats()['frames'], 0)

    def test_base_store_is_not_relaxed_for_development_source(self):
        result = self.frame()
        raw = self.provider.read_frame_asset(result)
        self.assert_code('UNSUPPORTED_VERSION', VisualContextStore(now_ms=self.clock).remember_frame, 's', result, asset_bytes=raw)
        self.assert_code('INVALID_PAYLOAD', DeviceTestVisualContextStore(now_ms=self.clock).remember_frame, 's', result, asset_bytes=raw)

    def test_development_store_keeps_source_and_actual_time(self):
        result = self.frame()
        frame = self.store.remember_frame('s', result, asset_bytes=self.provider.read_frame_asset(result), retain_copy=True)
        self.assertEqual(frame['producer']['role'], ROLE)
        self.assertEqual(frame['evidence_mode'], EVIDENCE)
        self.assertEqual(frame['capture_time_basis'], 'host_capture_return')
        self.assertEqual(frame['captured_mono_ms'], 1004)
        self.assertEqual(frame['width_px'], 20)

    def test_retained_copy_outlives_provider_lease_without_time_refresh(self):
        result = self.frame()
        raw = self.provider.read_frame_asset(result)
        self.store.remember_frame('s', result, asset_bytes=raw, retain_copy=True)
        self.clock.value += self.provider.lease_ms + 1
        self.assert_code('STALE_FRAME', self.provider.read_frame_asset, result)
        self.assertEqual(self.store.image_bytes('s', result['data']['frame_id']), raw)
        frame = self.store.current_frame('s')
        self.assertTrue(frame['historical_reference'])
        self.assertEqual(frame['captured_mono_ms'], 1004)

    def test_store_checks_size_digest_session_and_source(self):
        original = self.frame()
        raw = self.provider.read_frame_asset(original)
        changes = [lambda x:x['data'].update(width_px=21),
                   lambda x:x['data']['asset'].update(size_bytes=1),
                   lambda x:x['data']['asset'].update(sha256='0'*64),
                   lambda x:x.update(session_id='other'),
                   lambda x:x['producer'].update(role='main_control'),
                   lambda x:x.update(evidence_mode='live'),
                   lambda x:x['data']['asset'].update(transport='file'),
                   lambda x:x['data'].update(source_kind='upload')]
        for change in changes:
            changed = deepcopy(original);change(changed)
            self.assert_code('INVALID_FRAME_REF', self.store.remember_frame, 's', changed, asset_bytes=raw)
        self.assertEqual(self.store.snapshot('s')['frames'], [])

    def test_store_rejects_unproven_sensor_time_or_invalid_host_window(self):
        original = self.frame(); raw = self.provider.read_frame_asset(original)
        for patch in ({'sensor_capture_mono_ms':1004}, {'capture_time_basis':'sensor'},
                      {'host_capture_window':{'call_start_mono_ms':1005,'return_mono_ms':1004}},
                      {'captured_mono_ms':1000}):
            changed = deepcopy(original);changed['data'].update(patch)
            self.assert_code('INVALID_FRAME_REF', self.store.remember_frame, 's', changed, asset_bytes=raw)

    def test_development_store_preserves_original_main_control_validation(self):
        from test_visual_bridge import envelope, picture
        raw = picture()
        original = envelope('fixture', raw)
        frame = self.store.remember_frame('s', original, asset_bytes=raw, retain_copy=True)
        self.assertEqual(frame['producer']['role'], 'test_fixture')
        changed = deepcopy(original);changed['producer']['role'] = ROLE
        self.assert_code('INVALID_PAYLOAD', self.store.remember_frame, 's', changed, asset_bytes=raw)

    def test_bridge_followup_reuses_and_refresh_captures(self):
        bridge = VisualBridge(self.store, lambda:self.provider)
        first = bridge.resolve('s', {'question':'这是什么'})
        followed = bridge.resolve('s', {'question':'它是什么颜色'})
        self.assertEqual(first['frame_id'], followed['frame_id'])
        self.assertEqual(first['image'], followed['image'])
        self.assertEqual(self.camera.calls, 1)
        self.assertEqual(followed['frame_context']['used_frame']['evidence_mode'], EVIDENCE)
        fresh = bridge.resolve('s', {'question':'现在呢，重新看'})
        self.assertNotEqual(fresh['frame_id'], first['frame_id'])
        self.assertEqual(self.camera.calls, 2)
        pinned = bridge.resolve('s', {'frame_id':first['frame_id']})
        self.assertEqual(pinned['frame_id'], first['frame_id'])
        self.assertEqual(self.camera.calls, 2)

    def test_bridge_camera_off_is_error_not_main_control_fallback(self):
        self.camera.running = False
        bridge = VisualBridge(self.store, lambda:self.provider)
        self.assert_code('NO_FRAME', bridge.resolve, 's', {'question':'重新看'})
        self.camera.start.assert_not_called()

    def test_simultaneous_capture_is_bounded(self):
        started, release = threading.Event(), threading.Event()
        original = self.camera.capture_still
        def blocked():
            started.set()
            self.assertTrue(release.wait(2))
            return original()
        self.camera.capture_still = blocked
        result=[];thread=threading.Thread(target=lambda:result.append(self.frame()), daemon=True)
        try:
            thread.start();self.assertTrue(started.wait(1))
            self.assertEqual(self.frame()['error']['code'], 'BUSY')
        finally:
            release.set();thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]['status'], 'ok')
        self.assertEqual(self.camera.calls, 1)

    def test_configure_requires_explicit_store_and_keeps_formal_fallback(self):
        fallback = mock.Mock(return_value='main-adapter')
        bridge = VisualBridge(VisualContextStore(), fallback)
        self.assertIsNone(configure_standalone_visual(bridge, 'main-control', self.camera, fallback))
        self.assertEqual(bridge.main_control_getter(), 'main-adapter')
        self.assertRaises(ValueError, configure_standalone_visual, bridge, SOURCE, self.camera, fallback)
        bridge = VisualBridge(self.store, fallback)
        provider = configure_standalone_visual(bridge, SOURCE, self.camera, fallback)
        self.assertIsInstance(provider, StandaloneFrameProvider)
        self.assertIs(bridge.main_control_getter(), provider)
        self.camera.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
