"""独立树莓派相机测试帧桥接；不是主控 FrameProvider 的实现或空间事实。

只有显式 standalone-device-test 模式可启用。相机由调用方启动，资产只在
有界内存中保存。captured_mono_ms 的依据是 host_capture_return，传感器
采集时间未知；host_capture_window 只是主机调用窗口，不能当传感器曝光时间。
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import math
import threading
import time
import uuid

from PIL import Image
from visual_context import VisualContextError, VisualContextStore


SCHEMA = 'standalone.device-frame.v1'
SOURCE = 'standalone-device-test'
ROLE = 'standalone_device'
EVIDENCE = 'development_live'


def _require(ok, message, code='INVALID_FRAME_REF'):
    if not ok:
        raise VisualContextError(code, message)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonempty(value, limit):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _pixels(content, max_pixels=16_000_000):
    _require(isinstance(content, bytes) and content, '独立相机资产字节为空')
    try:
        with Image.open(io.BytesIO(content)) as image:
            _require(image.format == 'PNG' and image.mode == 'RGB', '独立相机资产必须是RGB PNG')
            width, height = image.size
            _require(width > 0 and height > 0 and width * height <= max_pixels, '独立相机图片超出像素预算')
            image.verify()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, VisualContextError):
            raise
        raise VisualContextError('INVALID_FRAME_REF', '独立相机图片字节无法解析') from exc
    return width, height


class StandaloneFrameProvider:
    """capture_still() 按需取图；pinned 仅返回本会话原帧，绝不重新拍摄。"""
    def __init__(self, camera, *, lease_ms=10_000, max_frames=8,
                 max_frame_bytes=8 * 1024 * 1024, max_cached_bytes=16 * 1024 * 1024,
                 max_pixels=16_000_000, now_ms=None):
        for value in (lease_ms, max_frames, max_frame_bytes, max_cached_bytes, max_pixels):
            _require(type(value) is int and value > 0, '独立相机容量和期限必须是正整数', 'INVALID_PAYLOAD')
        self.camera = camera
        self.lease_ms = lease_ms
        self.max_frames = max_frames
        self.max_frame_bytes = max_frame_bytes
        self.max_cached_bytes = max_cached_bytes
        self.max_pixels = max_pixels
        self.now_ms = now_ms or (lambda: time.monotonic_ns() / 1_000_000)
        self.instance_id = 'standalone-device-' + uuid.uuid4().hex
        self.clock_id = 'host-monotonic-' + self.instance_id
        self._frames = OrderedDict()
        self._retired = OrderedDict()
        self._cached_bytes = 0
        self._lock = threading.RLock()
        self._capture_lock = threading.Lock()

    def _failure(self, code, message, session_id=None):
        return {'schema_version': SCHEMA, 'type': 'frame.result', 'status': 'unavailable',
                'producer': {'role': ROLE, 'instance_id': self.instance_id},
                'evidence_mode': EVIDENCE, 'image_source': SOURCE, 'clock_id': self.clock_id,
                'session_id': session_id, 'data': None,
                'error': {'code': code, 'message': message, 'retryable': code in ('NO_FRAME', 'BUSY')}}

    def _expire_locked(self):
        now = self.now_ms()
        for fid in list(self._frames):
            record = self._frames[fid]
            if now >= record['result']['data']['asset']['available_until_mono_ms']:
                self._cached_bytes -= len(record['content'])
                self._retired[fid] = record['result']['session_id']
                del self._frames[fid]
        while len(self._retired) > self.max_frames * 2:
            self._retired.popitem(last=False)

    def acquire_frame(self, *, mode='latest', purpose='vqa', frame_id=None, session_id=None, **kwargs):
        if not _nonempty(session_id, 128) or mode not in ('latest', 'pinned') or purpose not in ('vqa', 'ocr', 'compare'):
            return self._failure('INVALID_PAYLOAD', '独立相机请求的会话、模式或用途无效', session_id)
        if kwargs:
            return self._failure('INVALID_PAYLOAD', '独立相机不支持未声明的帧请求参数', session_id)
        if mode == 'pinned':
            if not _nonempty(frame_id, 256):
                return self._failure('INVALID_FRAME_REF', '指定历史帧必须提供frame_id', session_id)
            with self._lock:
                self._expire_locked()
                record = self._frames.get(frame_id)
                if record is None or record['result']['session_id'] != session_id:
                    code = 'STALE_FRAME' if self._retired.get(frame_id) == session_id else 'NO_FRAME'
                    return self._failure(code, '指定独立相机帧不存在或租期已过；不会换成新图', session_id)
                return deepcopy(record['result'])
        if frame_id not in (None, ''):
            return self._failure('INVALID_FRAME_REF', 'latest不能携带旧frame_id', session_id)
        if not self._capture_lock.acquire(blocking=False):
            return self._failure('BUSY', '独立相机正在处理另一取图请求', session_id)
        try:
            with self._lock:
                self._expire_locked()
                if len(self._frames) >= self.max_frames:
                    return self._failure('BUSY', '独立相机有效资产占满容量，请等待租期释放', session_id)
            try:
                status = self.camera.status() if self.camera is not None else {}
                if not isinstance(status, dict) or status.get('running') is not True or status.get('available') is False:
                    return self._failure('NO_FRAME', '独立测试摄像头尚未开启或不可用，不会自动启动', session_id)
                start = self.now_ms()
                image = self.camera.capture_still()
                returned = self.now_ms()
            except Exception:
                return self._failure('NO_FRAME', '独立测试摄像头取图失败', session_id)
            if image is None:
                return self._failure('NO_FRAME', '独立测试摄像头没有返回图片', session_id)
            if not isinstance(image, Image.Image):
                return self._failure('INVALID_FRAME_REF', '摄像头必须返回实际PIL图片', session_id)
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > self.max_pixels:
                return self._failure('INVALID_FRAME_REF', '独立相机图片超出像素预算', session_id)
            if not (_finite(start) and _finite(returned) and 0 <= start <= returned):
                return self._failure('CLOCK_UNVERIFIED', '独立相机主机时钟不连续', session_id)
            try:
                encoded = io.BytesIO()
                image.convert('RGB').save(encoded, format='PNG')
                content = encoded.getvalue()
            except Exception:
                return self._failure('INVALID_FRAME_REF', '独立相机图片编码失败', session_id)
            if not 0 < len(content) <= self.max_frame_bytes:
                return self._failure('INVALID_FRAME_REF', '独立相机压缩图片超出单帧预算', session_id)
            with self._lock:
                self._expire_locked()
                if self._cached_bytes + len(content) > self.max_cached_bytes:
                    return self._failure('BUSY', '独立相机资产总字节预算不足', session_id)
                now = self.now_ms()
                if not _finite(now) or now < returned:
                    return self._failure('CLOCK_UNVERIFIED', '独立相机主机时钟倒退', session_id)
                fid = 'device-test-' + uuid.uuid4().hex
                digest = hashlib.sha256(content).hexdigest()
                result = {'schema_version': SCHEMA, 'type': 'frame.result', 'status': 'ok', 'error': None,
                          'producer': {'role': ROLE, 'instance_id': self.instance_id},
                          'evidence_mode': EVIDENCE, 'image_source': SOURCE, 'clock_id': self.clock_id,
                          'session_id': session_id, 'timestamp_mono_ms': now,
                          'data': {'frame_id': fid, 'captured_mono_ms': returned,
                                   'capture_time_basis': 'host_capture_return', 'sensor_capture_mono_ms': None,
                                   'host_capture_window': {'call_start_mono_ms': start, 'return_mono_ms': returned},
                                   'source_kind': 'camera', 'is_historical': False,
                                   'width_px': width, 'height_px': height, 'color_space': 'RGB',
                                   'parent_frame_id': None, 'transform_to_parent': None,
                                   'quality': {'sharpness': 'unknown', 'glare': 'unknown', 'exposure': 'unknown'},
                                   'asset': {'asset_id': 'asset-' + fid, 'transport': 'standalone_memory',
                                             'encoding': 'png', 'sha256': digest, 'size_bytes': len(content),
                                             'available_until_mono_ms': now + self.lease_ms}}}
                self._frames[fid] = {'result': result, 'content': content}
                self._cached_bytes += len(content)
                return deepcopy(result)
        finally:
            self._capture_lock.release()

    def read_frame_asset(self, result):
        _require(isinstance(result, dict) and isinstance(result.get('data'), dict), '独立相机结果格式无效')
        frame_id = result['data'].get('frame_id')
        _require(isinstance(frame_id, str), '独立相机结果缺少帧身份')
        with self._lock:
            self._expire_locked()
            record = self._frames.get(frame_id)
            if record is None:
                code = 'STALE_FRAME' if self._retired.get(frame_id) == result.get('session_id') else 'NO_FRAME'
                raise VisualContextError(code, '独立相机资产不存在或已释放')
            _require(result == record['result'], '独立相机结果被修改或来自其他来源')
            content = record['content']
            _require(hashlib.sha256(content).hexdigest() == result['data']['asset']['sha256'], '独立相机资产摘要不符')
            return content

    def stats(self):
        with self._lock:
            self._expire_locked()
            return {'source': SOURCE, 'evidence_mode': EVIDENCE, 'instance_id': self.instance_id,
                    'frames': len(self._frames), 'cached_bytes': self._cached_bytes,
                    'max_frames': self.max_frames, 'max_cached_bytes': self.max_cached_bytes,
                    'lease_ms': self.lease_ms, 'capture_time_basis': 'host_capture_return',
                    'sensor_capture_time_known': False}


class DeviceTestVisualContextStore(VisualContextStore):
    """只为显式独立设备测试接受开发来源；主控验证仍走原父类。"""
    def __init__(self, *, allow_standalone_device=False, **kwargs):
        if type(allow_standalone_device) is not bool:
            raise ValueError('allow_standalone_device必须为布尔值')
        super().__init__(**kwargs)
        self.allow_standalone_device = allow_standalone_device

    def remember_frame(self, session_id, frame_result, *, asset_bytes, relation='new',
                       object_id=None, slot=None, retain_copy=False):
        is_dev = isinstance(frame_result, dict) and frame_result.get('schema_version') == SCHEMA
        if not is_dev:
            return super().remember_frame(session_id, frame_result, asset_bytes=asset_bytes,
                                          relation=relation, object_id=object_id, slot=slot,
                                          retain_copy=retain_copy)
        _require(self.allow_standalone_device, '独立设备帧未显式启用', 'INVALID_PAYLOAD')
        _require(frame_result.get('type') == 'frame.result' and frame_result.get('status') == 'ok', '独立相机未返回可用帧', 'NO_FRAME')
        _require(frame_result.get('image_source') == SOURCE and frame_result.get('evidence_mode') == EVIDENCE,
                 '独立相机证据模式不符')
        producer = frame_result.get('producer')
        _require(isinstance(producer, dict) and producer.get('role') == ROLE and _nonempty(producer.get('instance_id'), 256),
                 '独立相机来源不符，不得冒充主控')
        _require(frame_result.get('session_id') == session_id, '独立相机帧属于其他会话')
        clock = frame_result.get('clock_id')
        _require(_nonempty(clock, 256), '独立相机缺少主机时钟身份', 'CLOCK_UNVERIFIED')
        data = frame_result.get('data')
        _require(isinstance(data, dict) and data.get('source_kind') == 'camera' and data.get('color_space') == 'RGB', '独立相机帧来源或颜色无效')
        _require(data.get('capture_time_basis') == 'host_capture_return' and data.get('sensor_capture_mono_ms') is None,
                 '独立相机不能伪称传感器采集时间')
        window = data.get('host_capture_window')
        _require(isinstance(window, dict), '独立相机缺少主机取图窗口')
        start, end = window.get('call_start_mono_ms'), window.get('return_mono_ms')
        _require(_finite(start) and _finite(end) and 0 <= start <= end <= self.now_ms() and data.get('captured_mono_ms') == end,
                 '独立相机主机取图时间无效')
        _require(type(data.get('is_historical')) is bool and data['is_historical'] is False,
                 '独立相机原始帧历史标记无效')
        asset = data.get('asset')
        _require(isinstance(asset, dict) and asset.get('transport') == 'standalone_memory' and asset.get('encoding') == 'png',
                 '独立相机资产传输无效')
        _require(isinstance(asset_bytes, bytes) and 0 < len(asset_bytes) <= self.max_frame_bytes and asset.get('size_bytes') == len(asset_bytes), '独立相机资产大小不符或超过预算')
        width, height = _pixels(asset_bytes)
        _require(data.get('width_px') == width and data.get('height_px') == height, '独立相机帧描述与实际图片尺寸不符')
        return self._remember(session_id, data, asset_bytes=asset_bytes, evidence_mode=EVIDENCE,
                              producer=producer, clock_id=clock, relation=relation,
                              object_id=object_id, slot=slot, retain_copy=retain_copy)


def configure_standalone_visual(bridge, image_source, camera, fallback_getter):
    """原store须先显式构造；切换getter不会替换已注册路由使用的store。"""
    if image_source == SOURCE:
        if not isinstance(bridge.store, DeviceTestVisualContextStore) or not bridge.store.allow_standalone_device:
            raise ValueError('独立相机模式需要显式启用DeviceTestVisualContextStore')
        provider = StandaloneFrameProvider(camera, max_frame_bytes=bridge.max_image_bytes, max_pixels=bridge.max_pixels)
        bridge.main_control_getter = lambda: provider
        return provider
    if not callable(fallback_getter):
        raise ValueError('非独立测试模式需要原FrameProvider getter')
    bridge.main_control_getter = fallback_getter
    return None
