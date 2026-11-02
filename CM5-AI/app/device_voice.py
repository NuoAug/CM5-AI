# -*- coding: utf-8 -*-
"""设备端常开语音：arecord → KWS → VAD 切句 → 本地 /api/voice。

浏览器仍保留为调试界面；真机语音不经过浏览器麦克风。
正式模式取图由服务端通过主控 FrameProvider 完成；只有显式
IMAGE_SOURCE=standalone-device-test 时本机相机才随语音抓图。
半双工：播报期间暂停采集，因此不支持播报中可靠打断。
"""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Optional

import requests
from PIL import Image

import kws
from device_io import AlsaCapture, DeviceCamera, encode_wav
from output import OUTPUT, PRIORITY_CONFIRM

VOICE_URL = "http://127.0.0.1:8090/api/voice"


def jpeg_b64(img: Image.Image, max_side: int = 1600, quality: int = 88) -> str:
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DeviceVoice:
    def __init__(
        self,
        camera: Optional[DeviceCamera] = None,
        capture: Optional[AlsaCapture] = None,
        url: str = VOICE_URL,
        session_id: str = "device",
        capture_images: bool = True,
    ):
        self.camera = camera
        self.capture_images = capture_images
        self.capture = capture or AlsaCapture()
        self.url = url
        self.session_id = session_id
        self.gate = kws.WakeGate(armed_seconds=15.0)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._paused = False
        self._sink_present = False
        self._resume_timer: Optional[threading.Timer] = None
        self._last_error = None
        self._last_command_at = 0.0
        self._seq = 0

    def available(self) -> bool:
        return self.capture.available() and kws.status().get("ready", False)

    def status(self) -> dict:
        return {
            "available": self.available(),
            "running": self._running,
            "paused": self._paused,
            "sink_present": self._sink_present,
            "capture": self.capture.running,
            "kws": kws.status(),
            "error": self._last_error,
            "last_command_age_s": round(time.monotonic() - self._last_command_at, 1) if self._last_command_at else None,
        }

    def set_sink_present(self, present: bool) -> None:
        self._sink_present = bool(present)

    def start(self) -> bool:
        if self._running:
            return True
        if not self.capture.available():
            self._last_error = "arecord 不可用"
            return False
        if not kws.status().get("ready", False):
            self._last_error = "KWS 模型未就绪"
            return False
        if not self.capture.start():
            self._last_error = "麦克风采集启动失败"
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="device-voice", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running:
            chunk = self.capture.read()
            if chunk is None:
                time.sleep(0.05)
                continue
            try:
                event = self.gate.feed(chunk)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"KWS 异常：{exc}"
                time.sleep(0.2)
                continue
            if event.get("wake"):
                OUTPUT.push(
                    "我在，请说", priority=PRIORITY_CONFIRM, source="device",
                    reason="wake", ttl=2.0, dedupe_key="device-wake",
                )
            command = event.get("command")
            if command is not None and len(command):
                threading.Thread(target=self._handle_command, args=(command,), daemon=True).start()

    def _handle_command(self, samples) -> None:
        self.pause()
        self._last_command_at = time.monotonic()
        body = {"audio": "data:audio/wav;base64," + base64.b64encode(encode_wav(samples)).decode("ascii"),
                "session_id": self.session_id, "wake_detected": True}
        if self.capture_images and self.camera is not None and self.camera.status().get("running"):
            still = self.camera.capture_still()
            if still is not None:
                self._seq += 1
                body["image"] = "data:image/jpeg;base64," + jpeg_b64(still)
                body["image_vlm"] = "data:image/jpeg;base64," + jpeg_b64(still, 800)
                body["frame_id"] = f"dev-{self._seq}"
                body["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                body["image_source"] = "device_camera"
        try:
            r = requests.post(self.url, json=body, timeout=120)
            j = r.json() if r.content else {}
            if not r.ok:
                raise RuntimeError(j.get("error") or f"HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            OUTPUT.push(
                "设备语音请求失败，请稍后再试", priority=PRIORITY_CONFIRM,
                source="device", reason="voice_error", ttl=4.0,
            )
            self.resume()
            return
        if not self._sink_present:
            self.resume()
        elif not j.get("spoken", False):
            self.resume()
        else:
            # 播报由 DeviceSink 暂停/恢复麦克风；这里只做超时兜底。
            if self._resume_timer is not None:
                self._resume_timer.cancel()
            self._resume_timer = threading.Timer(8.0, self.resume)
            self._resume_timer.daemon = True
            self._resume_timer.start()

    def pause(self) -> None:
        with self._lock:
            if self._paused:
                return
            self._paused = True
        self.capture.stop()
        self.gate.reset()

    def resume(self) -> None:
        with self._lock:
            if not self._running or not self._paused:
                return
            self._paused = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.capture.start()

    def stop(self) -> None:
        self._running = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.capture.stop()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
