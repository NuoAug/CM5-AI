# -*- coding: utf-8 -*-
"""CM5 设备侧摄像头与 ALSA 音频适配。

没有真机时这些类只返回不可用状态；不会用随机数据冒充传感器。
"""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def picamera_available() -> bool:
    return importlib.util.find_spec("picamera2") is not None


def audio_tools() -> dict:
    return {
        "arecord": shutil.which("arecord"),
        "aplay": shutil.which("aplay"),
    }


class DeviceCamera:
    """Picamera2 CSI 采集：低分辨率连续帧给引导，高清帧按需给 OCR/VLM。"""

    def __init__(self, high_size=(1280, 720), low_size=(512, 384), fps=5.0):
        self.high_size = high_size
        self.low_size = low_size
        self.fps = max(1.0, fps)
        self._picam = None
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._use_lores = True
        self._seq = 0
        self._error = None
        self._callback: Optional[Callable] = None

    def available(self) -> bool:
        return picamera_available()

    def status(self) -> dict:
        return {
            "available": self.available(),
            "running": self._running,
            "high_size": list(self.high_size),
            "low_size": list(self.low_size),
            "fps": self.fps,
            "use_lores": self._use_lores,
            "error": self._error,
        }

    def start(self, callback: Callable) -> bool:
        if not self.available():
            self._error = "picamera2 未安装"
            return False
        if self._running:
            self._callback = callback
            return True
        try:
            from picamera2 import Picamera2

            self._picam = Picamera2()
            try:
                config = self._picam.create_video_configuration(
                    main={"size": self.high_size, "format": "RGB888"},
                    lores={"size": self.low_size, "format": "RGB888"},
                    controls={"FrameRate": self.fps},
                )
                self._picam.configure(config)
                self._use_lores = True
            except Exception:
                # 部分平台 lores 不支持 RGB888，退回只采 main 并缩放。
                config = self._picam.create_video_configuration(
                    main={"size": self.high_size, "format": "RGB888"},
                    controls={"FrameRate": self.fps},
                )
                self._picam.configure(config)
                self._use_lores = False
            self._picam.start()
            if self._use_lores:
                try:
                    probe = self._picam.capture_array("lores")
                    if probe is None or probe.ndim != 3 or probe.shape[0] < 10 or probe.shape[2] < 3:
                        raise RuntimeError("lores 不是 RGB 三通道输出")
                except Exception:
                    self._use_lores = False
        except Exception as exc:  # noqa: BLE001
            self._error = f"摄像头启动失败：{exc}"
            self._stop_camera()
            return False
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="device-camera", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while self._running:
            t0 = time.monotonic()
            try:
                with self._lock:
                    key = "lores" if self._use_lores else "main"
                    arr = self._picam.capture_array(key)
                frame = Image.fromarray(np.ascontiguousarray(arr[:, :, :3]))
                if not self._use_lores and max(frame.size) != max(self.low_size):
                    frame = frame.resize(self.low_size, Image.BILINEAR)
                self._seq += 1
                if self._callback:
                    self._callback(
                        frame,
                        f"cam-{self._seq}",
                        time.monotonic(),
                        {"driver": "picamera2", "lores": self._use_lores},
                    )
            except Exception as exc:  # noqa: BLE001
                self._error = f"取帧失败：{exc}"
                time.sleep(0.2)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

    def capture_still(self) -> Optional[Image.Image]:
        if not self._running:
            return None
        try:
            with self._lock:
                arr = self._picam.capture_array("main")
            return Image.fromarray(np.ascontiguousarray(arr[:, :, :3]))
        except Exception as exc:  # noqa: BLE001
            self._error = f"高清取帧失败：{exc}"
            return None

    def _stop_camera(self) -> None:
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
        except Exception:  # noqa: BLE001
            pass
        self._picam = None

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._stop_camera()


class AlsaCapture:
    """arecord 原始 PCM 采集，16 kHz / 单声道 / 16-bit。"""

    def __init__(self, device: str = "", sample_rate: int = 16000, chunk_frames: int = 1600):
        self.device = device or os.environ.get("ALSA_INPUT", "plughw:0,0")
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()

    def available(self) -> bool:
        return bool(audio_tools()["arecord"])

    def start(self) -> bool:
        if not self.available() or self._proc is not None:
            return self._proc is not None
        exe = audio_tools()["arecord"]
        try:
            self._proc = subprocess.Popen(
                [exe, "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            return True
        except Exception:  # noqa: BLE001
            self._proc = None
            return False

    def read(self) -> Optional[np.ndarray]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        nbytes = self.chunk_frames * 2
        try:
            raw = proc.stdout.read(nbytes)
        except Exception:  # noqa: BLE001
            return None
        if not raw or len(raw) < nbytes:
            return None
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    @property
    def running(self) -> bool:
        return self._proc is not None


class AlsaPlayer:
    """aplay 播放 WAV；可被危险提示打断。"""

    def __init__(self, device: str = ""):
        self.device = device or os.environ.get("ALSA_OUTPUT", "plughw:0,0")
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()

    def available(self) -> bool:
        return bool(audio_tools()["aplay"])

    def play(self, wav_bytes: bytes, cancel_event: threading.Event) -> bool:
        if not self.available():
            return False
        exe = audio_tools()["aplay"]
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        Path(path).write_bytes(wav_bytes)
        try:
            proc = subprocess.Popen(
                [exe, "-q", "-D", self.device, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001
            try:
                os.unlink(path)
            except OSError:
                pass
            return False
        with self._lock:
            self._proc = proc
        try:
            while proc.poll() is None:
                if cancel_event.is_set():
                    proc.kill()
                    break
                time.sleep(0.05)
        except Exception:  # noqa: BLE001
            pass
        finally:
            with self._lock:
                self._proc = None
            try:
                os.unlink(path)
            except OSError:
                pass
        return proc.returncode == 0

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


class DeviceSink:
    """把统一输出总线接到 Piper + aplay；播报前暂停麦克风，避免自听回环。"""

    def __init__(
        self,
        synth: Callable[[str], bytes],
        player: AlsaPlayer,
        before_speak: Optional[Callable] = None,
        after_speak: Optional[Callable] = None,
    ):
        self.synth = synth
        self.player = player
        self.before_speak = before_speak
        self.after_speak = after_speak
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._speaking = False

    def speak(self, text: str, cancel_event: threading.Event) -> None:
        with self._lock:
            self._cancel = cancel_event
            self._speaking = True
        if self.before_speak:
            self.before_speak()
        try:
            if cancel_event.is_set():
                return
            wav = self.synth(text)
            if cancel_event.is_set():
                return
            self.player.play(wav, cancel_event)
        finally:
            self._speaking = False
            if self.after_speak:
                self.after_speak()

    def stop(self) -> None:
        with self._lock:
            self._cancel.set()
        self.player.stop()


def encode_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_b64(samples: np.ndarray, sample_rate: int = 16000) -> str:
    return "data:audio/wav;base64," + base64.b64encode(encode_wav(samples, sample_rate)).decode("ascii")
