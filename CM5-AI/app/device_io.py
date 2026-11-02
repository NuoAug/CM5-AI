# -*- coding: utf-8 -*-
"""CM5 设备侧摄像头与 ALSA 音频适配。

没有真机时这些类只返回不可用状态；不会用随机数据冒充传感器。
"""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import re
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


def alsa_card_present(device: str, tool: str) -> bool:
    """检查配置的 ALSA 卡是否真实存在；USB 音频掉线时不要假装可采集。"""
    exe = shutil.which(tool)
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-l"], capture_output=True, text=True, timeout=3).stdout or ""
    except Exception:  # noqa: BLE001
        return False
    m = re.search(r"CARD=([^,]+)", device)
    if m:
        return m.group(1).strip().lower() in out.lower()
    m = re.search(r"plughw:(\d+),", device)
    if m:
        return f"card {m.group(1)}:" in out
    return bool(out.strip())


class DeviceCamera:
    """Picamera2 CSI 采集：低分辨率连续帧用于预览，高清帧按需给 OCR/VLM。"""

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
        self._latest = None
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

    def start(self, callback: Optional[Callable] = None) -> bool:
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
                captured_mono = time.monotonic()
                captured = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with self._lock:
                    self._latest = (frame, f"cam-{self._seq}", captured, captured_mono)
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

    def preview_frame(self):
        """只返回正在采集且足够新鲜的帧；断流后不展示旧画面。"""
        with self._lock:
            if not self._running or self._latest is None:
                return None
            img, fid, captured, mono = self._latest
            if time.monotonic() - mono > 1.5:
                return None
            return img.copy(), fid, captured, mono

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
        return alsa_card_present(self.device, "arecord")

    def start(self) -> bool:
        if not self.available():
            return False
        if self._proc is not None and self._proc.poll() is None:
            return True
        self._proc = None
        exe = audio_tools()["arecord"]
        try:
            self._proc = subprocess.Popen(
                [exe, "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            time.sleep(0.25)
            if self._proc.poll() is not None:
                self._proc = None
                return False
            return True
        except Exception:  # noqa: BLE001
            self._proc = None
            return False

    def read(self) -> Optional[np.ndarray]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        if proc.poll() is not None:
            self._proc = None
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
        proc = self._proc
        if proc is not None and proc.poll() is not None:
            self._proc = None
            return False
        return proc is not None


class AlsaPlayer:
    """aplay 连续 PCM 播放通道；可被危险提示打断。

    逐段合成时把 PCM 写入同一个 aplay 进程，避免每句重新启动进程产生空隙。
    如果无法建立流式通道，仍可通过 play() 逐文件播放作为兜底。
    """

    def __init__(self, device: str = ""):
        self.device = device or os.environ.get("ALSA_OUTPUT", "plughw:0,0")
        self._proc: Optional[subprocess.Popen] = None
        self._stream_proc: Optional[subprocess.Popen] = None
        self._stream_rate = 0
        self._stream_channels = 0
        self._lock = threading.RLock()

    def available(self) -> bool:
        return alsa_card_present(self.device, "aplay")

    def _open_stream(self, rate: int, channels: int) -> bool:
        exe = audio_tools()["aplay"]
        try:
            proc = subprocess.Popen(
                [exe, "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE",
                 "-r", str(int(rate)), "-c", str(int(channels))],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception:  # noqa: BLE001
            return False
        with self._lock:
            self._stream_proc = proc
            self._stream_rate = int(rate)
            self._stream_channels = int(channels)
        return True

    def feed(self, wav_bytes: bytes, cancel_event: threading.Event) -> bool:
        """把一段 WAV 的 PCM 追加到连续播放通道。"""
        if cancel_event.is_set() or not self.available():
            return False
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            pcm = wf.readframes(wf.getnframes())
        if width != 2:
            return False
        with self._lock:
            proc = self._stream_proc
            stale = proc is None or proc.poll() is not None or rate != self._stream_rate \
                or channels != self._stream_channels
        if stale:
            self.end_stream()
            if not self._open_stream(rate, channels):
                return False
            with self._lock:
                proc = self._stream_proc
        try:
            proc.stdin.write(pcm)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, AttributeError):
            self.end_stream()
            # 播放进程可能被设备重置；重建一次，避免整句丢失。
            if cancel_event.is_set() or not self._open_stream(rate, channels):
                return False
            with self._lock:
                proc = self._stream_proc
            try:
                proc.stdin.write(pcm)
                proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError, AttributeError):
                self.end_stream()
                return False

    def end_stream(self) -> None:
        with self._lock:
            proc = self._stream_proc
            self._stream_proc = None
            self._stream_rate = 0
            self._stream_channels = 0
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

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
        """取消/抢占：立即断开播放进程，不等缓冲区排空。"""
        with self._lock:
            proc = self._proc
            self._proc = None
            stream = self._stream_proc
            self._stream_proc = None
            self._stream_rate = 0
            self._stream_channels = 0
        for item in (proc, stream):
            if item is None:
                continue
            try:
                if item.stdin:
                    item.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                item.kill()
            except Exception:  # noqa: BLE001
                pass


class DeviceSink:
    """把统一输出总线接到 Piper + 连续 PCM 播放。

    同一 utterance 的多个片段只暂停一次麦克风，最后一段播完再恢复；
    新回答或危险提示到来时由 OutputBus 调用 stop() 结束当前通道。
    """

    def __init__(
        self,
        synth: Callable[[str], bytes],
        player: AlsaPlayer,
        before_speak: Optional[Callable] = None,
        after_speak: Optional[Callable] = None,
        status_cb: Optional[Callable] = None,
    ):
        self.synth = synth
        self.player = player
        self.before_speak = before_speak
        self.after_speak = after_speak
        self.status_cb = status_cb
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._utterance = ""
        self._speaking = False
        self._payload = None

    def _status(self, payload, state, basis=None, reason=""):
        if self.status_cb is None or not payload:
            return
        try:
            self.status_cb(payload, state, basis, reason)
        except Exception:  # noqa: BLE001
            pass

    def speak(self, payload, cancel_event: threading.Event) -> None:
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            utterance = str(payload.get("utterance_id") or "")
            is_final = bool(payload.get("is_final", True))
        else:
            text, utterance, is_final = str(payload or ""), "", True
        if not text:
            return
        self._payload = payload if isinstance(payload, dict) else {"text": text}
        with self._lock:
            self._cancel = cancel_event
            new_utterance = utterance and utterance != self._utterance
            if new_utterance:
                if self._speaking:
                    self._finish_utterance()
                self._utterance = utterance
            if not self._speaking:
                self._speaking = True
                if self.before_speak:
                    self.before_speak()
        ok = False
        outcome = ""
        try:
            if cancel_event.is_set():
                outcome = "cancelled"
                return
            wav = self.synth(text)
            if cancel_event.is_set():
                outcome = "cancelled"
                return
            if not self.player.feed(wav, cancel_event) and not cancel_event.is_set():
                raise RuntimeError("ALSA 连续播放写入失败")
            ok = True
            self._status(self._payload, "started", "device_write")
            if is_final:
                outcome = "completed"
        except Exception:  # noqa: BLE001
            outcome = "cancelled" if cancel_event.is_set() else "failed"
            raise
        finally:
            if is_final or cancel_event.is_set() or not ok:
                with self._lock:
                    if self._speaking:
                        self._finish_utterance(outcome or "failed")

    def _finish_utterance(self, outcome: str = "") -> None:
        self.player.end_stream()
        payload = self._payload
        self._utterance = ""
        self._speaking = False
        if outcome:
            self._status(payload, outcome, "device_write" if outcome == "completed" else None,
                         "sink_error" if outcome == "failed" else "")
        if self.after_speak:
            self.after_speak()

    def stop(self) -> None:
        # 先置取消并立即杀掉播放进程，再拿锁；否则 sink 线程正在 end_stream 排空时会阻塞取消。
        self._cancel.set()
        self.player.stop()
        with self._lock:
            if self._speaking:
                self._finish_utterance("cancelled")


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
