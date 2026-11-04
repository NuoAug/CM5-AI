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
import select
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
            # Picamera2/libcamera名称按内存位序：BGR888数组为[R,G,B]，
            # RGB888数组为[B,G,R]。这里送PIL，必须请求BGR888避免红蓝互换。
            # 官方映射：raspberrypi/picamera2 request.py::_get_pil_mode。
            try:
                config = self._picam.create_video_configuration(
                    main={"size": self.high_size, "format": "BGR888"},
                    lores={"size": self.low_size, "format": "BGR888"},
                    controls={"FrameRate": self.fps},
                )
                self._picam.configure(config)
                self._use_lores = True
            except Exception:
                # 部分平台 lores 不支持 BGR888，退回只采 main 并缩放。
                config = self._picam.create_video_configuration(
                    main={"size": self.high_size, "format": "BGR888"},
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


class AlsaCaptureError(RuntimeError):
    def __init__(self, message: str, returncode: Optional[int] = None):
        super().__init__(message)
        self.returncode = returncode


class AlsaCapture:
    """arecord 原始 PCM 采集，16 kHz / 单声道 / 16-bit。"""

    def __init__(self, device: str = "", sample_rate: int = 16000, chunk_frames: int = 1600):
        self.device = device or os.environ.get("ALSA_INPUT", "plughw:0,0")
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self._proc: Optional[subprocess.Popen] = None
        self._retiring: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._generation = 0
        self.last_error: Optional[AlsaCaptureError] = None

    def available(self) -> bool:
        return alsa_card_present(self.device, "arecord")

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return True
            generation = self._generation
        available = self.available()
        with self._lock:
            if generation != self._generation:
                return False
            if self.running:
                return True
            if self._retiring is not None:
                if self._retiring.poll() is None:
                    self.last_error = AlsaCaptureError("旧 arecord 尚未退出，等待设备释放")
                    return False
                self._terminate(self._retiring)
            if not available:
                self.last_error = AlsaCaptureError("麦克风设备或 arecord 不可用")
                return False
            old_proc = self._proc
            self._proc = None
            try:
                proc = subprocess.Popen(
                    [audio_tools()["arecord"], "-q", "-D", self.device, "-f", "S16_LE",
                     "-r", str(self.sample_rate), "-c", "1", "-t", "raw"],
                    stdout=subprocess.PIPE,
                    # 驱动诊断进入服务日志。无额外 stderr 管道、线程或缓冲队列。
                    bufsize=0,
                )
            except (OSError, TypeError) as exc:
                self.last_error = AlsaCaptureError(f"启动 arecord 失败：{exc}")
                proc = None
            self._proc = proc
        self._terminate(old_proc)
        if proc is None:
            return False
        time.sleep(0.25)
        with self._lock:
            if self._proc is not proc:
                return False
            if proc.poll() is None:
                self.last_error = None
                return True
        self._fail(proc, "arecord 启动后退出")
        return False

    def _terminate(self, proc) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if proc.poll() is None:
            # 例如内核 I/O 暂未解除。保留身份，禁止重开占用同一采集端。
            return
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        with self._lock:
            if self._retiring is proc:
                self._retiring = None

    def _fail(self, proc, reason: str) -> Optional[AlsaCaptureError]:
        with self._lock:
            if self._proc is not proc:
                # 显式停止或其他代次已经接管，只能清理自己持有的进程。
                return None
            returncode = proc.poll()
            if returncode is not None:
                reason += f"（退出码 {returncode}）"
            error = AlsaCaptureError(reason, returncode)
            self.last_error = error
            self._retiring = proc
            self._proc = None
        self._terminate(proc)
        return error

    @staticmethod
    def _read_pipe(proc, count: int, deadline: float) -> bytes:
        if os.name == "posix":
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                raise TimeoutError("2 秒内未收到完整 PCM 块")
        return proc.stdout.read(count)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            proc = self._proc
        if proc is None:
            return None
        nbytes = self.chunk_frames * 2
        raw = bytearray()
        deadline = time.monotonic() + 2.0
        try:
            while len(raw) < nbytes:
                if proc.poll() is not None:
                    raise OSError("arecord 已退出")
                block = self._read_pipe(proc, nbytes - len(raw), deadline)
                if not block:
                    raise EOFError("arecord 输出已结束")
                raw.extend(block)
        except (OSError, ValueError, EOFError) as exc:
            error = self._fail(proc, f"麦克风采集中断：{exc}")
            if error is not None:
                raise error from exc
            return None
        with self._lock:
            if self._proc is not proc:
                return None
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            proc = self._proc or self._retiring
            self._retiring = proc
            self._proc = None
        self._terminate(proc)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None


class AlsaPlayer:
    """唯一 aplay 输出端；进程启动、写入和排空都检查取消代次。"""
    def __init__(self, device: str = ""):
        self.device = device or os.environ.get("ALSA_OUTPUT", "plughw:0,0")
        self._proc: Optional[subprocess.Popen] = None
        self._stream_proc: Optional[subprocess.Popen] = None
        self._stream_rate = 0
        self._stream_channels = 0
        self._lock = threading.RLock()
        self._stop_generation = 0
        self._draining = set()

    def available(self) -> bool:
        return alsa_card_present(self.device, "aplay")

    @staticmethod
    def _terminate(proc):
        # 终止优先于关管道，使阻塞写入/排空退出；不拿播放器锁等待进程。
        try:
            proc.kill()
        except Exception:
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

    def _discard_stream(self, proc):
        with self._lock:
            if self._stream_proc is proc:
                self._stream_proc = None
                self._stream_rate = self._stream_channels = 0
            self._draining.add(proc)
        try:
            self._terminate(proc)
        finally:
            with self._lock:
                self._draining.discard(proc)

    def _open_stream(self, rate: int, channels: int, *, generation=None, cancel_event=None) -> bool:
        with self._lock:
            generation = self._stop_generation if generation is None else generation
            if generation != self._stop_generation or (cancel_event and cancel_event.is_set()):
                return False
        exe = audio_tools()["aplay"]
        try:
            proc = subprocess.Popen(
                [exe, "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE",
                 "-r", str(int(rate)), "-c", str(int(channels))],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, bufsize=0)
        except Exception:
            return False
        with self._lock:
            stale = generation != self._stop_generation or (cancel_event and cancel_event.is_set())
            if not stale and self._stream_proc is None:
                self._stream_proc = proc
                self._stream_rate = int(rate)
                self._stream_channels = int(channels)
                return True
            reusable = (not stale and self._stream_proc is not None and self._stream_proc.poll() is None
                        and self._stream_rate == rate and self._stream_channels == channels)
        # stop可能发生在Popen尚未返回时；只结束本次迟到进程，不碰新代次的进程。
        self._terminate(proc)
        return reusable

    def _stream_live(self, proc, generation, cancel_event):
        with self._lock:
            return (not cancel_event.is_set() and generation == self._stop_generation
                    and self._stream_proc is proc)

    def feed(self, wav_bytes: bytes, cancel_event: threading.Event) -> bool:
        """把一段 WAV 的 PCM 分块追加到连续通道；失败不重放已写出的片段。"""
        with self._lock:
            generation = self._stop_generation
        if cancel_event.is_set() or not self.available():
            return False
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            pcm = wf.readframes(wf.getnframes())
        if width != 2:
            return False
        with self._lock:
            proc = self._stream_proc
            stale = proc is None or proc.poll() is not None or rate != self._stream_rate or channels != self._stream_channels
        if stale:
            if proc is not None:
                self._end_stream(proc)
            if not self._open_stream(rate, channels, generation=generation, cancel_event=cancel_event):
                return False
            with self._lock:
                if generation != self._stop_generation:
                    return False
                proc = self._stream_proc
        if proc is None:
            return False
        try:
            offset = 0
            while offset < len(pcm):
                if not self._stream_live(proc, generation, cancel_event):
                    self._discard_stream(proc)
                    return False
                # 原始管道写入不持播放器锁，stop可随时杀进程解除阻塞。
                written = proc.stdin.write(pcm[offset:offset + 4096])
                if not isinstance(written, int) or written <= 0:
                    raise OSError("PCM写入没有进展")
                offset += written
            if not self._stream_live(proc, generation, cancel_event):
                self._discard_stream(proc)
                return False
            proc.stdin.flush()
            if not self._stream_live(proc, generation, cancel_event):
                self._discard_stream(proc)
                return False
            return True
        except (BrokenPipeError, OSError, AttributeError, ValueError):
            self._discard_stream(proc)
            return False

    def _end_stream(self, proc):
        with self._lock:
            if self._stream_proc is proc:
                self._stream_proc = None
                self._stream_rate = self._stream_channels = 0
            # 保留可被stop找到的引用，直到声卡排空/进程退出。
            self._draining.add(proc)
        try:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=60)
            except Exception:
                self._terminate(proc)
        finally:
            with self._lock:
                self._draining.discard(proc)

    def end_stream(self) -> None:
        with self._lock:
            proc = self._stream_proc
        if proc is not None:
            self._end_stream(proc)

    def play(self, wav_bytes: bytes, cancel_event: threading.Event) -> bool:
        with self._lock:
            generation = self._stop_generation
        if cancel_event.is_set() or not self.available():
            return False
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        proc = None
        try:
            Path(path).write_bytes(wav_bytes)
            if cancel_event.is_set():
                return False
            proc = subprocess.Popen([audio_tools()["aplay"], "-q", "-D", self.device, path],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self._lock:
                stale = generation != self._stop_generation or cancel_event.is_set()
                if not stale:
                    self._proc = proc
            if stale:
                self._terminate(proc)
                return False
            while proc.poll() is None:
                if cancel_event.is_set() or generation != self._stop_generation:
                    self._terminate(proc)
                    break
                time.sleep(0.05)
            return proc.returncode == 0 and not cancel_event.is_set() and generation == self._stop_generation
        except Exception:
            if proc is not None:
                self._terminate(proc)
            return False
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop(self) -> None:
        """只捕获并终止调用时已有的进程，包括正在排空的旧通道。"""
        with self._lock:
            self._stop_generation += 1
            processes = set(self._draining)
            processes.update(p for p in (self._proc, self._stream_proc) if p is not None)
            self._proc = self._stream_proc = None
            self._stream_rate = self._stream_channels = 0
            self._draining.clear()
        for proc in processes:
            self._terminate(proc)


def _payload_prompt_id(payload) -> str:
    """读取输出事件里的固定提示 ID；缺省返回空串。"""
    if not isinstance(payload, dict):
        return ""
    direct = payload.get("prompt_id")
    if direct:
        return str(direct)
    meta = payload.get("meta")
    if isinstance(meta, dict) and meta.get("prompt_id"):
        return str(meta["prompt_id"])
    return ""


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
        emergency_synth: Optional[Callable] = None,
        cancel_synth: Optional[Callable] = None,
        prompts=None,
    ):
        self.synth = synth
        self.emergency_synth = emergency_synth or synth
        self.cancel_synth = cancel_synth
        self.prompts = prompts
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

    def prepare(self, payload, cancel_event):
        if cancel_event.is_set():
            return b''
        prompt_id = _payload_prompt_id(payload)
        if prompt_id:
            store = self.prompts
            if store is not None:
                wav = store.read(prompt_id)
                if wav:
                    return wav
        synth = self.emergency_synth if payload.get('priority', 3) == 0 else self.synth
        return synth(str(payload.get('text') or ''))

    def speak(self, payload, cancel_event: threading.Event) -> None:
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            utterance = str(payload.get("utterance_id") or "")
            is_final = bool(payload.get("is_final", True))
        else:
            text, utterance, is_final = str(payload or ""), "", True
        if not text:
            if isinstance(payload, dict) and payload.get('type') == 'speech_end':
                with self._lock:
                    if self._speaking and utterance == self._utterance:
                        self._finish_utterance('completed')
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
            wav = payload.get("_prepared_wav") if isinstance(payload, dict) else None
            if wav is None:
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

    def cancel_synthesis(self):
        if self.cancel_synth:
            self.cancel_synth()

    def stop(self) -> None:
        # 先置取消并立即杀掉播放进程，再拿锁；否则 sink 线程正在 end_stream 排空时会阻塞取消。
        self._cancel.set()
        self.cancel_synthesis()
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
