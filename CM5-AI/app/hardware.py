# -*- coding: utf-8 -*-
"""硬件抽象：真机、回放和测试接收器共用同一组接口。

真机类直接复用 device_io，避免两套实现漂移：
  LiveCamera = DeviceCamera
  LiveMicrophone = AlsaCapture
  DeviceAudioOutput = AlsaPlayer
回放类只读录制数据，不连接真实生产服务、不触发远端播放。
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
from PIL import Image

import device_io

LiveCamera = device_io.DeviceCamera
LiveMicrophone = device_io.AlsaCapture
DeviceAudioOutput = device_io.AlsaPlayer


class ManualClock:
    """可注入时钟：回放和确定性测试用，单位为秒。"""

    def __init__(self, start: float = 0.0):
        self._t = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._t += float(seconds)
            return self._t


class ReplayCamera:
    """按清单回放帧，可注入丢帧、过期、延迟。

    frames: [{"path": Path, "frame_id": str, "captured_at": str, "meta": dict}]
    drop_every: 每 N 帧丢 1 帧（0 表示不丢）
    stale_every: 每 N 帧把 captured_mono 提前 stale_ms，模拟过期帧
    """

    def __init__(
        self,
        frames: list[dict],
        clock: Optional[ManualClock] = None,
        drop_every: int = 0,
        stale_every: int = 0,
        stale_ms: int = 600,
        delay_ms: int = 0,
    ):
        self.frames = frames
        self.clock = clock or ManualClock()
        self.drop_every = drop_every
        self.stale_every = stale_every
        self.stale_ms = stale_ms
        self.delay_ms = delay_ms

    def iter_frames(self) -> Iterator[tuple[Image.Image, str, float, dict]]:
        for idx, item in enumerate(self.frames):
            if self.delay_ms:
                self.clock.advance(self.delay_ms / 1000.0)
            if self.drop_every and idx % self.drop_every == self.drop_every - 1:
                self.clock.advance(0.25)
                continue
            path = Path(item["path"])
            img = Image.open(path).convert("RGB")
            captured = self.clock.now()
            if self.stale_every and idx % self.stale_every == self.stale_every - 1:
                captured -= self.stale_ms / 1000.0
            self.clock.advance(0.25)
            yield img, str(item.get("frame_id") or f"replay-{idx}"), captured, dict(item.get("meta") or {})


class ReplayMicrophone:
    """把 WAV 切成固定块；可原速/加速，可注入断流与静音。

    drop_after: 在该块序号后停止输出（模拟设备断流）
    silence_every: 每 N 块替换为静音（模拟遮挡/失声）
    """

    def __init__(
        self,
        wav_path: Path,
        chunk_frames: int = 1600,
        sample_rate: int = 16000,
        speed: float = 0.0,
        drop_after: Optional[int] = None,
        silence_every: int = 0,
    ):
        self.wav_path = Path(wav_path)
        self.chunk_frames = chunk_frames
        self.sample_rate = sample_rate
        self.speed = speed
        self.drop_after = drop_after
        self.silence_every = silence_every

    def _read(self) -> np.ndarray:
        with wave.open(str(self.wav_path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            raw = w.readframes(n)
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if sr != self.sample_rate and x.size:
            n2 = int(round(x.size * self.sample_rate / sr))
            x = np.interp(
                np.linspace(0, x.size - 1, n2, dtype=np.float32),
                np.arange(x.size, dtype=np.float32),
                x,
            ).astype(np.float32)
        return x

    def iter_chunks(self) -> Iterator[np.ndarray]:
        x = self._read()
        for idx, start in enumerate(range(0, x.size, self.chunk_frames)):
            if self.drop_after is not None and idx >= self.drop_after:
                return
            chunk = x[start:start + self.chunk_frames]
            if self.silence_every and idx % self.silence_every == self.silence_every - 1:
                chunk = np.zeros_like(chunk)
            if self.speed > 0:
                # 原速回放：按音频时长等待，便于观察实时行为
                time.sleep(len(chunk) / self.sample_rate / self.speed)
            yield chunk


class TestAudioSink:
    """测试接收器：把音频写到目录，不调用 ALSA 或远端播放。"""

    def __init__(self, out_dir: Path, prefix: str = "tts"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._seq = 0

    def feed(self, wav_bytes: bytes, cancel_event=None) -> bool:
        self._seq += 1
        path = self.out_dir / f"{self.prefix}-{self._seq:03d}.wav"
        path.write_bytes(wav_bytes)
        return True

    def play(self, wav_bytes: bytes, cancel_event=None) -> bool:
        return self.feed(wav_bytes, cancel_event)

    def end_stream(self) -> None:
        return None

    def stop(self) -> None:
        return None
