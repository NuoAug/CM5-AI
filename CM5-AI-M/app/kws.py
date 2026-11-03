# -*- coding: utf-8 -*-
"""sherpa-onnx 中文关键词检测（KWS）。

模型为 sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01，
Apache-2.0。关键词文件用拼音 token 描述，见 config/keywords.txt。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "kws" / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
MODEL_DIR = Path(os.environ.get("KWS_MODEL_DIR") or DEFAULT_MODEL_DIR)
KEYWORDS_FILE = Path(
    os.environ.get("KWS_KEYWORDS")
    or (ROOT / "app" / "config" / "keywords.txt")
)
KWS_THREADS = int(os.environ.get("KWS_THREADS", "1"))
KWS_SCORE = float(os.environ.get("KWS_SCORE", "2.0"))
KWS_THRESHOLD = float(os.environ.get("KWS_THRESHOLD", "0.10"))
KWS_TRAILING_BLANKS = int(os.environ.get("KWS_TRAILING_BLANKS", "8"))
KWS_USE_INT8 = os.environ.get("KWS_INT8", "0").strip().lower() in ("1", "true", "yes")
SAMPLE_RATE = 16000

_LOCK = threading.Lock()
_engine: Optional["KwsEngine"] = None
_load_error: Optional[str] = None


def _pick(pattern: str) -> Optional[Path]:
    files = sorted(MODEL_DIR.glob(pattern))
    if KWS_USE_INT8:
        int8 = [p for p in files if ".int8." in p.name]
        if int8:
            return int8[0]
    plain = [p for p in files if ".int8." not in p.name]
    return plain[0] if plain else (files[0] if files else None)


class KwsEngine:
    def __init__(self) -> None:
        self.tokens = MODEL_DIR / "tokens.txt"
        self.encoder = _pick("encoder-*.onnx")
        self.decoder = _pick("decoder-*.onnx")
        self.joiner = _pick("joiner-*.onnx")
        self.keywords = KEYWORDS_FILE
        self.spotter = None
        self.load_error = None

    def missing(self) -> list[str]:
        need = [self.tokens, self.encoder, self.decoder, self.joiner, self.keywords]
        return [str(p) for p in need if p is None or not p.is_file()]

    def available(self) -> bool:
        return not self.missing()

    def load(self) -> bool:
        if self.spotter is not None:
            return True
        if self.load_error:
            return False
        missing = self.missing()
        if missing:
            self.load_error = "缺少 KWS 文件：" + "，".join(missing)
            return False
        try:
            self.spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(self.tokens),
                encoder=str(self.encoder),
                decoder=str(self.decoder),
                joiner=str(self.joiner),
                keywords_file=str(self.keywords),
                num_threads=max(1, KWS_THREADS),
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                max_active_paths=4,
                keywords_score=KWS_SCORE,
                keywords_threshold=KWS_THRESHOLD,
                num_trailing_blanks=KWS_TRAILING_BLANKS,
                provider="cpu",
            )
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"KWS 加载失败：{exc}"
            self.spotter = None
            return False
        return True

    def new_stream(self):
        if not self.load():
            return None
        return self.spotter.create_stream()

    def feed(self, stream, samples: np.ndarray) -> list[str]:
        """向一个持续流送音频，返回本次解码检测到的关键词。"""
        if stream is None or self.spotter is None:
            return []
        hits = []
        samples = np.asarray(samples, dtype=np.float32)
        stream.accept_waveform(SAMPLE_RATE, samples)
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)
            result = self.spotter.get_result(stream)
            if result:
                hits.append(result)
        return hits

    def detect(self, samples: np.ndarray, tail_padding: float = 0.5) -> list[str]:
        """离线检测：整段音频 + 尾部静音，再刷新解码。"""
        if not self.load():
            return []
        stream = self.spotter.create_stream()
        hits = []
        x = np.asarray(samples, dtype=np.float32)
        for i in range(0, len(x), 1600):
            hits.extend(self.feed(stream, x[i:i + 1600]))
        if tail_padding > 0:
            hits.extend(self.feed(stream, np.zeros(int(SAMPLE_RATE * tail_padding), dtype=np.float32)))
        stream.input_finished()
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)
            result = self.spotter.get_result(stream)
            if result:
                hits.append(result)
        return hits


def engine() -> KwsEngine:
    global _engine, _load_error
    with _LOCK:
        if _engine is None:
            _engine = KwsEngine()
            if not _engine.available():
                _load_error = "缺少 KWS 文件：" + "，".join(_engine.missing())
            else:
                _engine.load()
                _load_error = _engine.load_error
        return _engine


def status() -> dict:
    eng = engine()
    ready = bool(eng.spotter is not None)
    return {
        "ready": ready,
        "model": str(MODEL_DIR),
        "keywords": str(KEYWORDS_FILE),
        "threads": KWS_THREADS,
        "int8": KWS_USE_INT8,
        "score": KWS_SCORE,
        "threshold": KWS_THRESHOLD,
        "trailing_blanks": KWS_TRAILING_BLANKS,
        "error": _load_error or eng.load_error,
    }


def detect(samples: np.ndarray) -> dict:
    eng = engine()
    t0 = time.perf_counter()
    hits = eng.detect(samples)
    return {
        "keyword": hits[0] if hits else "",
        "hits": hits,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "ready": eng.spotter is not None,
        "error": _load_error or eng.load_error,
    }


class AudioRingBuffer:
    """短音频环形缓冲：保留唤醒词之前的音频，避免丢掉指令开头。"""

    def __init__(self, seconds: float = 3.0, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.max_samples = int(seconds * sample_rate)
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        with self._lock:
            self._chunks.append(x)
            self._samples += x.size
            while self._samples > self.max_samples and self._chunks:
                head = self._chunks[0]
                excess = self._samples - self.max_samples
                if head.size <= excess:
                    self._chunks.pop(0)
                    self._samples -= head.size
                else:
                    self._chunks[0] = head[excess:]
                    self._samples -= excess

    def snapshot(self, seconds: Optional[float] = None) -> np.ndarray:
        with self._lock:
            data = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        if seconds is not None:
            data = data[-int(seconds * self.sample_rate):]
        return data.astype(np.float32, copy=False)

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._samples = 0


class WakeGate:
    """常开 KWS + 简单能量 VAD 的半双工语音门。

    只负责唤醒与切出一句指令；意图解析由 intent 模块完成。
    """

    def __init__(self, engine_obj: Optional[KwsEngine] = None, armed_seconds: float = 15.0,
                 buffer_commands: bool = True):
        self.kws = engine_obj or engine()
        self.buffer_commands = bool(buffer_commands)
        self.stream = None
        self.ring = AudioRingBuffer(seconds=3.0)
        self.sample_rate = SAMPLE_RATE
        self.armed_until = 0.0
        self.armed_seconds = armed_seconds
        self.command: list[np.ndarray] = []
        self.voiced = 0
        self.silence = 0.0
        self.speaking = False
        self.threshold = float(os.environ.get("KWS_VAD_RMS", "0.012"))
        self.silence_seconds = float(os.environ.get("KWS_VAD_SILENCE", "0.7"))
        self.max_command_seconds = float(os.environ.get("KWS_MAX_COMMAND", "8.0"))

    def reset(self) -> None:
        self.command = []
        self.voiced = 0
        self.silence = 0.0
        self.speaking = False

    def feed(self, samples: np.ndarray) -> dict:
        """返回 {wake: bool, command: np.ndarray|None, keyword: str}。"""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.ring.append(x)
        if self.stream is None:
            self.stream = self.kws.new_stream()
        hits = self.kws.feed(self.stream, x)
        now = time.monotonic()
        wake = False
        keyword = ""
        if hits:
            keyword = hits[0]
            self.stream = self.kws.new_stream()
            if now >= self.armed_until:
                self.armed_until = now + self.armed_seconds
                wake = True
                self.reset()

        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        command = None
        if not self.buffer_commands:
            # 流式ASR自行处理端点；这里只保留唤醒与有界前滚，不再重复拼整段命令。
            return {"wake": wake, "keyword": keyword, "command": None,
                    "armed": now < self.armed_until, "rms": round(rms, 4),
                    "preroll": self.ring.snapshot(0.8) if wake else None}
        if now < self.armed_until:
            if rms >= self.threshold:
                self.speaking = True
                self.silence = 0.0
                self.voiced += x.size
                self.command.append(x)
            elif self.speaking:
                self.silence += x.size / self.sample_rate
                self.command.append(x)
                if self.silence >= self.silence_seconds:
                    spoken = self.voiced / self.sample_rate
                    data = np.concatenate(self.command) if self.command else np.zeros(0, dtype=np.float32)
                    if spoken >= 0.5:
                        command = data
                    self.reset()
                    self.ring.clear()
            if self.voiced / self.sample_rate >= self.max_command_seconds:
                data = np.concatenate(self.command) if self.command else np.zeros(0, dtype=np.float32)
                command = data
                self.reset()
                self.ring.clear()
        return {
            "wake": wake,
            "keyword": keyword,
            "command": command,
            "armed": now < self.armed_until,
            "rms": round(rms, 4),
        }
