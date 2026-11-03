# -*- coding: utf-8 -*-
"""原生流式 Paraformer 适配：逐块输入同一 OnlineStream，默认不加载模型。

只接收 16 kHz 单声道归一化样本，不保留整段录音。每个原生流受最长话语
时长限制；超限不会伪造最终话语，必须显式 reset 后继续。模型不提供词级时间戳。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import threading
import time
import uuid
import weakref

import numpy as np


class StreamingASRError(RuntimeError):
    pass


class UtteranceLimitError(StreamingASRError):
    pass


@dataclass(frozen=True)
class StreamingASRConfig:
    model_dir: Path
    num_threads: int = 1
    sample_rate: int = 16000
    max_chunk_samples: int = 16000
    max_utterance_seconds: float = 30.0
    max_sessions: int = 2
    trailing_silence_seconds: float = 1.2
    silence_only_seconds: float = 2.4
    tail_padding_seconds: float = 0.66

    def __post_init__(self):
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        if self.sample_rate != 16000:
            raise ValueError("流式输入必须为 16000 Hz")
        for value in (self.num_threads, self.max_chunk_samples, self.max_sessions):
            if type(value) is not int or value < 1:
                raise ValueError("线程数、音频块上限和会话上限必须是正整数")
        for value in (self.max_utterance_seconds, self.trailing_silence_seconds,
                      self.silence_only_seconds, self.tail_padding_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("时长配置必须为有限正数")
        if self.tail_padding_seconds > 2:
            raise ValueError("结束补零最多两秒")


@dataclass(frozen=True)
class ASREvent:
    text: str
    is_final: bool
    utterance_id: str
    sequence: int
    audio_time_s: float
    produced_mono_ms: float
    reason: str

    @property
    def kind(self):
        return "final" if self.is_final else "partial"

    def to_dict(self):
        return dict(asdict(self), kind=self.kind, word_timestamps=None)


class StreamingASR:
    def __init__(self, config: StreamingASRConfig, *, engine=None):
        self.config = config
        self._engine = engine
        self._lock = threading.RLock()
        self._sessions = weakref.WeakSet()

    def load(self):
        with self._lock:
            if self._engine is not None:
                return self
            directory = self.config.model_dir
            files = {"tokens": directory / "tokens.txt", "encoder": directory / "encoder.int8.onnx",
                     "decoder": directory / "decoder.int8.onnx"}
            if not all(path.is_file() for path in files.values()):
                raise StreamingASRError("流式 Paraformer 缺少 tokens/encoder.int8/decoder.int8")
            import sherpa_onnx
            self._engine = sherpa_onnx.OnlineRecognizer.from_paraformer(
                **{name: str(path) for name, path in files.items()},
                num_threads=self.config.num_threads, sample_rate=16000,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=self.config.silence_only_seconds,
                rule2_min_trailing_silence=self.config.trailing_silence_seconds,
                # 本适配器在上限前拒绝超长话语，不把时长截断冒充用户说完。
                rule3_min_utterance_length=self.config.max_utterance_seconds + 10,
                decoding_method="greedy_search", provider="cpu")
            return self

    def create_session(self):
        with self._lock:
            if self._engine is None:
                raise StreamingASRError("模型尚未显式 load")
            if len(self._sessions) >= self.config.max_sessions:
                raise StreamingASRError("流式会话数达到上限")
            session = StreamingASRSession(self)
            self._sessions.add(session)
            return session

    def status(self):
        with self._lock:
            return {"backend": "sherpa-onnx-online-paraformer", "ready": self._engine is not None,
                    "sample_rate": 16000, "active_sessions": len(self._sessions),
                    "max_sessions": self.config.max_sessions, "word_timestamps_supported": False}

    def close(self):
        with self._lock:
            for session in list(self._sessions):
                session.close()
            self._engine = None


class StreamingASRSession:
    def __init__(self, owner):
        self._owner = owner
        self._stream = owner._engine.create_stream()
        self._closed = False
        self._finished = False
        self._overflowed = False
        self._total_samples = 0
        self._utterance_samples = 0
        self._last_text = ""
        self._sequence = 0
        self.utterance_id = "asr-" + uuid.uuid4().hex

    def _ensure_open(self):
        if self._closed:
            raise StreamingASRError("会话已关闭")
        if self._overflowed:
            raise UtteranceLimitError("话语超限，需显式 reset；未生成最终话语")
        if self._finished:
            raise StreamingASRError("输入已结束，需 reset 创建新话语")

    def _decode(self):
        engine = self._owner._engine
        # 有界块通常只需少量步骤；防止失效运行器无限报告 ready。
        for _ in range(512):
            if not engine.is_ready(self._stream):
                return
            engine.decode_stream(self._stream)
        raise StreamingASRError("运行器未能在有界解码步数内消化音频")

    def _event(self, text, final, reason):
        self._sequence += 1
        return ASREvent(text, final, self.utterance_id, self._sequence,
                        self._total_samples / 16000, time.monotonic_ns() / 1_000_000, reason)

    def _new_utterance(self):
        # 新建原生流，释放旧特征缓存，而不是长期保留已处理的录音特征。
        self._stream = self._owner._engine.create_stream()
        self._utterance_samples = 0
        self._last_text = ""
        self._finished = self._overflowed = False
        self.utterance_id = "asr-" + uuid.uuid4().hex

    def feed(self, samples, *, sample_rate=16000):
        with self._owner._lock:
            self._ensure_open()
            if sample_rate != 16000:
                raise ValueError("feed 只接受 16000 Hz；设备重采样须在 AudioInput 完成")
            if not hasattr(samples, "__len__") or len(samples) > self._owner.config.max_chunk_samples:
                raise ValueError("音频块超过上限")
            values = np.asarray(samples)
            if values.ndim != 1 or values.dtype != np.float32:
                raise ValueError("输入必须是单声道 float32 归一化样本")
            if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
                raise ValueError("样本必须有限且在 [-1,1] 内")
            if not len(values):
                return []
            if self._utterance_samples + len(values) > self._owner.config.max_utterance_seconds * 16000:
                self._stream = None
                self._overflowed = True
                raise UtteranceLimitError("话语超限，旧音频已释放；未生成最终话语")
            values = np.ascontiguousarray(values)
            self._stream.accept_waveform(16000, values)
            self._total_samples += len(values)
            self._utterance_samples += len(values)
            self._decode()
            engine = self._owner._engine
            text = str(engine.get_result(self._stream) or "").strip()
            events = []
            if text and text != self._last_text:
                events.append(self._event(text, False, "incremental"))
                self._last_text = text
            if engine.is_endpoint(self._stream):
                if text:
                    events.append(self._event(text, True, "endpoint"))
                self._new_utterance()
            return events

    def finish(self):
        with self._owner._lock:
            if self._finished and not self._closed:
                return []
            self._ensure_open()
            if self._utterance_samples == 0:
                self._stream = None
                self._finished = True
                return []
            padding = np.zeros(round(self._owner.config.tail_padding_seconds * 16000), dtype=np.float32)
            self._stream.accept_waveform(16000, padding)
            self._stream.input_finished()
            self._decode()
            text = str(self._owner._engine.get_result(self._stream) or "").strip()
            events = [self._event(text, True, "input_finished")] if text else []
            self._stream = None
            self._finished = True
            return events

    def reset(self):
        with self._owner._lock:
            if self._closed:
                raise StreamingASRError("会话已关闭")
            self._new_utterance()

    def close(self):
        with self._owner._lock:
            self._stream = None
            self._closed = True
            self._owner._sessions.discard(self)
