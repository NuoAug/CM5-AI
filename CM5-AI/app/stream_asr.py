# -*- coding: utf-8 -*-
"""流式 Paraformer：边收音频边解码，区分临时结果与最终结果。

模型：sherpa-onnx-streaming-paraformer-bilingual-zh-en（Apache-2.0），
本机使用 int8 encoder/decoder，加载后约 230 MB。不是把整段录音反复切碎识别。
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
DEFAULT_DIR = ROOT / "models" / "asr-streaming" / "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
MODEL_DIR = Path(os.environ.get("ASR_STREAM_MODEL_DIR") or DEFAULT_DIR)
THREADS = int(os.environ.get("ASR_STREAM_THREADS", "1"))
USE_INT8 = os.environ.get("ASR_STREAM_INT8", "1").strip().lower() in ("1", "true", "yes")
SAMPLE_RATE = 16000

_LOCK = threading.RLock()
_RECOGNIZER = None
_LOAD_ERROR: Optional[str] = None
_LOAD_MS: Optional[int] = None
_LAST_USED = 0.0


def choose(name: str) -> Path:
    int8 = MODEL_DIR / f"{name}.int8.onnx"
    plain = MODEL_DIR / f"{name}.onnx"
    if USE_INT8 and int8.exists():
        return int8
    return plain if plain.exists() else int8


def files() -> dict:
    return {
        "tokens": MODEL_DIR / "tokens.txt",
        "encoder": choose("encoder"),
        "decoder": choose("decoder"),
    }


def missing() -> list[str]:
    return [str(p) for p in files().values() if not p.is_file()]


def ensure():
    """按需加载；模型缺失时明确报错，不退回假流式。"""
    global _RECOGNIZER, _LOAD_ERROR, _LOAD_MS, _LAST_USED
    with _LOCK:
        if _RECOGNIZER is not None:
            _LAST_USED = time.monotonic()
            return _RECOGNIZER
        if _LOAD_ERROR:
            raise RuntimeError(_LOAD_ERROR)
        lack = missing()
        if lack:
            _LOAD_ERROR = "缺少流式 ASR 文件：" + "，".join(lack)
            raise RuntimeError(_LOAD_ERROR)
        t0 = time.perf_counter()
        try:
            f = files()
            _RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=str(f["tokens"]),
                encoder=str(f["encoder"]),
                decoder=str(f["decoder"]),
                num_threads=max(1, THREADS),
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                enable_endpoint_detection=False,
                decoding_method="greedy_search",
                provider="cpu",
            )
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERROR = f"流式 ASR 加载失败：{exc}"
            _RECOGNIZER = None
            raise RuntimeError(_LOAD_ERROR) from exc
        _LOAD_MS = int((time.perf_counter() - t0) * 1000)
        _LAST_USED = time.monotonic()
        return _RECOGNIZER


def create():
    rec = ensure()
    return rec.create_stream()


def accept(stream, samples: np.ndarray) -> str:
    """送入一小块音频并返回当前临时文本；解码在锁内串行执行。"""
    global _LAST_USED
    rec = ensure()
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    with _LOCK:
        stream.accept_waveform(SAMPLE_RATE, x)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        _LAST_USED = time.monotonic()
        return (rec.get_result(stream) or "").strip()


def finish(stream) -> str:
    global _LAST_USED
    rec = ensure()
    with _LOCK:
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        _LAST_USED = time.monotonic()
        return (rec.get_result(stream) or "").strip()


def release() -> bool:
    global _RECOGNIZER, _LAST_USED
    with _LOCK:
        if _RECOGNIZER is None:
            return False
        _RECOGNIZER = None
        _LAST_USED = 0.0
        return True


def status() -> dict:
    f = files()
    return {
        "ready": _RECOGNIZER is not None,
        "available": not missing(),
        "model": str(MODEL_DIR),
        "encoder": str(f["encoder"]),
        "decoder": str(f["decoder"]),
        "int8": USE_INT8,
        "threads": THREADS,
        "load_ms": _LOAD_MS,
        "error": _LOAD_ERROR,
    }
