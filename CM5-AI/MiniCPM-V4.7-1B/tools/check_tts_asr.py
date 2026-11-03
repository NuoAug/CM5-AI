# -*- coding: utf-8 -*-
"""用本地 Paraformer 回听 TTS 样本，辅助发现漏字/读错。

自动转写只能做筛查，不能代替人工试听；指标是字符相似度与关键数字/字母是否保留。
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_tts import TEXTS  # noqa: E402

ASR_DIR = ROOT / "models" / "asr" / "sherpa-onnx-paraformer-zh-small-2024-03-09"


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = np.interp(
            np.linspace(0, len(data), n, endpoint=False),
            np.arange(len(data)),
            data,
        ).astype(np.float32)
    return data


def clean(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=[
        str(ROOT / "data" / "tts-samples" / "piper"),
        str(ROOT / "data" / "tts-samples" / "matcha"),
        str(ROOT / "data" / "tts-samples" / "kokoro"),
    ])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "data" / "tts-asr-check.json"))
    args = parser.parse_args()

    import sherpa_onnx
    engine = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(ASR_DIR / "model.int8.onnx"),
        tokens=str(ASR_DIR / "tokens.txt"),
        num_threads=args.threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )

    rows = []
    for d in args.dirs:
        base = Path(d)
        if not base.is_dir():
            continue
        for wav in sorted(base.glob("*.wav")):
            key = wav.stem.split("-", 1)[-1]
            ref = TEXTS.get(key, "")
            if not ref:
                continue
            stream = engine.create_stream()
            stream.accept_waveform(16000, load_wav(wav))
            engine.decode_stream(stream)
            hyp = (stream.result.text or "").strip()
            a, b = clean(ref), clean(hyp)
            ratio = round(difflib.SequenceMatcher(None, a, b).ratio(), 3) if a else 0.0
            rows.append({
                "wav": str(wav),
                "key": key,
                "ref": ref,
                "asr": hyp,
                "char_ratio": ratio,
                "latin_tokens_kept": all(
                    tok.lower() in hyp.lower()
                    for tok in re.findall(r"[A-Za-z][A-Za-z0-9]*", ref)
                    if len(tok) >= 2
                ),
            })
            print(f"{wav.name:24s} ratio={ratio:.3f} ref={ref[:18]} | asr={hyp[:40]}")
    Path(args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
