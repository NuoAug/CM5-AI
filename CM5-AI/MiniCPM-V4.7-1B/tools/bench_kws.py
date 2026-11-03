# -*- coding: utf-8 -*-
"""KWS 离线基准：对正向/负向 wav 目录逐个检测，统计命中率、延迟与每小时误唤醒估计。

必须使用真人录音；TTS 合成音频不能代表唤醒样本。
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import kws  # noqa: E402


def read_wav(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        n2 = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data), n2, endpoint=False), np.arange(len(data)), data)
    return data.astype(np.float32), len(data) / 16000


def files(directory: str):
    if not directory:
        return []
    return sorted(Path(directory).glob("*.wav"))


def run(directory, engine):
    rows = []
    total_duration = 0.0
    for path in files(directory):
        samples, duration = read_wav(path)
        total_duration += duration
        result = engine.detect(samples)
        rows.append({"file": path.name, "duration_s": round(duration, 2), "hits": result})
    return rows, total_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", default="", help="包含唤醒词的真人 wav 目录")
    parser.add_argument("--negative", default="", help="不含唤醒词的真人 wav 目录")
    parser.add_argument("--keywords", default="", help="覆盖关键词文件；空则用 app/config/keywords.txt")
    parser.add_argument("--threshold", type=float, default=kws.KWS_THRESHOLD)
    parser.add_argument("--score", type=float, default=kws.KWS_SCORE)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    engine = kws.KwsEngine()
    if args.keywords:
        engine.keywords = Path(args.keywords)
    kws.KWS_THRESHOLD = args.threshold
    kws.KWS_SCORE = args.score
    if not engine.load():
        print(json.dumps({"error": engine.load_error or "KWS 未就绪"}, ensure_ascii=False))
        return 1

    pos, pos_dur = run(args.positive, engine)
    neg, neg_dur = run(args.negative, engine)
    pos_hits = sum(1 for row in pos if row["hits"])
    neg_hits = sum(1 for row in neg if row["hits"])
    result = {
        "keywords": str(engine.keywords),
        "threshold": args.threshold,
        "score": args.score,
        "positive": {
            "files": len(pos),
            "hits": pos_hits,
            "miss_rate": round(1 - pos_hits / len(pos), 3) if pos else None,
            "duration_s": round(pos_dur, 1),
            "rows": pos,
        },
        "negative": {
            "files": len(neg),
            "false_accepts": neg_hits,
            "duration_s": round(neg_dur, 1),
            "false_accepts_per_hour": round(neg_hits / neg_dur * 3600, 2) if neg_dur else None,
            "rows": neg,
        },
        "note": "样本必须为真人录音；样本量小，误唤醒率只能作为初步估计。",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
