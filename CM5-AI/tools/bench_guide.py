# -*- coding: utf-8 -*-
"""引导模型基准：真实推理延迟分位数与进程内存增量。

Windows 结果只能用于开发对比，不能当作 CM5 性能。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import guide  # noqa: E402


def pct(values, p):
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def mem_mb():
    info = psutil.Process().memory_info()
    return round(getattr(info, "private", info.rss) / 1024 ** 2, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(ROOT / "samples" / "scene-corridor.jpg"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=guide.GUIDE_THREADS)
    parser.add_argument("--input-size", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = guide.CFG
    if args.input_size:
        w, h = [int(x) for x in args.input_size.lower().split("x")]
        cfg = replace(cfg, input_w=w, input_h=h)
        guide.INPUT_W, guide.INPUT_H = w, h
    guide.GUIDE_THREADS = args.threads
    guide._session = None
    guide._load_error = None

    img = Image.open(args.image).convert("RGB")
    before = mem_mb()
    t0 = time.perf_counter()
    guide.analyze_image(img, cfg=cfg)
    load_ms = int((time.perf_counter() - t0) * 1000)
    after_load = mem_mb()

    for _ in range(max(0, args.warmup)):
        guide.analyze_image(img, cfg=cfg)

    values = []
    decisions = {}
    for _ in range(max(1, args.frames)):
        obs = guide.analyze_image(img, cfg=cfg)
        ms = obs.get("timing", {}).get("inference_ms")
        if isinstance(ms, (int, float)):
            values.append(int(ms))
        key = f"{obs.get('raw_direction')}/{obs.get('raw_reason')}"
        decisions[key] = decisions.get(key, 0) + 1

    after = mem_mb()
    result = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "is_cm5": platform.machine().lower() in ("aarch64", "arm64"),
            "note": "Windows x86 结果不能当作 CM5 性能",
        },
        "model": guide.status(),
        "input_size": f"{cfg.input_w}x{cfg.input_h}",
        "threads": args.threads,
        "frames": args.frames,
        "latency_ms": {
            "p50": pct(values, 0.50) if values else None,
            "p95": pct(values, 0.95) if values else None,
            "p99": pct(values, 0.99) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        },
        "load_ms": load_ms,
        "memory_mb": {"before": before, "after_load": after_load, "after_frames": after, "delta": round(after - before, 1)},
        "decisions": decisions,
        "image": str(args.image),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
