# -*- coding: utf-8 -*-
"""VLM 视觉路径基准：同图同 prompt，比较输入尺寸的 prefill/predicted/wall。

只在本机运行；Windows/AMD64 结果不能当作 CM5 性能。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = (
    "看图，用一到两句说清正前方从近到远有什么。"
    "如果画面主要是文字、标牌或屏幕，就照原文读出最醒目的文字，不要编造人物、车辆、台阶或道路。"
    "没有明显障碍就说“前方没看到明显障碍”。不要分点、不要换行。"
)


def encode_image(path: Path, max_side: int) -> tuple[str, float]:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    t0 = time.perf_counter()
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, (time.perf_counter() - t0) * 1000


def pct(values, p):
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def run_one(url, key, img_b64, prompt, max_tokens=72):
    payload = {
        "model": "qwen3.5-0.8b",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens, "temperature": 0.1, "top_p": 0.9,
        "cache_prompt": False,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(url + "/v1/chat/completions", json=payload,
                      headers={"Authorization": f"Bearer {key}"}, timeout=300)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    return {
        "wall_ms": wall_ms,
        "prefill_ms": timings.get("prompt_ms"),
        "decode_ms": timings.get("predicted_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": data["choices"][0].get("finish_reason"),
        "answer": (data["choices"][0]["message"].get("content") or "")[:160],
        "jpeg_kb": round(len(img_b64) * 0.75 / 1024, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18101")
    parser.add_argument("--key", default="")
    parser.add_argument("--sizes", default="384,448,512")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--images", default=str(ROOT / "samples/scene-corridor.jpg") + "," + str(ROOT / "samples/scene-tactile-straight.jpg"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    key = args.key or __import__("os").environ.get("VLM_API_KEY", "")
    if not key:
        print("缺少 VLM_API_KEY")
        return 1
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    images = [Path(x) for x in args.images.split(",") if x.strip()]
    rows = []
    for image in images:
        for size in sizes:
            b64, enc_ms = encode_image(image, size)
            # 预热一次，不计入样本
            run_one(args.url, key, b64, args.prompt)
            samples = [run_one(args.url, key, b64, args.prompt) for _ in range(max(1, args.runs))]
            for s in samples:
                s.update({
                    "image": image.name, "max_side": size, "encode_ms": round(enc_ms, 1),
                })
                rows.append(s)
            walls = [s["wall_ms"] for s in samples]
            prefills = [s["prefill_ms"] for s in samples if isinstance(s["prefill_ms"], (int, float))]
            print(f"{image.name} size={size} wall_p50={pct(walls, .5)} prefill_p50={pct(prefills, .5)} jpeg_kb={samples[0]['jpeg_kb']} prompt_tokens={samples[0]['prompt_tokens']}")
    result = {"url": args.url, "rows": rows, "note": "AMD64 Windows 结果，不是 CM5 性能"}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    out = Path(args.out) if args.out else (ROOT / "data" / "vlm-bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
