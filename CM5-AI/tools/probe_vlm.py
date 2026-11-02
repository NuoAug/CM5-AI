# -*- coding: utf-8 -*-
"""VLM 单图探测：逐尺寸测 prompt/prefill/decode/wall，用于 CM5 模型与分辨率取舍。

不启动服务，只向已运行的 llama-server 发请求；每次请求 cache_prompt=False，避免缓存掩盖冷帧。

用法：
  python tools/probe_vlm.py --image /tmp/cam.jpg --sizes 256,320,384,512 --max-tokens 8
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image

DEFAULT_PROMPT = (
    "看图，用一到两句说清正前方从近到远有什么。"
    "如果画面主要是文字、标牌或屏幕，就照原文读出最醒目的文字，不要编造人物、车辆、台阶或道路。"
    "没有明显障碍就说“前方没看到明显障碍”。不要分点、不要换行。"
)


def encode(path: Path, size: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > size:
        scale = size / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def one(url: str, key: str, img_b64: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "vlm",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens, "temperature": 0.1, "top_p": 0.9,
        "cache_prompt": False, "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(url + "/v1/chat/completions", json=payload,
                      headers={"Authorization": f"Bearer {key}"}, timeout=600)
    wall = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    d = r.json()
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    return {
        "wall_ms": wall,
        "prefill_ms": t.get("prompt_ms"),
        "decode_ms": t.get("predicted_ms"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "answer": (d["choices"][0]["message"].get("content") or "")[:120],
        "finish": d["choices"][0].get("finish_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--key", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--sizes", default="256,320,384,512")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    key = args.key or os.environ.get("VLM_API_KEY", "")
    if not key:
        print("缺少 API key：--key 或 VLM_API_KEY")
        return 1
    rows = []
    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        b64 = encode(Path(args.image), size)
        for run in range(max(1, args.runs)):
            row = one(args.url, key, b64, DEFAULT_PROMPT, args.max_tokens)
            row.update({"max_side": size, "run": run, "image": str(args.image)})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out) if args.out else Path("data") / "probe-vlm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"url": args.url, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
