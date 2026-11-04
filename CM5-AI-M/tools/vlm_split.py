# -*- coding: utf-8 -*-
"""板端 VLM 拆测：文本无图基线 vs 同图不同输入尺寸，拆分预填充与解码耗时。

只读服务：调用 llama-server /v1/chat/completions，不经过 app/server.py。
用法示例（板端）：

    PYTHONPATH=/home/mylin/CM5-AI-M/app /home/mylin/CM5-AI-M/app/.venv/bin/python \
        tools/vlm_split.py --image samples/scene-corridor.jpg \
        --sizes 384,448,512 --max-tokens 8 --out data/vlm-split.json

结果里的 prompt_ms/predicted_ms 来自服务端 timings；prompt_n/predicted_n 为 token 数。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = (
    "看图，用一到两句说清画面里有什么。"
    "方位只说左/中/右，远近只说近/远；看不清就直说，不猜、不补常识；"
    "不要判断有无障碍、安全、可通行性或实际距离。不要分点、不要换行。"
)


def encode_image(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def post(url: str, key: str, payload: dict, timeout: int = 600) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        url + "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    return {
        "wall_ms": wall_ms,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": data["choices"][0].get("finish_reason"),
        "answer": (data["choices"][0]["message"].get("content") or "")[:160],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--key", default=os.environ.get("VLM_API_KEY", ""))
    parser.add_argument("--image", default=str(ROOT / "samples/scene-corridor.jpg"))
    parser.add_argument("--sizes", default="384,448,512")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--text-max-tokens", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default=str(ROOT / "data" / "vlm-split.json"))
    args = parser.parse_args()
    if not args.key:
        print("缺少 VLM_API_KEY")
        return 1
    image = Path(args.image)
    rows = []
    started = datetime.now(timezone.utc).isoformat()

    text_payload = {
        "model": "minicpm-v",
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.text_max_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
        "cache_prompt": False,
        "stream": False,
    }
    text_row = post(args.url, args.key, text_payload)
    text_row.update({"kind": "text_only", "image": None, "max_side": None,
                     "jpeg_kb": 0.0, "encode_ms": 0.0})
    rows.append(text_row)
    print("text_only", json.dumps(text_row, ensure_ascii=False))

    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        t0 = time.perf_counter()
        b64 = encode_image(image, size)
        encode_ms = round((time.perf_counter() - t0) * 1000, 1)
        payload = {
            "model": "minicpm-v",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": args.prompt},
            ]}],
            "max_tokens": args.max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "cache_prompt": False,
            "stream": False,
        }
        row = post(args.url, args.key, payload)
        row.update({"kind": "image", "image": image.name, "max_side": size,
                    "jpeg_kb": round(len(b64) * 0.75 / 1024, 1), "encode_ms": encode_ms})
        rows.append(row)
        print(f"image size={size}", json.dumps(row, ensure_ascii=False))

    result = {
        "url": args.url,
        "image": str(image),
        "prompt": args.prompt,
        "sizes": [int(x) for x in args.sizes.split(",") if x.strip()],
        "max_tokens": args.max_tokens,
        "text_max_tokens": args.text_max_tokens,
        "rows": rows,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "note": "CM5 板端结果；只读 llama-server，不经 app/server.py",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
