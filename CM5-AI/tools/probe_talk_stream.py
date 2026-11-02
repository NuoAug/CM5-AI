# -*- coding: utf-8 -*-
"""测量 /api/talk SSE 的首段、全文与片段间隔（客户端观察值，不是服务端内部耗时）。

用法：
  python tools/probe_talk_stream.py --image samples/camera_still.jpg --task scene
  python tools/probe_talk_stream.py --task chat --question 你好
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8090"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="")
    parser.add_argument("--task", default="scene")
    parser.add_argument("--question", default="")
    parser.add_argument("--item", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    body = {"task": args.task, "request_id": "probe-talk"}
    if args.image:
        body["image"] = "data:image/jpeg;base64," + base64.b64encode(Path(args.image).read_bytes()).decode("ascii")
        body["frame_id"] = "probe-talk"
        body["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.question:
        body["question"] = args.question
    if args.item:
        body["item"] = args.item

    t0 = time.perf_counter()
    first = None
    segments = []
    done = None
    with requests.post(BASE + "/api/talk", json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = json.loads(raw[5:].strip())
            now = int((time.perf_counter() - t0) * 1000)
            if "segment" in payload:
                text = (payload["segment"] or {}).get("text") or ""
                if text:
                    if first is None:
                        first = now
                    segments.append({"ms": now, "text": text})
            elif payload.get("done") is not None:
                done = now
                break

    result = {
        "task": args.task,
        "has_image": bool(args.image),
        "first_segment_ms": first,
        "done_ms": done,
        "segments": segments,
        "answer": "".join(s["text"] for s in segments),
        "note": "客户端观察值；首段包含排队、prefill 与首个完整语义片段",
    }
    out = Path(args.out) if args.out else Path("data") / "probe-talk.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "segments"}, ensure_ascii=False, indent=2))
    print("segments:", [(s["ms"], s["text"]) for s in segments[:6]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
