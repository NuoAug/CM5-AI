# -*- coding: utf-8 -*-
"""引导与 VLM 并发测试：问答期间持续采样引导是否被饿死。

输出：问答首段/全文、引导推理 P50/P95/最大值、帧龄、处理/丢弃增量、状态变化。
用于比较 VLM 线程数与调度策略，不是危险场景的替代品。

用法：
  python tools/test_concurrency.py --image samples/camera_still.jpg --out results/concurrency.json
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8090"


def pct(v, p):
    if not v:
        return None
    a = sorted(v)
    return a[min(len(a) - 1, max(0, int(len(a) * p) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", default="scene")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    args = parser.parse_args()

    requests.post(BASE + "/api/guide/start", json={"source": "device"}, timeout=30).raise_for_status()
    time.sleep(args.warmup_s)

    stop = threading.Event()
    samples = []

    def sampler():
        while not stop.is_set():
            try:
                g = requests.get(BASE + "/api/guide/status", timeout=10).json()
                st = g.get("stats") or {}
                samples.append({
                    "t": round(time.monotonic(), 3),
                    "infer": st.get("last_inference_ms"),
                    "frame_age": g.get("frame_age_ms"),
                    "processed": st.get("processed"),
                    "dropped": st.get("dropped"),
                    "stuck": g.get("stuck"),
                    "state": (g.get("decision") or {}).get("state"),
                    "reason": (g.get("decision") or {}).get("reason"),
                })
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time.sleep(1.0)

    body = {
        "task": args.task, "request_id": "concurrency",
        "image": "data:image/jpeg;base64," + base64.b64encode(Path(args.image).read_bytes()).decode("ascii"),
        "frame_id": "concurrency",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.perf_counter()
    first = None
    segments = []
    done = None
    with requests.post(BASE + "/api/talk", json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            p = json.loads(raw[5:].strip())
            now = int((time.perf_counter() - t0) * 1000)
            if "segment" in p:
                txt = (p["segment"] or {}).get("text") or ""
                if txt:
                    if first is None:
                        first = now
                    segments.append({"ms": now, "text": txt})
            elif p.get("done") is not None:
                done = now
                break

    time.sleep(1.0)
    stop.set()
    th.join(timeout=2)
    requests.post(BASE + "/api/guide/stop", json={"reason": "concurrency"}, timeout=20)

    infers = [s["infer"] for s in samples if isinstance(s["infer"], (int, float))]
    ages = [s["frame_age"] for s in samples if isinstance(s["frame_age"], int)]
    processed = [s["processed"] for s in samples if isinstance(s["processed"], int)]
    dropped = [s["dropped"] for s in samples if isinstance(s["dropped"], int)]
    result = {
        "talk": {"task": args.task, "first_segment_ms": first, "done_ms": done,
                 "segments": len(segments), "answer": "".join(s["text"] for s in segments)},
        "guide": {
            "samples": len(samples),
            "infer_p50_ms": pct(infers, 0.5), "infer_p95_ms": pct(infers, 0.95),
            "infer_max_ms": max(infers) if infers else None,
            "frame_age_p95_ms": pct(ages, 0.95), "frame_age_max_ms": max(ages) if ages else None,
            "processed_delta": (max(processed) - min(processed)) if processed else None,
            "dropped_delta": (max(dropped) - min(dropped)) if dropped else None,
            "stuck_samples": sum(1 for s in samples if s.get("stuck")),
            "states": sorted({(s.get("state"), s.get("reason")) for s in samples}, key=str),
        },
        "note": "并发性能数据；不能替代真人危险场景测试",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
