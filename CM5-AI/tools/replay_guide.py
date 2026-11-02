# -*- coding: utf-8 -*-
"""引导离线回放：逐张图片跑真实模型，输出 JSONL、混淆统计与延迟分位数。

用法示例：
  python tools/replay_guide.py --manifest testsets/guide_cases.json
  python tools/replay_guide.py --input-size 512x384 --out data/replay-512.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import guide  # noqa: E402


def percentile(values, p):
    if not values:
        return None
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "testsets" / "guide_cases.json"))
    parser.add_argument("--samples", default=str(ROOT / "samples"))
    parser.add_argument("--model", default="")
    parser.add_argument("--input-size", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cases = manifest.get("cases", manifest if isinstance(manifest, list) else [])
    samples = Path(args.samples)
    if args.model:
        guide.MODEL_PATH = Path(args.model)
    cfg = guide.CFG
    if args.input_size:
        w, h = [int(x) for x in args.input_size.lower().split("x")]
        cfg = replace(cfg, input_w=w, input_h=h)
        guide.INPUT_W, guide.INPUT_H = w, h

    out_path = Path(args.out) if args.out else (ROOT / "data" / f"guide-replay-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".summary.json")

    rows = []
    latencies = []
    passed = 0
    danger_cases = 0
    danger_missed = 0
    false_stops = 0
    state_errors = 0

    for idx, case in enumerate(cases):
        images = case.get("sequence") or [case["image"]]
        expected_dir = case.get("expected_direction")
        expected_reason = case.get("expected_reason")
        expect_danger = bool(case.get("expect_danger"))
        if len(images) > 1:
            sm = guide.GuideStateMachine(cfg)
            decisions = []
            observations = []
            for j, name in enumerate(images):
                img = Image.open(samples / name).convert("RGB")
                obs = guide.analyze_image(img, cfg=cfg, frame={"frame_id": f"seq-{idx}-{j}"})
                decision = sm.update(obs, now=j * 0.3)
                decisions.append(decision)
                observations.append(obs)
            final = decisions[-1]
            raw = final["state"]
            raw_reason = final["reason"]
            timing_values = [o.get("timing", {}).get("inference_ms") for o in observations]
            used_state_machine = True
        else:
            img = Image.open(samples / images[0]).convert("RGB")
            obs = guide.analyze_image(img, cfg=cfg, frame={"frame_id": f"single-{idx}"})
            final = None
            raw = obs.get("raw_direction")
            raw_reason = obs.get("raw_reason")
            timing_values = [obs.get("timing", {}).get("inference_ms")]
            used_state_machine = False

        for value in timing_values:
            if isinstance(value, (int, float)):
                latencies.append(int(value))
        ok = (expected_dir is None or raw == expected_dir) and (expected_reason is None or raw_reason == expected_reason)
        if ok:
            passed += 1
        if expect_danger:
            danger_cases += 1
            if raw != "stop":
                danger_missed += 1
        elif raw == "stop" and expected_dir != "stop":
            false_stops += 1
        if used_state_machine and expected_dir and raw != expected_dir:
            state_errors += 1
        rows.append(
            {
                "case": idx,
                "images": images,
                "data_kind": manifest.get("data_kind", "unknown"),
                "expected": {"direction": expected_dir, "reason": expected_reason, "danger": expect_danger},
                "actual": {"direction": raw, "reason": raw_reason, "state_machine": used_state_machine},
                "pass": ok,
                "note": case.get("note", ""),
                "quality": observations[0].get("quality") if used_state_machine else obs.get("quality"),
                "raw": final.get("raw_direction") if used_state_machine and final else raw,
                "raw_reason": final.get("raw_reason") if used_state_machine and final else raw_reason,
                "model": obs.get("model", {}),
                "sensor": obs.get("sensor", {}),
            }
        )

    write_jsonl(out_path, rows)
    summary = {
        "manifest": str(Path(args.manifest)),
        "data_kind": manifest.get("data_kind", "unknown"),
        "model": guide.status(),
        "input_size": f"{cfg.input_w}x{cfg.input_h}",
        "cases": len(cases),
        "pass": passed,
        "fail": len(cases) - passed,
        "danger_cases": danger_cases,
        "danger_missed": danger_missed,
        "false_stops": false_stops,
        "state_machine_errors": state_errors,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "note": "AI 生成样本只作补充；真实数据需按采集批次划分训练/测试。",
        "out": str(out_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
