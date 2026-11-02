# -*- coding: utf-8 -*-
"""分析 30 分钟/8 小时 soak 数据：推理、帧龄、状态分布、资源与健康检查。

用法：
  python tools/soak_analysis.py --run-dir ~/cm5-ai/test-runs/<id>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def pct(values, p):
    if not values:
        return None
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    rows = list(csv.DictReader((run / "resource.csv").open(encoding="utf-8")))
    metrics = []
    for line in (run / "metrics.jsonl").open(encoding="utf-8"):
        try:
            metrics.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def vals(key, cast=float):
        out = []
        for row in rows:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    out.append(cast(value))
                except ValueError:
                    pass
        return out

    infer = vals("guide_last_infer_ms")
    processed = vals("guide_processed", int)
    dropped = vals("guide_dropped", int)
    ages = [m.get("guide", {}).get("frame_age_ms") for m in metrics if isinstance(m.get("guide", {}).get("frame_age_ms"), int)]
    states: dict[str, int] = {}
    for m in metrics:
        g = m.get("guide") or {}
        key = f"{g.get('state')}/{g.get('reason')}"
        states[key] = states.get(key, 0) + 1

    result = {
        "run_dir": str(run),
        "samples": len(rows),
        "duration_s": vals("mono_ms", int)[-1] / 1000 if vals("mono_ms", int) else None,
        "guide_infer_ms": {
            "p50": pct(infer, 0.5), "p95": pct(infer, 0.95), "max": max(infer) if infer else None, "n": len(infer),
        },
        "guide_frame_age_ms": {
            "p50": pct(ages, 0.5), "p95": pct(ages, 0.95), "max": max(ages) if ages else None, "n": len(ages),
        },
        "guide_processed_delta": (max(processed) - min(processed)) if processed else None,
        "guide_dropped_delta": (max(dropped) - min(dropped)) if dropped else None,
        "states": states,
        "mem_avail_min_mb": min(vals("mem_avail_mb")) if vals("mem_avail_mb") else None,
        "swap_used_max_mb": max(vals("swap_used_mb")) if vals("swap_used_mb") else None,
        "temp_max_c": max(vals("temp_c")) if vals("temp_c") else None,
        "arm_mhz_min": min(vals("arm_mhz")) if vals("arm_mhz") else None,
        "throttled_values": sorted({r.get("throttled") for r in rows if r.get("throttled")}),
        "health_ok_pct": round(100 * sum(1 for r in rows if r.get("health_ok") == "True") / len(rows), 1) if rows else None,
        "status_ms_p50": pct(vals("status_ms"), 0.5),
        "status_ms_p95": pct(vals("status_ms"), 0.95),
        "note": "P50/P95 为采样近似值；资源采样本身有 10–60 ms 开销",
    }
    out = run / "results" / "soak-analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
