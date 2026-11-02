# -*- coding: utf-8 -*-
"""CM5 稳定性与资源采样：按固定间隔记录整机/进程资源、服务和引导状态。

只读采样 + 可选低频请求，不修改服务状态（除 mixed 场景会启停引导）。
输出 resource.csv、metrics.jsonl、results/soak.json。

用法：
  python tools/cm5_soak.py --run-id <id> --out-dir test-runs/<id> --duration 3600 --interval 2
  python tools/cm5_soak.py --duration 28800 --scenario mixed   # 8 小时
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("CM5_API", "http://127.0.0.1:8090")


def pss_kb(proc) -> int:
    try:
        with open(f"/proc/{proc.pid}/smaps_rollup", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    try:
        return int(proc.memory_info().rss / 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def find_procs():
    llama = app = None
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            cmdline = " ".join(p.info["cmdline"] or [])
            if name.startswith("llama-server"):
                llama = p
            elif "server.py" in cmdline and "/cm5-ai/" in cmdline:
                app = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return llama, app


def vcgencmd(args):
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=3).stdout
        return out.strip()
    except Exception:  # noqa: BLE001
        return ""


def api(path, method="GET", payload=None, timeout=10):
    try:
        r = requests.request(method, BASE + path, json=payload, timeout=timeout)
        return r.status_code, (r.json() if r.content else {})
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--duration", type=int, default=600, help="秒；8 小时用 28800")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--scenario", choices=["idle", "mixed"], default="idle")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S%z") + "_soak"
    out = Path(args.out_dir) if args.out_dir else (ROOT / "test-runs" / run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)

    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    fields = [
        "iso_time", "mono_ms", "mem_total_mb", "mem_avail_mb", "swap_used_mb",
        "llama_rss_kb", "llama_pss_kb", "app_rss_kb", "app_pss_kb",
        "temp_c", "throttled", "arm_mhz", "load1",
        "app_cpu_pct", "llama_cpu_pct", "mc_connected", "mc_scene_seq",
        "mc_scene_stale", "output_pushed", "output_cancelled", "output_pending",
        "health_ok", "status_ms",
    ]
    csv_path = out / "resource.csv"
    t_start = time.monotonic()
    samples = []
    errors = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        next_tick = time.monotonic()
        while not stop["flag"] and time.monotonic() - t_start < args.duration:
            t0 = time.monotonic()
            iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            llama, app = find_procs()
            llama_p = psutil.Process(llama.pid) if llama else None
            app_p = psutil.Process(app.pid) if app else None
            temp = vcgencmd(["measure_temp"])
            try:
                temp_c = float(temp.split("=")[1].split("'")[0])
            except Exception:  # noqa: BLE001
                temp_c = None
            thr = vcgencmd(["get_throttled"])
            freq = vcgencmd(["measure_clock", "arm"])
            try:
                arm_mhz = round(int(freq.split("=")[1]) / 1e6)
            except Exception:  # noqa: BLE001
                arm_mhz = None
            t_status = time.monotonic()
            code_h, health = api("/api/health", timeout=5)
            status_ms = int((time.monotonic() - t_status) * 1000)
            if code_h not in (200, 503):
                errors += 1
            _, status = api("/api/status", timeout=10)
            mc = (status or {}).get("main_control") or {}
            mc_scene = mc.get("scene") or {}
            out_stats = (status or {}).get("output") or {}
            row = {
                "iso_time": iso,
                "mono_ms": int((time.monotonic() - t_start) * 1000),
                "mem_total_mb": round(vm.total / 1024 ** 2),
                "mem_avail_mb": round(vm.available / 1024 ** 2),
                "swap_used_mb": round(swap.used / 1024 ** 2),
                "llama_rss_kb": round(llama_p.memory_info().rss / 1024) if llama_p else 0,
                "llama_pss_kb": pss_kb(llama_p) if llama_p else 0,
                "app_rss_kb": round(app_p.memory_info().rss / 1024) if app_p else 0,
                "app_pss_kb": pss_kb(app_p) if app_p else 0,
                "temp_c": temp_c,
                "throttled": thr.replace("throttled=", ""),
                "arm_mhz": arm_mhz,
                "load1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
                "app_cpu_pct": round(app_p.cpu_percent(interval=None), 1) if app_p else 0,
                "llama_cpu_pct": round(llama_p.cpu_percent(interval=None), 1) if llama_p else 0,
                "mc_connected": bool(mc.get("connected")),
                "mc_scene_seq": mc_scene.get("sequence"),
                "mc_scene_stale": bool(mc_scene.get("stale")),
                "output_pushed": out_stats.get("pushed"),
                "output_cancelled": out_stats.get("cancelled"),
                "output_pending": out_stats.get("pending"),
                "health_ok": health.get("ok") if isinstance(health, dict) else None,
                "status_ms": status_ms,
            }
            writer.writerow(row)
            f.flush()
            samples.append(row)

            if args.scenario == "mixed":
                action = random.choice(["tts", "scene", "status"])
                if action == "tts":
                    api("/api/tts", "POST", {"text": "这是测试播报。"}, timeout=30)
                elif action == "scene":
                    api("/api/main-control/scene", "GET", timeout=10)

            with (out / "metrics.jsonl").open("a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "t": row["mono_ms"], "iso": iso, "status": row,
                    "health": health, "main_control": {
                        "connected": mc.get("connected"),
                        "scene": mc_scene,
                        "last_error": mc.get("last_error"),
                    },
                }, ensure_ascii=False) + "\n")

            elapsed = time.monotonic() - t0
            next_tick += args.interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _vals(key):
        return [s[key] for s in samples if isinstance(s.get(key), (int, float))]

    def _pct(values, p):
        if not values:
            return None
        arr = sorted(values)
        return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]

    summary = {
        "run_id": run_id,
        "scenario": args.scenario,
        "duration_s": int(time.monotonic() - t_start),
        "requested_duration_s": args.duration,
        "samples": len(samples),
        "interval_s": args.interval,
        "completed": not stop["flag"] or time.monotonic() - t_start >= args.duration,
        "api_errors": errors,
        "mem_avail_min_mb": min(_vals("mem_avail_mb")) if samples else None,
        "swap_used_max_mb": max(_vals("swap_used_mb")) if samples else None,
        "llama_pss_max_mb": round(max(_vals("llama_pss_kb")) / 1024) if samples else None,
        "app_pss_max_mb": round(max(_vals("app_pss_kb")) / 1024) if samples else None,
        "temp_max_c": max(_vals("temp_c")) if samples else None,
        "arm_mhz_min": min(_vals("arm_mhz")) if samples else None,
        "throttled_values": sorted({s["throttled"] for s in samples if s.get("throttled")}),
        "health_ok_pct": round(100 * sum(1 for s in samples if s.get("health_ok")) / len(samples), 1) if samples else None,
        "status_ms_p50": _pct(_vals("status_ms"), 0.5),
        "status_ms_p95": _pct(_vals("status_ms"), 0.95),
        "main_control_connected_pct": round(100 * sum(1 for s in samples if s.get("mc_connected")) / len(samples), 1) if samples else None,
        "output_pushed_delta": (max(_vals("output_pushed")) - min(_vals("output_pushed"))) if samples else None,
        "output_cancelled_delta": (max(_vals("output_cancelled")) - min(_vals("output_cancelled"))) if samples else None,
        "measuring_overhead_note": "采样本身每轮读取 /api/health 与 /api/status，已记录 status_ms",
    }
    (out / "results" / "soak.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
