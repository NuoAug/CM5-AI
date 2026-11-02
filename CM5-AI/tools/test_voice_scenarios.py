# -*- coding: utf-8 -*-
"""CM5 语音业务主线：用 Piper 合成指令驱动 /api/voice，逐项记录意图、回答与延迟。

边界：
- 合成语音验证业务链路与意图，不代替真人唤醒准确率。
- 每项都从设备摄像头抓取新帧，保证使用当前画面。
- 结果写入 results/voice-scenarios.jsonl 与 voice-summary.json。
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8090"
PIPER = ROOT / "runtime/piper/piper"
PIPER_MODEL = ROOT / "models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx"

SCENARIOS = [
    ("wake_only", "小冰小冰", "唤醒反馈"),
    ("wake_command", "小冰小冰，前面有什么", "唤醒词与指令连续说"),
    ("guide_start", "小冰小冰，开始引导", "开始引导"),
    ("scene", "小冰小冰，前面有什么", "当前画面描述"),
    ("read_text", "小冰小冰，读一下文字", "OCR 读字"),
    ("read_date", "小冰小冰，读日期", "日期识别"),
    ("find_item", "小冰小冰，帮我找杯子", "找物品（同音压力）"),
    ("find_item_alt", "小冰小冰，帮我找水杯", "找物品（避同音）"),
    ("guide_status", "小冰小冰，现在能走吗", "引导状态快路径"),
    ("speech_stop", "小冰小冰，停止播报", "停止播报"),
    ("guide_stop", "小冰小冰，关闭引导", "关闭引导"),
    ("ambiguous", "小冰小冰，那个东西呢", "不明确指令"),
    ("negative", "小冰小冰，不要打开摄像头", "否定指令（放最后，会关摄像头）"),
]


def synth(text: str, out: Path) -> dict:
    if not out.exists():
        t0 = time.perf_counter()
        p = subprocess.run([str(PIPER), "-m", str(PIPER_MODEL), "-f", str(out), "--length_scale", "1.0"],
                           input=text, text=True, capture_output=True, timeout=120)
        return {"ok": p.returncode == 0 and out.exists(), "gen_ms": int((time.perf_counter() - t0) * 1000)}
    return {"ok": True, "gen_ms": 0, "cached": True}


def fresh_image(samples: Path) -> tuple[str, str, str]:
    r = requests.get(BASE + "/api/camera/still", timeout=20)
    r.raise_for_status()
    samples.mkdir(parents=True, exist_ok=True)
    path = samples / f"voice-{time.time_ns()}.jpg"
    path.write_bytes(r.content)
    return ("data:image/jpeg;base64," + base64.b64encode(r.content).decode("ascii"),
            r.headers.get("X-Frame-Id", ""), r.headers.get("X-Captured-At", ""))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", default="", help="只跑指定 case key")
    args = parser.parse_args()
    out = Path(args.out_dir)
    samples = out / "samples"
    results = out / "results"
    samples.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    rows = []

    for key, text, purpose in SCENARIOS:
        if args.only and key != args.only:
            continue
        wav = samples / f"voice-{key}.wav"
        s = synth(text, wav)
        image_b64, frame_id, captured_at = fresh_image(samples)
        body = {
            "audio": "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode("ascii"),
            "image": image_b64, "image_vlm": image_b64,
            "session_id": "device", "wake_detected": True,
            "request_id": f"scenario-{key}", "frame_id": frame_id,
            "captured_at": captured_at, "image_source": "device_camera",
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(BASE + "/api/voice", json=body, timeout=240)
            wall = int((time.perf_counter() - t0) * 1000)
            data = r.json() if r.content else {}
            row = {
                "case": key, "purpose": purpose, "text": text,
                "http": r.status_code, "wall_ms": wall,
                "intent": data.get("intent"), "action": data.get("action"),
                "answer": data.get("answer"), "spoken": data.get("spoken"),
                "latency_ms": data.get("latency_ms"), "frame_id": frame_id,
                "audio_kind": "synthetic_tts",
                "synth": s,
            }
            if key in ("guide_start", "guide_stop", "speech_stop", "guide_status"):
                g = requests.get(BASE + "/api/guide/status", timeout=10).json()
                row["guide_running_after"] = g.get("running")
                row["guide_state_after"] = (g.get("decision") or {}).get("state")
        except Exception as exc:  # noqa: BLE001
            row = {"case": key, "purpose": purpose, "text": text, "error": str(exc), "wall_ms": int((time.perf_counter() - t0) * 1000)}
        rows.append(row)
        with (results / "voice-scenarios.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: row.get(k) for k in ("case", "http", "wall_ms", "intent", "action", "spoken", "answer", "guide_running_after", "error")}, ensure_ascii=False))

    (results / "voice-summary.json").write_text(json.dumps({"rows": rows, "note": "合成语音，非真人唤醒"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
