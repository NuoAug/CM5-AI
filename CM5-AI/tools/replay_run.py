# -*- coding: utf-8 -*-
"""Windows 本机回放：用 CM5 实机录制的帧和音频复现识别与流程问题。

边界：
- 回放只读 test-runs 目录，不连接生产服务，不触发远端播放。
- 人为延迟/丢帧只验证时序逻辑，不等于 CM5 的 CPU、热降频、AEC 或 ALSA 行为。
- 跨平台数值差异用关键行为断言和合理容差比较，不强求自由文本逐字一致。

用法：
  python tools/replay_run.py --run-dir test-runs/<run_id>
  python tools/replay_run.py --run-dir <run> --drop-every 3 --stale-every 4 --delay-ms 120
  python tools/replay_run.py --run-dir <run> --compare-run test-runs/<another>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import guide  # noqa: E402
import intent  # noqa: E402
from hardware import ManualClock, ReplayCamera, ReplayMicrophone, TestAudioSink  # noqa: E402


def pct(values, p):
    if not values:
        return None
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    frames, audio = [], []
    for p in sorted((run_dir / "samples").glob("*.jpg")):
        frames.append({"path": str(p.relative_to(run_dir)), "frame_id": p.stem, "kind": "recorded_frame"})
    for p in sorted((run_dir / "samples").glob("*.wav")):
        audio.append({"path": str(p.relative_to(run_dir)), "kind": "recorded_audio"})
    return {"run_id": run_dir.name, "frames": frames, "audio": audio, "source": "inferred"}


def replay_guide(run_dir: Path, manifest: dict, drop_every: int, stale_every: int, delay_ms: int) -> dict:
    frames = [dict(f, path=str(run_dir / f["path"])) for f in manifest.get("frames", []) if (run_dir / f["path"]).exists()]
    if not frames:
        return {"ok": False, "error": "没有可用帧"}
    clock = ManualClock()
    cam = ReplayCamera(frames, clock=clock, drop_every=drop_every, stale_every=stale_every, delay_ms=delay_ms)
    sm = guide.GuideStateMachine()
    rows, latencies, faults = [], [], []
    for idx, (img, fid, captured_mono, meta) in enumerate(cam.iter_frames()):
        t0 = time.perf_counter()
        obs = guide.analyze_image(img, frame={"frame_id": fid, "captured_at": meta.get("captured_at", ""), "source": "replay"})
        infer_ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(infer_ms)
        decision = sm.update(obs, clock.now())
        rows.append({
            "index": idx,
            "frame_id": fid,
            "path": str(Path(frames[idx]["path"])),
            "state": decision.get("state"),
            "reason": decision.get("reason"),
            "raw_direction": obs.get("raw_direction"),
            "raw_reason": obs.get("raw_reason"),
            "quality": (obs.get("quality") or {}).get("reasons"),
            "metrics": obs.get("metrics"),
            "infer_ms": infer_ms,
            "virtual_time_s": round(clock.now(), 3),
        })
        if decision.get("changed"):
            faults.append({"frame_id": fid, "state": decision.get("state"), "reason": decision.get("reason")})
    return {
        "ok": bool(rows),
        "mode": "replay",
        "frames_input": len(frames),
        "frames_processed": len(rows),
        "dropped": len(frames) - len(rows),
        "latency_p50_ms": pct(latencies, 0.5),
        "latency_p95_ms": pct(latencies, 0.95),
        "states": [r["state"] for r in rows],
        "changes": faults,
        "rows": rows,
        "model": guide.status(),
        "note": "replay 的延迟是本机测量，不能代表 CM5 性能",
    }


def load_asr():
    import sherpa_onnx
    asr_dir = ROOT / "models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09"
    return sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(asr_dir / "model.int8.onnx"),
        tokens=str(asr_dir / "tokens.txt"),
        num_threads=2, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
    )


def replay_voice(run_dir: Path, manifest: dict, drop_after, silence_every: int) -> dict:
    items = [a for a in manifest.get("audio", []) if (run_dir / a["path"]).exists()]
    if not items:
        return {"ok": False, "error": "没有可用音频"}
    engine = load_asr()
    rows = []
    for item in items:
        mic = ReplayMicrophone(run_dir / item["path"], speed=0.0, drop_after=drop_after, silence_every=silence_every)
        chunks = list(mic.iter_chunks())
        if not chunks:
            rows.append({"path": item["path"], "error": "断流后没有音频", "text": "", "intent": None})
            continue
        x = np.concatenate(chunks)
        stream = engine.create_stream()
        stream.accept_waveform(16000, x)
        t0 = time.perf_counter()
        engine.decode_stream(stream)
        text = (stream.result.text or "").strip()
        task = intent.parse(text, has_context=False)
        rows.append({
            "path": item["path"],
            "kind": item.get("kind", "recorded"),
            "duration_s": round(x.size / 16000, 3),
            "rms": round(float(np.sqrt(np.mean(x ** 2))), 4),
            "text": text,
            "expected_text": item.get("expected_text", ""),
            "intent": task.intent,
            "slots": task.slots,
            "needs_image": task.needs_image,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        })
    return {
        "ok": True,
        "mode": "replay",
        "items": rows,
        "note": "合成音频只验证识别链路；真人语音需在 CM5 现场录制后回放",
    }


def compare(a: Path, b: Path) -> dict:
    def _load(p):
        f = p / "replay-out" / "summary.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    sa, sb = _load(a), _load(b)
    ga = (sa.get("guide") or {}).get("rows", [])
    gb = (sb.get("guide") or {}).get("rows", [])
    va = {r["path"]: r for r in (sa.get("voice") or {}).get("items", [])}
    vb = {r["path"]: r for r in (sb.get("voice") or {}).get("items", [])}
    guide_mismatch = [
        {"index": i, "a": ga[i].get("state"), "b": gb[i].get("state")}
        for i in range(min(len(ga), len(gb))) if ga[i].get("state") != gb[i].get("state")
    ]
    voice_mismatch = [
        {"path": k, "a": va[k].get("intent"), "b": vb[k].get("intent")}
        for k in sorted(set(va) & set(vb)) if va[k].get("intent") != vb[k].get("intent")
    ]
    lat_a = (sa.get("guide") or {}).get("latency_p50_ms")
    lat_b = (sb.get("guide") or {}).get("latency_p50_ms")
    return {
        "a": str(a), "b": str(b),
        "guide_frames": [len(ga), len(gb)],
        "guide_state_mismatch": guide_mismatch,
        "voice_intent_mismatch": voice_mismatch,
        "latency_p50_ratio": round(lat_b / lat_a, 2) if lat_a and lat_b else None,
        "note": "跨平台数值差异按行为断言比较；latency ratio 不能当作 CM5 加速结论",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--drop-every", type=int, default=0)
    parser.add_argument("--stale-every", type=int, default=0)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--mic-drop-after", type=int, default=None)
    parser.add_argument("--mic-silence-every", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--compare-run", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out) if args.out else (run_dir / "replay-out")
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(run_dir)
    manifest.setdefault("run_id", run_dir.name)

    g = replay_guide(run_dir, manifest, args.drop_every, args.stale_every, args.delay_ms)
    v = replay_voice(run_dir, manifest, args.mic_drop_after, args.mic_silence_every)

    with (out / "guide.jsonl").open("w", encoding="utf-8") as f:
        for row in g.get("rows", []):
            f.write(json.dumps({"mode": "replay", **row}, ensure_ascii=False) + "\n")
    with (out / "voice.jsonl").open("w", encoding="utf-8") as f:
        for row in v.get("items", []):
            f.write(json.dumps({"mode": "replay", **row}, ensure_ascii=False) + "\n")

    summary = {
        "run_id": manifest.get("run_id"),
        "mode": "replay",
        "faults": {
            "drop_every": args.drop_every, "stale_every": args.stale_every,
            "delay_ms": args.delay_ms, "mic_drop_after": args.mic_drop_after,
            "mic_silence_every": args.mic_silence_every,
        },
        "guide": {k: v2 for k, v2 in g.items() if k != "rows"},
        "voice": v,
    }
    if args.compare_run:
        summary["compare"] = compare(run_dir, Path(args.compare_run).resolve())
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "guide_states": summary["guide"].get("states"),
        "guide_latency_p50_ms": summary["guide"].get("latency_p50_ms"),
        "voice": [(r.get("path"), r.get("text"), r.get("intent")) for r in v.get("items", [])],
        "compare": summary.get("compare"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
