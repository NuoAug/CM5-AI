# -*- coding: utf-8 -*-
"""CM5 真实外设与接口验收：摄像头、麦克风、扬声器、KWS/ASR、业务接口。

所有结果写入 <out>/results/accept.json，样本写入 <out>/samples/。
需要真人参与的项只给出现场步骤，不用合成音频冒充唤醒准确率。

用法（在 CM5 上，使用发布目录的 venv）：
  python tools/cm5_accept.py --run-id <id> --out-dir test-runs/<id> --section all
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

BASE = os.environ.get("CM5_API", "http://127.0.0.1:8090")
ALSA_IN = os.environ.get("ALSA_INPUT", "plughw:CARD=Array,DEV=0")
ALSA_OUT = os.environ.get("ALSA_OUTPUT", "plughw:CARD=Array,DEV=0")
PIPER_BIN = Path(os.environ.get("PIPER_BIN") or (ROOT / "runtime/piper/piper"))
PIPER_MODEL = Path(os.environ.get("PIPER_MODEL") or (ROOT / "models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx"))
RUN_DIR = ROOT / "test-runs"
SAMPLES = RUN_DIR / "_samples"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_env(src: Path, dst: Path) -> None:
    """复制运行配置，但抹掉 API key；不采集任何密钥。"""
    if not src.exists():
        return
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("VLM_API_KEY="):
            lines.append("VLM_API_KEY=<redacted>")
        elif line.startswith(("WEB_HOST=", "WEB_PORT=", "VLM_URL=", "DATA_DIR=")):
            lines.append(line)
        else:
            lines.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(out: Path, run_id: str, data: dict, samples: Path) -> None:
    env_src = Path(os.environ.get("CM5_ENV_FILE") or (Path.home() / "cm5-ai/env/cm5-ai.env"))
    sanitize_env(env_src, out / "config" / "cm5-ai.env.sanitized")
    freeze = ROOT / "pip-freeze.txt"
    if freeze.exists():
        shutil.copy2(freeze, out / "config" / "pip-freeze.txt")

    frames, audio = [], []
    for p in sorted(samples.glob("camera_*.jpg")):
        frames.append({"path": str(p.relative_to(out)), "frame_id": p.stem, "kind": "real_device_camera",
                       "sha256": sha256_file(p), "bytes": p.stat().st_size})
    for p in sorted(samples.glob("*.wav")):
        kind = "synthetic_tts" if "synth" in p.name or "phrase" in p.name else "real_device_microphone"
        audio.append({"path": str(p.relative_to(out)), "kind": kind, "sha256": sha256_file(p), "bytes": p.stat().st_size})

    manifest = {
        "run_id": run_id,
        "created_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "timezone": time.strftime("%z"),
        "source": "raspberry_pi_cm5" if sys.platform.startswith("linux") else "windows_replay_host",
        "hardware": {
            "model": Path("/proc/device-tree/model").read_text(errors="ignore").strip() if Path("/proc/device-tree/model").exists() else None,
            "arch": os.uname().machine if hasattr(os, "uname") else None,
        },
        "frames": frames,
        "audio": audio,
        "config": {"env": "config/cm5-ai.env.sanitized", "env_sha256": sha256_file(out / "config" / "cm5-ai.env.sanitized") if (out / "config" / "cm5-ai.env.sanitized").exists() else None},
        "data_kind": {"real_device": True, "synthetic_tts": True, "replay": False},
        "manual_confirmations": {
            "speaker_audible_human": False,
            "human_wake_word_accuracy": False,
            "main_control_joint_test": False,
            "note": "需要现场人工确认；自动结果不能替代听辨与真人唤醒测试",
        },
        "notes": [
            "TTS 合成音频只验证链路，不能当作真人唤醒准确率",
            "回环录音只能证明声卡链路有能量输出，不能量化实际音质",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def pct(values, p):
    if not values:
        return None
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x.astype(np.float32)
    n = int(round(x.size * dst / src))
    idx = np.linspace(0, x.size - 1, n, dtype=np.float32)
    return np.interp(idx, np.arange(x.size, dtype=np.float32), x).astype(np.float32)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"只支持 16-bit WAV：{path}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def wav_b64(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def piper(text: str, out: Path, length_scale: float = 1.0) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(PIPER_BIN), "-m", str(PIPER_MODEL), "-f", str(out), "--length_scale", str(length_scale)],
        input=text, text=True, capture_output=True, timeout=120,
    )
    return {
        "ok": proc.returncode == 0 and out.exists(),
        "gen_ms": int((time.perf_counter() - t0) * 1000),
        "bytes": out.stat().st_size if out.exists() else 0,
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-1:] or [],
    }


def record(seconds: float, out: Path) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["arecord", "-q", "-D", ALSA_IN, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", str(out), "-d", str(int(seconds))],
        capture_output=True, text=True, timeout=seconds + 10,
    )
    dt = time.perf_counter() - t0
    info = {"ok": proc.returncode == 0 and out.exists(), "elapsed_ms": int(dt * 1000)}
    if out.exists():
        x, sr = read_wav(out)
        info.update({
            "rate": sr, "duration_s": round(x.size / sr, 3),
            "rms": round(float(np.sqrt(np.mean(x ** 2))), 4),
            "peak": round(float(np.max(np.abs(x))), 4),
            "clipped": int(np.sum(np.abs(x) > 0.999)),
        })
    return info


def cam_frames(count: int = 20, samples_dir: Optional[Path] = None) -> dict:
    """优先走服务端设备摄像头；服务未运行时才直接打开 Picamera2。

    服务端常驻摄像头时再开第二个 Picamera2 会失败，因此这里按实际状态分流。
    """
    import requests
    try:
        st = requests.get(BASE + "/api/device/status", timeout=8).json()
        running = bool((st.get("camera") or {}).get("running"))
    except Exception:  # noqa: BLE001
        st, running = {}, False

    if running:
        shots = []
        for idx in range(3):
            t0 = time.perf_counter()
            r = requests.get(BASE + "/api/camera/still", timeout=20)
            if r.status_code != 200:
                return {"ok": False, "error": f"抓图 HTTP {r.status_code}", "mode": "server_device_camera"}
            path = (samples_dir or Path(".")) / ("camera_still.jpg" if idx == 0 else f"camera_still_{idx + 1}.jpg")
            path.write_bytes(r.content)
            img = Image.open(io.BytesIO(r.content))
            shots.append({
                "ms": int((time.perf_counter() - t0) * 1000),
                "bytes": len(r.content),
                "size": list(img.size),
                "frame_id": r.headers.get("X-Frame-Id"),
                "captured_at": r.headers.get("X-Captured-At"),
                "sha256": sha256_file(path),
            })
            time.sleep(0.4)
        distinct = len({s["sha256"] for s in shots}) > 1
        return {
            "ok": True,
            "mode": "server_device_camera",
            "frames": len(shots),
            "distinct_frames": distinct,
            "non_frozen": distinct,
            "shots": shots,
            "device_camera": st.get("camera"),
            "images": {"still": Image.open(samples_dir / "camera_still.jpg")} if samples_dir else {},
        }

    import device_io
    frames = []
    intervals = []
    last = [None]

    def cb(img, fid, mono, meta):
        if last[0] is not None:
            intervals.append((mono - last[0]) * 1000)
        last[0] = mono
        frames.append({"frame_id": fid, "mono": mono, "size": img.size, "meta": meta, "image": img.copy()})

    cam = device_io.DeviceCamera(fps=10.0)
    t0 = time.perf_counter()
    if not cam.start(cb):
        return {"ok": False, "error": cam.status().get("error"), "available": cam.available()}
    start_ms = int((time.perf_counter() - t0) * 1000)
    deadline = time.time() + max(3.0, count / 8.0)
    while len(frames) < count and time.time() < deadline:
        time.sleep(0.05)
    still = cam.capture_still()
    status = cam.status()
    cam.stop()
    if not frames:
        return {"ok": False, "error": status.get("error") or "未取到帧", "start_ms": start_ms, "status": status}
    first = frames[0]["image"]
    sample = frames[len(frames) // 2]["image"]
    return {
        "ok": True,
        "mode": "direct_picamera2",
        "available": status.get("available"),
        "start_ms": start_ms,
        "frames": len(frames),
        "fps_est": round(1000.0 / float(np.mean(intervals)), 2) if intervals else None,
        "interval_p50_ms": round(pct(intervals, 0.5), 1) if intervals else None,
        "interval_p95_ms": round(pct(intervals, 0.95), 1) if intervals else None,
        "low_size": list(status.get("low_size") or []),
        "use_lores": status.get("use_lores"),
        "still_size": list(still.size) if still is not None else None,
        "first_frame_id": frames[0]["frame_id"],
        "last_frame_id": frames[-1]["frame_id"],
        "captured_at": now_iso(),
        "error": status.get("error"),
        "images": {"first": first, "mid": sample, "still": still},
    }


def speaker_loopback(samples_dir: Path) -> dict:
    tone_path = samples_dir / "speaker_tone.wav"
    phrase_path = samples_dir / "speaker_phrase.wav"
    loop_path = samples_dir / "speaker_loopback.wav"
    sr = 16000
    t = np.arange(int(sr * 2.0)) / sr
    tone = 0.35 * np.sin(2 * np.pi * 1000 * t)
    write_wav(tone_path, tone, sr)
    phrase = piper("红灯请停下等待，前面有台阶请小心。", phrase_path)
    rec = subprocess.Popen(
        ["arecord", "-q", "-D", ALSA_IN, "-f", "S16_LE", "-r", str(sr), "-c", "1", "-t", "wav", str(loop_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.8)
    play_t0 = time.perf_counter()
    play = subprocess.run(["aplay", "-q", "-D", ALSA_OUT, str(tone_path)], capture_output=True, text=True, timeout=30)
    play_ms = int((time.perf_counter() - play_t0) * 1000)
    time.sleep(0.8)
    rec.terminate()
    try:
        rec.wait(timeout=5)
    except subprocess.TimeoutExpired:
        rec.kill()
    result = {"tone_play_ok": play.returncode == 0, "tone_play_ms": play_ms, "phrase": phrase}
    if loop_path.exists():
        x, got_sr = read_wav(loop_path)
        result["loopback"] = {
            "rate": got_sr, "duration_s": round(x.size / got_sr, 2),
            "rms": round(float(np.sqrt(np.mean(x ** 2))), 4),
            "peak": round(float(np.max(np.abs(x))), 4),
        }
        if x.size >= got_sr:
            spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
            freqs = np.fft.rfftfreq(x.size, 1 / got_sr)
            band = (freqs > 950) & (freqs < 1050)
            total = float(np.sum(spec ** 2)) + 1e-12
            result["loopback"]["tone_band_ratio"] = round(float(np.sum(spec[band] ** 2)) / total, 4)
            result["loopback"]["dominant_hz"] = round(float(freqs[int(np.argmax(spec))]), 1)
    return result


def kws_check(samples_dir: Path) -> dict:
    import kws
    wav = samples_dir / "kws_wake_synth.wav"
    synth = piper("小冰小冰", wav, length_scale=1.1)
    if not synth.get("ok"):
        return {"ok": False, "error": "Piper 合成失败", "synth": synth}
    x, sr = read_wav(wav)
    x16 = resample(x, sr, 16000)
    result = kws.detect(x16)
    return {
        "ok": bool(result.get("hits")),
        "kind": "synthetic_tts",
        "note": "TTS 合成唤醒词只验证链路，不能代替真人唤醒准确率",
        "hits": result.get("hits"),
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
        "synth_gen_ms": synth.get("gen_ms"),
    }


def asr_check(samples_dir: Path) -> dict:
    import sherpa_onnx
    wav = samples_dir / "asr_command_synth.wav"
    if not wav.exists():
        synth = piper("小冰小冰，前面有什么", wav, length_scale=1.1)
        if not synth.get("ok"):
            return {"ok": False, "error": "Piper 合成失败"}
    x, sr = read_wav(wav)
    x16 = resample(x, sr, 16000)
    asr_dir = ROOT / "models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09"
    engine = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(asr_dir / "model.int8.onnx"),
        tokens=str(asr_dir / "tokens.txt"),
        num_threads=int(os.environ.get("ASR_THREADS", "1")),
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    stream = engine.create_stream()
    stream.accept_waveform(16000, x16)
    t0 = time.perf_counter()
    engine.decode_stream(stream)
    text = (stream.result.text or "").strip()
    return {
        "ok": bool(text),
        "kind": "synthetic_tts",
        "text": text,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


def http_json(method: str, path: str, payload=None, timeout=60, raw=False):
    import requests
    url = BASE + path
    t0 = time.perf_counter()
    r = requests.request(method, url, json=payload, timeout=timeout)
    dt = int((time.perf_counter() - t0) * 1000)
    if raw:
        return {"status": r.status_code, "ms": dt, "headers": dict(r.headers), "bytes": len(r.content)}
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = r.text[:400]
    return {"status": r.status_code, "ms": dt, "body": body}


def api_checks(samples_dir: Path) -> dict:
    out = {}
    out["health"] = http_json("GET", "/api/health", timeout=10)
    out["status"] = http_json("GET", "/api/status", timeout=15)
    still = samples_dir / "camera_still.jpg"
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(still.read_bytes()).decode("ascii") if still.exists() else ""
    if img_b64:
        out["ocr"] = http_json("POST", "/api/ocr", {"image": img_b64}, timeout=60)
        out["tts"] = http_json("POST", "/api/tts", {"text": "主控接口测试播报。", "engine": "piper"}, timeout=60)
        out["talk"] = http_json("POST", "/api/talk", {
            "task": "scene", "image": img_b64, "frame_id": "accept-talk", "request_id": "accept-talk",
        }, timeout=180)
    voice_wav = samples_dir / "asr_command_synth.wav"
    if voice_wav.exists() and img_b64:
        out["voice"] = http_json("POST", "/api/voice", {
            "audio": wav_b64(voice_wav), "image": img_b64,
            "session_id": "accept", "wake_detected": True,
            "request_id": "accept-voice", "frame_id": "accept-voice",
            "captured_at": now_iso(),
        }, timeout=180)
    out["cancel"] = http_json("POST", "/api/output/cancel", {"min_priority": 1}, timeout=20)
    return out


def main_control_check() -> dict:
    """只验证适配器连通与场景查询；替身/未配置不算真实主控联调通过。"""
    status = http_json("GET", "/api/main-control/status", timeout=10)
    body = status.get("body") or {}
    info = body.get("status") or {}
    scene = http_json("GET", "/api/main-control/scene", timeout=10)
    return {
        "ok": bool(info.get("connected")) and not info.get("fixture_allowed"),
        "configured": bool(info.get("configured")),
        "connected": bool(info.get("connected")),
        "master_instance_id": info.get("master_instance_id"),
        "clock_id": info.get("clock_id"),
        "evidence_mode": info.get("evidence_mode"),
        "scene": scene,
        "note": "未配置或仅替身连接时不得当作真实主控联调通过",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--section", choices=["device", "api", "all"], default="all")
    args = parser.parse_args()
    out = Path(args.out_dir) if args.out_dir else (RUN_DIR / args.run_id)
    samples = out / "samples"
    results = out / "results"
    samples.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    data = {"run_id": args.run_id, "started_at": now_iso(), "sections": {}}
    events = []
    t_all = time.perf_counter()

    if args.section in ("device", "all"):
        t0 = time.perf_counter()
        cam = cam_frames(20, samples)
        if cam.get("ok"):
            for key, img in (cam.pop("images") or {}).items():
                if img is not None:
                    path = samples / f"camera_{key}.jpg"
                    img.save(path, format="JPEG", quality=90)
        cam["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        data["sections"]["camera"] = cam
        events.append({"at": now_iso(), "event": "camera_done", "ok": cam.get("ok")})

        t0 = time.perf_counter()
        mic = record(5.0, samples / "mic_ambient.wav")
        mic["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        data["sections"]["microphone"] = mic
        events.append({"at": now_iso(), "event": "mic_done", "ok": mic.get("ok")})

        t0 = time.perf_counter()
        spk = speaker_loopback(samples)
        spk["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        data["sections"]["speaker"] = spk
        events.append({"at": now_iso(), "event": "speaker_done", "ok": spk.get("tone_play_ok")})

        data["sections"]["kws"] = kws_check(samples)
        data["sections"]["asr"] = asr_check(samples)

    if args.section in ("api", "all"):
        data["sections"]["api"] = api_checks(samples)
        data["sections"]["main_control"] = main_control_check()

    data["elapsed_ms"] = int((time.perf_counter() - t_all) * 1000)
    data["finished_at"] = now_iso()
    result_path = results / "accept.json"
    if result_path.exists():
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            old.setdefault("sections", {}).update(data["sections"])
            old["finished_at"] = data["finished_at"]
            old.setdefault("started_at", data["started_at"])
            data = old
        except Exception:  # noqa: BLE001
            pass
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(out, args.run_id, data, samples)
    with (out / "events.jsonl").open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(json.dumps({
        "run_id": args.run_id,
        "camera": data["sections"].get("camera", {}).get("ok"),
        "mic": data["sections"].get("microphone", {}).get("ok"),
        "speaker_play": data["sections"].get("speaker", {}).get("tone_play_ok"),
        "speaker_loopback_tone_ratio": data["sections"].get("speaker", {}).get("loopback", {}).get("tone_band_ratio"),
        "kws_synth_hits": data["sections"].get("kws", {}).get("hits"),
        "asr_synth_text": data["sections"].get("asr", {}).get("text"),
        "api_status": (data["sections"].get("api", {}).get("health") or {}).get("status"),
        "main_control": data["sections"].get("main_control", {}).get("ok"),
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
