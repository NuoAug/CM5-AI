# -*- coding: utf-8 -*-
"""TTS 基准：Piper 常驻路径与 sherpa-onnx 候选同文本对比。

保存同文本 WAV 到 data/tts-samples，便于人工试听；自动数字分数不能代替音质判断。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from tts_worker import PiperWorker  # noqa: E402

TEXTS = {
    "danger_5_15": "前方有台阶，请停下。",
    "short_5_15": "继续直行。",
    "normal_20_50": "前方近处是黄色盲道，往远是灰色路面，左边是建筑物。",
    "numbers": "生产日期2026年3月15日，保质期12个月，距离3.5米，型号A12B。",
    "mixed": "电梯在左边，按钮上写着B2，温度26.5度。",
    "long_100_200": (
        "正前方和近处没有明显障碍，脚下是连续的浅灰色地砖。"
        "左侧是黄色盲道，沿盲道向前延伸约十几步，右侧是深灰色沥青路面和白色路缘石。"
        "再往前可以看到一排低矮护栏和几盏路灯，远处有行人和车辆通过。"
        "如果需要继续前进，请保持直行，并用盲杖确认脚下和两侧边界。"
    ),
}


def pct(values, p):
    arr = sorted(values)
    return arr[min(len(arr) - 1, max(0, int(len(arr) * p) - 1))]


def save_wav(path: Path, data: bytes) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 1)


def save_samples(path: Path, samples: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
        return len(pcm) / float(sr)


def bench_piper(out_dir: Path, texts: dict, threads: int):
    import server
    server.TTS_IDLE_SECONDS = 0
    worker = PiperWorker(server.PIPER_BIN, server.PIPER_MODEL, out_dir / "tmp", timeout_s=90)
    rows = []
    for idx, (name, text) in enumerate(texts.items()):
        t0 = time.perf_counter()
        data, gen_ms = worker.synth(text, 1.0)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        path = out_dir / f"{idx:02d}-{name}.wav"
        audio_s = save_wav(path, data)
        rows.append({
            "backend": "piper-persistent", "name": name, "chars": len(text),
            "gen_ms": gen_ms, "wall_ms": wall_ms, "audio_s": round(audio_s, 2),
            "rtf": round(gen_ms / 1000 / max(audio_s, 0.01), 3),
            "cold": worker.status()["synth_count"] == 1,
            "wav": str(path),
        })
    status = worker.status()
    worker.stop()
    return rows, status


def bench_piper_subprocess(out_dir: Path, texts: dict, threads: int):
    import server
    rows = []
    for idx, (name, text) in enumerate(texts.items()):
        t0 = time.perf_counter()
        with __import__("tempfile").TemporaryDirectory() as d:
            import subprocess
            out = Path(d) / "x.wav"
            proc = subprocess.run(
                [str(server.PIPER_BIN), "-m", str(server.PIPER_MODEL), "-f", str(out),
                 "--length_scale", "1.0", "--sentence_silence", "0.12"],
                input=text.encode("utf-8"), capture_output=True, timeout=90,
                cwd=str(server.PIPER_BIN.parent),
            )
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"piper subprocess failed: {proc.stderr[-300:]}")
            data = out.read_bytes()
        gen_ms = int((time.perf_counter() - t0) * 1000)
        path = out_dir / f"{idx:02d}-{name}.wav"
        audio_s = save_wav(path, data)
        rows.append({
            "backend": "piper-subprocess", "name": name, "chars": len(text),
            "gen_ms": gen_ms, "wall_ms": gen_ms, "audio_s": round(audio_s, 2),
            "rtf": round(gen_ms / 1000 / max(audio_s, 0.01), 3),
            "cold": True, "wav": str(path),
        })
    return rows, {"engine": "piper-subprocess", "load_ms": None}


def find_candidate_dir(prefix: str):
    base = ROOT / "models" / "tts-candidates"
    if not base.is_dir():
        return None
    for d in sorted(base.glob(prefix + "*")):
        if list(d.glob("model*.onnx")) or (d / "model-steps-3.onnx").is_file():
            return d
    return None


def _sherpa_run(tts, texts: dict, out_dir: Path, backend: str, sample_rate_hint: int):
    rows = []
    for idx, (name, text) in enumerate(texts.items()):
        t0 = time.perf_counter()
        audio = tts.generate(text, sid=0, speed=1.0)
        gen_ms = int((time.perf_counter() - t0) * 1000)
        path = out_dir / f"{idx:02d}-{name}.wav"
        sr = int(audio.sample_rate or sample_rate_hint)
        audio_s = save_samples(path, np.asarray(audio.samples, dtype=np.float32), sr)
        rows.append({
            "backend": backend, "name": name, "chars": len(text),
            "gen_ms": gen_ms, "wall_ms": gen_ms, "audio_s": round(audio_s, 2),
            "rtf": round(gen_ms / 1000 / max(audio_s, 0.01), 3),
            "cold": idx == 0, "wav": str(path),
            "sample_rate": sr,
        })
    return rows


def bench_kokoro(out_dir: Path, texts: dict, threads: int, sid: int):
    import sherpa_onnx
    d = find_candidate_dir("kokoro")
    if d is None:
        raise RuntimeError("未找到 Kokoro 候选模型（models/tts-candidates/kokoro*）")
    model = next(iter(sorted(d.glob("model*.onnx")))
                 ) if list(d.glob("model*.onnx")) else d / "model.onnx"
    voices = d / "voices.bin"
    tokens = d / "tokens.txt"
    # v1.1 同时包含中英词典，中文文本里的字母/型号需要英文词典。
    lexicon = ",".join(
        str(p) for p in (d / "lexicon-us-en.txt", d / "lexicon-zh.txt") if p.is_file()
    ) or str(d / "lexicon.txt")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(model), voices=str(voices), tokens=str(tokens),
                lexicon=lexicon, lang="zh", length_scale=1.0,
                data_dir=str(d / "espeak-ng-data") if (d / "espeak-ng-data").is_dir() else "",
                dict_dir=str(d / "dict") if (d / "dict").is_dir() else "",
            ),
            num_threads=threads, provider="cpu",
        ),
        max_num_sentences=1,
        silence_scale=0.2,
        rule_fsts=",".join(
            str(p) for p in (d / "phone-zh.fst", d / "date-zh.fst", d / "number-zh.fst")
            if p.is_file()
        ),
    )
    t0 = time.perf_counter()
    tts = sherpa_onnx.OfflineTts(cfg)
    load_ms = int((time.perf_counter() - t0) * 1000)
    # v1.1 前三个说话人是英文；中文从 sid=3 (zf_001) 开始。
    if sid <= 0:
        sid = 3
    rows = []
    for idx, (name, text) in enumerate(texts.items()):
        t0 = time.perf_counter()
        audio = tts.generate(text, sid=sid, speed=1.0)
        gen_ms = int((time.perf_counter() - t0) * 1000)
        path = out_dir / f"{idx:02d}-{name}.wav"
        audio_s = save_samples(path, np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
        rows.append({
            "backend": "kokoro-int8-v1.1", "name": name, "chars": len(text),
            "gen_ms": gen_ms, "wall_ms": gen_ms, "audio_s": round(audio_s, 2),
            "rtf": round(gen_ms / 1000 / max(audio_s, 0.01), 3),
            "cold": idx == 0, "wav": str(path), "sample_rate": audio.sample_rate,
            "sid": sid,
        })
    return rows, {"engine": "kokoro-int8-v1.1", "load_ms": load_ms, "model": str(model), "sid": sid}


def bench_matcha(out_dir: Path, texts: dict, threads: int):
    import sherpa_onnx
    d = find_candidate_dir("matcha")
    if d is None:
        raise RuntimeError("未找到 Matcha 候选模型（models/tts-candidates/matcha*）")
    vocoder = ROOT / "models" / "tts-candidates" / "vocos-22khz-univ.onnx"
    if not vocoder.is_file():
        raise RuntimeError(f"缺少声码器：{vocoder}")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(d / "model-steps-3.onnx"),
                vocoder=str(vocoder),
                lexicon=str(d / "lexicon.txt"),
                tokens=str(d / "tokens.txt"),
                dict_dir=str(d / "dict") if (d / "dict").is_dir() else "",
            ),
            num_threads=threads, provider="cpu",
        ),
        max_num_sentences=1,
        silence_scale=0.2,
        rule_fsts=",".join(
            str(p) for p in (d / "phone.fst", d / "date.fst", d / "number.fst") if p.is_file()
        ),
    )
    t0 = time.perf_counter()
    tts = sherpa_onnx.OfflineTts(cfg)
    load_ms = int((time.perf_counter() - t0) * 1000)
    rows = _sherpa_run(tts, texts, out_dir, "matcha-zh-baker", 22050)
    return rows, {"engine": "matcha-zh-baker", "load_ms": load_ms, "model": str(d), "vocoder": str(vocoder)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default="piper",
        choices=["piper", "kokoro", "matcha", "piper-subprocess"],
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sid", type=int, default=0, help="Kokoro 说话人编号，v1.1 中文从 3 开始")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    texts = dict(TEXTS)
    if args.text_file:
        texts = {}
        for i, line in enumerate(Path(args.text_file).read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line:
                texts[f"custom_{i:02d}"] = line
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "data" / "tts-samples" / args.backend)
    if args.backend == "piper":
        rows, status = bench_piper(out_dir, texts, args.threads)
    elif args.backend == "piper-subprocess":
        rows, status = bench_piper_subprocess(out_dir, texts, args.threads)
    elif args.backend == "kokoro":
        rows, status = bench_kokoro(out_dir, texts, args.threads, args.sid)
    else:
        rows, status = bench_matcha(out_dir, texts, args.threads)
    result = {"backend": args.backend, "threads": args.threads, "status": status, "rows": rows}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    (ROOT / "data" / f"tts-bench-{args.backend}.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
