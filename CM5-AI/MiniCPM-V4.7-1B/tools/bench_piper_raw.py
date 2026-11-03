# -*- coding: utf-8 -*-
"""测量 Piper 1.2.0 `--output_raw` 的首个 PCM 与总合成时间。

用途：核实“当前安装的 Piper 是否真的在句内分块输出”。如果 first_byte_ms
接近 total_ms，说明仍是整句合成后才输出，不能据此声称句内流式。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER_BIN = ROOT / "runtime" / "piper" / "piper" / ("piper.exe" if __import__("os").name == "nt" else "piper")
PIPER_MODEL = ROOT / "models" / "tts" / "piper-zh_CN-huayan-medium" / "zh_CN-huayan-medium.onnx"

TEXTS = {
    "short_5_15": "继续直行。",
    "normal_20_50": "前方近处是黄色盲道，往远是灰色路面，左边是建筑物。",
    "long_100_200": (
        "正前方和近处没有明显障碍，脚下是连续的浅灰色地砖。"
        "左侧是黄色盲道，沿盲道向前延伸约十几步，右侧是深灰色沥青路面和白色路缘石。"
        "再往前可以看到一排低矮护栏和几盏路灯，远处有行人和车辆通过。"
        "如果需要继续前进，请保持直行，并用盲杖确认脚下和两侧边界。"
    ),
}


def run_one(name: str, text: str, timeout_s: float = 60.0) -> dict:
    proc = subprocess.Popen(
        [str(PIPER_BIN), "-m", str(PIPER_MODEL), "--output_raw",
         "--length_scale", "1.0", "--sentence_silence", "0.12"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=str(PIPER_BIN.parent),
    )
    t0 = time.perf_counter()
    first = {"ms": None}
    done = threading.Event()
    state = {"bytes": 0}

    def reader():
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            if first["ms"] is None:
                first["ms"] = (time.perf_counter() - t0) * 1000
            state["bytes"] += len(chunk)
        done.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    proc.stdin.write((text + "\n").encode("utf-8"))
    proc.stdin.flush()
    proc.stdin.close()
    done.wait(timeout_s)
    timed_out = not done.is_set()
    if timed_out:
        proc.kill()
    proc.wait(timeout=5)
    total_ms = (time.perf_counter() - t0) * 1000
    audio_s = state["bytes"] / (22050 * 2)
    return {
        "name": name, "chars": len(text),
        "first_pcm_ms": round(first["ms"]) if first["ms"] is not None else None,
        "total_ms": round(total_ms),
        "audio_s": round(audio_s, 2),
        "rtf": round(total_ms / 1000 / max(audio_s, 0.01), 3),
        "intra_sentence_streaming": bool(
            first["ms"] is not None and (total_ms - first["ms"]) > 300
        ),
        "timed_out": timed_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "piper-raw-bench.json"))
    args = parser.parse_args()
    rows = [run_one(name, text) for name, text in TEXTS.items()]
    result = {
        "piper_version": subprocess.run(
            [str(PIPER_BIN), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "hardware": "Windows x86 dev machine (not CM5)",
        "rows": rows,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
