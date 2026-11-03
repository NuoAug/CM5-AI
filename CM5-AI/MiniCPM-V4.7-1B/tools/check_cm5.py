# -*- coding: utf-8 -*-
"""CM5 环境体检：架构、内存、存储、温度/欠压、摄像头、音频、模型与线程配置。

只读；不启动服务、不加载模型。Windows 上运行会明确报告不是 CM5。
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def cmd(cargs):
    try:
        return subprocess.run(cargs, capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    data = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "is_aarch64": platform.machine().lower() in ("aarch64", "arm64"),
        "cpu_count": psutil.cpu_count(),
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
            "available_gb": round(psutil.virtual_memory().available / 1024 ** 3, 2),
        },
        "disk": {},
        "pi": {
            "temperature_c": None,
            "throttled": None,
            "model": cmd(["cat", "/proc/device-tree/model"]) if Path("/proc/device-tree/model").exists() else None,
        },
        "devices": {
            "picamera2": shutil.which("libcamera-hello") is not None or importlib.util.find_spec("picamera2") is not None,
            "arecord": shutil.which("arecord"),
            "aplay": shutil.which("aplay"),
        },
        "models": {},
        "config": {
            key: os.environ.get(key, "(默认)")
            for key in (
                "MEM_BUDGET_GB", "VLM_THREADS", "VLM_CTX",
                "KWS_THREADS", "ASR_THREADS", "OCR_INTRA_THREADS", "VOICE_SOURCE", "OUTPUT_SINK",
            )
        },
        "warnings": [],
    }
    for path in (ROOT, ROOT / "data", Path("/")):
        try:
            usage = shutil.disk_usage(path)
            data["disk"][str(path)] = {
                "free_gb": round(usage.free / 1024 ** 3, 2),
                "total_gb": round(usage.total / 1024 ** 3, 2),
            }
        except Exception:  # noqa: BLE001
            pass

    if shutil.which("vcgencmd"):
        out = cmd(["vcgencmd", "measure_temp"])
        if "temp=" in out:
            data["pi"]["temperature_c"] = float(out.split("=")[1].split("'")[0])
        out = cmd(["vcgencmd", "get_throttled"])
        if "throttled=" in out:
            data["pi"]["throttled"] = out.split("=")[1]
            if data["pi"]["throttled"] != "0x0":
                data["warnings"].append("检测到欠压/降频标志，先解决供电与散热")
    else:
        data["warnings"].append("无 vcgencmd：当前不是树莓派或未安装 raspberrypi utils")

    for name, rel in (
        ("vlm_model", "vlm/MiniCPM-V 4.7 1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf"),
        ("vlm_mmproj", "vlm/MiniCPM-V 4.7 1B/mmproj-model-f16.gguf"),
        ("asr", "asr/sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx"),
        ("piper", "tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx"),
    ):
        p = MODELS / rel
        data["models"][name] = {"exists": p.exists(), "mb": round(p.stat().st_size / 1024 ** 2, 1) if p.exists() else None}
        if not p.exists():
            data["warnings"].append(f"缺少模型 {rel}")

    if not data["is_aarch64"]:
        data["warnings"].append("当前不是 aarch64；性能与线程配置不能代表 CM5")
    if data["memory"]["total_gb"] < 3.8 and data["is_aarch64"]:
        data["warnings"].append("可用内存低于 4GB 标称，检查是否被其它服务占用")
    if data["memory"]["available_gb"] < 1.2:
        data["warnings"].append("可用内存低于 1.2GB，加载模型前先释放内存")

    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
