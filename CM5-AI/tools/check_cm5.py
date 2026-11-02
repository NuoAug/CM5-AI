# -*- coding: utf-8 -*-
"""CM5 环境体检与部署前快照：只读，输出 JSON 和人可读 Markdown。

默认把快照写到 <ROOT>/test-runs/<run_id>/environment.{json,md}。
Windows 上运行会明确标注“不是 CM5”，不能当作真机结论。

用法：
  python tools/check_cm5.py --run-id 20261102T143500+0800_preflight --hash
  python tools/check_cm5.py --out-dir D:/tmp/snap
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def run(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "").strip(), (p.stderr or "").strip(), p.returncode
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}", -1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_throttled(raw: str) -> dict:
    try:
        value = int(raw, 16)
    except (TypeError, ValueError):
        return {"raw": raw, "flags": [], "current_fault": None, "historical": None}
    flags = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
    current = [name for bit, name in THROTTLE_BITS.items() if bit < 16 and value & (1 << bit)]
    return {
        "raw": raw,
        "flags": flags,
        "current_fault": bool(current),
        "current_flags": current,
        "historical": bool(value & 0xFFFF0000),
    }


def parse_asound() -> list[dict]:
    text = Path("/proc/asound/cards").read_text(encoding="utf-8", errors="ignore")
    cards = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+\[(\S+)\s*\]:\s*(\S+)\s*-\s*(\S+)", line)
        if m:
            cards.append({"index": int(m.group(1)), "id": m.group(2), "driver": m.group(3), "name": m.group(4)})
    return cards


def package_versions(names: list[str]) -> dict:
    out = {}
    for name in names:
        stdout, _, code = run(["dpkg-query", "-W", "-f=${Version}", name])
        if code == 0 and stdout:
            out[name] = stdout
    return out


def venv_versions(venv: Path) -> dict:
    pip = venv / "bin" / "pip"
    if not pip.exists():
        return {}
    stdout, _, code = run([str(pip), "list", "--format=json", "--disable-pip-version-check"], timeout=20)
    if code != 0:
        return {}
    try:
        return {row["name"].lower(): row["version"] for row in json.loads(stdout)}
    except Exception:  # noqa: BLE001
        return {}


def collect(hash_models: bool) -> dict:
    is_linux = sys.platform.startswith("linux")
    machine = platform.machine().lower()
    data = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "machine": machine,
            "is_aarch64": machine in ("aarch64", "arm64"),
            "kernel": platform.release(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "is_cm5_like": False,
            "hardware_model": None,
        },
        "os": {},
        "cpu": {},
        "memory": {},
        "storage": {},
        "thermal": {},
        "throttle": {},
        "power": {},
        "camera": {},
        "audio": {},
        "usb": [],
        "network": {},
        "time": {},
        "services": {},
        "ports": [],
        "processes": [],
        "runtimes": {},
        "models": {},
        "warnings": [],
        "notes": [],
    }

    if is_linux:
        for key, path in (
            ("hardware_model", "/proc/device-tree/model"),
            ("os_release", "/etc/os-release"),
        ):
            p = Path(path)
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="ignore").strip().replace("\x00", "")
                if key == "hardware_model":
                    data["platform"]["hardware_model"] = text
                    data["platform"]["is_cm5_like"] = "Compute Module 5" in text
                else:
                    data["os"] = dict(
                        line.split("=", 1)
                        for line in text.splitlines()
                        if "=" in line
                    )
        stdout, _, _ = run(["lscpu"], timeout=5)
        for line in stdout.splitlines():
            if ":" in line:
                k, v = [x.strip() for x in line.split(":", 1)]
                if k in ("Architecture", "CPU(s)", "Model name", "Thread(s) per core", "Core(s) per cluster",
                         "CPU max MHz", "CPU min MHz", "CPU(s) scaling MHz", "BogoMIPS"):
                    data["cpu"][k] = v
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if gov.exists():
            data["cpu"]["governor"] = gov.read_text().strip()
        cur = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if cur.exists():
            data["cpu"]["cur_khz"] = int(cur.read_text().strip() or 0)

        stdout, _, _ = run(["free", "-m"])
        for line in stdout.splitlines():
            parts = line.split()
            if parts[:1] == ["Mem:"]:
                data["memory"].update(
                    total_mb=int(parts[1]), used_mb=int(parts[2]), free_mb=int(parts[3]),
                    shared_mb=int(parts[4]), buff_cache_mb=int(parts[5]), available_mb=int(parts[6]),
                )
            elif parts[:1] == ["Swap:"]:
                data["memory"].update(swap_total_mb=int(parts[1]), swap_used_mb=int(parts[2]), swap_free_mb=int(parts[3]))
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    data["memory"]["MemTotal_kb"] = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    data["memory"]["MemAvailable_kb"] = int(line.split()[1])
        stdout, _, _ = run(["sh", "-c", "swapon --show 2>/dev/null || cat /proc/swaps"], timeout=5)
        data["memory"]["swap_devices"] = stdout

        stdout, _, _ = run(["df", "-h", "/", "/boot/firmware"], timeout=5)
        data["storage"]["df"] = stdout
        data["storage"]["root_free_gb"] = round(shutil.disk_usage("/").free / 1024 ** 3, 2)
        data["storage"]["root_total_gb"] = round(shutil.disk_usage("/").total / 1024 ** 3, 2)

        if shutil.which("vcgencmd"):
            stdout, _, _ = run(["vcgencmd", "measure_temp"])
            m = re.search(r"temp=([0-9.]+)", stdout)
            if m:
                data["thermal"]["temperature_c"] = float(m.group(1))
            stdout, _, _ = run(["vcgencmd", "get_throttled"])
            m = re.search(r"throttled=(0x[0-9a-fA-F]+)", stdout)
            if m:
                data["throttle"] = decode_throttled(m.group(1))
            stdout, _, _ = run(["vcgencmd", "measure_clock", "arm"])
            m = re.search(r"frequency\(0\)=([0-9]+)", stdout)
            if m:
                data["thermal"]["arm_hz"] = int(m.group(1))
            for name, args in (("core_voltage", ["vcgencmd", "measure_volts", "core"]),
                               ("sdram_voltage", ["vcgencmd", "measure_volts", "sdram_c"])):
                stdout, _, _ = run(args)
                if "=" in stdout:
                    data["power"][name] = stdout.split("=", 1)[1].strip()
            stdout, _, _ = run(["vcgencmd", "pmic_read_adc"], timeout=10)
            data["power"]["pmic_adc"] = stdout[:4000]
        else:
            data["warnings"].append("无 vcgencmd：不是树莓派系统或未安装 raspberrypi utils")
        temp_zone = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_zone.exists():
            data["thermal"]["soc_temp_c"] = int(temp_zone.read_text().strip() or 0) / 1000.0

        if shutil.which("rpicam-hello"):
            stdout, _, code = run(["rpicam-hello", "--list-cameras"], timeout=15)
            data["camera"]["list"] = stdout
            data["camera"]["available"] = "Available cameras" in stdout
        elif shutil.which("libcamera-hello"):
            stdout, _, code = run(["libcamera-hello", "--list-cameras"], timeout=15)
            data["camera"]["list"] = stdout
            data["camera"]["available"] = "Available cameras" in stdout
        else:
            data["camera"]["available"] = False
            data["camera"]["list"] = ""
            data["warnings"].append("未找到 rpicam-hello/libcamera-hello")

        stdout, _, _ = run(["arecord", "-l"], timeout=5)
        data["audio"]["arecord_list"] = stdout
        stdout, _, _ = run(["aplay", "-l"], timeout=5)
        data["audio"]["aplay_list"] = stdout
        data["audio"]["cards"] = parse_asound()

        stdout, _, _ = run(["lsusb"], timeout=5)
        data["usb"] = [line for line in stdout.splitlines() if line.strip()]
        stdout, _, _ = run(["lsusb", "-v", "-d", "2886:001a"], timeout=10)
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith(("bcdDevice", "iManufacturer", "iProduct", "iSerial", "bNumInterfaces")):
                data["audio"].setdefault("usb_descriptor", []).append(line)

        stdout, _, _ = run(["ip", "-br", "addr"], timeout=5)
        data["network"]["addr"] = stdout

        stdout, _, _ = run(["timedatectl"], timeout=5)
        data["time"]["timedatectl"] = stdout
        stdout, _, _ = run(["timedatectl", "timesync-status"], timeout=5)
        data["time"]["timesync"] = stdout
        data["time"]["now"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        data["time"]["monotonic_note"] = "进程内耗时使用 time.monotonic()；不跨设备相减未同步的墙钟时间"

        stdout, _, _ = run(["systemctl", "--user", "list-units", "--type=service", "--all", "--no-pager"], timeout=8)
        data["services"]["user"] = [
            line.strip() for line in stdout.splitlines() if ".service" in line and ("loaded" in line)
        ]
        stdout, _, _ = run(["systemctl", "list-units", "--type=service", "--state=running", "--no-pager"], timeout=8)
        data["services"]["system_running"] = [
            line.split()[0] for line in stdout.splitlines() if line.startswith(" ") and ".service" in line
        ]

        stdout, _, _ = run(["ss", "-tlnp"], timeout=8)
        for line in stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 4:
                data["ports"].append({"local": parts[3], "process": " ".join(parts[5:])})

        stdout, _, _ = run(["ps", "-eo", "pid,ppid,user,comm,%cpu,%mem,rss,etime,args", "--sort=-%mem"], timeout=8)
        for line in stdout.splitlines()[1:16]:
            parts = line.split(None, 8)
            if len(parts) >= 8:
                data["processes"].append({
                    "pid": int(parts[0]) if parts[0].isdigit() else None,
                    "user": parts[2], "comm": parts[3],
                    "cpu_percent": parts[4], "mem_percent": parts[5],
                    "rss_kb": parts[6], "elapsed": parts[7],
                    "args": parts[8] if len(parts) > 8 else "",
                })

    else:
        data["os"] = {"note": platform.platform()}
        data["warnings"].append("当前不是 Linux/树莓派；快照不能作为 CM5 结论")
        data["cpu"]["CPU(s)"] = str(os.cpu_count() or 0)
        vm = None
        try:
            import psutil
            vm = psutil.virtual_memory()
            data["memory"].update(total_mb=round(vm.total / 1024 ** 2), available_mb=round(vm.available / 1024 ** 2))
        except Exception:  # noqa: BLE001
            pass

    # ---- runtimes ----
    llama = next((p for p in (ROOT / "runtime/llama.cpp/llama-server",
                              ROOT / "runtime/llama.cpp/build/bin/llama-server",
                              ROOT / "runtime/llama.cpp/bin/llama-server",
                              ROOT / "runtime/llama.cpp-src/build4/bin/llama-server") if p.exists()), None)
    if llama:
        stdout, stderr, _ = run([str(llama), "--version"], timeout=10)
        data["runtimes"]["llama_server"] = {"path": str(llama), "version": (stdout or stderr).splitlines()[:3]}
    else:
        data["runtimes"]["llama_server"] = {"path": None, "note": "未找到 llama-server"}
    piper_default = (ROOT / "runtime/piper/piper/piper.exe") if os.name == "nt" else (ROOT / "runtime/piper/piper")
    piper = Path(os.environ.get("PIPER_BIN") or piper_default)
    data["runtimes"]["piper"] = {"path": str(piper) if piper.exists() else None}
    data["runtimes"]["python_packages"] = venv_versions(ROOT / "app/.venv")
    if is_linux:
        data["runtimes"]["packages"] = package_versions([
            "python3", "python3-venv", "python3-picamera2", "alsa-utils",
            "libopenblas0", "libgomp1", "libcamera-tools",
        ])
        auth, _, _ = run(["ldd", "--version"])
        data["platform"]["glibc"] = auth.splitlines()[0] if auth else None

    # ---- models ----
    vlm_model = Path(os.environ.get("VLM_MODEL") or (MODELS / "vlm/minicpm-v-4.7-1b/MiniCPM-V-4_7-1B-Q4_K_M.gguf"))
    vlm_mmproj = Path(os.environ.get("VLM_MMPROJ") or (MODELS / "vlm/minicpm-v-4.7-1b/mmproj-model-f16.gguf"))
    model_paths = [
        ("vlm", vlm_model),
        ("vlm_mmproj", vlm_mmproj),
        ("ocr_det", MODELS / "ocr/pp-ocrv6-small/PP-OCRv6_det_small.onnx"),
        ("ocr_rec", MODELS / "ocr/pp-ocrv6-small/PP-OCRv6_rec_small.onnx"),
        ("ocr_cls", MODELS / "ocr/pp-ocrv6-small/ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
        ("asr", MODELS / "asr/sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx"),
        ("piper", Path(os.environ.get("PIPER_MODEL") or (MODELS / "tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx"))),
        ("piper_json", Path(os.environ.get("PIPER_MODEL") or (MODELS / "tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx")).with_suffix(".onnx.json")),
    ]
    for name, p in model_paths:
        item = {"path": str(p), "exists": p.exists()}
        if p.exists():
            item["bytes"] = p.stat().st_size
            item["sha256"] = sha256(p) if hash_models else "（未计算，加 --hash）"
        else:
            data["warnings"].append(f"缺少模型 {p}")
        data["models"][name] = item
    kws_dir = Path(os.environ.get("KWS_MODEL_DIR") or (MODELS / "kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"))
    kws_files = sorted(kws_dir.glob("*.onnx")) if kws_dir.exists() else []
    data["models"]["kws"] = {
        "dir": str(kws_dir),
        "exists": kws_dir.exists(),
        "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha256(f) if hash_models else "（未计算）"} for f in kws_files],
    }
    if not kws_files:
        data["warnings"].append("KWS 模型缺失，唤醒会退回 ASR 文本匹配")

    if not data["platform"]["is_aarch64"]:
        data["warnings"].append("当前不是 aarch64；线程与性能配置不能代表 CM5")
    avail = data["memory"].get("available_mb") or 0
    if is_linux and avail and avail < 1200:
        data["warnings"].append(f"可用内存仅 {avail} MB，小于 1200 MB，部署前先停掉旧实例")

    return data


def to_markdown(data: dict) -> str:
    p = data["platform"]
    lines = [
        "# CM5 环境快照",
        "",
        f"- 采集时间：`{data['collected_at']}`（UTC `{data['collected_at_utc']}`）",
        f"- 主机名：`{data['host']}`",
        f"- 平台：`{p['system']}` / `{p['machine']}` / 内核 `{p['kernel']}`",
        f"- 硬件型号：`{p['hardware_model']}`",
        f"- aarch64：`{p['is_aarch64']}`；CM5 识别：`{p['is_cm5_like']}`",
        f"- Python：`{p['python']}`；glibc：`{p.get('glibc')}`",
        "",
        "## 系统与资源",
        "",
        "| 项目 | 值 |",
        "|---|---|",
    ]
    for k, v in data["cpu"].items():
        lines.append(f"| CPU {k} | `{v}` |")
    for k, v in data["memory"].items():
        if k != "swap_devices":
            lines.append(f"| 内存 {k} | `{v}` |")
    for k, v in data["storage"].items():
        if k != "df":
            lines.append(f"| 存储 {k} | `{v}` |")
    for k, v in data["thermal"].items():
        lines.append(f"| 温度/频率 {k} | `{v}` |")
    th = data.get("throttle") or {}
    lines.append(f"| 欠压/降频原始 | `{th.get('raw')}` |")
    lines.append(f"| 当前故障 | `{th.get('current_flags')}` |")
    lines.append(f"| 历史记录 | `{th.get('flags')}` |")
    lines.append(f"| 当前是否故障 | `{th.get('current_fault')}` |")

    lines += ["", "## 摄像头", "", "```text", data["camera"].get("list") or "(无)", "```", ""]
    lines += ["## 音频设备", "", "### arecord -l", "```text", data["audio"].get("arecord_list") or "(无)", "```",
              "", "### aplay -l", "```text", data["audio"].get("aplay_list") or "(无)", "```"]
    if data["audio"].get("usb_descriptor"):
        lines += ["", "### USB 描述符", "```text", *data["audio"]["usb_descriptor"], "```"]
    lines += ["", "## 网络", "", "```text", data["network"].get("addr") or "(Windows)", "```", ""]

    lines += ["## 时间", "", "```text", data["time"].get("timedatectl") or "(无 timedatectl)", "", data["time"].get("timesync") or "", "```",
              "", f"- 说明：{data['time'].get('monotonic_note')}", ""]

    lines += ["## 服务与端口", "", "| 监听 | 进程 |", "|---|---|"]
    for item in data["ports"]:
        lines.append(f"| `{item['local']}` | `{item['process']}` |")
    lines += ["", "### 用户级服务", "```text", *([s for s in data["services"].get("user", [])] or ["(无)"]), "```", ""]

    lines += ["## 进程（内存前 15）", "", "| PID | 用户 | 命令 | CPU% | MEM% | RSS(KB) | 运行时长 |", "|---|---|---|---|---|---|---|"]
    for proc in data["processes"]:
        lines.append(f"| {proc['pid']} | {proc['user']} | `{proc['comm']}` | {proc['cpu_percent']} | {proc['mem_percent']} | {proc['rss_kb']} | {proc['elapsed']} |")

    lines += ["", "## 运行时与模型", ""]
    rt = data["runtimes"]
    lines.append(f"- llama-server：`{rt.get('llama_server')}`")
    lines.append(f"- piper：`{rt.get('piper')}`")
    if rt.get("packages"):
        lines += ["", "| 系统包 | 版本 |", "|---|---|"]
        for k, v in rt["packages"].items():
            lines.append(f"| {k} | `{v}` |")
    if rt.get("python_packages"):
        keep = ("flask", "numpy", "onnxruntime", "sherpa-onnx", "rapidocr", "pillow", "psutil", "requests", "waitress")
        lines += ["", "| Python 包 | 版本 |", "|---|---|"]
        for k in keep:
            if k in rt["python_packages"]:
                lines.append(f"| {k} | `{rt['python_packages'][k]}` |")

    lines += ["", "| 模型 | 存在 | 字节 | SHA256 |", "|---|---|---|---|"]
    for name, item in data["models"].items():
        if name == "kws":
            for f in item.get("files", []):
                lines.append(f"| kws/{f['name']} | {item['exists']} | {f['bytes']} | `{f['sha256']}` |")
        else:
            lines.append(f"| {name} | {item['exists']} | {item.get('bytes', '')} | `{item.get('sha256', '')}` |")

    lines += ["", "## 警告与待确认", ""]
    if data["warnings"]:
        lines += [f"- {w}" for w in data["warnings"]]
    else:
        lines.append("- 无")
    lines += ["", "> 本快照只读采集，不包含任何密钥、令牌或完整敏感环境变量。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--hash", action="store_true", help="对模型文件计算 SHA256（1.7GB 约需数秒）")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S%z") + "_snapshot"
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "test-runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.hash)
    data["run_id"] = run_id
    (out_dir / "environment.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "environment.md").write_text(to_markdown(data), encoding="utf-8")

    print(json.dumps({
        "run_id": run_id,
        "out_dir": str(out_dir),
        "is_cm5_like": data["platform"]["is_cm5_like"],
        "warnings": data["warnings"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
