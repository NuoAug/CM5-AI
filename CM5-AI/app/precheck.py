"""启动前资源预检：只读检查文件、依赖、端口，不启动服务，不开相机麦克风。"""

import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA_DIR = Path(os.environ.get("DATA_DIR") or (ROOT / "data"))

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))


def check_deps():
    missing = []
    for mod in ("flask", "numpy", "psutil", "requests", "PIL", "rapidocr", "sherpa_onnx", "onnxruntime"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    record("Python 依赖", not missing, "缺少 " + ",".join(missing) if missing else "")


def check_models():
    piper_model = Path(
        os.environ.get("PIPER_MODEL")
        or (MODELS / "tts" / "piper-zh_CN-huayan-medium" / "zh_CN-huayan-medium.onnx")
    )
    kws_dir = Path(
        os.environ.get("KWS_MODEL_DIR")
        or (MODELS / "kws" / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
    )
    required = [
        MODELS / "vlm" / "minicpm-v-4.7-1b" / "MiniCPM-V-4_7-1B-Q4_K_M.gguf",
        MODELS / "vlm" / "minicpm-v-4.7-1b" / "mmproj-model-f16.gguf",
        MODELS / "ocr" / "pp-ocrv6-small" / "PP-OCRv6_det_small.onnx",
        MODELS / "ocr" / "pp-ocrv6-small" / "PP-OCRv6_rec_small.onnx",
        MODELS / "ocr" / "pp-ocrv6-small" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        piper_model,
        piper_model.with_suffix(".onnx.json"),
        MODELS / "asr" / "sherpa-onnx-paraformer-zh-small-2024-03-09" / "model.int8.onnx",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    record("模型文件", not missing, "缺少 " + "，".join(missing) if missing else "")
    kws_required = [
        kws_dir / "tokens.txt",
        ROOT / "app" / "config" / "keywords.txt",
    ]
    glob_required = [
        (kws_dir, "encoder-*.onnx"),
        (kws_dir, "decoder-*.onnx"),
        (kws_dir, "joiner-*.onnx"),
    ]
    kws_missing = [str(p.relative_to(ROOT)) for p in kws_required if not p.exists()]
    for directory, pattern in glob_required:
        if not list(directory.glob(pattern)):
            kws_missing.append(str(directory.relative_to(ROOT)) + "/" + pattern)
    if kws_missing:
        print("WARN KWS 模型未就绪，语音唤醒退回 ASR 文本匹配： " + "，".join(kws_missing))
    else:
        print("PASS KWS 唤醒模型")


def check_llama():
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    env_path = os.environ.get("LLAMA_SERVER")
    candidates = ([Path(env_path)] if env_path else []) + [
        ROOT / "runtime" / "llama.cpp" / name,
        ROOT / "runtime" / "llama.cpp" / "build" / "bin" / name,
        ROOT / "runtime" / "llama.cpp" / "bin" / name,
        ROOT / "runtime" / "llama.cpp-src" / "build4" / "bin" / name,
    ]
    found = next((p for p in candidates if p.exists()), None)
    detail = "未找到"
    if found is not None:
        try:
            detail = str(found.relative_to(ROOT))
        except ValueError:
            detail = str(found)
    record("llama-server 可执行", found is not None, detail)


def check_piper():
    name = "piper.exe" if os.name == "nt" else "piper"
    exe = Path(os.environ.get("PIPER_BIN") or (ROOT / "runtime" / "piper" / "piper" / name))
    detail = "未找到"
    if exe.exists():
        try:
            detail = str(exe.relative_to(ROOT))
        except ValueError:
            detail = str(exe)
    record("piper 可执行", exe.exists(), detail)


def check_main_control():
    url = os.environ.get("MAIN_CONTROL_URL", "").strip()
    source = os.environ.get("IMAGE_SOURCE", "main_control").strip().lower()
    if source not in ("main_control", "standalone-device-test", "browser"):
        record("取图来源配置", False, f"IMAGE_SOURCE={source}")
        return
    if source == "main_control" and not url:
        print("WARN 主控未配置（MAIN_CONTROL_URL 为空）：行走查询与正式取图不可用，纯聊天仍可用")
    else:
        print(f"PASS 取图来源 {source}  MAIN_CONTROL_URL={url or '(未配置)'}")


def check_ports():
    if os.environ.get("PRECHECK_ALLOW_BUSY", "0").lower() in ("1", "true", "yes") or "--allow-busy" in sys.argv:
        print("SKIP 端口检查（systemd 启动时模型服务可能已占用 8091）")
        return
    busy = []
    for port in (8090, 8091):
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(str(port))
    record("端口 8090/8091 空闲", not busy, "已被占用 " + ",".join(busy) if busy else "")


def check_data_dir():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        probe = DATA_DIR / ".precheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        record("数据目录可写", True, str(DATA_DIR))
    except Exception as exc:
        record("数据目录可写", False, str(exc))


def check_python():
    ok = sys.version_info >= (3, 10)
    record("Python 版本 >= 3.10", ok, sys.version.split()[0])


def check_devices():
    import importlib.util
    import shutil
    picamera = importlib.util.find_spec("picamera2") is not None
    print(("PASS" if picamera else "WARN") + " Picamera2  " + ("可导入" if picamera else "未安装；真机 CSI 采集不可用"))
    arecord = shutil.which("arecord")
    aplay = shutil.which("aplay")
    print(("PASS" if arecord else "WARN") + " arecord  " + (arecord or "未找到；设备端语音采集不可用"))
    print(("PASS" if aplay else "WARN") + " aplay  " + (aplay or "未找到；设备端播报不可用"))


def check_env():
    keys = (
        "VLM_THREADS", "VLM_CTX", "KWS_THREADS", "KWS_SCORE", "KWS_THRESHOLD", "KWS_MODEL_DIR",
        "ASR_THREADS", "OCR_INTRA_THREADS", "MEM_BUDGET_GB", "VOICE_SOURCE", "OUTPUT_SINK",
        "IMAGE_SOURCE", "MAIN_CONTROL_URL", "MAIN_CONTROL_ALLOW_FIXTURE",
    )
    print("环境变量：")
    for key in keys:
        print("  %-24s %s" % (key, os.environ.get(key, "(默认)")))


def main():
    print("资源预检（只读，不启动服务）\n")
    check_python()
    check_deps()
    check_models()
    check_llama()
    check_piper()
    check_main_control()
    check_devices()
    check_env()
    check_ports()
    check_data_dir()
    failed = [name for name, ok in RESULTS if not ok]
    print("\n合计 %d 项，失败 %d 项" % (len(RESULTS), len(failed)))
    if failed:
        print("未通过：" + "、".join(failed))
        print("提示：模型缺失时先按 README 的模型清单补齐；端口占用说明已有实例在运行。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
