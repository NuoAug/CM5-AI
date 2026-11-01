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
    for mod in ("flask", "numpy", "psutil", "requests", "PIL", "rapidocr", "sherpa_onnx"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    record("Python 依赖", not missing, "缺少 " + ",".join(missing) if missing else "")


def check_models():
    required = [
        MODELS / "vlm" / "minicpm-v-4.7-1b" / "MiniCPM-V-4_7-1B-Q4_K_M.gguf",
        MODELS / "vlm" / "minicpm-v-4.7-1b" / "mmproj-model-f16.gguf",
        MODELS / "ocr" / "pp-ocrv6-small" / "PP-OCRv6_det_small.onnx",
        MODELS / "ocr" / "pp-ocrv6-small" / "PP-OCRv6_rec_small.onnx",
        MODELS / "ocr" / "pp-ocrv6-small" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        MODELS / "tts" / "vits-melo-tts-zh_en" / "model.onnx",
        MODELS / "asr" / "sherpa-onnx-paraformer-zh-small-2024-03-09" / "model.int8.onnx",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    record("模型文件", not missing, "缺少 " + "，".join(missing) if missing else "")


def check_llama():
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    candidates = [
        ROOT / "runtime" / "llama.cpp" / name,
        ROOT / "runtime" / "llama.cpp-src" / "build4" / "bin" / name,
    ]
    found = next((p for p in candidates if p.exists()), None)
    record("llama-server 可执行", found is not None, str(found.relative_to(ROOT)) if found else "未找到")


def check_ports():
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


def main():
    print("资源预检（只读，不启动服务）\n")
    check_python()
    check_deps()
    check_models()
    check_llama()
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
