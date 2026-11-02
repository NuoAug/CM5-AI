# -*- coding: utf-8 -*-
"""生成本机唯一发布清单：来源、版本、用途、许可状态、字节数与 SHA256。

只扫描本文件列出的路径，不遍历整个仓库；输出 docs/模型与工具发布清单.md。
标记 release=False 的条目是本地对照/历史资产，不进入目标板部署。
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "docs" / "模型与工具发布清单.md"

ENTRIES = [
    # ---- 正式发布：VLM ----
    {
        "path": "models/vlm/minicpm-v-4.7-1b/MiniCPM-V-4_7-1B-Q4_K_M.gguf",
        "role": "VLM 语言权重（基线）",
        "source": "https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf",
        "license": "Apache-2.0（模型卡）",
        "license_status": "已核实",
        "release": True,
        "note": "CM5 实测视觉编码约 118s/帧，只作质量/兼容基线，不作实时默认",
    },
    {
        "path": "models/vlm/minicpm-v-4.7-1b/mmproj-model-f16.gguf",
        "role": "VLM 视觉投影（基线）",
        "source": "https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf",
        "license": "Apache-2.0（模型卡）",
        "license_status": "已核实",
        "release": True,
        "note": "与语言权重同仓；必须配套加载",
    },
    # ---- 正式发布：ASR / KWS / OCR / TTS ----
    {
        "path": "models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09/*",
        "role": "离线整句 ASR",
        "source": "ModelScope crazyant/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-onnx（Paraformer 上游 FunASR）",
        "license": "Apache-2.0（FunASR/Paraformer 上游）",
        "license_status": "上游已核实；转换制品未单独声明，需复核",
        "release": True,
        "note": "INT8；按需加载，闲置释放；不是原生流式",
    },
    {
        "path": "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/*.onnx",
        "role": "唤醒词 KWS",
        "source": "https://huggingface.co/csukuangfj/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        "license": "Apache-2.0",
        "license_status": "已核实（仓库 README）",
        "release": True,
        "note": "中文拼音关键词；关键词表见 app/config/keywords.txt",
    },
    {
        "path": "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
        "role": "KWS 词表",
        "source": "同上",
        "license": "Apache-2.0",
        "license_status": "已核实",
        "release": True,
        "note": "",
    },
    {
        "path": "models/ocr/pp-ocrv6-small/*.onnx",
        "role": "中文 OCR（检测/识别/方向）",
        "source": "PaddleOCR PP-OCRv6 small（RapidOCR 调用）",
        "license": "Apache-2.0（PaddleOCR）",
        "license_status": "上游已核实；实际转换制品需按下载页复核",
        "release": True,
        "note": "det/rec/cls 三个 ONNX；不把 OCR 任务交给 VLM 自由生成",
    },
    {
        "path": "models/tts/piper-zh_CN-huayan-medium/*",
        "role": "中文 TTS（Piper VITS）",
        "source": "https://huggingface.co/rhasspy/piper-voices/tree/main/zh/zh_CN/huayan/medium",
        "license": "音色数据许可 Unknown；Piper 代码 MIT",
        "license_status": "未闭合，禁止对外分发",
        "release": True,
        "note": "研发与内部测试可用；商用/再分发前必须替换或取得授权",
    },
    # ---- 正式发布：运行器 ----
    {
        "path": "runtime/llama.cpp/llama-server.exe",
        "role": "Windows 开发 VLM 运行器",
        "source": "llama.cpp 0.7.0-dev build 1 commit 8ca4060（本机构建）",
        "license": "MIT",
        "license_status": "已核实",
        "release": False,
        "note": "仅 Windows 开发；ARM64 运行器需在 CM5 或交叉构建，不能直接使用 EXE",
    },
    {
        "path": "runtime/llama.cpp/llama-mtmd-cli.exe",
        "role": "Windows 多模态调试工具",
        "source": "同上",
        "license": "MIT",
        "license_status": "已核实",
        "release": False,
        "note": "只用于本机对照，不部署到 CM5",
    },
    {
        "path": "runtime/piper/piper/piper.exe",
        "role": "Windows Piper 运行器",
        "source": "Piper 1.2.0（本机解压制品）",
        "license": "MIT（运行器）；音色另计",
        "license_status": "运行器已核实；音色未闭合",
        "release": False,
        "note": "CM5 需要 ARM64 piper 可执行文件",
    },
    # ---- 本地对照/历史，不发布 ----
    {
        "path": "models/vlm/smolvlm-256m/*.gguf",
        "role": "更小 VLM 对照候选",
        "source": "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
        "license": "Apache-2.0",
        "license_status": "已核实（模型卡）",
        "release": False,
        "note": "仅 M3 对照；未验证中文自然度与视觉事实，不替换基线",
    },
    {
        "path": "models/tts-candidates/kokoro-int8-multi-lang-v1_1/*",
        "role": "TTS 对照候选（已淘汰）",
        "source": "sherpa-onnx Kokoro int8 制品",
        "license": "需按仓库与音色分别复核",
        "license_status": "未核实",
        "release": False,
        "note": "实测合成过慢，已从默认路径移除；不得进入发布",
    },
    {
        "path": "models/tts-candidates/matcha-icefall-zh-baker/*",
        "role": "TTS 对照候选",
        "source": "sherpa-onnx Matcha icefall zh baker 制品",
        "license": "需按仓库复核",
        "license_status": "未核实",
        "release": False,
        "note": "仅作对照；未进入默认路径",
    },
    {
        "path": "data/archive/segformer-b0-ade-20261102/model.onnx",
        "role": "历史独立引导分割模型（越界）",
        "source": "NVIDIA SegFormer-B0 ADE20K ONNX",
        "license": "NVIDIA 非商业研究许可",
        "license_status": "不满足本项目分发条件",
        "release": False,
        "note": "M0 从当前交付删除；历史版本由 archive/pre-interaction-v11-20261102 保留",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expand(pattern: str) -> list[Path]:
    base = ROOT
    if any(ch in pattern for ch in "*?["):
        parent = ROOT / Path(pattern).parent
        return sorted(p for p in parent.glob(Path(pattern).name) if p.is_file())
    path = ROOT / pattern
    return [path] if path.is_file() else []


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GiB"


def main() -> int:
    rows, missing = [], []
    for entry in ENTRIES:
        files = expand(entry["path"])
        if not files:
            missing.append(entry["path"])
            continue
        for path in files:
            rows.append((entry, path, path.stat().st_size, sha256(path)))

    release_bytes = sum(size for e, _, size, _ in rows if e["release"])
    local_bytes = sum(size for e, _, size, _ in rows if not e["release"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = [
        "# 模型与工具发布清单",
        "",
        f"生成时间：{now}（UTC）  ",
        "生成脚本：`CM5-AI/tools/make_release_manifest.py`  ",
        "主机：Windows 开发机；目标板：Raspberry Pi CM5 4GB / 32GB eMMC / aarch64  ",
        "口径：字节数与 SHA256 来自当前本机实际文件；许可状态区分“已核实 / 上游已核实 / 未闭合 / 未核实”。",
        "",
        "## 发布规则",
        "",
        "- 只有本清单 `发布=是` 的条目允许进入 CM5 运行目录。",
        "- 树莓派上自行下载或残留的模型不得作为隐式备用来沿用。",
        "- Windows EXE/DLL 与 `.venv` 不能作为 ARM64 运行环境；ARM64 运行器单独构建并记录。",
        "- 许可未闭合的音色、OCR 转换制品和 TTS 候选不得对外分发。",
        "",
        "## 正式发布条目",
        "",
        "| 文件 | 用途 | 字节数 | SHA256 | 来源 | 许可 | 许可状态 | 备注 |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for entry, path, size, digest in rows:
        if not entry["release"]:
            continue
        rel = path.relative_to(ROOT).as_posix()
        lines.append(
            f"| `{rel}` | {entry['role']} | {size} ({fmt_bytes(size)}) | `{digest}` | "
            f"{entry['source']} | {entry['license']} | {entry['license_status']} | {entry['note']} |"
        )

    lines += [
        "",
        f"发布条目合计：{fmt_bytes(release_bytes)}。",
        "",
        "## 本地对照/历史条目（不发布）",
        "",
        "| 文件 | 用途 | 字节数 | SHA256 | 许可状态 | 处理 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for entry, path, size, digest in rows:
        if entry["release"]:
            continue
        rel = path.relative_to(ROOT).as_posix()
        lines.append(
            f"| `{rel}` | {entry['role']} | {size} ({fmt_bytes(size)}) | `{digest}` | "
            f"{entry['license_status']} | {entry['note']} |"
        )

    lines += [
        "",
        f"本地非发布条目合计：{fmt_bytes(local_bytes)}。",
        "",
        "## 依赖与系统组件",
        "",
        "| 组件 | 本机版本 | 目标板要求 | 说明 |",
        "| --- | --- | --- | --- |",
        "| Python | 3.12.10（runtime/python） | 系统 Python 或同版本 ARM64 虚拟环境 | Windows 运行时不可拷贝到 CM5 |",
        "| llama.cpp | 0.7.0-dev build 1 commit 8ca4060（Windows） | ARM64 源码构建，版本与参数单独记录 | GGUF 权重可复用，运行器不可复用 |",
        "| Piper | 1.2.0（Windows） | ARM64 piper 可执行文件 | 音色许可未闭合 |",
        "| Flask / requests / Pillow / numpy / psutil | 见 app/requirements.txt | 同左 | 纯 Python 依赖 |",
        "| rapidocr / onnxruntime / sherpa-onnx | 见 deploy/requirements-cm5.txt | ARM64 wheel 或源码构建 | 需在 CM5 上实际安装验证 |",
        "",
        "## 未闭合项",
        "",
        "- Piper `zh_CN-huayan-medium` 训练数据许可为 Unknown：研发可用，最终分发前必须替换为许可明确的中文 TTS。",
        "- Paraformer 转换制品与 PP-OCRv6 实际 ONNX 转换页未逐一保存许可原文，需在发布前复核。",
        "- 本清单不包含主控、体型侧资产；CM5 上的主控/体型文件在清理时单独保护。",
        "",
    ]
    if missing:
        lines += ["## 缺失条目", ""] + [f"- `{m}`" for m in missing] + [""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT} rows={len(rows)} release={fmt_bytes(release_bytes)} missing={len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
