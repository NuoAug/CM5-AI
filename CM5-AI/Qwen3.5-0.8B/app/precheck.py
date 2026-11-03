"""Candidate precheck: independent assets and recorded real validation; no fallback."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from candidate_paths import ROOT, resolve_local, validate_environment, validate_profile

ASR = "models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09/"
STREAMING = "models/asr-streaming/sherpa-onnx-streaming-paraformer-bilingual-zh-en/"
OCR = "models/ocr/pp-ocrv6-small/"
TTS = "models/tts/piper-zh_CN-huayan-medium/"
KWS = "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/"
INTERACTION_ASSETS = {
    "asr_model": ASR + "model.int8.onnx", "asr_tokens": ASR + "tokens.txt",
    "streaming_encoder": STREAMING + "encoder.int8.onnx", "streaming_decoder": STREAMING + "decoder.int8.onnx",
    "streaming_tokens": STREAMING + "tokens.txt",
    "ocr_det": OCR + "PP-OCRv6_det_small.onnx", "ocr_rec": OCR + "PP-OCRv6_rec_small.onnx",
    "ocr_cls": OCR + "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "tts_model": TTS + "zh_CN-huayan-medium.onnx", "tts_config": TTS + "zh_CN-huayan-medium.onnx.json",
    "kws_tokens": KWS + "tokens.txt", "kws_keywords": "app/config/keywords.txt",
}
REQUIRED_RUNTIME_CHECKS = ("offline_asr_decoded", "streaming_asr_decoded", "ocr_image_read",
                           "tts_nonempty_pcm", "kws_stream_decoded")


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect_checks(profile, *, vlm_only=False, root=ROOT):
    checks = {}
    assets = {}
    for key in ("model_file", "mmproj_file", "runner_file" if os.name == "nt" else "runner_file_posix"):
        assets[key] = resolve_local(profile[key], root=root)
        checks[key] = nonempty(assets[key])
    if not vlm_only:
        assets.update({name: resolve_local(value, root=root) for name, value in INTERACTION_ASSETS.items()})
        assets["piper_binary"] = resolve_local("runtime/piper/piper/" + ("piper.exe" if os.name == "nt" else "piper"), root=root)
        for part in ("encoder", "decoder", "joiner"):
            found = sorted(resolve_local(KWS, root=root).glob(part + "-*.onnx"))
            assets["kws_" + part] = found[0] if found else resolve_local(KWS + part + "-missing.onnx", root=root)
        for name, path in assets.items():
            checks[name] = nonempty(path)
        checks["profile_interaction_installed"] = profile.get("complete_interaction_assets_installed") is True
        receipt = {}
        try:
            receipt = json.loads(resolve_local("models/interaction-ready.json", root=root).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        version = (root / "VERSION").read_text(encoding="utf-8").strip()
        checks["interaction_real_validation_receipt"] = (
            receipt.get("profile_id") == profile.get("profile_id") and receipt.get("version") == version
            and receipt.get("readiness_type") == "real_runtime_validation"
            and all(receipt.get("runtime_checks", {}).get(name) is True for name in REQUIRED_RUNTIME_CHECKS))
        recorded = receipt.get("asset_sha256", {})
        checks["receipt_covers_current_assets"] = all(
            nonempty(path) and recorded.get(path.relative_to(root).as_posix()) == sha256(path)
            for path in assets.values())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-busy", action="store_true", help="Compatibility only; never takes over processes")
    args = parser.parse_args()
    try:
        validate_environment()
        profile = validate_profile()
        checks = collect_checks(profile, vlm_only=args.vlm_only)
        modules = ("requests", "PIL") if args.vlm_only else ("flask", "numpy", "psutil", "requests", "PIL", "rapidocr", "sherpa_onnx", "onnxruntime")
        for module in modules:
            checks["dependency_" + module] = importlib.util.find_spec(module) is not None
        data = {"profile_id": profile["profile_id"], "stage": "candidate", "ready": all(checks.values()),
                "scope": "vlm_only" if args.vlm_only else "interaction", "checks": checks,
                "physical_audio_validated": False, "target_board_validated": False}
    except (ValueError, OSError, KeyError) as exc:
        data = {"stage": "candidate", "ready": False, "error": str(exc), "checks": {"path_boundary": False}}
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        for name, ok in data["checks"].items(): print(("PASS " if ok else "UNAVAILABLE ") + name)
        print("candidate ready=" + str(data["ready"]) + "; not quality, physical audio or target-board acceptance")
    return 0 if data["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
