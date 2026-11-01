# -*- coding: utf-8 -*-
"""
CM5 边缘 AI 本地测试服务
- VLM：调用本机 llama-server（MiniCPM-V 4.7 1B，OpenAI 兼容接口）
- OCR：RapidOCR + PP-OCRv6_small（ONNX Runtime）
- TTS：sherpa-onnx；默认 MeloTTS 中文（VITS fp32），可选 Kokoro-82M-v1.1-zh（int8）
"""

import base64
import io
import json
import os
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import requests
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OCR_DIR = MODELS / "ocr" / "pp-ocrv6-small"
TTS_DIR = MODELS / "tts" / "kokoro-int8-multi-lang-v1_1"
TTS_MELO_DIR = MODELS / "tts" / "vits-melo-tts-zh_en"
ASR_DIR = MODELS / "asr" / "sherpa-onnx-paraformer-zh-small-2024-03-09"
VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8091")
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "samples"

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")

OCR_LOCK = threading.Lock()
TTS_LOCK = threading.Lock()
VLM_LOCK = threading.Lock()
ASR_LOCK = threading.Lock()

ocr_engine = None
tts_engine = None
kokoro_engine = None
asr_engine = None
init_error = {"ocr": None, "tts": None, "kokoro": None, "asr": None}

MEM_BUDGET_GB = float(os.environ.get("MEM_BUDGET_GB", "3.6"))


def _used_bytes(mem_info):
    """Windows 用提交内存更稳定；工作集会被系统修剪，读数会虚低。"""
    return getattr(mem_info, "private", mem_info.rss)


def model_memory_bytes():
    """只统计模型侧占用：llama-server 常驻 + 本服务进程。"""
    app = _used_bytes(psutil.Process(os.getpid()).memory_info())
    vlm = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name.startswith("llama-server"):
                vlm += _used_bytes(proc.info["memory_info"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return app, vlm


def over_budget():
    app, vlm = model_memory_bytes()
    return (app + vlm) > MEM_BUDGET_GB * 1024 ** 3


EYES_RULE = (
    "你是视障人士的AI眼睛。用1到2句口语中文回答，不分点、不加括号。"
    "方位只说左/中/右，远近只说近/远。"
    "只说画面里看得见的；看不清、被挡住、没看到就直说，不猜、不补常识。"
    "不报米数、不报秒数，不说能不能走、能不能过马路。"
)

EYES_TASKS = {
    "scene": {
        "prompt": (
            "看图，说清正前方从近到远有什么：有人、车、台阶、路沿、坑或障碍物就先说它们；"
            "然后说最主要的三样东西，每样带左/中/右和近/远。"
            "没有明显障碍就先说“前方没看到明显障碍”。不要重复方位词，不要罗列全部物体。"
            "例如：前方没看到明显障碍，近处中间是一张床，左右各有一只猫。"
        ),
        "max_tokens": 72,
        "fallback": "看不清",
    },
    "toilet": {
        "prompt": (
            "只找卫生间、洗手间、厕所、WC 或 Toilet 的标识、门牌或门，"
            "看到了就说在左/中/右、近还是远。"
            "没看到就只说“画面里没看到卫生间标识”。"
            "不要描述画面里别的物体，不要按建筑布局推断。"
            "例如：卫生间标识在右边，很近。"
        ),
        "max_tokens": 32,
        "fallback": "画面里没看到卫生间标识",
    },
    "exit": {
        "prompt": (
            "只找出口、安全出口、EXIT 的标识、箭头或门，"
            "看到了就说在左/中/右、近还是远，箭头朝哪边就说朝哪边。"
            "没看到就只说“画面里没看到出口标识”。"
            "不要描述画面里别的物体。"
            "例如：出口在左边，比较远，箭头朝左。"
        ),
        "max_tokens": 32,
        "fallback": "画面里没看到出口标识",
    },
    "item": {
        "prompt": (
            "画面里有“{item}”吗？有就说在左/中/右、近还是远、有没有被挡住。"
            "没看到就只说“画面里没看到{item}”。"
            "不要描述画面里别的物体。"
            "例如：{item}在右前方，没有被挡住。"
        ),
        "max_tokens": 32,
        "fallback": "画面里没看到{item}",
    },
    "text": {
        "prompt": (
            "只读出画面里最大、最清楚的那块文字，按原文顺序念，不改写、不解释、不翻译；"
            "看不清就说“看不清”。"
        ),
        "max_tokens": 64,
        "fallback": "看不清",
    },
    "followup": {
        "prompt": (
            "只根据下面这段刚才的描述回答，不要补充描述里没有的东西。\n"
            "描述：{context}\n问题：{question}\n"
            "用一句口语回答，分不清就说“刚才没看清”。"
        ),
        "max_tokens": 40,
        "fallback": "刚才没看清",
    },
}

EYES_NEGATIVE = ("没看到", "没有", "看不到", "没发现")
EYES_TARGET = {
    "toilet": ("卫生间", "洗手间", "厕所", "WC", "Toilet"),
    "exit": ("出口", "EXIT"),
}


def settle_eyes_answer(task, answer, fallback, item=""):
    """二值查找任务：模型说没找到时，一律只念兜底句。"""
    if not answer:
        return fallback
    targets = (item,) if task == "item" else EYES_TARGET.get(task, ())
    low = answer.lower()
    hit = any(t and t.lower() in low for t in targets)
    if hit and any(word in answer for word in EYES_NEGATIVE):
        return fallback
    return answer


def build_eyes_prompt(task, item="", context="", question=""):
    spec = EYES_TASKS[task]
    return EYES_RULE + "\n\n" + spec["prompt"].format(
        item=item, context=context, question=question
    )


PROMPTS = {
    "scene": (
        "请用中文描述这张图片：\n"
        "1) 画面里有哪些主要物体（最多 8 个，每个只列一次，不要重复）；\n"
        "2) 这是什么场景；\n"
        "3) 有没有明显的危险或异常。\n"
        "只描述看得见的内容，不要猜测。"
    ),
    "spatial": (
        "请只输出 JSON，不要输出其他文字。所有 name 和 relations 必须用中文。格式：\n"
        '{"objects":[{"name":"物体","position":"left/center/right/top/bottom",'
        '"occluded":true或false}],"relations":["物体A在物体B左边", "..."]}\n'
        "看不清或不确定的字段填 unknown，不要猜。"
    ),
    "ocr_help": "请读取图片中的中文文字，按行输出；看不清的字用 □ 代替。",
}


def init_ocr():
    global ocr_engine
    try:
        ocr_engine = RapidOCR(
            params={
                "Global.model_root_dir": str(OCR_DIR),
                "Det.model_path": str(OCR_DIR / "PP-OCRv6_det_small.onnx"),
                "Cls.model_path": str(OCR_DIR / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
                "Rec.model_path": str(OCR_DIR / "PP-OCRv6_rec_small.onnx"),
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": ModelType.SMALL,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.model_type": ModelType.SMALL,
            }
        )
        print("[init] OCR ready")
    except Exception as exc:  # noqa: BLE001
        init_error["ocr"] = str(exc)
        print("[init] OCR failed:", exc)


def init_tts():
    global tts_engine
    try:
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(TTS_MELO_DIR / "model.onnx"),
                    lexicon=str(TTS_MELO_DIR / "lexicon.txt"),
                    tokens=str(TTS_MELO_DIR / "tokens.txt"),
                    dict_dir=str(TTS_MELO_DIR / "dict"),
                ),
                num_threads=4,
            ),
            rule_fsts=",".join(
                str(TTS_MELO_DIR / name)
                for name in ("phone.fst", "date.fst", "number.fst", "new_heteronym.fst")
            ),
            max_num_sentences=1,
        )
        tts_engine = sherpa_onnx.OfflineTts(cfg)
        print("[init] MeloTTS ready, sample_rate=", tts_engine.sample_rate)
    except Exception as exc:  # noqa: BLE001
        init_error["tts"] = str(exc)
        print("[init] MeloTTS failed:", exc)


def load_kokoro():
    global kokoro_engine
    if kokoro_engine is not None:
        return kokoro_engine
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(TTS_DIR / "model.int8.onnx"),
                voices=str(TTS_DIR / "voices.bin"),
                tokens=str(TTS_DIR / "tokens.txt"),
                data_dir=str(TTS_DIR / "espeak-ng-data"),
                lexicon=str(TTS_DIR / "lexicon-us-en.txt") + "," + str(TTS_DIR / "lexicon-zh.txt"),
            ),
            num_threads=4,
        ),
        rule_fsts=",".join(
            str(TTS_DIR / name) for name in ("number-zh.fst", "phone-zh.fst", "date-zh.fst")
        ),
        max_num_sentences=1,
    )
    kokoro_engine = sherpa_onnx.OfflineTts(cfg)
    print("[init] Kokoro ready")
    return kokoro_engine


def init_asr():
    global asr_engine
    try:
        asr_engine = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(ASR_DIR / "model.int8.onnx"),
            tokens=str(ASR_DIR / "tokens.txt"),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        print("[init] ASR ready")
    except Exception as exc:  # noqa: BLE001
        init_error["asr"] = str(exc)
        print("[init] ASR failed:", exc)


def decode_image(data: str) -> Image.Image:
    if not data:
        raise ValueError("缺少图片数据")
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def image_to_jpeg_b64(img: Image.Image, max_side: int = 800) -> str:
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def vlm_health() -> bool:
    try:
        r = requests.get(VLM_URL + "/health", timeout=2)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def call_vlm(img_b64, prompt, max_tokens=256, temperature=0.3):
    content = []
    if img_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}}
        )
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": "minicpm-v",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(VLM_URL + "/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    answer = (msg.get("content") or "").strip()
    for tag in ("<think>", "</think>"):
        answer = answer.replace(tag, "")
    return {"answer": answer.strip(), "latency_ms": int((time.time() - t0) * 1000)}


def parse_ocr_result(res):
    lines = []
    for i, text in enumerate(res.txts or []):
        score = float(res.scores[i]) if res.scores is not None and i < len(res.scores) else None
        box = res.boxes[i].tolist() if res.boxes is not None and i < len(res.boxes) else None
        lines.append({"text": text, "score": score, "box": box})
    return lines


def pick_main_text_block(lines):
    """只保留面积最大的那块文字：水平重叠、垂直相距不到两个行高的行算同一块。"""
    rects = []
    for line in lines:
        if line.get("box"):
            xs = [p[0] for p in line["box"]]
            ys = [p[1] for p in line["box"]]
            rects.append((line, (min(xs), min(ys), max(xs), max(ys))))
    if not rects:
        return "\n".join(line["text"] for line in lines)
    _, (ax0, ay0, ax1, ay1) = max(
        rects, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
    )
    line_h = max(1, ay1 - ay0)
    keep = []
    for line, (x0, y0, x1, y1) in rects:
        if min(ax1, x1) - max(ax0, x0) <= 0:
            continue
        gap = max(0, max(ay0, y0) - min(ay1, y1))
        if gap <= line_h * 2:
            keep.append((y0, x0, line["text"]))
    keep.sort()
    return "\n".join(text for _, _, text in keep)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    app_rss, vlm_rss = model_memory_bytes()
    total = app_rss + vlm_rss
    return jsonify(
        {
            "vlm": {"up": vlm_health(), "url": VLM_URL},
            "ocr": {"ready": ocr_engine is not None, "error": init_error["ocr"]},
            "asr": {"ready": asr_engine is not None, "error": init_error["asr"]},
            "tts": {
                "ready": tts_engine is not None,
                "engine": "melo",
                "kokoro_ready": kokoro_engine is not None,
                "error": init_error["tts"],
                "kokoro_error": init_error["kokoro"],
            },
            "memory": {
                "model_gb": round(vlm_rss / 1024 ** 3, 2),
                "app_gb": round(app_rss / 1024 ** 3, 2),
                "sum_gb": round(total / 1024 ** 3, 2),
                "budget_gb": MEM_BUDGET_GB,
                "over_budget": total > MEM_BUDGET_GB * 1024 ** 3,
            },
            "cpu_percent": psutil.cpu_percent(interval=None),
        }
    )


@app.post("/api/vlm")
def api_vlm():
    if over_budget():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "scene")
    prompt = body.get("prompt") or PROMPTS.get(mode) or PROMPTS["scene"]
    max_tokens = int(body.get("max_tokens", 256))
    try:
        img = decode_image(body.get("image", ""))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"图片解析失败：{exc}"}), 400

    img_b64 = image_to_jpeg_b64(img)
    (DATA_DIR / "last-request.jpg").write_bytes(
        base64.b64decode(img_b64)
    )
    if not vlm_health():
        return jsonify({"error": "VLM 服务未启动（127.0.0.1:8091）"}), 503
    try:
        with VLM_LOCK:
            result = call_vlm(img_b64, prompt, max_tokens=max_tokens)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"VLM 调用失败：{exc}"}), 500


def run_eyes(task, item="", context="", question="", image_b64=""):
    """统一执行入口，HTTP 路由和语音链路都走这里。"""
    spec = EYES_TASKS.get(task)
    if spec is None:
        raise ValueError(f"未知任务：{task}")
    if task == "item" and not item:
        raise ValueError("缺少物品名称")
    if task == "followup" and not (context and question):
        raise ValueError("缺少描述或问题")

    prompt = build_eyes_prompt(task, item=item, context=context, question=question)
    fallback = spec["fallback"].format(item=item, context=context, question=question)
    max_tokens = spec["max_tokens"]
    t0 = time.time()

    if task == "followup":
        if not vlm_health():
            raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
        with VLM_LOCK:
            out = call_vlm(None, prompt, max_tokens, temperature=0.1)
        return {
            "task": task,
            "source": "vlm",
            "answer": out["answer"] or fallback,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    try:
        img = decode_image(image_b64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片解析失败：{exc}")

    if task == "text" and ocr_engine is not None:
        img_path = DATA_DIR / "eyes-text.png"
        img.save(img_path)
        try:
            with OCR_LOCK:
                res = ocr_engine(str(img_path))
            text = pick_main_text_block(parse_ocr_result(res))
        except Exception as exc:  # noqa: BLE001
            print("[eyes] OCR 失败，改用 VLM：", exc)
            text = ""
        if text:
            return {
                "task": task,
                "source": "ocr",
                "answer": text,
                "latency_ms": int((time.time() - t0) * 1000),
            }

    img_b64 = image_to_jpeg_b64(img)
    if not vlm_health():
        raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
    with VLM_LOCK:
        out = call_vlm(img_b64, prompt, max_tokens, temperature=0.1)
    return {
        "task": task,
        "source": "vlm",
        "answer": settle_eyes_answer(task, out["answer"], fallback, item),
        "latency_ms": int((time.time() - t0) * 1000),
    }


@app.post("/api/eyes")
def api_eyes():
    if over_budget():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    body = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(
            run_eyes(
                body.get("task", "scene"),
                item=str(body.get("item") or "").strip(),
                context=str(body.get("context") or "").strip(),
                question=str(body.get("question") or "").strip(),
                image_b64=body.get("image", ""),
            )
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"识别失败：{exc}"}), 500


WAKE_PATTERN = re.compile(r"小\s*[v薇微威伟]", re.I)


def decode_wav(data_b64):
    if not data_b64:
        raise ValueError("缺少音频数据")
    if data_b64.startswith("data:"):
        data_b64 = data_b64.split(",", 1)[1]
    raw = base64.b64decode(data_b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("只支持 16-bit PCM 音频")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sr != 16000 and len(samples):
        n = int(len(samples) * 16000 / sr)
        samples = np.interp(
            np.linspace(0, len(samples), n, endpoint=False),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    return samples


def asr_transcribe(samples):
    if asr_engine is None:
        raise RuntimeError("ASR 未初始化：" + str(init_error["asr"]))
    with ASR_LOCK:
        stream = asr_engine.create_stream()
        stream.accept_waveform(16000, samples)
        asr_engine.decode_stream(stream)
        return (stream.result.text or "").strip()


def extract_item(cmd):
    for pattern in (r"找(?:一下|一找)?(.+)", r"有没有(.+)"):
        m = re.search(pattern, cmd)
        if m:
            name = re.sub(r"[吗呢啊吧]", "", m.group(1)).strip()
            if name:
                return name
    return ""


def parse_voice_intent(cmd):
    """规则匹配意图，不额外加载模型。"""
    if any(w in cmd for w in ("卫生间", "洗手间", "厕所", "WC", "wc", "toilet", "Toilet")):
        return "toilet", ""
    if "出口" in cmd:
        return "exit", ""
    if any(w in cmd for w in ("读", "念", "什么字", "写的什么", "文字", "上面的字")):
        return "text", ""
    if "找" in cmd or "有没有" in cmd:
        item = extract_item(cmd)
        return ("item", item) if item else ("scene", "")
    if any(w in cmd for w in ("有什么", "看看", "前面", "周围", "描述")):
        return "scene", ""
    return "followup", ""


@app.post("/api/voice")
def api_voice():
    if asr_engine is None:
        return jsonify({"error": "ASR 未初始化：" + str(init_error["asr"])}), 503
    if over_budget():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    body = request.get_json(force=True, silent=True) or {}
    t0 = time.time()

    try:
        samples = decode_wav(body.get("audio", ""))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"音频解析失败：{exc}"}), 400
    if len(samples) < 4800:
        return jsonify({"text": "", "intent": "short", "answer": ""})

    try:
        text = asr_transcribe(samples)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"语音识别失败：{exc}"}), 500

    if not WAKE_PATTERN.search(text):
        return jsonify(
            {"text": text, "intent": "none", "answer": "", "latency_ms": int((time.time() - t0) * 1000)}
        )

    cmd = re.sub(r"[，。！？、,.!?\s]", "", WAKE_PATTERN.sub("", text))
    if not cmd or len(cmd) <= 3:
        return jsonify(
            {"text": text, "intent": "wake", "answer": "我在，请说", "latency_ms": int((time.time() - t0) * 1000)}
        )

    intent, item = parse_voice_intent(cmd)
    context = str(body.get("context") or "").strip()
    if intent == "followup" and not context:
        return jsonify(
            {"text": text, "intent": intent, "answer": "先看一下画面再说", "latency_ms": int((time.time() - t0) * 1000)}
        )

    try:
        if intent == "followup":
            result = run_eyes("followup", context=context, question=cmd)
        else:
            result = run_eyes(intent, item=item, image_b64=body.get("image", ""))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"识别失败：{exc}"}), 500

    return jsonify(
        {
            "text": text,
            "intent": intent,
            "item": item,
            "source": result["source"],
            "answer": result["answer"],
            "latency_ms": int((time.time() - t0) * 1000),
        }
    )


@app.post("/api/ocr")
def api_ocr():
    if ocr_engine is None:
        return jsonify({"error": "OCR 未初始化：" + str(init_error["ocr"])}), 503
    body = request.get_json(force=True, silent=True) or {}
    try:
        img = decode_image(body.get("image", ""))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"图片解析失败：{exc}"}), 400

    img_path = DATA_DIR / "last-ocr.png"
    img.save(img_path)
    try:
        t0 = time.time()
        with OCR_LOCK:
            res = ocr_engine(str(img_path))
        latency = int((time.time() - t0) * 1000)
        lines = parse_ocr_result(res)
        full_text = "\n".join(item["text"] for item in lines)
        return jsonify({"text": full_text, "lines": lines, "latency_ms": latency})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"OCR 失败：{exc}"}), 500


@app.post("/api/tts")
def api_tts():
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "文本为空"}), 400
    if len(text) > 160:
        text = text[:160]
    engine = (body.get("engine") or "melo").lower()
    sid = int(body.get("sid", 45))
    speed = float(body.get("speed", 1.0))
    try:
        t0 = time.time()
        with TTS_LOCK:
            if engine == "kokoro":
                try:
                    synth = load_kokoro()
                except Exception as exc:  # noqa: BLE001
                    init_error["kokoro"] = str(exc)
                    return jsonify({"error": "Kokoro 初始化失败：" + str(exc)}), 503
                audio = synth.generate(text, sid=sid, speed=speed)
                sr = synth.sample_rate
            else:
                if tts_engine is None:
                    return jsonify({"error": "MeloTTS 未初始化：" + str(init_error["tts"])}), 503
                audio = tts_engine.generate(text, sid=0, speed=speed)
                sr = tts_engine.sample_rate
            gen_ms = int((time.time() - t0) * 1000)
        samples = np.clip(np.asarray(audio.samples, dtype=np.float32) * 32767, -32768, 32767)
        pcm = samples.astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        buf.seek(0)
        resp = send_file(buf, mimetype="audio/wav", as_attachment=False, download_name="tts.wav")
        resp.headers["X-Gen-Ms"] = str(gen_ms)
        resp.headers["X-Audio-Seconds"] = str(round(len(audio.samples) / sr, 2))
        resp.headers["X-TTS-Engine"] = engine
        return resp
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"TTS 失败：{exc}"}), 500


@app.get("/api/sample/<path:name>")
def sample(name: str):
    return send_from_directory(SAMPLES_DIR, name)


if __name__ == "__main__":
    threading.Thread(target=init_ocr, daemon=True).start()
    threading.Thread(target=init_tts, daemon=True).start()
    threading.Thread(target=init_asr, daemon=True).start()
    time.sleep(1)
    app.run(host="127.0.0.1", port=8090, threaded=True, debug=False)
