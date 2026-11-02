# -*- coding: utf-8 -*-
"""
CM5 边缘 AI 本地测试服务
- VLM：调用本机 llama-server（MiniCPM-V 4.7 1B，OpenAI 兼容接口）
- OCR：RapidOCR + PP-OCRv6_small（ONNX Runtime）
- TTS：Piper 中文神经语音（zh_CN-huayan-medium，子进程按需合成）
"""

import base64
import gc
import hashlib
import io
import json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import atexit
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import psutil
import requests
from flask import Flask, Response, after_this_request, jsonify, request, send_file, send_from_directory, stream_with_context
from PIL import Image
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
import sherpa_onnx

import guide
import intent
import kws
import device_io
import device_voice
from guide_service import GUIDE
from output import OUTPUT, PRIORITY_ANSWER, PRIORITY_CONFIRM, PRIORITY_DANGER, PRIORITY_GUIDE
from stream_text import TTS_NORM_VERSION, normalize_for_tts, segment_text, SemanticSegmenter
from tts_worker import PiperWorker, TTSInterrupted
from trace import TraceStore

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OCR_DIR = MODELS / "ocr" / "pp-ocrv6-small"
PIPER_DIR = MODELS / "tts" / "piper-zh_CN-huayan-medium"
PIPER_BIN = Path(
    os.environ.get("PIPER_BIN")
    or (ROOT / "runtime" / "piper" / "piper" / ("piper.exe" if os.name == "nt" else "piper"))
)
PIPER_MODEL = Path(
    os.environ.get("PIPER_MODEL")
    or (PIPER_DIR / "zh_CN-huayan-medium.onnx")
)
ASR_DIR = MODELS / "asr" / "sherpa-onnx-paraformer-zh-small-2024-03-09"
VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8091")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
DATA_DIR = Path(os.environ.get("DATA_DIR") or (ROOT / "data"))
TTS_CACHE_DIR = DATA_DIR / "tts-cache"
SAMPLES_DIR = ROOT / "samples"
DEBUG_SNAPSHOT = os.environ.get("DEBUG_SNAPSHOT", "0").strip().lower() in ("1", "true", "yes")
TTS_SMOOTH = os.environ.get("TTS_SMOOTH", "0").strip().lower() in ("1", "true", "yes")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "persistent").strip().lower()
TTS_IDLE_SECONDS = float(os.environ.get("TTS_IDLE_SECONDS", "0"))
TTS_CACHE_VERSION = "piper-persistent-v3"
_PIPER_WORKER = None

# 线程数可配置：默认值与现状一致，CM5 上按实测再调，不做绑核
ASR_THREADS = int(os.environ.get("ASR_THREADS", "4"))
OCR_INTRA_THREADS = int(os.environ.get("OCR_INTRA_THREADS", "4"))
OCR_INTER_THREADS = int(os.environ.get("OCR_INTER_THREADS", "1"))
FRAME_MAX_AGE = float(os.environ.get("FRAME_MAX_AGE", "20"))
VLM_IMAGE_MAX_SIDE = int(os.environ.get("VLM_IMAGE_MAX_SIDE", "512"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"
TRACES = TraceStore(LOG_DIR / "traces.jsonl")


def _setup_logger():
    logger = logging.getLogger("cm5")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


log = _setup_logger()


class _QuietAccessFilter(logging.Filter):
    NOISE = ("/api/status", "/favicon.ico", "/static/", "/@vite/client")

    def filter(self, record):
        return not any(part in record.getMessage() for part in self.NOISE)


logging.getLogger("werkzeug").addFilter(_QuietAccessFilter())


def b64_md5(data):
    """日志用图片指纹：确认请求实际带的是哪张图。"""
    if not data:
        return "-"
    raw = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        return hashlib.md5(base64.b64decode(raw)).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "bad"


app = Flask(__name__, static_folder="static", static_url_path="/static")

OCR_LOCK = threading.Lock()
TTS_LOCK = threading.Lock()
VLM_LOCK = threading.Lock()
ASR_LOCK = threading.Lock()

ocr_engine = None
asr_engine = None
asr_last_used = 0.0
init_error = {"ocr": None, "asr": None}

MEM_BUDGET_GB = float(os.environ.get("MEM_BUDGET_GB", "3.6"))
MEM_MIN_AVAILABLE_MB = float(os.environ.get("MEM_MIN_AVAILABLE_MB", "400"))
MEM_SAMPLE_TTL = float(os.environ.get("MEM_SAMPLE_TTL", "10"))
_MEM_CACHE = {"ts": 0.0, "value": None}
VOICE_SOURCE = os.environ.get("VOICE_SOURCE", "browser").strip().lower()
GUIDE_SOURCE = os.environ.get("GUIDE_SOURCE", "auto").strip().lower()
OUTPUT_SINK = os.environ.get("OUTPUT_SINK", "browser").strip().lower()

DEVICE_CAMERA = device_io.DeviceCamera()
DEVICE_VOICE = None
DEVICE_SINK = None


def _on_device_frame(img, frame_id, captured_mono, meta):
    """Picamera2 连续帧直接送入服务端引导循环，不经过浏览器。"""
    st = GUIDE.status()
    if not st.get("running") or st.get("mode") != "device":
        return
    GUIDE.submit_frame(
        img,
        frame_id=frame_id,
        source="device",
        captured_mono=captured_mono,
        meta=meta or {},
    )


def device_status():
    return {
        "voice_source": VOICE_SOURCE,
        "guide_source": GUIDE_SOURCE,
        "output_sink": OUTPUT_SINK,
        "camera": DEVICE_CAMERA.status(),
        "voice": DEVICE_VOICE.status() if DEVICE_VOICE is not None else {
            "available": False, "running": False, "error": "设备语音未启用",
        },
        "audio": device_io.audio_tools(),
        "sink": DEVICE_SINK is not None,
    }


def default_guide_source():
    if GUIDE_SOURCE == "browser":
        return "browser"
    if GUIDE_SOURCE == "device" or DEVICE_CAMERA.available():
        return "device"
    return "browser"


def start_device_services():
    """按环境变量启用 CM5 设备摄像头、ALSA 语音与播报。"""
    global DEVICE_VOICE, DEVICE_SINK
    if DEVICE_CAMERA.available() and (VOICE_SOURCE == "device" or GUIDE_SOURCE == "device"):
        DEVICE_CAMERA.start(_on_device_frame)
    if VOICE_SOURCE != "device":
        return
    if DEVICE_VOICE is None:
        DEVICE_VOICE = device_voice.DeviceVoice(camera=DEVICE_CAMERA)
    if OUTPUT_SINK == "alsa" and device_io.AlsaPlayer().available() and not piper_error():
        player = device_io.AlsaPlayer()
        DEVICE_SINK = device_io.DeviceSink(
            synth=lambda text: piper_wav(text)[0],
            player=player,
            before_speak=DEVICE_VOICE.pause,
            after_speak=DEVICE_VOICE.resume,
        )
        OUTPUT.attach_sink(DEVICE_SINK)
        DEVICE_VOICE.set_sink_present(True)
        log.info("[device] ALSA sink enabled")
    if not DEVICE_VOICE.start():
        log.warning("[device] voice not started: %s", DEVICE_VOICE.status().get("error"))
    else:
        log.info("[device] voice listener started")


def stop_device_services():
    global DEVICE_VOICE, DEVICE_SINK
    if DEVICE_VOICE is not None:
        DEVICE_VOICE.stop()
    if DEVICE_SINK is not None:
        OUTPUT.detach_sink()
    DEVICE_CAMERA.stop()


def _used_bytes(proc):
    """运行时用 RSS，读取廉价；共享库会重复计入，属保守上界。

    PSS 只在测试报告里用 /proc/<pid>/smaps_rollup 单独测量，不进入请求路径。
    """
    try:
        return proc.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def model_memory_bytes(fresh: bool = False):
    """统计模型侧 RSS：llama-server + 本服务进程 + 短暂 TTS/播放子进程。

    默认 10 秒缓存；RSS 读取本身很轻，但避免每 1 秒轮询重复遍历进程。
    """
    now = time.monotonic()
    cached = _MEM_CACHE.get("value")
    if not fresh and cached is not None and now - _MEM_CACHE.get("ts", 0.0) < MEM_SAMPLE_TTL:
        return cached
    app = _used_bytes(psutil.Process(os.getpid()))
    vlm = 0
    tts = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name.startswith("llama-server"):
                vlm += _used_bytes(proc)
            elif name.startswith(("piper", "aplay", "arecord")):
                tts += _used_bytes(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    value = (app, vlm, tts)
    _MEM_CACHE.update(ts=now, value=value)
    return value


def over_budget(fresh: bool = False):
    """以整机可用内存为准；RSS 只用于显示，不把共享库重复相加当硬门槛。"""
    if psutil.virtual_memory().available < MEM_MIN_AVAILABLE_MB * 1024 ** 2:
        return True
    app, vlm, tts = model_memory_bytes(fresh=fresh)
    return (app + vlm + tts) > MEM_BUDGET_GB * 1024 ** 3 * 1.25


def release_low_priority_models():
    """内存超预算时先释放按需加载的 ASR；引导与 VLM 不在这里卸载。"""
    global asr_engine
    released = False
    with ASR_LOCK:
        if asr_engine is not None:
            asr_engine = None
            released = True
    if released:
        gc.collect()
        log.warning("[memory] budget exceeded, released ASR")
    return released


def budget_guard():
    if not over_budget():
        return True
    release_low_priority_models()
    return not over_budget()


def system_metrics():
    """整机内存、温度与树莓派欠压/降频标志；无 vcgencmd 时明确返回未知。"""
    vm = psutil.virtual_memory()
    data = {
        "mem_total_gb": round(vm.total / 1024 ** 3, 2),
        "mem_available_gb": round(vm.available / 1024 ** 3, 2),
        "mem_percent": round(vm.percent, 1),
        "temperature_c": None,
        "throttled": None,
        "throttled_flags": None,
    }
    try:
        import shutil
        import subprocess as sp
        vc = shutil.which("vcgencmd")
        if vc:
            out = sp.run([vc, "measure_temp"], capture_output=True, text=True, timeout=2).stdout
            m = re.search(r"temp=([0-9.]+)", out or "")
            if m:
                data["temperature_c"] = float(m.group(1))
            out = sp.run([vc, "get_throttled"], capture_output=True, text=True, timeout=2).stdout
            m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out or "")
            if m:
                flags = int(m.group(1), 16)
                data["throttled"] = flags != 0
                data["throttled_flags"] = m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return data


# ---------- CPU：只统计模型侧进程，按整机逻辑核心归一化 ----------
CPU_LOCK = threading.Lock()
_cpu_sample = {"ts": None, "ticks": 0.0, "percent": None}


def model_cpu_ticks():
    """本服务进程 + llama-server 的累计 CPU 秒数。"""
    total = 0.0
    try:
        t = psutil.Process(os.getpid()).cpu_times()
        total += t.user + t.system
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for proc in psutil.process_iter(["name", "cpu_times"]):
        try:
            if (proc.info["name"] or "").lower().startswith("llama-server"):
                t = proc.info["cpu_times"]
                total += t.user + t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def model_cpu_percent():
    """模型侧 CPU 占整机百分比；首次采样或间隔不足 1 秒时返回上次值。"""
    now = time.time()
    ticks = model_cpu_ticks()
    with CPU_LOCK:
        prev_ts = _cpu_sample["ts"]
        if prev_ts is None or now - prev_ts < 1.0:
            if prev_ts is None:
                _cpu_sample["ts"] = now
                _cpu_sample["ticks"] = ticks
            return _cpu_sample["percent"]
        percent = (ticks - _cpu_sample["ticks"]) / (now - prev_ts) / (psutil.cpu_count() or 1) * 100
        percent = round(min(max(percent, 0.0), 100.0), 1)
        _cpu_sample.update(ts=now, ticks=ticks, percent=percent)
        return percent


EYES_RULE = (
    "你是视障人士的AI助手。先说最重要的事实，再用一到两句口语中文回答；不分点、不加括号、不用 Markdown。"
    "方位只说左/中/右，远近只说近/远；不要重复同一方位。"
    "只说画面里看得见的；看不清、被挡住、没看到就直说，不猜、不补常识。"
    "不报米数、不报秒数，不说能不能走、能不能过马路。"
)

CHAT_RULE = (
    "你是陪伴视障人士的中文语音助手。先说结论，再用两到三句口语中文回答，不分点、不加括号、不用 Markdown。"
    "不要描述画面，也不要提眼睛或视力；不要重复开场白。"
    "不编造事实，不假装能查询实时信息。"
    "如果对方问新闻、天气或实时数据，直接说明没有联网查询能力，请他看手机或问身边的人。"
    "涉及安全或医疗时只给一般性建议，并提醒找专业人员。"
)

DETAIL_RULE = (
    "你是视障人士的AI助手。先用一句话说正前方和近处有没有障碍，再补充主要物体的位置和外观；总共三到五句口语中文，不分点、不加括号、不用 Markdown。"
    "方位只说左/中/右，远近只说近/远，不重复同一方位；看不清就直说，不猜、不补常识。"
    "不报米数、不报秒数，不说能不能走、能不能过马路。"
)

FOLLOW_RULE = (
    "你是视障人士的AI助手。先直接回答对方的问题，用一到两句口语中文，不分点、不加括号、不用 Markdown。"
    "给了画面就结合画面回答，只用能确认的信息；看不清、被挡住、没看到就直说，不猜。"
    "方位只说左/中/右，远近只说近/远，不报米数、不报秒数，不说能不能走。"
)

NO_FRAME_REPLY = "当前没有画面，请先拍一张或打开相机"

RULES_BY_TASK = {
    "chat": CHAT_RULE,
    "detail": DETAIL_RULE,
    "compare": DETAIL_RULE,
    "followup": FOLLOW_RULE,
}

REALTIME_WORDS = ("新闻", "天气", "股价", "比分", "热搜", "今天发生", "实时", "现在几点")
CHAT_REALTIME_REPLY = "我这边没有联网，查不到今天的实时信息，你可以用手机看一下。"

EYES_TASKS = {
    "scene": {
        "prompt": (
            "看图，用一到两句、总共不超过40个字，说清正前方从近到远有什么。"
            "如果画面主要是文字、标牌或屏幕，就照原文读出最醒目的文字，不要编造人物、车辆、台阶或道路。"
            "最后一句说明有没有明显障碍；没有就说“前方没看到明显障碍”。不要分点、不要换行。"
        ),
        "max_tokens": 48,
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
            "孤立数字和年份不要扩写成生产日期、保质期或型号；"
            "看不清就说“看不清”。"
        ),
        "max_tokens": 64,
        "fallback": "看不清",
    },
    "followup": {
        "prompt": (
            "根据刚才的对话回答，如果同时给了当前画面，可以结合画面补充细节。"
            "不要补充对话和画面里都没有的东西。\n"
            "对话：{context}\n问题：{question}\n"
            "用一到两句口语中文，分不清就说“刚才没看清”。"
        ),
        "max_tokens": 48,
        "fallback": "刚才没看清",
    },
    "detail": {
        "prompt": (
            "详细描述这张图片：先说正前方和近处有没有障碍物，再说主要物体的位置、外观和相互位置关系，"
            "用三到五句口语中文，不要分点。看不清的地方直接说看不清，不要猜。"
        ),
        "max_tokens": 160,
        "fallback": "看不清",
    },
    "compare": {
        "prompt": (
            "比较画面里“{item}”和另一个同类物体：先说各自在左/中/右、近还是远，"
            "再说大小、颜色或遮挡上的差别，两三句口语中文。看不清就说看不清。"
        ),
        "max_tokens": 120,
        "fallback": "看不清",
    },
    "chat": {
        "prompt": (
            "对方想和你聊几句，顺着他的话回应，可以问一句他现在的感受。"
            "不要描述画面，也不要提眼睛或视力。"
        ),
        "max_tokens": 120,
        "fallback": "我在听，你慢慢说",
    },
    "date": {
        "prompt": "只读出生产日期、保质期和批次，看不清就说看不清。",
        "max_tokens": 64,
        "fallback": "日期没看清，请把标签对准光线，靠近一点",
    },
}

EYES_NEGATIVE = ("没看到", "没有", "看不到", "没发现")
EYES_TARGET = {
    "toilet": ("卫生间", "洗手间", "厕所", "WC", "Toilet"),
    "exit": ("出口", "EXIT"),
}

NEG_PATTERN = "没看到|看不到|没发现|看不见|没有看到|没有|不在|没在"
OBSCURED_NEG = re.compile(r"没(有)?被(挡住|遮挡|遮住|遮盖)")


REPEAT_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{1,16})(?:[，,、]?\1){2,}")


def is_degenerate_repetition(text: str) -> bool:
    """检测小模型常见复读：长文本字符多样性过低，或同一短语连续出现 3 次。"""
    s = (text or "").strip()
    body = [c for c in s if c not in "，。；！？,.;!?：: \n"]
    if len(body) >= 30 and len(set(body)) / len(body) < 0.28:
        return True
    return bool(REPEAT_PATTERN.search(s))


def collapse_repetition(text: str) -> str:
    """只收敛明显复读：连续重复短语压成一次；多样性过低时只留第一句。"""
    s = (text or "").strip()
    s = REPEAT_PATTERN.sub(r"\1", s)
    body = [c for c in s if c not in "，。；！？,.;!?：: \n"]
    if len(body) >= 30 and len(set(body)) / len(body) < 0.28:
        first = re.split(r"[。；！？!?\n]", s, maxsplit=1)[0].strip()
        return first if len(first) >= 4 else ""
    return s


def repair_truncated_answer(text: str) -> str:
    """生成被 max_tokens 截断时，去掉不完整的尾句；只剩几个字时交给兜底句。"""
    s = (text or "").strip()
    if not s or s[-1] in "。！？!?":
        return s
    cut = max(s.rfind(x) for x in "。！？")
    if cut >= 8:
        return s[:cut + 1]
    return "" if len(s) < 10 else s


def settle_eyes_answer(task, answer, fallback, item=""):
    """二值查找任务：只有“否定词紧挨目标词”才收敛成兜底句。

    “水杯在右边，没有被挡住”里的“没有被”是描述遮挡，不是没找到，先剥掉再判断。
    """
    if not answer:
        return fallback
    targets = (item,) if task == "item" else EYES_TARGET.get(task, ())
    if not targets:
        return answer
    cleaned = OBSCURED_NEG.sub("", answer)
    for target in targets:
        if not target:
            continue
        t = re.escape(target)
        pat = re.compile(rf"({NEG_PATTERN})[^，。；,]{{0,8}}{t}|{t}[^，。；,]{{0,8}}({NEG_PATTERN})")
        if pat.search(cleaned):
            return fallback
    return answer


def build_eyes_prompt(task, item="", context="", question=""):
    spec = EYES_TASKS[task]
    rule = RULES_BY_TASK.get(task, EYES_RULE)
    return rule + "\n\n" + spec["prompt"].format(
        item=item, context=context, question=question
    )


def prompt_for(task, item="", context="", question=""):
    if task == "chat":
        prompt = build_eyes_prompt("chat")
        if context:
            prompt += f"\n\n刚才聊过：{context}"
        prompt += f"\n\n用户说：{question}"
        return prompt
    if task == "followup":
        return build_eyes_prompt("followup", context=context, question=question)
    return build_eyes_prompt(task, item=item)


# ---------- 会话状态：按 session_id 隔离，有界 + TTL，换画面只标记失效 ----------
SESSION_LOCK = threading.Lock()
SESSIONS = {}
MAX_SESSIONS = 8
SESSION_TTL = 600
MAX_HISTORY_TURNS = 3
MAX_HISTORY_CHARS = 360
MAX_CONTEXT_CHARS = 320
ARMED_SECONDS = float(os.environ.get("ARMED_SECONDS", "15"))
MIN_VOICE_SECONDS = float(os.environ.get("MIN_VOICE_SECONDS", "0.6"))
MIN_VOICE_RMS = float(os.environ.get("MIN_VOICE_RMS", "0.008"))


def get_session(session_id):
    sid = (session_id or "default").strip() or "default"
    now = time.time()
    with SESSION_LOCK:
        expired = [k for k, v in SESSIONS.items() if now - v["updated"] > SESSION_TTL]
        for k in expired:
            SESSIONS.pop(k, None)
        sess = SESSIONS.get(sid)
        if sess is None:
            if len(SESSIONS) >= MAX_SESSIONS:
                oldest = min(SESSIONS, key=lambda k: SESSIONS[k]["updated"])
                SESSIONS.pop(oldest, None)
            sess = {"frame_id": "", "history": [], "updated": now, "armed_until": 0.0, "last_cmd": "", "last_cmd_at": 0.0}
            SESSIONS[sid] = sess
        sess["updated"] = now
        return sess


def session_begin(session_id, frame_id):
    """换画面只把旧的视觉事实标记失效，聊天历史保留。"""
    sess = get_session(session_id)
    fid = (frame_id or "").strip()
    with SESSION_LOCK:
        if fid and fid != sess["frame_id"]:
            sess["frame_id"] = fid
            for item in sess["history"]:
                if item.get("visual"):
                    item["stale"] = True
        sess["updated"] = time.time()
    return sess


def session_remember(session_id, question, answer, visual=False, frame_id=None):
    if not question:
        return
    sess = get_session(session_id)
    with SESSION_LOCK:
        stale = bool(frame_id) and frame_id != sess["frame_id"]
        sess["history"].append(
            {
                "q": question[:60],
                "a": (answer or "")[:90],
                "visual": visual,
                "frame": frame_id or "",
                "stale": stale,
            }
        )
        sess["history"] = sess["history"][-MAX_HISTORY_TURNS:]
        while sum(len(h["q"]) + len(h["a"]) for h in sess["history"]) > MAX_HISTORY_CHARS:
            sess["history"].pop(0)
        sess["updated"] = time.time()


def history_text(session_id, include_visual=True):
    sess = get_session(session_id)
    with SESSION_LOCK:
        parts = []
        for item in sess["history"]:
            if item.get("visual") and not include_visual:
                continue
            tag = "（旧画面已失效）" if item.get("stale") else ""
            parts.append(f"{tag}问：{item['q']} 答：{item['a']}")
        return "；".join(parts)


def session_context(session_id, client_context="", include_visual=True):
    """服务端历史优先；无当前画面时不使用客户端视觉上下文，避免复述旧画面。"""
    server = history_text(session_id, include_visual=include_visual)
    if include_visual:
        text = server if server else (client_context or "").strip()
    else:
        text = server
    return text[-MAX_CONTEXT_CHARS:]


# ---------- 日期标签解析（只靠 OCR，值不合格就拒猜） ----------
DATE_FIELD_RULES = (
    ("生产日期", ("生产日期", "制造日期", "MFG", "PROD")),
    ("包装日期", ("包装日期",)),
    ("到期日", ("保质期至", "有效期至", "到期日", "此日期前", "EXP")),
    ("保质期", ("保质期限", "保质期", "有效期")),
    ("批号", ("批号", "批次", "LOT", "BATCH")),
)
DATE_RE = re.compile(r"(20\d{2})\s*[年\-./]\s*(\d{1,2})\s*[月\-./]\s*(\d{1,2})")
PLACEHOLDER_RE = re.compile(r"见(瓶底|瓶身|瓶盖|包装|盒底|喷码|标签|说明|外包装)")
DURATION_RE = re.compile(r"\d+\s*(个月|月|年|天)")
MIN_CONF = 0.65


def _valid_date(y, m, d):
    try:
        return 2000 <= int(y) <= 2099 and 1 <= int(m) <= 12 and 1 <= int(d) <= 31
    except ValueError:
        return False


def _box_rect(line):
    box = line.get("box")
    if not box:
        return None
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _nearby_line(rows, idx):
    """按 box 找垂直相邻且水平重叠的行，不依赖列表顺序。"""
    anchor = _box_rect(rows[idx])
    if not anchor:
        return None
    ax0, ay0, ax1, ay1 = anchor
    height = max(1, ay1 - ay0)
    best, best_gap = None, None
    for j, line in enumerate(rows):
        if j == idx:
            continue
        rect = _box_rect(line)
        if not rect:
            continue
        x0, y0, x1, y1 = rect
        if min(ax1, x1) - max(ax0, x0) <= 0:
            continue
        gap = max(0, max(ay0, y0) - min(ay1, y1))
        if gap > height * 1.6:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = line, gap
    return best


def _parse_value(text):
    """返回 (value, kind)，不合格一律 None。"""
    if not text or PLACEHOLDER_RE.search(text):
        return None, None
    m = DATE_RE.search(text)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if not _valid_date(y, mo, d):
            return None, None
        return f"{y}年{int(mo)}月{int(d)}日", "date"
    dm = DURATION_RE.search(text)
    if dm:
        return dm.group(0).replace(" ", ""), "duration"
    return None, None


def parse_date_label(lines, min_score=MIN_CONF):
    """标签与数值都要可信；同字段冲突、占位语、非法日期一律拒猜。"""
    rows = [l for l in lines if (l.get("text") or "").strip()]
    found, weak = {}, {}

    for idx, line in enumerate(rows):
        text = line["text"].strip()
        score = line.get("score")
        for field, keys in DATE_FIELD_RULES:
            match = None
            for k in keys:
                match = re.search(re.escape(k), text, re.I)
                if match:
                    break
            if not match:
                continue
            rest = text[match.end():].strip(" ：:　-")
            value, kind = _parse_value(rest)
            used_score = score
            if value is None:
                near = _nearby_line(rows, idx)
                if near is not None:
                    value, kind = _parse_value((near.get("text") or "").strip())
                    if value is not None:
                        used_score = near.get("score")
            if value is None:
                weak[field] = True
                continue
            if used_score is None or used_score < min_score:
                weak[field] = True
                continue
            actual = "到期日" if field == "保质期" and kind == "date" else field
            found.setdefault(actual, []).append(value)

    fields, missing, conflicts = {}, [], []
    for field, _ in DATE_FIELD_RULES:
        values = found.get(field, [])
        if not values:
            missing.append(field)
            continue
        uniq = set(values)
        if len(uniq) > 1:
            conflicts.append(field)
            continue
        if field in weak:
            missing.append(field)
            continue
        fields[field] = values[0]

    parts = [f"{k} {v}" for k, v in fields.items()]
    if conflicts:
        parts.append(f"{'、'.join(conflicts)}有多个不一致的值，没法确定")
    if missing and fields:
        parts.append(f"{'、'.join(missing)}没看清")
    if not fields and not conflicts:
        return None
    if not parts:
        return None
    return {"answer": "；".join(parts), "fields": fields, "missing": missing, "conflicts": conflicts}


# ---------- 长文本分句：不丢内容，超长硬切，保留原标点 ----------
def split_sentences(text, max_len=90):
    raw = re.split(r"([。！？；\n])", text or "")
    pieces, buf = [], ""
    for i in range(0, len(raw), 2):
        seg = raw[i].strip()
        punct = raw[i + 1] if i + 1 < len(raw) else ""
        if not seg:
            continue
        piece = seg + (punct if punct and punct != "\n" else "。")
        if len(buf) + len(piece) <= max_len:
            buf += piece
        else:
            if buf:
                pieces.append(buf)
            buf = piece
    if buf:
        pieces.append(buf)

    out = []
    for piece in pieces:
        while len(piece) > max_len:
            window = piece[:max_len]
            cut = max(window.rfind("，"), window.rfind("、"))
            cut = cut + 1 if cut >= max_len // 3 else max_len
            out.append(piece[:cut])
            piece = piece[cut:]
        if piece:
            out.append(piece)
    return out


# ---------- 重任务并发上限：引导运行时把 VLM/OCR 并发压到 1 ----------
ACTIVE_LOCK = threading.Lock()
ACTIVE = {"heavy": 0, "light": 0}
HEAVY_LIMIT = 3
GUIDE_HEAVY_LIMIT = 1
LIGHT_LIMIT = 2


def try_enter(kind: str = "heavy"):
    with ACTIVE_LOCK:
        if kind == "light":
            if ACTIVE["light"] >= LIGHT_LIMIT:
                return False
            ACTIVE["light"] += 1
            return True
        limit = GUIDE_HEAVY_LIMIT if GUIDE.status().get("running") else HEAVY_LIMIT
        if ACTIVE["heavy"] >= limit:
            return False
        ACTIVE["heavy"] += 1
        return True


def leave(kind: str = "heavy"):
    with ACTIVE_LOCK:
        key = "light" if kind == "light" else "heavy"
        ACTIVE[key] = max(0, ACTIVE[key] - 1)


def active_stats():
    with ACTIVE_LOCK:
        return {
            "heavy": ACTIVE["heavy"],
            "light": ACTIVE["light"],
            "heavy_limit": GUIDE_HEAVY_LIMIT if GUIDE.status().get("running") else HEAVY_LIMIT,
            "light_limit": LIGHT_LIMIT,
            "guide_running": bool(GUIDE.status().get("running")),
        }


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
                "EngineConfig.onnxruntime.intra_op_num_threads": OCR_INTRA_THREADS,
                "EngineConfig.onnxruntime.inter_op_num_threads": OCR_INTER_THREADS,
            }
        )
        log.info("[init] OCR ready")
    except Exception as exc:  # noqa: BLE001
        init_error["ocr"] = str(exc)
        log.error("[init] OCR failed: %s", exc)


def piper_error():
    """Piper 可执行或音色缺失时返回原因，否则返回空串。"""
    missing = []
    if not PIPER_BIN.exists():
        missing.append(f"可执行文件 {PIPER_BIN}")
    if not PIPER_MODEL.exists():
        missing.append(f"模型 {PIPER_MODEL}")
    if not PIPER_MODEL.with_suffix(".onnx.json").exists():
        missing.append("模型配置 .onnx.json")
    return "；".join(missing)


def prune_tts_cache(max_bytes=50 * 1024 * 1024):
    """缓存超过上限时按修改时间删最旧的，避免无限增长。"""
    try:
        files = sorted(TTS_CACHE_DIR.glob("piper-*.wav"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= max_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("tts cache prune failed: %s", exc)


def smooth_piper_wav(wav_bytes, max_pause=0.15, long_pause=0.28):
    """压掉 Piper 输出里过长的句内停顿，让播报更连贯；短停顿保持不变。"""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            params = wf.getparams()
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sr = params.framerate
        if not sr or samples.size == 0:
            return wav_bytes
        frame = max(1, int(sr * 0.02))
        count = len(samples) // frame
        rms = np.array([
            np.sqrt(np.mean(samples[i * frame:(i + 1) * frame].astype(np.float32) ** 2))
            for i in range(count)
        ])
        quiet = rms < 200
        out = []
        i = 0
        while i < count:
            if quiet[i]:
                j = i
                while j < count and quiet[j]:
                    j += 1
                if (j - i) * 0.02 >= long_pause:
                    keep = int(max_pause * sr)
                    out.append(samples[i * frame:i * frame + keep])
                else:
                    out.append(samples[i * frame:j * frame])
                i = j
            else:
                j = i
                while j < count and not quiet[j]:
                    j += 1
                out.append(samples[i * frame:j * frame])
                i = j
        out.append(samples[count * frame:])
        merged = np.concatenate(out) if out else samples
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(params.nchannels)
            wf.setsampwidth(params.sampwidth)
            wf.setframerate(params.framerate)
            wf.writeframes(merged.astype(np.int16).tobytes())
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        log.warning("piper silence smoothing failed: %s", exc)
        return wav_bytes


def piper_worker():
    global _PIPER_WORKER
    if _PIPER_WORKER is None:
        _PIPER_WORKER = PiperWorker(
            PIPER_BIN, PIPER_MODEL, DATA_DIR / "tts-tmp",
            timeout_s=float(os.environ.get("TTS_TIMEOUT", "90")),
            idle_seconds=TTS_IDLE_SECONDS,
        )
    return _PIPER_WORKER


def piper_wav(text, speed=1.0):
    """常驻 Piper worker 合成 WAV；缓存键包含模型、参数和文本规范版本。"""
    err = piper_error()
    if err:
        raise RuntimeError("Piper 未就绪：" + err)
    text = normalize_for_tts(text)
    if not text:
        raise ValueError("TTS 文本为空")
    try:
        length_scale = 1.0 / max(0.5, min(2.0, float(speed or 1.0)))
    except (TypeError, ValueError):
        length_scale = 1.0
    model_stat = PIPER_MODEL.stat()
    key = hashlib.sha1(
        f"{TTS_CACHE_VERSION}|{TTS_NORM_VERSION}|{TTS_BACKEND}|{int(TTS_SMOOTH)}|{PIPER_MODEL.name}|"
        f"{model_stat.st_size}|{int(model_stat.st_mtime)}|{length_scale:.3f}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    cache_path = TTS_CACHE_DIR / f"piper-{key}.wav"
    hit = cache_path.exists() and cache_path.stat().st_size > 1024
    if hit:
        data = cache_path.read_bytes()
        gen_ms = 0
    else:
        if TTS_BACKEND == "subprocess":
            data, gen_ms = piper_subprocess_synth(text, length_scale)
        else:
            data, gen_ms = piper_worker().synth(text, length_scale)
        if TTS_SMOOTH:
            data = smooth_piper_wav(data)
        cache_path.write_bytes(data)
        prune_tts_cache()
    with wave.open(io.BytesIO(data), "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate() or 1)
        sample_rate = wf.getframerate()
    log.info(
        "tts engine=%s cache=%s chars=%d gen=%sms audio=%.2fs smooth=%s",
        tts_engine_name(), "hit" if hit else "miss", len(text), gen_ms, duration, int(TTS_SMOOTH),
    )
    return data, duration, gen_ms, hit


def tts_engine_name() -> str:
    return "piper-subprocess" if TTS_BACKEND == "subprocess" else "piper-persistent"


def piper_subprocess_synth(text: str, length_scale: float) -> tuple[bytes, int]:
    """旧路径：每次启动 Piper。保留为内存紧张或 worker 异常时的回滚选项。"""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="piper-") as tmp:
        out = Path(tmp) / "out.wav"
        proc = subprocess.run(
            [str(PIPER_BIN), "-m", str(PIPER_MODEL), "-f", str(out),
             "--length_scale", f"{length_scale:.3f}", "--sentence_silence", "0.12"],
            input=text.encode("utf-8"), capture_output=True, timeout=90,
            cwd=str(PIPER_BIN.parent),
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"Piper 合成失败：{proc.stderr[-200:]!r}")
        data = out.read_bytes()
    return data, int((time.perf_counter() - t0) * 1000)


def piper_tts(text, speed=1.0):
    data, duration, gen_ms, hit = piper_wav(text, speed)
    with wave.open(io.BytesIO(data), "rb") as wf:
        sample_rate = wf.getframerate()
    resp = Response(data, mimetype="audio/wav")
    resp.headers["X-Gen-Ms"] = str(gen_ms)
    resp.headers["X-Audio-Seconds"] = str(round(duration, 2))
    resp.headers["X-TTS-Engine"] = tts_engine_name()
    resp.headers["X-TTS-Cache"] = "hit" if hit else "miss"
    resp.headers["X-Audio-Sample-Rate"] = str(sample_rate)
    return resp


def prewarm_tts():
    """常用引导/确认短语预生成，避免第一次危险提示还要等 Piper 启动。"""
    phrases = (
        "继续直行", "向左一点", "向右一点",
        "前方有台阶，请停下", "前方是道路，请停下", "前方有人，请停下",
        "前方有障碍，请停下", "前方无法通行，请停下", "前方通道不连续，请停下",
        "前方通道太窄，请停下", "脚下没有可靠的可走地面，请停下",
        "画面质量不足，请停下", "画面判断不确定，请停下",
        "画面已过期，请停下", "引导模型无响应，请停下",
        "我在，请说", "引导已停止",
    )
    for text in phrases:
        try:
            piper_wav(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("tts prewarm failed %r: %s", text, exc)
            return
    log.info("tts prewarm done phrases=%d", len(phrases))


def init_asr():
    """按需加载：只有出现语音请求时才占用内存。"""
    global asr_engine
    try:
        asr_engine = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(ASR_DIR / "model.int8.onnx"),
            tokens=str(ASR_DIR / "tokens.txt"),
            num_threads=ASR_THREADS,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        init_error["asr"] = None
        log.info("[init] ASR ready")
    except Exception as exc:  # noqa: BLE001
        init_error["asr"] = str(exc)
        log.error("[init] ASR failed: %s", exc)


def ensure_asr():
    global asr_engine, asr_last_used
    with ASR_LOCK:
        if asr_engine is None:
            init_asr()
        if asr_engine is None:
            raise RuntimeError("ASR 初始化失败：" + str(init_error["asr"]))
        asr_last_used = time.time()
        return asr_engine


ASR_IDLE_SECONDS = int(os.environ.get("ASR_IDLE_SECONDS", "180"))


def asr_idle_reaper():
    """闲置一段时间后释放 ASR，把内存还给系统。"""
    global asr_engine
    while True:
        time.sleep(20)
        if asr_engine is None or time.time() - asr_last_used < ASR_IDLE_SECONDS:
            continue
        with ASR_LOCK:
            if asr_engine is not None and time.time() - asr_last_used >= ASR_IDLE_SECONDS:
                asr_engine = None
                gc.collect()
                log.info("[asr] 空闲释放")


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


def vlm_image_b64(data: str, max_side: int = VLM_IMAGE_MAX_SIDE) -> str:
    """尽量复用客户端已压缩的 JPEG，避免 decode→resize→encode 的重复开销。"""
    if not data:
        raise ValueError("缺少图片数据")
    raw = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        binary = base64.b64decode(raw)
        probe = Image.open(io.BytesIO(binary))
        if probe.format == "JPEG" and max(probe.size) <= max_side:
            return raw
    except Exception:  # noqa: BLE001
        binary = None
    if binary is None:
        raise ValueError("图片解码失败")
    img = Image.open(io.BytesIO(binary)).convert("RGB")
    return image_to_jpeg_b64(img, max_side)


def save_debug_image(name, img):
    """调试快照默认关闭；开启时用唯一文件名，避免并发写到同一文件串图。"""
    if not DEBUG_SNAPSHOT:
        return None
    path = DATA_DIR / f"{time.time_ns()}-{threading.get_ident()}-{name}"
    img.save(path)
    log.info("[snapshot] %s", path.name)
    return path


def frame_age_seconds(captured_at):
    """解析前端给的拍摄时间，用于真正的过期检查。"""
    if not captured_at:
        return None
    try:
        ts = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


# ---------- 帧标识：同一 frame_id 不允许对应两张不同图片 ----------
FRAME_LOCK = threading.Lock()
FRAME_REGISTRY = {}


def canonical_frame_id(frame_id, image_b64):
    """客户端漏传或复用时生成新 id；同 id 换图时自动改名并记日志。"""
    md5 = b64_md5(image_b64)
    fid = (frame_id or "").strip() or f"f-{uuid.uuid4().hex[:12]}"
    with FRAME_LOCK:
        old = FRAME_REGISTRY.get(fid)
        if old is not None and old != md5:
            new = f"{fid}-{uuid.uuid4().hex[:8]}"
            log.warning("frame id reused frame=%s old=%s new=%s replaced=%s", fid, old, md5, new)
            FRAME_REGISTRY[new] = md5
            fid = new
        else:
            FRAME_REGISTRY[fid] = md5
        while len(FRAME_REGISTRY) > 64:
            FRAME_REGISTRY.pop(next(iter(FRAME_REGISTRY)))
    return fid


SAFE_CLAIM = re.compile(
    r"没有(明显)?(危险|异常|障碍|问题)|可以(安全)?(走|通行|通过|前进)|无障碍|畅通|安全地"
)
OBSTACLE_WORDS = (
    "自行车", "垃圾桶", "台阶", "楼梯", "路障", "椅子", "桌子", "障碍", "栏杆",
    "柱子", "箱子", "车辆", "汽车", "电动车", "婴儿车", "手推车",
)


def guard_vision_answer(task, answer):
    """引导提示危险时，禁止 VLM 同时说“没有危险/可以走”。"""
    if not answer or task in ("read_text", "read_date", "toilet", "exit", "item"):
        return answer
    # 场景/详细描述自己列了障碍物，却仍说“没看到明显障碍”：以列出的物体为准，改成警告。
    if task in ("scene", "detail") and "没看到明显障碍" in answer and any(w in answer for w in OBSTACLE_WORDS):
        answer = answer.replace("前方没看到明显障碍", "前方有障碍物，请小心")
        answer = answer.replace("没有明显障碍", "有障碍物，请小心")
    st = GUIDE.status()
    if not st.get("running"):
        return answer
    decision = st.get("decision") or {}
    state = decision.get("state")
    reason = decision.get("reason")
    if state not in ("stop", "unavailable"):
        return answer
    if reason in ("no_frame", "stopped"):
        return answer
    if not SAFE_CLAIM.search(answer):
        return answer
    if reason in ("stairs", "road", "person", "obstacle", "blocked", "ahead", "narrow", "depth_obstacle"):
        return "引导已提示前方危险，请先停下；现在不做画面安全判断。"
    return "引导画面当前不可靠，不能确认安全，请先停下或重拍。"


def temperature_for(task):
    return 0.4 if task == "chat" else 0.1


def vlm_headers():
    return {"Authorization": f"Bearer {VLM_API_KEY}"} if VLM_API_KEY else {}


def vlm_health() -> bool:
    try:
        r = requests.get(VLM_URL + "/health", headers=vlm_headers(), timeout=2)
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
    r = requests.post(VLM_URL + "/v1/chat/completions", json=payload, headers=vlm_headers(), timeout=300)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    answer = (msg.get("content") or "").strip()
    for tag in ("<think>", "</think>"):
        answer = answer.replace(tag, "")
    answer = collapse_repetition(answer)
    finish_reason = data["choices"][0].get("finish_reason")
    if finish_reason == "length":
        answer = repair_truncated_answer(answer)
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    prefill_ms = timings.get("prompt_ms")
    decode_ms = timings.get("predicted_ms")
    log.info(
        "vlm tokens prompt=%s completion=%s prefill_ms=%s decode_ms=%s wall_ms=%s finish=%s",
        usage.get("prompt_tokens"), usage.get("completion_tokens"),
        prefill_ms, decode_ms, int((time.time() - t0) * 1000),
        data["choices"][0].get("finish_reason"),
    )
    return {
        "answer": answer.strip(),
        "latency_ms": int((time.time() - t0) * 1000),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def stream_vlm(prompt, img_b64=None, max_tokens=160, temperature=0.3):
    """流式取增量文本，供分句 TTS 边生成边播报。"""
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
        "stream": True,
    }
    with requests.post(VLM_URL + "/v1/chat/completions", json=payload, headers=vlm_headers(), stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {}).get("content") or ""
            for tag in ("<think>", "</think>"):
                delta = delta.replace(tag, "")
            if delta:
                yield delta


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


def meaningful_ocr_text(text: str) -> bool:
    """过滤键盘单键、噪声等无意义单字符；单个汉字（如“停”）仍算有效。"""
    s = (text or "").strip()
    if not s:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", s)
    return any(len(t) >= 2 for t in tokens)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/api/status")
def status():
    app_rss, vlm_rss, tts_rss = model_memory_bytes()
    total = app_rss + vlm_rss + tts_rss
    cpu_percent = model_cpu_percent()
    return jsonify(
        {
            "vlm": {"up": vlm_health(), "url": VLM_URL},
            "ocr": {"ready": ocr_engine is not None, "error": init_error["ocr"]},
            "asr": {"ready": asr_engine is not None, "lazy": True, "error": init_error["asr"]},
            "tts": {
                "ready": not piper_error(),
                "engine": tts_engine_name(),
                "backend": TTS_BACKEND,
                "worker": piper_worker().status() if TTS_BACKEND != "subprocess"
                else {"alive": False, "engine": "piper-subprocess", "synth_count": None},
                "smooth": TTS_SMOOTH,
                "error": piper_error() or None,
            },
            "kws": kws.status(),
            "guide": GUIDE.status(),
            "guide_model": guide.status(),
            "output": OUTPUT.stats(),
            "active": active_stats(),
            "system": system_metrics(),
            "device": device_status(),
            "traces": {"active": TRACES.active_count(), "summary": TRACES.summary()},
            "memory": {
                "measure": "rss",
                "model_gb": round(vlm_rss / 1024 ** 3, 2),
                "app_gb": round(app_rss / 1024 ** 3, 2),
                "tts_gb": round(tts_rss / 1024 ** 3, 2),
                "sum_gb": round(total / 1024 ** 3, 2),
                "available_mb": round(psutil.virtual_memory().available / 1024 ** 2),
                "min_available_mb": MEM_MIN_AVAILABLE_MB,
                "budget_gb": MEM_BUDGET_GB,
                "over_budget": over_budget(),
            },
            "cpu": {
                "model_percent": cpu_percent,
                "cores": psutil.cpu_count() or 1,
            },
        }
    )


@app.get("/api/metrics")
def api_metrics():
    kind = (request.args.get("kind") or "").strip() or None
    try:
        limit = int(request.args.get("limit", "100"))
    except ValueError:
        limit = 100
    return jsonify({
        "summary": TRACES.summary(),
        "active": TRACES.active_count(),
        "records": TRACES.records(limit=max(1, min(limit, 500)), kind=kind),
        "tts": piper_worker().status(),
        "output": OUTPUT.stats(),
    })


@app.post("/api/trace/event")
def api_trace_event():
    body = request.get_json(force=True, silent=True) or {}
    trace_id = str(body.get("trace_id") or "").strip()
    stage = str(body.get("stage") or "").strip()
    allowed = {
        "play_start", "underrun", "tts_failed", "cancel", "client_vad_end",
        "client_capture_end", "client_request_start",
    }
    if not trace_id or stage not in allowed:
        return jsonify({"ok": False}), 400
    data = {k: v for k, v in body.items() if k not in ("trace_id", "stage", "at_ms")}
    if isinstance(body.get("at_ms"), (int, float)):
        data["wall_ms"] = int(body["at_ms"])
    TRACES.mark(trace_id, stage, **data)
    return jsonify({"ok": True})


@app.get("/api/health")
def health():
    """给 systemd/健康检查用的轻量接口，不触发模型加载。"""
    st = GUIDE.status()
    ok = (not piper_error()) and st.get("model", {}).get("ready", False)
    return jsonify(
        {
            "ok": ok,
            "guide_running": bool(st.get("running")),
            "guide_state": (st.get("decision") or {}).get("state"),
            "guide_reason": (st.get("decision") or {}).get("reason"),
            "memory_over_budget": over_budget(),
        }
    ), (200 if ok else 503)


@app.post("/api/vlm")
def api_vlm():
    if not budget_guard():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        body = request.get_json(force=True, silent=True) or {}
        mode = body.get("mode", "scene")
        prompt = body.get("prompt") or PROMPTS.get(mode) or PROMPTS["scene"]
        max_tokens = int(body.get("max_tokens", 256))
        try:
            img = decode_image(body.get("image", ""))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"图片解析失败：{exc}"}), 400

        img_b64 = image_to_jpeg_b64(img)
        save_debug_image("request.jpg", img)
        if not vlm_health():
            return jsonify({"error": "VLM 服务未启动（127.0.0.1:8091）"}), 503
        with VLM_LOCK:
            result = call_vlm(img_b64, prompt, max_tokens=max_tokens)
        log.info(
            "vlm mode=%s md5=%s latency=%sms answer=%r",
            mode, b64_md5(body.get("image", "")), result.get("latency_ms"),
            (result.get("answer") or "")[:160],
        )
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        log.exception("vlm failed")
        return jsonify({"error": f"VLM 调用失败：{exc}"}), 500
    finally:
        leave()


def run_eyes(task, item="", context="", question="", image_b64="", image_vlm_b64="", frame_id="", session_id="", captured_at="", trace_id=""):
    """统一执行入口，HTTP 路由和语音链路都走这里。"""
    spec = EYES_TASKS.get(task)
    if spec is None:
        raise ValueError(f"未知任务：{task}")
    if task in ("item", "compare") and not item:
        raise ValueError("缺少物品名称")
    if task == "followup" and not question:
        raise ValueError("缺少问题")

    t0 = time.time()
    session_begin(session_id, frame_id)

    age = frame_age_seconds(captured_at)
    if age is not None and age > FRAME_MAX_AGE and task not in ("chat", "followup"):
        raise ValueError("画面已过期，请重新拍一张再看")

    if task == "chat":
        if any(w in question for w in REALTIME_WORDS):
            session_remember(session_id, question, CHAT_REALTIME_REPLY)
            return {
                "task": task,
                "source": "rule",
                "answer": CHAT_REALTIME_REPLY,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        if not vlm_health():
            raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
        TRACES.mark(trace_id, "vlm_start", task="chat")
        with VLM_LOCK:
            out = call_vlm(
                None,
                prompt_for("chat", context=history_text(session_id), question=question),
                spec["max_tokens"],
                temperature=temperature_for(task),
            )
        if out.get("prefill_ms") is not None:
            TRACES.mark(trace_id, "prefill_ms", value_ms=out["prefill_ms"], image=False, task="chat")
        if out.get("decode_ms") is not None:
            TRACES.mark(trace_id, "decode_ms", value_ms=out["decode_ms"], task="chat")
        TRACES.mark(trace_id, "vlm_end", task="chat", latency_ms=out.get("latency_ms"))
        answer = out["answer"] or spec["fallback"]
        session_remember(session_id, question, answer)
        return {
            "task": task,
            "source": "vlm",
            "answer": answer,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    if task == "followup":
        age = frame_age_seconds(captured_at)
        img = None
        if (image_vlm_b64 or image_b64) and age is not None and age <= FRAME_MAX_AGE:
            try:
                img = vlm_image_b64(image_vlm_b64 or image_b64)
            except Exception:  # noqa: BLE001
                img = None
        ctx = session_context(session_id, context, include_visual=True)
        if not img:
            session_remember(session_id, question, NO_FRAME_REPLY, visual=False)
            return {
                "task": task,
                "source": "rule",
                "answer": NO_FRAME_REPLY,
                "has_image": False,
                "frame_age_s": round(age, 1) if age is not None else None,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        prompt = build_eyes_prompt("followup", context=ctx, question=question)
        if not vlm_health():
            raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
        TRACES.mark(trace_id, "vlm_start", task="followup")
        with VLM_LOCK:
            out = call_vlm(img, prompt, spec["max_tokens"], temperature=temperature_for(task))
        if out.get("prefill_ms") is not None:
            TRACES.mark(trace_id, "prefill_ms", value_ms=out["prefill_ms"], image=True, task="followup")
        if out.get("decode_ms") is not None:
            TRACES.mark(trace_id, "decode_ms", value_ms=out["decode_ms"], task="followup")
        TRACES.mark(trace_id, "vlm_end", task="followup", latency_ms=out.get("latency_ms"))
        answer = out["answer"] or spec["fallback"]
        session_remember(session_id, question, answer, visual=bool(img), frame_id=frame_id)
        return {
            "task": task,
            "source": "vlm",
            "answer": answer,
            "has_image": bool(img),
            "frame_age_s": round(age, 1) if age is not None else None,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    try:
        img = decode_image(image_b64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片解析失败：{exc}")

    if task == "date":
        if ocr_engine is None:
            raise RuntimeError("OCR 未初始化：" + str(init_error["ocr"]))
        save_debug_image("date.png", img)
        TRACES.mark(trace_id, "ocr_start", task="date")
        with OCR_LOCK:
            res = ocr_engine(img)
        TRACES.mark(trace_id, "ocr_end", task="date")
        parsed = parse_date_label(parse_ocr_result(res))
        if parsed is None:
            return {
                "task": task,
                "source": "ocr",
                "answer": spec["fallback"],
                "fields": {},
                "missing": [name for name, _ in DATE_FIELD_RULES],
                "latency_ms": int((time.time() - t0) * 1000),
            }
        session_remember(session_id, "读日期", parsed["answer"], visual=True, frame_id=frame_id)
        return {
            "task": task,
            "source": "ocr",
            "answer": parsed["answer"],
            "fields": parsed["fields"],
            "missing": parsed["missing"],
            "latency_ms": int((time.time() - t0) * 1000),
        }

    if task == "text" and ocr_engine is not None:
        save_debug_image("text.png", img)
        TRACES.mark(trace_id, "ocr_start", task="text")
        try:
            with OCR_LOCK:
                res = ocr_engine(img)
            text = pick_main_text_block(parse_ocr_result(res))
            if not meaningful_ocr_text(text):
                log.info("[eyes] OCR 只得到无意义短文本 %r，改用 VLM", text)
                text = ""
        except Exception as exc:  # noqa: BLE001
            log.warning("[eyes] OCR 失败，改用 VLM：%s", exc)
            text = ""
        TRACES.mark(trace_id, "ocr_end", task="text", chars=len(text))
        if text:
            session_remember(session_id, "读文字", text, visual=True, frame_id=frame_id)
            return {
                "task": task,
                "source": "ocr",
                "answer": text,
                "latency_ms": int((time.time() - t0) * 1000),
            }

    vlm_img = image_vlm_b64 or image_b64
    try:
        img_b64 = vlm_image_b64(vlm_img)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片解析失败：{exc}")
    if not vlm_health():
        raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
    TRACES.mark(trace_id, "vlm_start", task=task)
    with VLM_LOCK:
        out = call_vlm(img_b64, prompt_for(task, item, context, question), spec["max_tokens"], temperature=temperature_for(task))
    if out.get("prefill_ms") is not None:
        TRACES.mark(trace_id, "prefill_ms", value_ms=out["prefill_ms"], image=bool(img_b64), task=task)
    if out.get("decode_ms") is not None:
        TRACES.mark(trace_id, "decode_ms", value_ms=out["decode_ms"], task=task)
    TRACES.mark(trace_id, "vlm_end", task=task, latency_ms=out.get("latency_ms"))
    answer = settle_eyes_answer(task, out["answer"], spec["fallback"].format(item=item), item)
    session_remember(session_id, item or task, answer, visual=True, frame_id=frame_id)
    return {
        "task": task,
        "source": "vlm",
        "answer": answer,
        "latency_ms": int((time.time() - t0) * 1000),
    }


def finish_trace_after_request(trace_id: str) -> None:
    @after_this_request
    def _finish(resp):
        status = "ok" if resp.status_code < 400 else "error"
        TRACES.finish(trace_id, status=status, error="" if status == "ok" else str(resp.status_code))
        return resp


def push_answer_segments(answer, task, source, utterance_id=None, priority=PRIORITY_ANSWER, meta=None, trace_id=""):
    """把完整答案按语义片段推入输出总线；每段独立 TTS，不重播全文。"""
    answer = (answer or "").strip()
    if not answer:
        return None
    uid = utterance_id or OUTPUT.new_utterance_id("ans")
    pieces = segment_text(answer)
    total = len(pieces)
    for idx, piece in enumerate(pieces, 1):
        if idx == 1 and trace_id:
            TRACES.mark(trace_id, "first_segment", chars=len(piece), segments=total)
        OUTPUT.push_segment(
            piece,
            utterance_id=uid,
            segment_id=f"{uid}-{idx}",
            sequence=idx,
            is_final=(idx == total),
            priority=priority,
            source=source,
            reason=task,
            ttl=8.0,
            preempt=(idx == 1),
            deadline_s=3.0 if idx == 1 else 8.0,
            meta={**(meta or {}), "task": task, "segments": total, "answer": answer[:260], "request_id": trace_id},
        )
    return uid


@app.post("/api/talk")
def api_talk():
    """流式分句输出：边生成边推句，前端逐句合成播放。"""
    if not budget_guard():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429

    body = request.get_json(force=True, silent=True) or {}
    task = body.get("task", "chat")
    spec = EYES_TASKS.get(task)
    if spec is None or task == "date":
        leave()
        return jsonify({"error": f"任务不支持流式：{task}"}), 400

    item = str(body.get("item") or "").strip()
    question = str(body.get("question") or "").strip()
    session_id = str(body.get("session_id") or "").strip()
    trace_id = str(body.get("request_id") or "").strip() or OUTPUT.new_utterance_id("talk")
    TRACES.start(trace_id, "talk", session_id=session_id, task=task)
    if isinstance(body.get("request_start_ms"), (int, float)):
        TRACES.mark(trace_id, "request_start", wall_ms=int(body["request_start_ms"]))
    finish_trace_after_request(trace_id)
    has_image = bool(body.get("image_vlm") or body.get("image"))
    frame_id = canonical_frame_id(
        str(body.get("frame_id") or "").strip(),
        body.get("image_vlm") or body.get("image") or "",
    ) if has_image else ""
    session_begin(session_id, frame_id)
    context = session_context(session_id, body.get("context"))

    img_b64 = None
    if task != "chat":
        src = body.get("image_vlm") or body.get("image")
        if src:
            try:
                img_b64 = vlm_image_b64(src)
            except Exception:  # noqa: BLE001
                img_b64 = None
    if task not in ("chat", "followup") and not img_b64:
        leave()
        return jsonify({"error": "图片解析失败：缺少图片"}), 400

    age = frame_age_seconds(str(body.get("captured_at") or "").strip())
    if age is not None and age > FRAME_MAX_AGE and task not in ("chat", "followup"):
        leave()
        return jsonify({"error": "画面已过期，请重新拍一张再看"}), 400
    if not vlm_health():
        leave()
        return jsonify({"error": "VLM 服务未启动（127.0.0.1:8091）"}), 503

    if task == "chat" and any(w in question for w in REALTIME_WORDS):
        def gen_rule():
            log.info("talk task=chat rule=realtime q=%r", question)
            try:
                session_remember(session_id, question, CHAT_REALTIME_REPLY)
                uid = OUTPUT.new_utterance_id("talk")
                pieces = segment_text(CHAT_REALTIME_REPLY)
                total = len(pieces)
                for idx, piece in enumerate(pieces, 1):
                    OUTPUT.push_segment(
                        piece, utterance_id=uid, segment_id=f"{uid}-{idx}", sequence=idx,
                        is_final=(idx == total), priority=PRIORITY_ANSWER, source="chat",
                        reason="realtime", ttl=8.0, preempt=(idx == 1), deadline_s=3.0,
                        meta={"request_id": trace_id},
                    )
                    if idx == 1:
                        TRACES.mark(trace_id, "first_segment", chars=len(piece), segments=total)
                    payload = {
                        "segment": {
                            "text": piece, "utterance_id": uid, "segment_id": f"{uid}-{idx}",
                            "sequence": idx, "is_final": idx == total,
                        },
                        "sentence": piece, "trace_id": trace_id,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                TRACES.mark(trace_id, "answer", chars=len(CHAT_REALTIME_REPLY), source="rule")
                yield f"data: {json.dumps({'done': True, 'answer': CHAT_REALTIME_REPLY, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
            finally:
                leave()

        return Response(
            stream_with_context(gen_rule()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    prompt = prompt_for(task, item, context, question)
    max_tokens = spec["max_tokens"]
    t0 = time.time()
    utterance_id = OUTPUT.new_utterance_id("talk")

    def generate():
        buf, full = "", ""
        stream_seg = SemanticSegmenter()
        sequence = 0
        emitted = ""
        blocked = False
        if not VLM_LOCK.acquire(timeout=30):
            yield f"data: {json.dumps({'error': '设备正忙，请稍后再试'}, ensure_ascii=False)}\n\n"
            leave()
            return
        inner = None

        def emit(piece, final):
            nonlocal sequence, emitted, blocked
            piece = (piece or "").strip()
            if not piece or blocked:
                return None
            if is_degenerate_repetition(emitted + piece):
                piece = "画面重复太多，没看清，请再拍一张"
                blocked = True
                final = True
            # 守卫看“已播出 + 本段”的累计文本，避免跨段拼出“可以走……不能通行”。
            probe = (emitted + piece)[-500:]
            safe = guard_vision_answer(task, probe)
            if safe != probe:
                piece = safe
                blocked = True
                final = True
            sequence += 1
            if sequence == 1:
                TRACES.mark(trace_id, "first_segment", chars=len(piece), segments=None)
            OUTPUT.push_segment(
                piece, utterance_id=utterance_id, segment_id=f"{utterance_id}-{sequence}",
                sequence=sequence, is_final=final, priority=PRIORITY_ANSWER, source="chat",
                reason=task, ttl=8.0, preempt=(sequence == 1), deadline_s=3.0 if sequence == 1 else 8.0,
                meta={"task": task, "segment": sequence, "request_id": trace_id},
            )
            emitted += piece
            return {
                "segment": {
                    "text": piece, "utterance_id": utterance_id,
                    "segment_id": f"{utterance_id}-{sequence}", "sequence": sequence,
                    "is_final": final,
                },
                "sentence": piece,
                "trace_id": trace_id,
            }

        try:
            inner = stream_vlm(prompt, img_b64, max_tokens, temperature=temperature_for(task))
            first_token = False
            for delta in inner:
                if not first_token and delta:
                    TRACES.mark(trace_id, "first_token", chars=len(delta))
                    first_token = True
                buf += delta
                full += delta
                for piece in stream_seg.feed(delta):
                    payload = emit(piece, False)
                    if payload:
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if blocked:
                    break
            if not blocked:
                tail = stream_seg.flush()
                for idx, piece in enumerate(tail, 1):
                    payload = emit(piece, idx == len(tail))
                    if payload:
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            full = collapse_repetition(full)
            answer = settle_eyes_answer(task, full, spec["fallback"].format(item=item), item)
            answer = guard_vision_answer(task, answer)
            session_remember(session_id, question or item or task, answer, visual=bool(img_b64), frame_id=frame_id)
            TRACES.mark(trace_id, "answer", chars=len(answer), source="vlm")
            if not emitted and answer:
                # VLM 返回空文本或只有兜底句时，至少播报一次最终答案。
                for idx, piece in enumerate(segment_text(answer)):
                    payload = emit(piece, True)
                    if payload:
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, 'answer': answer, 'utterance_id': utterance_id, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            log.warning("talk stream failed: %s", exc)
            yield f"data: {json.dumps({'error': str(exc), 'utterance_id': utterance_id}, ensure_ascii=False)}\n\n"
        finally:
            if inner is not None:
                inner.close()
            VLM_LOCK.release()
            leave()
            log.info(
                "talk task=%s session=%s frame=%s q=%r latency=%sms segments=%d answer=%r",
                task, session_id, frame_id, question[:80],
                int((time.time() - t0) * 1000), sequence, full[:160],
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/eyes")
def api_eyes():
    if not budget_guard():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        body = request.get_json(force=True, silent=True) or {}
        task = str(body.get("task", "scene"))
        item = str(body.get("item") or "").strip()
        question = str(body.get("question") or "").strip()
        image = body.get("image", "")
        image_vlm = body.get("image_vlm", "")
        has_image = bool(image or image_vlm)
        frame_id = canonical_frame_id(
            str(body.get("frame_id") or "").strip(),
            image_vlm or image,
        ) if has_image else ""
        session_id = str(body.get("session_id") or "").strip()
        captured_at = str(body.get("captured_at") or "").strip()
        trace_id = str(body.get("request_id") or "").strip() or OUTPUT.new_utterance_id("eyes")
        TRACES.start(trace_id, "eyes", session_id=session_id, task=task, has_image=has_image)
        if isinstance(body.get("request_start_ms"), (int, float)):
            TRACES.mark(trace_id, "request_start", wall_ms=int(body["request_start_ms"]))
        finish_trace_after_request(trace_id)
        result = run_eyes(
            task,
            item=item,
            context=str(body.get("context") or "").strip(),
            question=question,
            image_b64=image,
            image_vlm_b64=image_vlm,
            frame_id=frame_id,
            session_id=session_id,
            captured_at=captured_at,
            trace_id=trace_id,
        )
        result["answer"] = guard_vision_answer(task, result.get("answer", ""))
        TRACES.mark(trace_id, "answer", chars=len(result["answer"]), source=result.get("source"))
        push_answer_segments(
            result["answer"], task, "eyes",
            meta={"frame_id": frame_id, "request_id": trace_id},
            trace_id=trace_id,
        )
        result["frame_id"] = frame_id
        result["trace_id"] = trace_id
        log.info(
            "eyes task=%s source=%s item=%r question=%r frame=%s captured=%s age=%s session=%s image_source=%s full=%s vlm=%s latency=%sms answer=%r",
            task, result.get("source"), item, question, frame_id, captured_at,
            frame_age_seconds(captured_at), session_id,
            body.get("image_source") or "-", b64_md5(image), b64_md5(image_vlm),
            result.get("latency_ms"), (result.get("answer") or "")[:160],
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        log.exception("eyes failed")
        return jsonify({"error": f"识别失败：{exc}"}), 500
    finally:
        leave()


# 唤醒词规则收敛到 intent 模块，服务端仍保留同名别名给旧测试/脚本使用。
WAKE_PATTERN = intent.WAKE_PATTERN
ACK_ONLY = intent.ACK_ONLY
ASR_ALIASES = intent.ASR_ALIASES
FOLLOWUP_MARKER = intent.FOLLOW_MARKERS


def normalize_voice_text(text):
    return intent.normalize_text(text)


def parse_voice_intent(cmd, has_context=False):
    """兼容旧测试的二元返回；新语音链路用 intent.parse 的结构化任务。"""
    return intent.parse_legacy(cmd, has_context=has_context)


KWS_LOCK = threading.Lock()


def recent_speech_active(max_age=8.0):
    """只把最近 8 秒内的回答播报当作“正在播报”；确认语和引导提示不算。"""
    stats = OUTPUT.stats()
    age = stats.get("last_event_age_s")
    last = stats.get("last_event") or {}
    return bool(
        age is not None and age <= max_age
        and last.get("priority", 99) >= PRIORITY_ANSWER
        and last.get("source") != "guide"
    )


def voice_reply(answer, intent_name, **extra):
    payload = {
        "text": extra.pop("text", ""),
        "intent": intent_name,
        "answer": answer,
        "latency_ms": extra.pop("latency_ms", 0),
    }
    payload.update(extra)
    return jsonify(payload)


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
    return samples, sr


def asr_transcribe(samples):
    engine = ensure_asr()
    stream = engine.create_stream()
    stream.accept_waveform(16000, samples)
    engine.decode_stream(stream)
    return (stream.result.text or "").strip()


@app.post("/api/voice")
def api_voice():
    if not budget_guard():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    body = request.get_json(force=True, silent=True) or {}
    t0 = time.time()
    session_id = str(body.get("session_id") or "").strip()
    trace_id = str(body.get("request_id") or "").strip() or OUTPUT.new_utterance_id("voice")
    TRACES.start(trace_id, "voice", session_id=session_id, has_image=bool(body.get("image") or body.get("image_vlm")))
    if isinstance(body.get("request_start_ms"), (int, float)):
        TRACES.mark(trace_id, "request_start", wall_ms=int(body["request_start_ms"]))
    if isinstance(body.get("speech_end_ms"), (int, float)):
        TRACES.mark(trace_id, "speech_end", wall_ms=int(body["speech_end_ms"]))
    if isinstance(body.get("vad_end_ms"), (int, float)):
        TRACES.mark(trace_id, "vad_end", wall_ms=int(body["vad_end_ms"]))
    finish_trace_after_request(trace_id)
    sess = get_session(session_id)

    try:
        samples, src_sr = decode_wav(body.get("audio", ""))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"音频解析失败：{exc}"}), 400

    duration = len(samples) / 16000
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    if duration < MIN_VOICE_SECONDS or rms < MIN_VOICE_RMS:
        log.info("voice skip dur=%.1fs rms=%.4f", duration, rms)
        return voice_reply("", "silent", latency_ms=int((time.time() - t0) * 1000), spoken=False)

    try:
        text = asr_transcribe(samples)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"语音识别失败：{exc}"}), 500
    log.info("voice asr sr=%s dur=%.1fs rms=%.4f text=%r", src_sr, duration, rms, text)
    TRACES.mark(trace_id, "asr_end", text=text, duration_s=round(duration, 2), rms=round(rms, 4))
    if not text:
        log.info("voice empty dur=%.1fs rms=%.4f", duration, rms)
        return voice_reply("", "silent", latency_ms=int((time.time() - t0) * 1000), spoken=False)

    now = time.time()
    armed = now < sess["armed_until"]
    wake_detected = bool(body.get("wake_detected"))
    text_wake = bool(WAKE_PATTERN.search(text))
    has_wake = text_wake or wake_detected
    stripped = WAKE_PATTERN.sub("", text) if text_wake else text
    cmd = intent.clean_command(stripped)

    if has_wake:
        with SESSION_LOCK:
            sess["armed_until"] = now + ARMED_SECONDS

    if not cmd or ACK_ONLY.fullmatch(cmd):
        if has_wake:
            log.info("voice wake text=%r kws=%s", text, wake_detected)
            OUTPUT.push(
                "我在，请说", priority=PRIORITY_CONFIRM, source="voice",
                reason="wake", ttl=2.0, dedupe_key="voice-wake",
            )
            return voice_reply(
                "我在，请说", "wake", text=text,
                latency_ms=int((time.time() - t0) * 1000), spoken=True, armed=True,
            )
        if armed:
            log.info("voice ack text=%r", text)
            return voice_reply("", "ack", text=text, latency_ms=int((time.time() - t0) * 1000), spoken=False, armed=True)
        return voice_reply("", "none", text=text, latency_ms=int((time.time() - t0) * 1000), spoken=False, armed=False)

    if not has_wake and not armed:
        log.info("voice no-wake text=%r", text)
        return voice_reply("", "none", text=text, latency_ms=int((time.time() - t0) * 1000), spoken=False, armed=False)

    if cmd == sess["last_cmd"] and now - sess["last_cmd_at"] < 2.0:
        log.info("voice duplicate text=%r", text)
        return voice_reply("", "ack", text=text, latency_ms=int((time.time() - t0) * 1000), spoken=False, armed=True)

    has_context = bool(history_text(session_id))
    guide_active = bool(GUIDE.status().get("running"))
    task = intent.parse(
        cmd,
        has_context=has_context,
        guide_active=guide_active,
        speech_active=recent_speech_active(),
        session_id=session_id,
    )
    TRACES.mark(trace_id, "intent_end", intent=task.intent, confidence=task.confidence)
    with SESSION_LOCK:
        sess["armed_until"] = 0.0
        sess["last_cmd"] = cmd
        sess["last_cmd_at"] = now

    base = {
        "text": text,
        "cmd": cmd,
        "task": task.to_dict(),
        "trace_id": trace_id,
        "latency_ms": int((time.time() - t0) * 1000),
    }

    if task.intent == "speech_stop":
        OUTPUT.cancel(min_priority=PRIORITY_GUIDE, source="voice")
        log.info("voice action=speech_stop session=%s", session_id)
        return voice_reply("已停止播报", "speech_stop", spoken=False, cancelled=True, **base)

    if task.intent == "guide_stop":
        if guide_active:
            GUIDE.stop(announce=True)
            answer = "已停止引导"
        else:
            answer = "引导现在没有开始"
            OUTPUT.push(answer, priority=PRIORITY_CONFIRM, source="voice", reason="guide_stop", ttl=2.0)
        log.info("voice action=guide_stop session=%s", session_id)
        return voice_reply(answer, "guide_stop", spoken=True, guide=GUIDE.status(), **base)

    if task.intent == "guide_start":
        source = default_guide_source()
        if source == "device":
            if not DEVICE_CAMERA.available():
                return jsonify({"error": "Picamera2 不可用，无法使用设备摄像头引导"}), 503
            if not DEVICE_CAMERA.start(_on_device_frame):
                return jsonify({"error": DEVICE_CAMERA.status().get("error") or "摄像头启动失败"}), 503
        try:
            status = GUIDE.start(source)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        answer = "已开始引导，请保持摄像头画面" if status.get("needs_frames") else "已开始引导"
        OUTPUT.push(answer, priority=PRIORITY_CONFIRM, source="voice", reason="guide_start", ttl=3.0)
        log.info("voice action=guide_start source=%s session=%s needs_frames=%s", source, session_id, status.get("needs_frames"))
        return voice_reply(answer, "guide_start", spoken=True, guide=status, **base)

    if task.intent == "camera_off":
        if guide_active:
            GUIDE.stop(announce=False)
        DEVICE_CAMERA.stop()
        answer = "已停止引导，请关闭摄像头"
        OUTPUT.push(answer, priority=PRIORITY_CONFIRM, source="voice", reason="camera_off", ttl=2.0)
        log.info("voice action=camera_off session=%s", session_id)
        return voice_reply(answer, "camera_off", spoken=True, close_camera=True, **base)

    if task.intent == "clarify":
        answer = task.clarify or "请再说一遍。"
        OUTPUT.push(answer, priority=PRIORITY_CONFIRM, source="voice", reason="clarify", ttl=3.0)
        log.info("voice clarify cmd=%r answer=%r", cmd, answer)
        return voice_reply(answer, "clarify", spoken=True, **base)

    if task.intent == "unknown":
        answer = task.clarify or "我没听清，请再说一遍。"
        OUTPUT.push(answer, priority=PRIORITY_CONFIRM, source="voice", reason="unknown", ttl=3.0)
        log.info("voice unknown cmd=%r", cmd)
        return voice_reply(answer, "unknown", spoken=True, **base)

    if task.intent == "ack":
        return voice_reply("", "ack", spoken=False, **base)

    if task.intent == "guide_status":
        st = GUIDE.status()
        d = st.get("decision") or {}
        age_ms = st.get("frame_age_ms")
        if not st.get("running"):
            answer = "引导没有运行，不能判断能不能走"
        elif age_ms is None or age_ms > int(guide.FRAME_MAX_AGE * 1000 * 2):
            answer = "引导画面已过期，不能判断，请先停下"
        elif d.get("state") == "stop":
            answer = guide.REASON_TEXT.get(d.get("reason"), "引导提示前方有危险，请停下")
        elif d.get("state") == "left":
            answer = "引导判断当前向左一点，请用盲杖确认"
        elif d.get("state") == "right":
            answer = "引导判断当前向右一点，请用盲杖确认"
        elif d.get("state") == "straight":
            answer = "引导判断当前直行通道可用，请用盲杖确认"
        else:
            answer = "引导画面当前不可用，不能判断，请先停下"
        TRACES.mark(trace_id, "answer", chars=len(answer), source="guide")
        push_answer_segments(
            answer, "guide_status", "guide_status",
            priority=PRIORITY_CONFIRM, meta={"frame_id": st.get("frame_id", ""), "request_id": trace_id},
            trace_id=trace_id,
        )
        log.info("voice guide_status state=%s reason=%s age=%sms", d.get("state"), d.get("reason"), age_ms)
        return voice_reply(answer, "guide_status", spoken=True, guide=st, **base)

    if "stop_guide" in task.actions and guide_active:
        GUIDE.stop(announce=False)

    img = body.get("image", "")
    img_vlm = body.get("image_vlm", "")
    has_image = bool(img or img_vlm)
    frame_id = canonical_frame_id(str(body.get("frame_id") or "").strip(), img or img_vlm) if has_image else ""
    captured_at = str(body.get("captured_at") or "").strip()
    context = str(body.get("context") or "").strip()

    if task.needs_image and not (img or img_vlm):
        answer = NO_FRAME_REPLY if task.intent == "followup" else "还没有画面，请先拍一张或打开相机"
        OUTPUT.push(answer, priority=PRIORITY_CONFIRM, source="voice", reason="no_image", ttl=3.0)
        return voice_reply(answer, task.intent, spoken=True, frame_id=frame_id, **base)

    if not try_enter("heavy"):
        return jsonify({"error": "设备正忙，请稍后再试"}), 429

    try:
        if task.intent == "scene":
            result = run_eyes("scene", image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
        elif task.intent == "detail":
            result = run_eyes("detail", image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
        elif task.intent == "read_text":
            result = run_eyes("text", image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
        elif task.intent == "read_date":
            result = run_eyes("date", image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
        elif task.intent == "find_object":
            target = task.slots.get("target")
            if target in ("toilet", "exit"):
                result = run_eyes(target, image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
            else:
                result = run_eyes("item", item=task.slots.get("item", ""), image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
        elif task.intent == "compare":
            result = run_eyes("compare", item=task.slots.get("item", ""), image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id, session_id=session_id, captured_at=captured_at, trace_id=trace_id)
        elif task.intent == "followup":
            result = run_eyes(
                "followup", context=context, question=task.slots.get("question", cmd),
                image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id,
                session_id=session_id, captured_at=captured_at, trace_id=trace_id,
            )
        elif task.intent == "chat":
            result = run_eyes("chat", question=cmd, frame_id=frame_id, session_id=session_id, trace_id=trace_id)
        else:
            raise ValueError(f"任务未实现：{task.intent}")
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("voice failed")
        return jsonify({"error": f"识别失败：{exc}"}), 500
    finally:
        leave("heavy")

    eyes_task = result.get("task") or task.intent
    answer = guard_vision_answer(eyes_task, result["answer"])
    result["answer"] = answer
    TRACES.mark(trace_id, "answer", chars=len(answer), source=result.get("source"))
    push_answer_segments(
        answer, eyes_task, "voice",
        meta={"frame_id": frame_id, "request_id": trace_id},
        trace_id=trace_id,
    )

    log.info(
        "voice intent=%s item=%r armed=%s frame=%s captured=%s session=%s image_source=%s full=%s vlm=%s source=%s latency=%sms answer=%r text=%r",
        task.intent, task.slots.get("item", ""), armed, frame_id, captured_at, session_id,
        body.get("image_source") or "-", b64_md5(img), b64_md5(img_vlm),
        result["source"], int((time.time() - t0) * 1000), answer[:160], text,
    )
    return jsonify(
        {
            "text": text,
            "cmd": cmd,
            "task": task.to_dict(),
            "intent": task.intent,
            "item": task.slots.get("item", ""),
            "source": result["source"],
            "answer": answer,
            "frame_id": frame_id,
            "trace_id": trace_id,
            "latency_ms": int((time.time() - t0) * 1000),
            "spoken": True,
            "armed": False,
        }
    )


@app.post("/api/kws")
def api_kws():
    body = request.get_json(force=True, silent=True) or {}
    try:
        samples, _ = decode_wav(body.get("audio", ""))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"音频解析失败：{exc}"}), 400
    if len(samples) / 16000 < 0.3:
        return jsonify({"keyword": "", "hits": [], "latency_ms": 0, "ready": kws.status()["ready"]})
    with KWS_LOCK:
        result = kws.detect(samples)
    if result.get("error"):
        return jsonify({"error": result["error"], **result}), 503
    log.info("kws hits=%s latency=%sms", result.get("hits"), result.get("latency_ms"))
    return jsonify(result)


@app.post("/api/ocr")
def api_ocr():
    if ocr_engine is None:
        return jsonify({"error": "OCR 未初始化：" + str(init_error["ocr"])}), 503
    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        body = request.get_json(force=True, silent=True) or {}
        try:
            img = decode_image(body.get("image", ""))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"图片解析失败：{exc}"}), 400

        save_debug_image("ocr.png", img)
        t0 = time.time()
        with OCR_LOCK:
            res = ocr_engine(img)
        latency = int((time.time() - t0) * 1000)
        lines = parse_ocr_result(res)
        full_text = "\n".join(item["text"] for item in lines)
        log.info("ocr lines=%d chars=%d latency=%sms", len(lines), len(full_text), latency)
        return jsonify({"text": full_text, "lines": lines, "latency_ms": latency})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"OCR 失败：{exc}"}), 500
    finally:
        leave()


@app.post("/api/guide/start")
def api_guide_start():
    body = request.get_json(force=True, silent=True) or {}
    source = (body.get("source") or default_guide_source()).strip().lower()
    if source == "device":
        if not DEVICE_CAMERA.available():
            return jsonify({"error": "Picamera2 不可用，无法使用设备摄像头引导"}), 503
        if not DEVICE_CAMERA.start(_on_device_frame):
            return jsonify({"error": DEVICE_CAMERA.status().get("error") or "摄像头启动失败"}), 503
    try:
        status = GUIDE.start(source)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    log.info("guide start source=%s needs_frames=%s", source, status.get("needs_frames"))
    return jsonify(status)


@app.post("/api/guide/stop")
def api_guide_stop():
    body = request.get_json(force=True, silent=True) or {}
    reason = (body.get("reason") or "manual").strip()
    status = GUIDE.stop(reason=reason, announce=True)
    # 设备语音仍需要摄像头抓当前画面；只有摄像头仅因引导而启动时才随引导关闭。
    if status.get("mode") == "device" and VOICE_SOURCE != "device":
        DEVICE_CAMERA.stop()
    log.info("guide stop reason=%s", reason)
    return jsonify(status)


@app.get("/api/device/status")
def api_device_status():
    return jsonify(device_status())


@app.get("/api/camera/still")
def api_camera_still():
    """设备端高清抓图，供调试、OCR 验收和回放取样；不重新打开摄像头。"""
    st = DEVICE_CAMERA.status()
    if not st.get("running"):
        return jsonify({"error": "设备摄像头未运行"}), 409
    img = DEVICE_CAMERA.capture_still()
    if img is None:
        return jsonify({"error": DEVICE_CAMERA.status().get("error") or "取帧失败"}), 503
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    fid = f"still-{time.time_ns()}"
    captured = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Response(
        buf.getvalue(),
        mimetype="image/jpeg",
        headers={"X-Frame-Id": fid, "X-Captured-At": captured, "Cache-Control": "no-store"},
    )


@app.post("/api/guide/frame")
def api_guide_frame():
    if not GUIDE.status().get("running"):
        return jsonify({"error": "引导未开始"}), 409
    body = request.get_json(force=True, silent=True) or {}
    image = body.get("image", "")
    try:
        img = decode_image(image)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"图片解析失败：{exc}"}), 400
    fid = canonical_frame_id(str(body.get("frame_id") or "").strip(), image)
    result = GUIDE.submit_frame(
        img,
        frame_id=fid,
        captured_at=str(body.get("captured_at") or "").strip(),
        source=str(body.get("source") or "browser"),
    )
    return jsonify(result)


@app.get("/api/guide/status")
def api_guide_status():
    return jsonify(GUIDE.status())


@app.post("/api/guide")
def api_guide():
    """单帧调试/离线回放入口；不进入实时状态机。"""
    if not try_enter("heavy"):
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        body = request.get_json(force=True, silent=True) or {}
        image = body.get("image", "")
        img = decode_image(image)
        frame_id = canonical_frame_id(str(body.get("frame_id") or "").strip(), image)
        result = guide.analyze_image(img, frame={"frame_id": frame_id})
        m = result.get("metrics") or {}
        log.info(
            "guide raw state=%s reason=%s frame=%s image=%s quality=%s infer=%sms",
            result.get("raw_direction"), result.get("raw_reason"), frame_id, b64_md5(image),
            (result.get("quality") or {}).get("reasons"), (result.get("timing") or {}).get("inference_ms"),
        )
        result["frame_id"] = frame_id
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        log.exception("guide failed")
        return jsonify({"error": f"引导识别失败：{exc}"}), 500
    finally:
        leave("heavy")


@app.post("/api/output/cancel")
def api_output_cancel():
    """停止普通播报；危险提示仍可继续抢占，引导检测不受影响。"""
    body = request.get_json(force=True, silent=True) or {}
    min_priority = int(body.get("min_priority", PRIORITY_GUIDE))
    event = OUTPUT.cancel(min_priority=min_priority, source="browser")
    log.info("output cancel min_priority=%s", min_priority)
    return jsonify(event)


@app.get("/api/output/stream")
def api_output_stream():
    last_id = request.headers.get("Last-Event-ID") or request.args.get("last_event_id") or ""
    try:
        since = int(last_id) if last_id else None
    except ValueError:
        since = None
    q = OUTPUT.subscribe()

    def sse(item: dict) -> str:
        if item.get("id") is not None:
            return f"id: {item['id']}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    def generate():
        yield "retry: 3000\n\n"
        for item in OUTPUT.history_since(since):
            yield sse(item)
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield sse(item)
        except GeneratorExit:
            pass
        finally:
            OUTPUT.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tts")
def api_tts():
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()
    trace_id = str(body.get("trace_id") or "").strip()
    if not text:
        return jsonify({"error": "文本为空"}), 400
    if len(text) > 200:
        return jsonify({"error": "单句过长，请按句子分段合成", "max_len": 200}), 400
    engine = (body.get("engine") or "piper").lower()
    speed = float(body.get("speed", 1.0))
    if engine != "piper":
        return jsonify({"error": f"未知 TTS 引擎：{engine}"}), 400
    if trace_id:
        TRACES.mark(trace_id, "tts_start", chars=len(text))
    if not try_enter("light"):
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        with TTS_LOCK:
            resp = piper_tts(text, speed)
        if trace_id:
            TRACES.mark(
                trace_id, "tts_ready",
                value_ms=int(resp.headers.get("X-Gen-Ms") or 0),
                cache=resp.headers.get("X-TTS-Cache"),
                chars=len(text),
            )
        return resp
    except TTSInterrupted as exc:
        if trace_id:
            TRACES.mark(trace_id, "tts_failed", error=str(exc))
        return jsonify({"error": str(exc)}), 499
    except RuntimeError as exc:
        if trace_id:
            TRACES.mark(trace_id, "tts_failed", error=str(exc))
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        if trace_id:
            TRACES.mark(trace_id, "tts_failed", error=str(exc))
        return jsonify({"error": f"TTS 失败：{exc}"}), 500
    finally:
        leave("light")


@app.get("/api/sample/<path:name>")
def sample(name: str):
    return send_from_directory(SAMPLES_DIR, name)


if __name__ == "__main__":
    log.info(
        "start vlm=%s auth=%s budget=%.1fGB frame_max_age=%.0fs snapshot=%s voice=%s guide=%s sink=%s data=%s log=%s",
        VLM_URL, "key" if VLM_API_KEY else "none",
        MEM_BUDGET_GB, FRAME_MAX_AGE, DEBUG_SNAPSHOT,
        VOICE_SOURCE, GUIDE_SOURCE, OUTPUT_SINK, DATA_DIR, LOG_FILE,
    )
    atexit.register(lambda: log.info("web exit"))
    atexit.register(stop_device_services)
    atexit.register(lambda: _PIPER_WORKER.stop() if _PIPER_WORKER is not None else None)
    log.info("piper=%s", "ready" if not piper_error() else piper_error())
    threading.Thread(target=init_ocr, daemon=True).start()
    threading.Thread(target=asr_idle_reaper, daemon=True).start()
    if not piper_error():
        threading.Thread(target=prewarm_tts, daemon=True).start()
    start_device_services()
    time.sleep(1)
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8090"))
    app.run(host=host, port=port, threaded=True, debug=False)
