# -*- coding: utf-8 -*-
"""
CM5 边缘 AI 本地测试服务
- VLM：调用本机 llama-server（MiniCPM-V 4.7 1B，OpenAI 兼容接口）
- OCR：RapidOCR + PP-OCRv6_small（ONNX Runtime）
- TTS：Piper 中文神经语音（zh_CN-huayan-medium，子进程按需合成），可选 Kokoro-82M-v1.1-zh（int8）
"""

import base64
import gc
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import psutil
import requests
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
from PIL import Image
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OCR_DIR = MODELS / "ocr" / "pp-ocrv6-small"
TTS_DIR = MODELS / "tts" / "kokoro-int8-multi-lang-v1_1"
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
DATA_DIR = Path(os.environ.get("DATA_DIR") or (ROOT / "data"))
TTS_CACHE_DIR = DATA_DIR / "tts-cache"
SAMPLES_DIR = ROOT / "samples"
DEBUG_SNAPSHOT = os.environ.get("DEBUG_SNAPSHOT", "0").strip().lower() in ("1", "true", "yes")

# 线程数可配置：默认值与现状一致，CM5 上按实测再调，不做绑核
TTS_THREADS = int(os.environ.get("TTS_THREADS", "4"))
ASR_THREADS = int(os.environ.get("ASR_THREADS", "4"))
OCR_INTRA_THREADS = int(os.environ.get("OCR_INTRA_THREADS", "4"))
OCR_INTER_THREADS = int(os.environ.get("OCR_INTER_THREADS", "1"))
FRAME_MAX_AGE = float(os.environ.get("FRAME_MAX_AGE", "20"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


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
kokoro_engine = None
asr_engine = None
asr_last_used = 0.0
init_error = {"ocr": None, "kokoro": None, "asr": None}

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
    "你是视障人士的AI助手。用1到2句口语中文回答，不分点、不加括号。"
    "方位只说左/中/右，远近只说近/远。"
    "只说画面里看得见的；看不清、被挡住、没看到就直说，不猜、不补常识。"
    "不报米数、不报秒数，不说能不能走、能不能过马路。"
)

CHAT_RULE = (
    "你是陪伴视障人士的中文语音助手。用口语中文回答，两到四句，不分点、不加括号。"
    "不要描述画面，也不要提眼睛或视力。"
    "不编造事实，不假装能查询实时信息。"
    "如果对方问新闻、天气或实时数据，直接说明没有联网查询能力，请他看手机或问身边的人。"
    "涉及安全或医疗时只给一般性建议，并提醒找专业人员。"
)

DETAIL_RULE = (
    "你是视障人士的AI助手。用三到五句口语中文描述，不分点、不加括号。"
    "方位只说左/中/右，远近只说近/远。"
    "只说画面里看得见的；看不清就直说，不猜、不补常识。"
    "不报米数、不报秒数，不说能不能走、能不能过马路。"
)

FOLLOW_RULE = (
    "你是视障人士的AI助手。用一到两句口语中文回答，不分点、不加括号。"
    "给了画面就结合画面回答，只用能确认的信息；看不清、被挡住、没看到就直说，不猜。"
    "方位只说左/中/右，远近只说近/远，不报米数、不报秒数，不说能不能走。"
)

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
            "看图，用一到两句说清正前方从近到远有什么。"
            "如果画面主要是文字、标牌或屏幕，就照原文读出最醒目的文字，不要编造人物、车辆、台阶或道路。"
            "没有明显障碍就说“前方没看到明显障碍”。"
            "不要分点、不要换行。"
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
MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS = 480


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
            sess = {"frame_id": "", "history": [], "updated": now}
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


def history_text(session_id):
    sess = get_session(session_id)
    with SESSION_LOCK:
        parts = []
        for item in sess["history"]:
            tag = "（旧画面已失效）" if item.get("stale") else ""
            parts.append(f"{tag}问：{item['q']} 答：{item['a']}")
        return "；".join(parts)


def session_context(session_id, client_context=""):
    """服务端历史优先；客户端 context 只在无历史时兜底，不能绕过失效标记。"""
    server = history_text(session_id)
    return server if server else (client_context or "").strip()


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


# ---------- 重任务并发上限 ----------
ACTIVE_LOCK = threading.Lock()
ACTIVE = {"n": 0}
MAX_ACTIVE = 3  # 流式聊天 + 当前句 TTS + 预取下一句，避免预取被 429 丢弃


def try_enter():
    with ACTIVE_LOCK:
        if ACTIVE["n"] >= MAX_ACTIVE:
            return False
        ACTIVE["n"] += 1
        return True


def leave():
    with ACTIVE_LOCK:
        ACTIVE["n"] = max(0, ACTIVE["n"] - 1)


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


def piper_tts(text, speed=1.0):
    """调用 Piper 子进程合成 wav；带磁盘缓存，固定短语第二次 0 ms。"""
    err = piper_error()
    if err:
        raise RuntimeError("Piper 未就绪：" + err)
    try:
        length_scale = 1.0 / max(0.5, min(2.0, float(speed or 1.0)))
    except (TypeError, ValueError):
        length_scale = 1.0
    key = hashlib.sha1(
        f"piper|{PIPER_MODEL.name}|{PIPER_MODEL.stat().st_size}|{length_scale:.3f}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    cache_path = TTS_CACHE_DIR / f"piper-{key}.wav"
    hit = cache_path.exists() and cache_path.stat().st_size > 1024
    t0 = time.time()
    if hit:
        data = cache_path.read_bytes()
    else:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            proc = subprocess.run(
                [
                    str(PIPER_BIN), "-m", str(PIPER_MODEL), "-f", tmp,
                    "--length_scale", f"{length_scale:.3f}",
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=90,
                cwd=str(PIPER_BIN.parent),
            )
            if proc.returncode != 0 or os.path.getsize(tmp) < 1024:
                tail = (proc.stderr or b"").decode("utf-8", "ignore")[-300:]
                raise RuntimeError(f"Piper 合成失败：{tail}")
            data = Path(tmp).read_bytes()
            cache_path.write_bytes(data)
            prune_tts_cache()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    with wave.open(io.BytesIO(data), "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate() or 1)
    gen_ms = 0 if hit else int((time.time() - t0) * 1000)
    log.info(
        "tts engine=piper cache=%s chars=%d gen=%sms audio=%.2fs",
        "hit" if hit else "miss", len(text), gen_ms, duration,
    )
    resp = Response(data, mimetype="audio/wav")
    resp.headers["X-Gen-Ms"] = str(gen_ms)
    resp.headers["X-Audio-Seconds"] = str(round(duration, 2))
    resp.headers["X-TTS-Engine"] = "piper"
    resp.headers["X-TTS-Cache"] = "hit" if hit else "miss"
    return resp


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
            num_threads=TTS_THREADS,
        ),
        rule_fsts=",".join(
            str(TTS_DIR / name) for name in ("number-zh.fst", "phone-zh.fst", "date-zh.fst")
        ),
        max_num_sentences=1,
    )
    kokoro_engine = sherpa_onnx.OfflineTts(cfg)
    log.info("[init] Kokoro ready")
    return kokoro_engine


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


ASR_IDLE_SECONDS = 60


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


def temperature_for(task):
    return 0.4 if task == "chat" else 0.1


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
    with requests.post(VLM_URL + "/v1/chat/completions", json=payload, stream=True, timeout=300) as resp:
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


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    app_rss, vlm_rss = model_memory_bytes()
    total = app_rss + vlm_rss
    cpu_percent = model_cpu_percent()
    return jsonify(
        {
            "vlm": {"up": vlm_health(), "url": VLM_URL},
            "ocr": {"ready": ocr_engine is not None, "error": init_error["ocr"]},
            "asr": {"ready": asr_engine is not None, "lazy": True, "error": init_error["asr"]},
            "tts": {
                "ready": not piper_error(),
                "engine": "piper",
                "error": piper_error() or None,
                "kokoro_ready": kokoro_engine is not None,
                "kokoro_error": init_error["kokoro"],
            },
            "memory": {
                "model_gb": round(vlm_rss / 1024 ** 3, 2),
                "app_gb": round(app_rss / 1024 ** 3, 2),
                "sum_gb": round(total / 1024 ** 3, 2),
                "budget_gb": MEM_BUDGET_GB,
                "over_budget": total > MEM_BUDGET_GB * 1024 ** 3,
            },
            "cpu": {
                "model_percent": cpu_percent,
                "cores": psutil.cpu_count() or 1,
            },
        }
    )


@app.post("/api/vlm")
def api_vlm():
    if over_budget():
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


def run_eyes(task, item="", context="", question="", image_b64="", image_vlm_b64="", frame_id="", session_id="", captured_at=""):
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
        with VLM_LOCK:
            out = call_vlm(
                None,
                prompt_for("chat", context=history_text(session_id), question=question),
                spec["max_tokens"],
                temperature=temperature_for(task),
            )
        answer = out["answer"] or spec["fallback"]
        session_remember(session_id, question, answer)
        return {
            "task": task,
            "source": "vlm",
            "answer": answer,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    if task == "followup":
        ctx = session_context(session_id, context)
        prompt = build_eyes_prompt("followup", context=ctx, question=question)
        img = None
        if image_vlm_b64 or image_b64:
            try:
                img = image_to_jpeg_b64(decode_image(image_vlm_b64 or image_b64))
            except Exception:  # noqa: BLE001
                img = None
        if not vlm_health():
            raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
        with VLM_LOCK:
            out = call_vlm(img, prompt, spec["max_tokens"], temperature=temperature_for(task))
        answer = out["answer"] or spec["fallback"]
        session_remember(session_id, question, answer, visual=bool(img), frame_id=frame_id)
        return {
            "task": task,
            "source": "vlm",
            "answer": answer,
            "has_image": bool(img),
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
        with OCR_LOCK:
            res = ocr_engine(img)
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
        try:
            with OCR_LOCK:
                res = ocr_engine(img)
            text = pick_main_text_block(parse_ocr_result(res))
        except Exception as exc:  # noqa: BLE001
            log.warning("[eyes] OCR 失败，改用 VLM：%s", exc)
            text = ""
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
        img_b64 = image_to_jpeg_b64(decode_image(vlm_img))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片解析失败：{exc}")
    if not vlm_health():
        raise RuntimeError("VLM 服务未启动（127.0.0.1:8091）")
    with VLM_LOCK:
        out = call_vlm(img_b64, prompt_for(task, item, context, question), spec["max_tokens"], temperature=temperature_for(task))
    answer = settle_eyes_answer(task, out["answer"], spec["fallback"].format(item=item), item)
    session_remember(session_id, item or task, answer, visual=True, frame_id=frame_id)
    return {
        "task": task,
        "source": "vlm",
        "answer": answer,
        "latency_ms": int((time.time() - t0) * 1000),
    }


@app.post("/api/talk")
def api_talk():
    """流式分句输出：边生成边推句，前端逐句合成播放。"""
    if over_budget():
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
    frame_id = str(body.get("frame_id") or "").strip()
    session_begin(session_id, frame_id)
    context = session_context(session_id, body.get("context"))

    img_b64 = None
    if task != "chat":
        src = body.get("image_vlm") or body.get("image")
        if src:
            try:
                img_b64 = image_to_jpeg_b64(decode_image(src))
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
                chunk = json.dumps({"sentence": CHAT_REALTIME_REPLY}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"
                chunk = json.dumps({"done": True, "answer": CHAT_REALTIME_REPLY}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"
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

    def generate():
        buf, full = "", ""
        if not VLM_LOCK.acquire(timeout=30):
            yield f"data: {json.dumps({'error': '设备正忙，请稍后再试'}, ensure_ascii=False)}\n\n"
            leave()
            return
        inner = None
        try:
            inner = stream_vlm(prompt, img_b64, max_tokens, temperature=temperature_for(task))
            for delta in inner:
                buf += delta
                full += delta
                while True:
                    m = re.search(r"[。！？]", buf)
                    if not m:
                        break
                    sentence, buf = buf[: m.end()], buf[m.end() :]
                    if sentence.strip():
                        yield f"data: {json.dumps({'sentence': sentence.strip()}, ensure_ascii=False)}\n\n"
                if len(buf) >= 60:
                    yield f"data: {json.dumps({'sentence': buf.strip()}, ensure_ascii=False)}\n\n"
                    buf = ""
            if buf.strip():
                yield f"data: {json.dumps({'sentence': buf.strip()}, ensure_ascii=False)}\n\n"
            answer = settle_eyes_answer(task, full, spec["fallback"].format(item=item), item)
            session_remember(session_id, question or item or task, answer, visual=bool(img_b64), frame_id=frame_id)
            yield f"data: {json.dumps({'done': True, 'answer': answer}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            log.warning("talk stream failed: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            if inner is not None:
                inner.close()
            VLM_LOCK.release()
            leave()
            log.info(
                "talk task=%s session=%s frame=%s q=%r latency=%sms answer=%r",
                task, session_id, frame_id, question[:80],
                int((time.time() - t0) * 1000), full[:160],
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/eyes")
def api_eyes():
    if over_budget():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        body = request.get_json(force=True, silent=True) or {}
        task = str(body.get("task", "scene"))
        item = str(body.get("item") or "").strip()
        question = str(body.get("question") or "").strip()
        frame_id = str(body.get("frame_id") or "").strip()
        session_id = str(body.get("session_id") or "").strip()
        captured_at = str(body.get("captured_at") or "").strip()
        image = body.get("image", "")
        image_vlm = body.get("image_vlm", "")
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
        )
        log.info(
            "eyes task=%s source=%s item=%r question=%r frame=%s captured=%s session=%s image_source=%s full=%s vlm=%s latency=%sms answer=%r",
            task, result.get("source"), item, question, frame_id, captured_at, session_id,
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


# 唤醒词固定为「小冰」，兼容常见同音识别：小/晓/筱 + bing/bin 的多种写法
WAKE_HEAD = "小晓筱"
WAKE_TAIL = "冰兵并病丙饼秉炳柄禀斌宾滨彬缤濒"
WAKE_SEP = r"[\s，,。.、！!？?·]*"
WAKE_PATTERN = re.compile(rf"[{WAKE_HEAD}]{WAKE_SEP}[{WAKE_TAIL}]", re.I)


ACK_ONLY = re.compile(r"^(?:喂|哎|嗯|你好|您好|在吗|在不在|听得到吗|能听到吗)$")


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


def extract_item(cmd):
    for pattern in (r"找(?:一下|一找)?(.+)", r"有没有(.+)"):
        m = re.search(pattern, cmd)
        if m:
            name = re.sub(r"[吗呢啊吧]", "", m.group(1)).strip()
            if name:
                return name
    return ""


def extract_compare(cmd):
    m = re.search(r"(?:比较|对比)(?:一下)?(.+)", cmd)
    if m:
        return re.sub(r"[吗呢啊吧，。]", "", m.group(1)).strip()
    return ""


CAMERA_OFF = re.compile(
    r"(?:不要|别|不用)(?:再)?(?:打开|开启|用)?(?:摄像头|相机|镜头)"
    r"|(?:关闭|关掉|停用)(?:摄像头|相机)|别看了|不看东西"
)
TOPIC_SWITCH = re.compile(r"换个话题|聊点别的|只聊天|继续聊天|不看东西")
DEFINITION = re.compile(r"是什么意思|什么意思|怎么理解|如何定义")


def parse_voice_intent(cmd):
    """规则匹配意图，不额外加载模型。"""
    if CAMERA_OFF.search(cmd) or TOPIC_SWITCH.search(cmd):
        return "chat", ""
    if DEFINITION.search(cmd) and "眼前" not in cmd and "手里" not in cmd:
        return "chat", ""
    if any(w in cmd for w in ("卫生间", "洗手间", "厕所", "WC", "wc", "toilet", "Toilet")):
        return "toilet", ""
    if "出口" in cmd:
        return "exit", ""
    if any(w in cmd for w in ("日期", "生产", "保质期", "过期", "批号")):
        return "date", ""
    if any(w in cmd for w in ("读", "念", "什么字", "写的什么", "文字", "上面的字")):
        return "text", ""
    if any(w in cmd for w in ("比较", "对比", "哪个更")):
        return "compare", extract_compare(cmd)
    if any(w in cmd for w in ("详细", "仔细说", "具体说", "多说说", "多讲")):
        return "detail", ""
    if any(w in cmd for w in ("聊", "烦", "难受", "难过", "开心", "累", "陪我", "新闻", "天气", "心情")):
        return "chat", ""
    if "找" in cmd or "有没有" in cmd:
        item = extract_item(cmd)
        return ("item", item) if item else ("scene", "")
    if any(w in cmd for w in ("有什么", "看看", "前面", "前方", "周围", "描述")):
        return "scene", ""
    return "followup", ""


@app.post("/api/voice")
def api_voice():
    if over_budget():
        return jsonify({"error": f"模型内存已超出 {MEM_BUDGET_GB} GB 预算，拒绝本次请求"}), 503
    body = request.get_json(force=True, silent=True) or {}
    t0 = time.time()

    try:
        samples, src_sr = decode_wav(body.get("audio", ""))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"音频解析失败：{exc}"}), 400
    if len(samples) < 4800:
        return jsonify({"text": "", "intent": "short", "answer": ""})

    try:
        text = asr_transcribe(samples)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"语音识别失败：{exc}"}), 500
    log.info("voice asr sr=%s dur=%.1fs text=%r", src_sr, len(samples) / 16000, text)

    pattern = WAKE_PATTERN
    armed = bool(body.get("armed"))
    if not armed and not pattern.search(text):
        log.info("voice no-wake text=%r", text)
        return jsonify(
            {"text": text, "intent": "none", "answer": "", "latency_ms": int((time.time() - t0) * 1000)}
        )

    stripped = pattern.sub("", text) if pattern.search(text) else text
    cmd = re.sub(r"[，。！？、,.!?\s]", "", stripped)
    if not cmd or ACK_ONLY.fullmatch(cmd):
        log.info("voice wake text=%r", text)
        return jsonify(
            {"text": text, "intent": "wake", "answer": "我在，请说", "latency_ms": int((time.time() - t0) * 1000)}
        )

    intent, item = parse_voice_intent(cmd)
    context = str(body.get("context") or "").strip()
    frame_id = str(body.get("frame_id") or "").strip()
    session_id = str(body.get("session_id") or "").strip()
    captured_at = str(body.get("captured_at") or "").strip()
    img = body.get("image", "")
    img_vlm = body.get("image_vlm", "")

    if intent != "chat" and intent != "followup" and not (img or img_vlm):
        return jsonify(
            {
                "text": text,
                "intent": intent,
                "answer": "还没有画面，请先拍一张或打开相机",
                "latency_ms": int((time.time() - t0) * 1000),
            }
        )

    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        if intent == "followup":
            result = run_eyes(
                "followup", context=context, question=cmd,
                image_b64=img, image_vlm_b64=img_vlm, frame_id=frame_id,
                session_id=session_id, captured_at=captured_at,
            )
        elif intent == "chat":
            result = run_eyes("chat", question=cmd, frame_id=frame_id, session_id=session_id)
        else:
            result = run_eyes(
                intent, item=item, image_b64=img, image_vlm_b64=img_vlm,
                frame_id=frame_id, session_id=session_id, captured_at=captured_at,
            )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("voice failed")
        return jsonify({"error": f"识别失败：{exc}"}), 500
    finally:
        leave()

    log.info(
        "voice intent=%s item=%r frame=%s captured=%s session=%s image_source=%s full=%s vlm=%s source=%s latency=%sms answer=%r text=%r",
        intent, item, frame_id, captured_at, session_id,
        body.get("image_source") or "-", b64_md5(img), b64_md5(img_vlm),
        result["source"], int((time.time() - t0) * 1000), result["answer"][:160], text,
    )
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


@app.post("/api/tts")
def api_tts():
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "文本为空"}), 400
    if len(text) > 200:
        return jsonify({"error": "单句过长，请按句子分段合成", "max_len": 200}), 400
    engine = (body.get("engine") or "piper").lower()
    sid = int(body.get("sid", 45))
    speed = float(body.get("speed", 1.0))
    if engine not in ("piper", "kokoro"):
        return jsonify({"error": f"未知 TTS 引擎：{engine}"}), 400
    if not try_enter():
        return jsonify({"error": "设备正忙，请稍后再试"}), 429
    try:
        t0 = time.time()
        if engine == "piper":
            with TTS_LOCK:
                return piper_tts(text, speed)
        with TTS_LOCK:
            try:
                synth = load_kokoro()
            except Exception as exc:  # noqa: BLE001
                init_error["kokoro"] = str(exc)
                return jsonify({"error": "Kokoro 初始化失败：" + str(exc)}), 503
            audio = synth.generate(text, sid=sid, speed=speed)
            sr = synth.sample_rate
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
        log.info(
            "tts engine=%s chars=%d gen=%sms audio=%.2fs",
            engine, len(text), gen_ms, len(audio.samples) / sr,
        )
        resp = send_file(buf, mimetype="audio/wav", as_attachment=False, download_name="tts.wav")
        resp.headers["X-Gen-Ms"] = str(gen_ms)
        resp.headers["X-Audio-Seconds"] = str(round(len(audio.samples) / sr, 2))
        resp.headers["X-TTS-Engine"] = engine
        return resp
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"TTS 失败：{exc}"}), 500
    finally:
        leave()


@app.get("/api/sample/<path:name>")
def sample(name: str):
    return send_from_directory(SAMPLES_DIR, name)


if __name__ == "__main__":
    log.info(
        "start vlm=%s budget=%.1fGB frame_max_age=%.0fs snapshot=%s data=%s log=%s",
        VLM_URL, MEM_BUDGET_GB, FRAME_MAX_AGE, DEBUG_SNAPSHOT, DATA_DIR, LOG_FILE,
    )
    log.info("piper=%s", "ready" if not piper_error() else piper_error())
    threading.Thread(target=init_ocr, daemon=True).start()
    threading.Thread(target=asr_idle_reaper, daemon=True).start()
    time.sleep(1)
    app.run(host="127.0.0.1", port=8090, threaded=True, debug=False)
