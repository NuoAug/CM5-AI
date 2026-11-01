# -*- coding: utf-8 -*-
"""无盲道引导：SegFormer-B0 ADE20K 分割 + 保守方向决策（试验）。

只判断脚下到前方几米的可走地面，不做地图导航，不覆盖过马路、楼梯和扶梯。
"""

import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get("GUIDE_MODEL")
    or (ROOT / "models" / "guide" / "segformer-b0-ade" / "model.onnx")
)
INPUT_W, INPUT_H = 384, 288

# ADE20K 类别：只把这些当地面。道路单独判危险，不并入可走区域。
SAFE_IDS = {3, 11, 28, 52}       # floor / sidewalk / rug / path
ROAD_ID = 6
STAIR_IDS = {53, 59, 96, 121}    # stairs / stairway / escalator / step
PERSON_ID = 12

ROAD_STOP = 15.0
PERSON_STOP = 8.0
STAIR_STOP = 2.0
NEAR_MIN = 50.0
MID_MIN = 30.0
LANE_MIN = 60.0
SIDE_MIN = 55.0

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_SESSION_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_session = None
_input_name = None
_load_error = None


def available():
    return MODEL_PATH.is_file()


def load_error():
    return _load_error


def _get_session():
    global _session, _input_name, _load_error
    with _SESSION_LOCK:
        if _session is not None or _load_error is not None:
            return _session
        if not MODEL_PATH.is_file():
            _load_error = f"缺少引导模型：{MODEL_PATH}"
            return None
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("GUIDE_THREADS", "2"))
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            _session = ort.InferenceSession(
                str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            _input_name = _session.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            _load_error = f"引导模型加载失败：{exc}"
            _session = None
    return _session


def status():
    if _load_error:
        return {"ready": False, "engine": "segformer-b0-ade", "error": _load_error}
    return {"ready": available(), "engine": "segformer-b0-ade", "error": None}


def _ratio(mask, y0, y1, x0, x1):
    h, w = mask.shape
    return round(float(mask[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].mean()) * 100, 1)


def metrics_from_pred(pred):
    """pred 为 H×W 的类别图；所有区域都落在画面下半部。"""
    safe = np.isin(pred, list(SAFE_IDS))
    return {
        "left": _ratio(safe, 0.50, 1.0, 0.05, 0.36),
        "center": _ratio(safe, 0.50, 1.0, 0.33, 0.67),
        "right": _ratio(safe, 0.50, 1.0, 0.64, 0.95),
        "near": _ratio(safe, 0.75, 1.0, 0.25, 0.75),
        "mid": _ratio(safe, 0.55, 0.75, 0.25, 0.75),
        "road": _ratio(pred == ROAD_ID, 0.55, 1.0, 0.25, 0.75),
        "stairs": _ratio(np.isin(pred, list(STAIR_IDS)), 0.55, 1.0, 0.25, 0.75),
        "person": _ratio(pred == PERSON_ID, 0.55, 1.0, 0.25, 0.75),
    }


def decide(m):
    """保守状态机：先判危险，再判近处能否通行，最后才选左右。"""
    if m["stairs"] >= STAIR_STOP:
        return "stop", "stairs"
    if m["person"] >= PERSON_STOP:
        return "stop", "person"
    if m["road"] >= ROAD_STOP:
        return "stop", "road"
    if m["near"] < NEAR_MIN:
        return "stop", "blocked"
    if m["mid"] < MID_MIN:
        return "stop", "ahead"
    if m["center"] >= LANE_MIN:
        return "straight", "clear"
    side = "left" if m["left"] >= m["right"] else "right"
    if m[side] >= SIDE_MIN:
        return side, "obstacle"
    return "stop", "blocked"


def guide_frame(img):
    """输入 PIL 图像，返回方向、原因、分区占比与耗时。"""
    session = _get_session()
    if session is None:
        raise RuntimeError(_load_error or "引导模型不可用")

    t0 = time.perf_counter()
    small = img.convert("RGB").resize((INPUT_W, INPUT_H), Image.BILINEAR)
    x = (np.asarray(small, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    x = (x - MEAN) / STD
    with _RUN_LOCK:
        logits = session.run(None, {_input_name: x})[0]
    pred = logits[0].argmax(0)
    m = metrics_from_pred(pred)
    direction, reason = decide(m)
    return {
        "direction": direction,
        "reason": reason,
        "metrics": m,
        "input": f"{INPUT_W}x{INPUT_H}",
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
