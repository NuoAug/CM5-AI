# -*- coding: utf-8 -*-
"""无盲道局部引导：分割 → 连通通道分析 → 保守状态机。

只回答“脚下到前方这一段有没有可走通道”，不提供目的地导航，不判断过马路、
上下楼梯或扶梯。像素宽度未做几何标定，不能当作厘米宽度使用。
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get("GUIDE_MODEL")
    or (ROOT / "models" / "guide" / "segformer-b0-ade" / "model.onnx")
)
MODEL_NAME = "segformer-b0-ade20k"

INPUT_W = int(os.environ.get("GUIDE_INPUT_W", "384"))
INPUT_H = int(os.environ.get("GUIDE_INPUT_H", "288"))
GUIDE_THREADS = int(os.environ.get("GUIDE_THREADS", "2"))
CONF_MIN = float(os.environ.get("GUIDE_CONF_MIN", "0.55"))
ENTROPY_MAX = float(os.environ.get("GUIDE_ENTROPY_MAX", "0.55"))
UNKNOWN_MAX = float(os.environ.get("GUIDE_UNKNOWN_MAX", "0.65"))

# ADE20K 类别编号。只把明确的地面类别当作可走候选；其余一律不是“安全”。
SAFE_IDS = {3, 11, 28, 52}          # floor / sidewalk / rug / path
ROAD_ID = 6
STAIR_IDS = {53, 59, 96, 121}      # stairs / stairway / escalator / step
PERSON_ID = 12

# 可能挡路的物体类别。只用来做“前方有东西”的额外证据，不能替代地面连通性。
OBSTACLE_IDS = {
    19, 20, 21, 24, 30, 31, 33, 35, 42, 43, 45, 62, 67, 70, 71, 74, 76, 80,
    82, 83, 88, 93, 98, 102, 110, 111, 112, 115, 116, 127, 138,
}  # chair car sofa shelf armchair seat desk wardrobe column signboard counter
# bookcase book countertop stove computer boat bus light truck booth pole bottle
# van stool barrel basket bag minibike bicycle ashcan

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# 垂直分层：近、中、远。远层只做连续性参考，不单独决定直行。
NEAR_BAND = (0.78, 1.00)
MID_BAND = (0.62, 0.80)
FAR_BAND = (0.48, 0.62)
LANES = {
    "left": (0.05, 0.38),
    "center": (0.31, 0.69),
    "right": (0.62, 0.95),
}
DANGER_ROI = (0.50, 0.95, 0.10, 0.90)
OBSTACLE_ROI = (0.40, 0.85, 0.05, 0.95)
SEED_BOTTOM = 0.88

NEAR_RATIO_MIN = float(os.environ.get("GUIDE_NEAR_RATIO", "0.50"))
MID_RATIO_MIN = float(os.environ.get("GUIDE_MID_RATIO", "0.35"))
FAR_RATIO_MIN = float(os.environ.get("GUIDE_FAR_RATIO", "0.10"))
MIN_RUN_FRACTION = float(os.environ.get("GUIDE_MIN_RUN_FRACTION", "0.10"))
EDGE_MARGIN_FRACTION = float(os.environ.get("GUIDE_EDGE_MARGIN_FRACTION", "0.035"))
OBSTACLE_CENTER = float(os.environ.get("GUIDE_OBSTACLE_CENTER", "0.18"))
OBSTACLE_SIDE = float(os.environ.get("GUIDE_OBSTACLE_SIDE", "0.10"))
ROAD_STOP = float(os.environ.get("GUIDE_ROAD_STOP", "15.0"))
PERSON_STOP = float(os.environ.get("GUIDE_PERSON_STOP", "8.0"))
STAIR_STOP = float(os.environ.get("GUIDE_STAIR_STOP", "2.0"))

# 图像质量阈值。均为 384x288 灰度图口径，换分辨率必须重新标定。
QUALITY_LUMA_MIN = float(os.environ.get("GUIDE_LUMA_MIN", "40"))
QUALITY_LUMA_MAX = float(os.environ.get("GUIDE_LUMA_MAX", "225"))
QUALITY_OVEREXPOSED_MAX = float(os.environ.get("GUIDE_OVEREXPOSED_MAX", "0.40"))
QUALITY_UNDEREXPOSED_MAX = float(os.environ.get("GUIDE_UNDEREXPOSED_MAX", "0.50"))
QUALITY_CONTRAST_MIN = float(os.environ.get("GUIDE_CONTRAST_MIN", "20"))
QUALITY_BLUR_MIN = float(os.environ.get("GUIDE_BLUR_MIN", "20"))

DANGER_REASONS = {"stairs", "road", "person", "depth_obstacle"}
HOLD_SECONDS = float(os.environ.get("GUIDE_HOLD_SECONDS", "1.5"))
RECOVER_FRAMES = int(os.environ.get("GUIDE_RECOVER_FRAMES", "4"))
RECOVER_SECONDS = float(os.environ.get("GUIDE_RECOVER_SECONDS", "0.8"))
CONFIRM_FRAMES = int(os.environ.get("GUIDE_CONFIRM_FRAMES", "2"))
SWITCH_FRAMES = int(os.environ.get("GUIDE_SWITCH_FRAMES", "3"))
MIN_SWITCH_SECONDS = float(os.environ.get("GUIDE_MIN_SWITCH_SECONDS", "0.6"))
WATCHDOG_SECONDS = float(os.environ.get("GUIDE_WATCHDOG_SECONDS", "2.5"))
FRAME_MAX_AGE = float(os.environ.get("GUIDE_FRAME_MAX_AGE", "0.6"))
TARGET_FPS = float(os.environ.get("GUIDE_FPS", "4.0"))

REASON_TEXT = {
    "stairs": "前方有台阶，请停下",
    "road": "前方是道路，请停下",
    "person": "前方有人，请停下",
    "depth_obstacle": "前方有障碍，请停下",
    "obstacle": "前方有障碍，请停下",
    "blocked": "前方无法通行，请停下",
    "ahead": "前方通道不连续，请停下",
    "narrow": "前方通道太窄，请停下",
    "no_seed": "脚下没有可靠的可走地面，请停下",
    "low_quality": "画面质量不足，请停下",
    "uncertain": "画面判断不确定，请停下",
    "stale": "画面已过期，请停下",
    "model_stuck": "引导模型无响应，请停下",
    "model_missing": "引导模型不可用，请停下",
    "tilted": "手机或相机倾斜过大，请停下",
    "error": "引导异常，请停下",
    "no_frame": "等待画面",
    "stopped": "引导已停止",
    "recovering": "正在恢复通行判断，请稍等",
    "clear": "继续直行",
    "left": "向左一点",
    "right": "向右一点",
}

_SESSION_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None
_output_shape: Optional[list] = None
_load_error: Optional[str] = None
_session_sha256: Optional[str] = None


def available() -> bool:
    return MODEL_PATH.is_file()


def load_error() -> Optional[str]:
    return _load_error


def model_sha256() -> Optional[str]:
    return _session_sha256


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _get_session() -> Optional[ort.InferenceSession]:
    global _session, _input_name, _output_shape, _load_error, _session_sha256
    with _SESSION_LOCK:
        if _session is not None or _load_error is not None:
            return _session
        if not MODEL_PATH.is_file():
            _load_error = f"缺少引导模型：{MODEL_PATH}"
            return None
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, GUIDE_THREADS)
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            _session = ort.InferenceSession(
                str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            _input_name = _session.get_inputs()[0].name
            _output_shape = list(_session.get_outputs()[0].shape)
            _session_sha256 = _file_sha256(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            _load_error = f"引导模型加载失败：{exc}"
            _session = None
    return _session


def status() -> dict:
    ready = available() and _load_error is None
    return {
        "ready": ready,
        "loaded": _session is not None,
        "engine": MODEL_NAME,
        "model": str(MODEL_PATH),
        "model_sha256": _session_sha256,
        "input": f"{INPUT_W}x{INPUT_H}",
        "threads": GUIDE_THREADS,
        "error": _load_error,
        "output_shape_symbolic": _output_shape,
    }


# ---------- 可选传感器接口：没有硬件时明确返回不可用，不伪造数据 ----------
@dataclass
class DepthFrame:
    depth_m: Optional[np.ndarray] = None
    valid: bool = False
    calibrated: bool = False
    timestamp_mono: float = 0.0
    source: str = "none"
    note: str = "未接入深度传感器"


@dataclass
class ImuFrame:
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    timestamp_mono: float = 0.0
    source: str = "none"
    note: str = "未接入 IMU"

    @property
    def known(self) -> bool:
        return self.pitch_deg is not None and self.roll_deg is not None


def null_depth() -> DepthFrame:
    return DepthFrame()


def null_imu() -> ImuFrame:
    return ImuFrame()


@dataclass
class GuideConfig:
    input_w: int = INPUT_W
    input_h: int = INPUT_H
    conf_min: float = CONF_MIN
    entropy_max: float = ENTROPY_MAX
    unknown_max: float = UNKNOWN_MAX
    near_ratio_min: float = NEAR_RATIO_MIN
    mid_ratio_min: float = MID_RATIO_MIN
    far_ratio_min: float = FAR_RATIO_MIN
    min_run_fraction: float = MIN_RUN_FRACTION
    edge_margin_fraction: float = EDGE_MARGIN_FRACTION
    obstacle_center: float = OBSTACLE_CENTER
    obstacle_side: float = OBSTACLE_SIDE
    road_stop: float = ROAD_STOP
    person_stop: float = PERSON_STOP
    stair_stop: float = STAIR_STOP
    quality_luma_min: float = QUALITY_LUMA_MIN
    quality_luma_max: float = QUALITY_LUMA_MAX
    quality_over_max: float = QUALITY_OVEREXPOSED_MAX
    quality_under_max: float = QUALITY_UNDEREXPOSED_MAX
    quality_contrast_min: float = QUALITY_CONTRAST_MIN
    quality_blur_min: float = QUALITY_BLUR_MIN


CFG = GuideConfig()


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=0, keepdims=True), 1e-9)


def _percent(value: float) -> float:
    return round(float(value) * 100.0, 1)


def _laplacian_var(gray: np.ndarray) -> float:
    """灰度图拉普拉斯方差，作为模糊程度的近似。"""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * gray[dy:dy + h - 2, dx:dx + w - 2]
    return float(out.var())


def image_quality(img: Image.Image, cfg: GuideConfig = CFG) -> dict:
    """检查过暗、过曝、低对比度和模糊；返回原因列表。"""
    small = img.convert("L").resize((cfg.input_w, cfg.input_h), Image.BILINEAR)
    g = np.asarray(small, dtype=np.float32)
    mean = float(g.mean())
    p5, p95 = np.percentile(g, [5, 95])
    over = float(np.mean(g > 245))
    under = float(np.mean(g < 15))
    contrast = float(p95 - p5)
    blur = _laplacian_var(g)
    reasons = []
    if mean < cfg.quality_luma_min:
        reasons.append("too_dark")
    if mean > cfg.quality_luma_max:
        reasons.append("too_bright")
    if over > cfg.quality_over_max:
        reasons.append("overexposed")
    if under > cfg.quality_under_max:
        reasons.append("underexposed")
    if contrast < cfg.quality_contrast_min:
        reasons.append("low_contrast")
    if blur < cfg.quality_blur_min:
        reasons.append("blurred")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "mean_luma": round(mean, 1),
        "p5": round(float(p5), 1),
        "p95": round(float(p95), 1),
        "contrast": round(contrast, 1),
        "overexposed_frac": round(over, 3),
        "underexposed_frac": round(under, 3),
        "laplacian_var": round(blur, 1),
    }


def _band_stats(mask: np.ndarray, band: tuple, lane: tuple, margin: int, step: int = 2) -> dict:
    h, w = mask.shape
    y0, y1 = int(h * band[0]), int(h * band[1])
    x0, x1 = int(w * lane[0]), int(w * lane[1])
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return {"ratio": 0.0, "min_width_px": 0, "max_width_px": 0, "rows": 0}
    widths = []
    for row in sub[::step]:
        best = cur = 0
        for value in row:
            cur = cur + 1 if value else 0
            if cur > best:
                best = cur
        widths.append(max(0, best - 2 * margin))
    return {
        "ratio": round(float(sub.mean()), 3),
        "min_width_px": int(min(widths)) if widths else 0,
        "max_width_px": int(max(widths)) if widths else 0,
        "rows": len(widths),
    }


def _ratio(mask: np.ndarray, roi: tuple) -> float:
    h, w = mask.shape
    y0, y1, x0, x1 = roi
    sub = mask[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    return round(float(sub.mean()), 3) if sub.size else 0.0


def _seed_mask(safe: np.ndarray, cfg: GuideConfig) -> tuple[np.ndarray, dict]:
    """近处种子：画面最底部一小段里连续的可走地面。种子不可用就返回未知。"""
    h, w = safe.shape
    band = safe[int(h * SEED_BOTTOM):, :]
    if band.size == 0:
        return np.zeros_like(safe), {"ratio": 0.0, "width_px": 0, "ok": False}
    widths = []
    for row in band:
        best = cur = 0
        for value in row:
            cur = cur + 1 if value else 0
            if cur > best:
                best = cur
        widths.append(best)
    width = int(max(widths)) if widths else 0
    min_width = max(4, int(w * cfg.min_run_fraction))
    seed = np.zeros_like(safe)
    seed[int(h * SEED_BOTTOM):, :] = band
    info = {
        "ratio": round(float(band.mean()), 3),
        "width_px": width,
        "min_width_px": min_width,
        "ok": width >= min_width,
    }
    return seed, info


def _reachable(safe: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """从近处种子做 4 邻域连通，只保留与脚下相连的可走像素。"""
    h, w = safe.shape
    reach = np.zeros_like(safe, dtype=bool)
    q: deque = deque()
    ys, xs = np.nonzero(seed)
    for y, x in zip(ys, xs):
        reach[y, x] = True
        q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and safe[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                q.append((ny, nx))
    return reach


def analyze_pred(
    pred: np.ndarray,
    conf: np.ndarray,
    entropy: np.ndarray,
    cfg: GuideConfig = CFG,
    quality: Optional[dict] = None,
    depth: Optional[DepthFrame] = None,
    imu: Optional[ImuFrame] = None,
    timing: Optional[dict] = None,
    frame: Optional[dict] = None,
) -> dict:
    """对已得到的类别图做通道分析；单元测试直接调用它。"""
    h, w = pred.shape
    quality = quality or {"ok": True, "reasons": []}
    margin = max(1, int(w * cfg.edge_margin_fraction))
    min_run = max(4, int(w * cfg.min_run_fraction))

    safe = (
        np.isin(pred, list(SAFE_IDS))
        & (conf >= cfg.conf_min)
        & (entropy <= cfg.entropy_max)
    )
    unknown = (conf < cfg.conf_min) | (entropy > cfg.entropy_max)
    unknown_frac = float(unknown.mean())

    danger = {
        "road": _ratio(pred == ROAD_ID, DANGER_ROI),
        "stairs": _ratio(np.isin(pred, list(STAIR_IDS)), DANGER_ROI),
        "person": _ratio(pred == PERSON_ID, DANGER_ROI),
    }
    obstacle_mask = np.isin(pred, list(OBSTACLE_IDS))
    obstacles = {
        name: _ratio(obstacle_mask, (OBSTACLE_ROI[0], OBSTACLE_ROI[1], xrange[0], xrange[1]))
        for name, xrange in LANES.items()
    }

    seed, seed_info = _seed_mask(safe, cfg)
    reach = _reachable(safe, seed) if seed_info["ok"] else np.zeros_like(safe)

    lanes = {}
    for name, xrange in LANES.items():
        near = _band_stats(reach, NEAR_BAND, xrange, margin)
        mid = _band_stats(reach, MID_BAND, xrange, margin)
        far = _band_stats(reach, FAR_BAND, xrange, margin)
        obj = obstacles[name]
        near_ok = near["ratio"] >= cfg.near_ratio_min and near["min_width_px"] >= min_run
        mid_ok = mid["ratio"] >= cfg.mid_ratio_min and mid["min_width_px"] >= min_run
        far_ok = far["ratio"] >= cfg.far_ratio_min
        lane = {
            "near": near,
            "mid": mid,
            "far": far,
            "object_ratio": obj,
            "near_ok": bool(near_ok),
            "mid_ok": bool(mid_ok),
            "far_ok": bool(far_ok),
            "passable": bool(near_ok and mid_ok and far_ok),
        }
        lanes[name] = lane

    metrics = {
        "lanes": lanes,
        "danger": danger,
        "seed": seed_info,
        "unknown_fraction": round(unknown_frac, 3),
        "mean_confidence": round(float(conf.mean()), 3),
        "low_confidence_fraction": round(float((conf < cfg.conf_min).mean()), 3),
        "high_entropy_fraction": round(float((entropy > cfg.entropy_max).mean()), 3),
        "min_run_px": min_run,
        "edge_margin_px": margin,
        "calibrated": False,
        "width_unit": "pixel",
        # 旧版扁平字段，供旧页面/日志兼容；不要用于新决策。
        "left": _percent(lanes["left"]["near"]["ratio"]),
        "center": _percent(lanes["center"]["near"]["ratio"]),
        "right": _percent(lanes["right"]["near"]["ratio"]),
        "near": _percent(seed_info["ratio"]),
        "mid": _percent(lanes["center"]["mid"]["ratio"]),
        "road": _percent(danger["road"]),
        "stairs": _percent(danger["stairs"]),
        "person": _percent(danger["person"]),
        "object_center": _percent(obstacles["center"]),
    }

    sensor = {
        "depth": {
            "available": bool(depth and depth.valid),
            "calibrated": bool(depth and depth.calibrated),
            "source": depth.source if depth else "none",
            "note": depth.note if depth else "未接入深度传感器",
        },
        "imu": {
            "available": bool(imu and imu.known),
            "pitch_deg": imu.pitch_deg if imu else None,
            "roll_deg": imu.roll_deg if imu else None,
            "source": imu.source if imu else "none",
            "note": imu.note if imu else "未接入 IMU",
        },
    }

    result = {
        "direction": "unavailable",
        "reason": "no_seed",
        "raw_direction": "unavailable",
        "raw_reason": "no_seed",
        "metrics": metrics,
        "quality": quality,
        "sensor": sensor,
        "timing": timing or {},
        "frame": frame or {},
        "model": {
            "name": MODEL_NAME,
            "input": f"{cfg.input_w}x{cfg.input_h}",
            "threads": GUIDE_THREADS,
            "sha256": _session_sha256,
        },
        "label": "uncalibrated_free_space",
    }

    if not quality.get("ok", True):
        result.update(reason="low_quality", raw_reason="low_quality")
        return result
    if imu and imu.known:
        if abs(imu.pitch_deg) > 15.0 or abs(imu.roll_deg) > 10.0:
            result.update(reason="tilted", raw_reason="tilted")
            return result
    if danger["stairs"] * 100 >= cfg.stair_stop:
        result.update(direction="stop", reason="stairs", raw_direction="stop", raw_reason="stairs")
        return result
    if danger["person"] * 100 >= cfg.person_stop:
        result.update(direction="stop", reason="person", raw_direction="stop", raw_reason="person")
        return result
    if danger["road"] * 100 >= cfg.road_stop:
        result.update(direction="stop", reason="road", raw_direction="stop", raw_reason="road")
        return result
    if depth and depth.valid and depth.calibrated:
        try:
            d = np.asarray(depth.depth_m, dtype=np.float32)
            dh, dw = d.shape
            roi = d[int(dh * 0.55):int(dh * 0.95), int(dw * 0.25):int(dw * 0.75)]
            near = float(np.nanpercentile(roi, 10))
            result["metrics"]["depth_near_m"] = round(near, 2)
            if np.isfinite(near) and 0.0 < near < 0.5:
                result.update(
                    direction="stop", reason="depth_obstacle",
                    raw_direction="stop", raw_reason="depth_obstacle",
                )
                return result
        except Exception as exc:  # noqa: BLE001
            result["sensor"]["depth"]["note"] = f"深度数据处理失败：{exc}"

    if unknown_frac > cfg.unknown_max:
        result.update(reason="uncertain", raw_reason="uncertain")
        return result
    if not seed_info["ok"]:
        result.update(reason="no_seed", raw_reason="no_seed")
        return result

    center = lanes["center"]
    center_blocked = center["object_ratio"] >= cfg.obstacle_center
    if center["passable"] and not center_blocked:
        result.update(
            direction="straight", reason="clear",
            raw_direction="straight", raw_reason="clear",
        )
        return result

    # 中间不合格或前方有物体证据：先看左右是否有合格通道，再决定停下。
    candidates = []
    for name in ("left", "right"):
        lane = lanes[name]
        if lane["passable"] and lane["object_ratio"] < cfg.obstacle_side:
            score = min(lane["near"]["ratio"], lane["mid"]["ratio"])
            candidates.append((score, lane["mid"]["min_width_px"], name))
    if candidates:
        candidates.sort(reverse=True)
        name = candidates[0][2]
        other = "right" if name == "left" else "left"
        if len(candidates) > 1:
            score_diff = candidates[0][0] - candidates[1][0]
            width_diff = candidates[0][1] - candidates[1][1]
            if score_diff < 0.05 and width_diff < 2:
                result.update(
                    direction="stop", reason="narrow",
                    raw_direction="stop", raw_reason="narrow",
                )
                return result
        reason = "obstacle" if center_blocked else "narrow"
        result.update(
            direction=name, reason=reason,
            raw_direction=name, raw_reason=reason,
        )
        return result

    near_any = any(lanes[name]["near_ok"] for name in LANES)
    mid_any = any(lanes[name]["mid_ok"] for name in LANES)
    if not near_any:
        reason = "blocked"
    elif not mid_any:
        reason = "ahead"
    elif center_blocked or any(lanes[name]["object_ratio"] >= cfg.obstacle_side for name in ("left", "right")):
        reason = "obstacle"
    else:
        reason = "narrow"
    result.update(direction="stop", reason=reason, raw_direction="stop", raw_reason=reason)
    return result


def analyze_image(
    img: Image.Image,
    cfg: GuideConfig = CFG,
    frame: Optional[dict] = None,
    depth: Optional[DepthFrame] = None,
    imu: Optional[ImuFrame] = None,
) -> dict:
    """单帧完整分析：模型推理 + 质量 + 通道 + 传感器接口。"""
    session = _get_session()
    if session is None:
        return {
            "direction": "unavailable",
            "reason": "model_missing",
            "raw_direction": "unavailable",
            "raw_reason": "model_missing",
            "metrics": {},
            "quality": {"ok": False, "reasons": ["model_missing"]},
            "sensor": {"depth": {"available": False}, "imu": {"available": False}},
            "timing": {},
            "frame": frame or {},
            "model": status(),
        }

    t0 = time.perf_counter()
    rgb = img.convert("RGB")
    small = rgb.resize((cfg.input_w, cfg.input_h), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    x = (x - MEAN) / STD
    with _RUN_LOCK:
        logits = session.run(None, {_input_name: x})[0]
    if logits.ndim == 4:
        logits = logits[0]
    pred = logits.argmax(0)
    prob = _softmax(logits)
    conf = prob.max(0)
    entropy = -(prob * np.log(prob + 1e-9)).sum(0) / np.log(prob.shape[0])
    preprocess_ms = int((time.perf_counter() - t0) * 1000)

    quality = image_quality(rgb, cfg)
    timing = {
        "inference_ms": None,
        "preprocess_ms": preprocess_ms,
        "total_ms": None,
    }
    result = analyze_pred(
        pred, conf, entropy, cfg=cfg, quality=quality,
        depth=depth, imu=imu, timing=timing, frame=frame or {},
    )
    total_ms = int((time.perf_counter() - t0) * 1000)
    result["timing"].update(inference_ms=total_ms, total_ms=total_ms)
    result["model"]["sha256"] = _session_sha256
    return result


class GuideStateMachine:
    """把单帧 raw 结果变成稳定的引导指令。

    - 危险/无法通行立即 stop，不等待方向防抖。
    - 从 stop 恢复需要连续多帧且满足保持时间。
    - 左右切换有滞回，避免逐帧抖动。
    """

    def __init__(self, cfg: GuideConfig = CFG):
        self.cfg = cfg
        self.state = "unavailable"
        self.reason = "no_frame"
        self.raw_direction = "unavailable"
        self.raw_reason = "no_frame"
        self.candidate = None
        self.candidate_count = 0
        self.last_change = 0.0
        self.hold_until = 0.0
        self.clear_count = 0
        self.clear_start = 0.0
        self.seq = 0
        self.last_update = 0.0
        self.last_speech = ""
        self.last_speech_at = 0.0
        self.speech_repeat_seconds = float(os.environ.get("GUIDE_SPEECH_REPEAT", "4.0"))

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _decision(self, now: float, urgent: bool, changed: bool, holding: bool = False) -> dict:
        speech = REASON_TEXT.get(self.reason, "请停下")
        if self.state in ("left", "right"):
            speech = REASON_TEXT[self.state]
        elif self.state == "straight":
            speech = REASON_TEXT["clear"]
        elif self.state == "unavailable" and self.reason == "no_frame":
            speech = ""
        repeat = urgent and now - self.last_speech_at >= self.speech_repeat_seconds
        announce = bool(speech) and (changed or repeat)
        if announce:
            self.last_speech = speech
            self.last_speech_at = now
        return {
            "state": self.state,
            "direction": self.state,
            "reason": self.reason,
            "raw_direction": self.raw_direction,
            "raw_reason": self.raw_reason,
            "stable": True,
            "urgent": bool(urgent),
            "changed": bool(changed),
            "holding": bool(holding),
            "speech": speech if announce else "",
            "seq": self.seq,
            "updated_mono": now,
            "since_change_ms": int((now - self.last_change) * 1000) if self.last_change else None,
        }

    def update(self, obs: dict, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else now
        self.seq += 1
        self.last_update = now
        raw_dir = obs.get("raw_direction") or obs.get("direction") or "unavailable"
        raw_reason = obs.get("raw_reason") or obs.get("reason") or "error"
        self.raw_direction = raw_dir
        self.raw_reason = raw_reason
        prev_state, prev_reason = self.state, self.reason

        if raw_dir == "unavailable":
            self.state = "unavailable"
            self.reason = raw_reason
            self.candidate = None
            self.candidate_count = 0
            self.clear_count = 0
            self.clear_start = 0.0
            if prev_state != self.state or prev_reason != self.reason:
                self.last_change = now
            return self._decision(now, urgent=True, changed=True)

        if raw_dir == "stop":
            self.state = "stop"
            self.reason = raw_reason
            self.candidate = None
            self.candidate_count = 0
            self.clear_count = 0
            self.clear_start = 0.0
            if raw_reason in DANGER_REASONS or raw_reason in ("obstacle", "blocked", "ahead", "narrow"):
                self.hold_until = now + HOLD_SECONDS
            changed = prev_state != "stop" or prev_reason != raw_reason
            if changed:
                self.last_change = now
            return self._decision(now, urgent=True, changed=changed)

        # 非 stop：先处理从 stop 恢复
        if self.state == "stop":
            if now < self.hold_until:
                return self._decision(now, urgent=False, changed=False, holding=True)
            if self.clear_count == 0:
                self.clear_start = now
            self.clear_count += 1
            if self.clear_count < RECOVER_FRAMES or now - self.clear_start < RECOVER_SECONDS:
                return self._decision(now, urgent=False, changed=False, holding=True)
            # 恢复后仍需按普通候选切换，不直接放行
            self.state = "stop"
            self.reason = "recovering"
        else:
            self.clear_count = 0
            self.clear_start = 0.0

        if raw_dir == self.state:
            self.candidate = None
            self.candidate_count = 0
            if prev_reason != raw_reason:
                self.reason = raw_reason
                self.last_change = now
            return self._decision(now, urgent=False, changed=prev_reason != raw_reason)

        if raw_dir == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = raw_dir
            self.candidate_count = 1

        threshold = SWITCH_FRAMES if self.state in ("straight", "left", "right") else CONFIRM_FRAMES
        if self.candidate_count >= threshold and now - self.last_change >= MIN_SWITCH_SECONDS:
            self.state = raw_dir
            self.reason = raw_reason
            self.last_change = now
            self.candidate = None
            self.candidate_count = 0
            return self._decision(now, urgent=False, changed=True)
        return self._decision(now, urgent=False, changed=False)


def analyze_frame(img: Image.Image, frame: Optional[dict] = None) -> dict:
    """兼容旧调用的单帧入口；不带状态机，不用于实时控制。"""
    return analyze_image(img, frame=frame)


def decide(m: dict) -> tuple:
    """旧测试用的扁平阈值决策，仅保留兼容；新控制链路不使用。"""
    if m["stairs"] >= STAIR_STOP:
        return "stop", "stairs"
    if m["person"] >= PERSON_STOP:
        return "stop", "person"
    if m["road"] >= ROAD_STOP:
        return "stop", "road"
    if m["near"] < NEAR_RATIO_MIN * 100:
        return "stop", "blocked"
    if m["mid"] < MID_RATIO_MIN * 100:
        return "stop", "ahead"
    if m["center"] >= 60.0:
        return "straight", "clear"
    side = "left" if m["left"] >= m["right"] else "right"
    if m[side] >= 55.0:
        return side, "obstacle"
    return "stop", "blocked"
