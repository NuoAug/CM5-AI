"""Conservative pixel quality check for the Pi test VLM input."""
from PIL import Image


MAX_NEAR_BLACK_CHANNEL = 24
MAX_NEAR_BLACK_RANGE = 16
NEAR_BLACK_REPLY = "画面太暗，暂时看不清。请增加光线，检查镜头是否被挡住，再拍一张。"


def assess_image_quality(image: Image.Image) -> dict:
    """Check a decoded image or ROI without changing it or deciding OCR quality."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("不能检查空图片")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    # Inspect all pixels so a thumbnail cannot hide small labels or bright marks.
    extrema = rgb.getextrema()
    low = min(channel[0] for channel in extrema)
    high = max(channel[1] for channel in extrema)
    value_range = high - low
    blocked = high <= MAX_NEAR_BLACK_CHANNEL and value_range <= MAX_NEAR_BLACK_RANGE
    return {
        "blocked": blocked,
        "reason": "near_black" if blocked else None,
        "message": NEAR_BLACK_REPLY if blocked else None,
        "evidence": {
            "method": "rgb_extrema_v1",
            "image_size_px": list(rgb.size),
            "min_rgb_value": low,
            "max_rgb_value": high,
            "rgb_value_range": value_range,
            "max_near_black_channel": MAX_NEAR_BLACK_CHANNEL,
            "max_near_black_range": MAX_NEAR_BLACK_RANGE,
            "sharpness": "unknown",
        },
    }
