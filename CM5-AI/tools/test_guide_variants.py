# -*- coding: utf-8 -*-
"""引导受控图片变体与画面过期测试。

覆盖：清晰、暗光、模糊、低对比、中央遮挡；以及浏览器模式下帧过期。
这些是图片级受控测试，不是真实台阶/玻璃/路缘场景。

用法：
  python tools/test_guide_variants.py --image samples/scene-tactile-straight.jpg --out results/guide-variants.json
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter

BASE = "http://127.0.0.1:8090"


def b64(img: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def variants(path: Path) -> dict[str, Image.Image]:
    base = Image.open(path).convert("RGB")
    dark = ImageEnhance.Brightness(base).enhance(0.2)
    dark = ImageEnhance.Contrast(dark).enhance(0.6)
    blur = base.filter(ImageFilter.GaussianBlur(radius=5))
    low = ImageEnhance.Contrast(base).enhance(0.3)
    occ = base.copy()
    from PIL import ImageDraw
    d = ImageDraw.Draw(occ)
    w, h = occ.size
    d.rectangle([int(w * 0.25), int(h * 0.45), int(w * 0.75), int(h * 0.95)], fill=(0, 0, 0))
    return {"clear": base, "dark": dark, "blur": blur, "low_contrast": low, "central_occlusion": occ}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = {}
    for name, img in variants(Path(args.image)).items():
        t0 = time.perf_counter()
        r = requests.post(BASE + "/api/guide", json={"image": b64(img), "frame_id": f"variant-{name}"}, timeout=60)
        ms = int((time.perf_counter() - t0) * 1000)
        body = r.json() if r.content else {}
        rows[name] = {
            "http": r.status_code, "ms": ms,
            "raw_direction": body.get("raw_direction"), "raw_reason": body.get("raw_reason"),
            "quality": body.get("quality"), "timing": body.get("timing"),
            "error": body.get("error"),
        }
        print(name, rows[name]["http"], rows[name]["raw_direction"], rows[name]["raw_reason"], f"{ms}ms")

    # 画面过期：浏览器模式只收显式帧；设备帧不应再注入
    stale = {}
    try:
        requests.post(BASE + "/api/guide/start", json={"source": "browser"}, timeout=30).raise_for_status()
        requests.post(BASE + "/api/guide/frame", json={
            "image": b64(Image.open(args.image).convert("RGB")), "frame_id": "stale-1",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, timeout=30).raise_for_status()
        time.sleep(1.2)
        g = requests.get(BASE + "/api/guide/status", timeout=10).json()
        stale = {
            "running": g.get("running"), "mode": g.get("mode"),
            "frame_age_ms": g.get("frame_age_ms"),
            "state": (g.get("decision") or {}).get("state"),
            "reason": (g.get("decision") or {}).get("reason"),
            "frame_source": g.get("frame_source"),
        }
    finally:
        requests.post(BASE + "/api/guide/stop", json={"reason": "variant-test"}, timeout=20)

    result = {"image": args.image, "variants": rows, "stale": stale,
              "note": "受控图片测试；不能替代真实台阶/玻璃/路缘场景"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
