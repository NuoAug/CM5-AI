# -*- coding: utf-8 -*-
"""设备播报取消/抢占验证：在 DeviceSink 真实播放时调用 /api/output/cancel。

判定：取消请求延迟、aplay 子进程是否立即退出、取消后 sink_played 是否停止增长。
需要真实 ALSA 设备；在 CM5 上运行。
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import psutil
import requests

BASE = "http://127.0.0.1:8090"


def aplay_pids() -> list[int]:
    out = []
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info["name"] or "").lower() == "aplay":
                out.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def status() -> dict:
    return requests.get(BASE + "/api/status", timeout=10).json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="16k WAV，内容为一条长回答指令")
    parser.add_argument("--image", required=True, help="当前画面 JPEG")
    parser.add_argument("--wait-s", type=float, default=0.8, help="开始播放后等多久取消")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    audio = "data:audio/wav;base64," + base64.b64encode(Path(args.audio).read_bytes()).decode("ascii")
    image = "data:image/jpeg;base64," + base64.b64encode(Path(args.image).read_bytes()).decode("ascii")
    base_played = ((status().get("output") or {}).get("sink_played")) or 0
    t0 = time.perf_counter()
    r = requests.post(BASE + "/api/voice", json={
        "audio": audio, "image": image, "image_vlm": image,
        "session_id": "cancel-test", "wake_detected": True,
        "request_id": "cancel-test", "frame_id": "cancel-test",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, timeout=180)
    voice_ms = int((time.perf_counter() - t0) * 1000)
    body = r.json() if r.content else {}

    started = None
    polls = 0
    for _ in range(60):
        st = status()
        played = (st.get("output") or {}).get("sink_played") or 0
        pids = aplay_pids()
        if played > base_played or pids:
            started = time.perf_counter()
            break
        polls += 1
        time.sleep(0.25)

    pre = {"voice_status": r.status_code, "voice_ms": voice_ms, "sink_played": ((status().get("output") or {}).get("sink_played")),
           "aplay_pids": aplay_pids(), "answer_chars": len(body.get("answer") or ""), "spoken": body.get("spoken")}
    if started is None:
        result = {"ok": False, "error": "没有观察到播报开始", "pre": pre}
    else:
        time.sleep(max(0.0, args.wait_s))
        played_before = ((status().get("output") or {}).get("sink_played")) or 0
        tc = time.perf_counter()
        cr = requests.post(BASE + "/api/output/cancel", json={"min_priority": 1}, timeout=20)
        cancel_ms = int((time.perf_counter() - tc) * 1000)
        time.sleep(0.5)
        played_after = ((status().get("output") or {}).get("sink_played")) or 0
        time.sleep(1.0)
        played_later = ((status().get("output") or {}).get("sink_played")) or 0
        result = {
            "ok": cr.status_code == 200 and cancel_ms < 500 and played_later == played_after,
            "cancel_http": cr.status_code,
            "cancel_ms": cancel_ms,
            "sink_played_before": played_before,
            "sink_played_after_0_5s": played_after,
            "sink_played_after_1_5s": played_later,
            "aplay_pids_after": aplay_pids(),
            "pre": pre,
        }
    out = Path(args.out) if args.out else Path("results") / "cancel-playback.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
