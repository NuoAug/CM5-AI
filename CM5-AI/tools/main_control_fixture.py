# -*- coding: utf-8 -*-
"""主控替身：只回放明确样例，producer=test_fixture，evidence_mode=synthetic。

不是真实主控，不计算通行/距离/空间判断；仅用于验证 interaction.v0.1 协议、
取消、时效与负向路径。默认监听 127.0.0.1:18091，不对外暴露。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "interaction.v0.1"
CLOCK_ID = "fixture-clock-1"


def now_ms() -> int:
    return int(time.monotonic() * 1000)


class Fixture:
    def __init__(self, args):
        self.args = args
        self.instance_id = args.instance_id
        self.sequences = 0
        self.scene_queries = 0
        self.control_state = "idle"
        self.frame_asset = Path(args.frame).resolve()
        self.frame_sha = hashlib.sha256(self.frame_asset.read_bytes()).hexdigest()
        with Image.open(self.frame_asset) as img:
            self.frame_width, self.frame_height = img.size
        self.frame_captured = now_ms() - int(args.frame_age_ms)

    def envelope(self, msg_type: str, request: dict, *, status="ok", error=None, data=None) -> dict:
        return {
            "schema_version": SCHEMA,
            "type": msg_type,
            "message_id": f"msg-{uuid.uuid4().hex[:12]}",
            "reply_to": request.get("message_id"),
            "request_id": request.get("request_id"),
            "session_id": request.get("session_id"),
            "producer": {"role": "test_fixture", "instance_id": self.instance_id},
            "clock_id": CLOCK_ID,
            "timestamp_mono_ms": now_ms(),
            "evidence_mode": "synthetic",
            "status": status,
            "error": error,
            "data": data,
        }

    def capabilities(self, request):
        return self.envelope("capabilities.result", request, data={
            "supported_versions": [SCHEMA],
            "roles": ["test_fixture"],
            "modes": {"frame_provider": True, "scene_state": True, "control": True,
                      "target": True, "guidance": True},
            "resources": {"cpu": "fixture", "memory": "fixture"},
            "unimplemented": [],
            "note": "固定样例，不表示真实主控能力",
        })

    def frame(self, request):
        data = request.get("data") or {}
        mode = data.get("mode")
        if mode == "pinned" and not data.get("frame_id"):
            return self.envelope("frame.result", request, status="rejected",
                                 error={"code": "INVALID_PAYLOAD", "message": "pinned 缺少 frame_id",
                                        "retryable": False})
        return self.envelope("frame.result", request, data={
            "frame_id": "fixture-frame-001",
            "captured_mono_ms": self.frame_captured,
            "source_kind": "fixture",
            "is_historical": bool(self.args.historical),
            "width_px": self.frame_width,
            "height_px": self.frame_height,
            "color_space": "RGB",
            "parent_frame_id": None,
            "transform_to_parent": None,
            "quality": {"sharpness": "synthetic", "glare": "synthetic", "exposure": "synthetic"},
            "asset": {
                "asset_id": "fixture-asset-001",
                "transport": "file",
                "path": str(self.frame_asset),
                "encoding": "jpeg",
                "sha256": self.frame_sha,
                "available_until_mono_ms": now_ms() + 30000,
            },
        })

    def scene(self, request):
        self.sequences += 1
        self.scene_queries += 1
        stale = self.args.scene_mode == "stale"
        if self.args.scene_mode == "invalid":
            nav = {"mode": "paused", "decision": "pause",
                   "reason_code": "SPACE_UNCONFIRMED",
                   "speech_fact": "两侧空间还没有确认，请先停下。"}
            measurement = {"distance_m": None, "valid": False,
                           "valid_until_mono_ms": now_ms() + 400, "reason_code": "DEPTH_INVALID"}
        else:
            nav = {"mode": "walking", "decision": "straight",
                   "reason_code": "CLEAR_AHEAD",
                   "speech_fact": "主控样例：前方通道暂时可用，请用盲杖确认。"}
            measurement = {"distance_m": 1.2, "valid": True,
                           "valid_until_mono_ms": now_ms() + 400, "reason_code": None}
        return self.envelope("scene.result", request, data={
            "state_id": f"fixture-state-{self.sequences:03d}",
            "sequence": self.sequences,
            "observed_mono_ms": now_ms() - 50,
            "valid_until_mono_ms": now_ms() - 500 if stale else now_ms() + 400,
            "coordinate_frame": "body_forward",
            "navigation": nav,
            "objects": [{
                "track_id": "fixture-obstacle-1",
                "class_name": "静态实体",
                "bearing": "front",
                "measurement": measurement,
            }],
            "measurement": measurement,
            "sensor_status": {"rgb": "valid", "depth": "valid" if measurement["valid"] else "invalid",
                              "heading": "valid"},
        })

    def control(self, request):
        data = request.get("data") or {}
        action = data.get("action")
        allowed = {"start_guidance", "pause_guidance", "resume_guidance",
                   "end_guidance", "repeat_guidance"}
        if action not in allowed:
            return self.envelope("control.result", request, status="rejected",
                                 error={"code": "INVALID_PAYLOAD", "message": "动作不允许",
                                        "retryable": False})
        if self.args.control_mode == "rejected":
            return self.envelope("control.result", request, data={
                "accepted": False, "operation_state": "rejected", "actual_mode": self.control_state,
                "reason_code": "FIXTURE_REJECT", "state_id": f"fixture-state-{self.sequences:03d}"})
        target = {"start_guidance": "walking", "resume_guidance": "walking",
                  "pause_guidance": "paused", "end_guidance": "idle",
                  "repeat_guidance": self.control_state}.get(action, self.control_state)
        state = "pending" if self.args.control_mode == "pending" else "completed"
        if state == "completed":
            self.control_state = target
        return self.envelope("control.result", request, data={
            "accepted": True, "operation_state": state, "actual_mode": self.control_state,
            "reason_code": "FIXTURE_OK", "state_id": f"fixture-state-{self.sequences:03d}"})

    def target(self, request):
        return self.envelope("target.result", request, data={
            "match": "unmatched", "track_id": None, "valid_until_mono_ms": None,
            "reason_code": "FIXTURE_UNMATCHED",
        })

    def playback(self, request):
        # 事件型消息的构造回执；只证明替身收到，不代表真实出声。
        return self.envelope("playback.status", request, data={
            "recorded": True, "note": "fixture 仅记录收到播放状态",
        })


FIXTURE: Fixture | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _chunk(self, data: bytes):
        """SSE 必须用 chunked，否则客户端按固定块读取会等不到首条事件。"""
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):  # noqa: A003
        print("[fixture] " + fmt % args)

    def _send(self, payload: dict, status: int = 200):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/interaction/rpc":
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, 400)
            return
        if request.get("schema_version") != SCHEMA:
            self._send(FIXTURE.envelope("error", request, status="rejected",
                                        error={"code": "UNSUPPORTED_VERSION",
                                               "message": "schema_version 不支持",
                                               "retryable": False}), 400)
            return
        kind = request.get("type")
        dispatch = {
            "capabilities.get": FIXTURE.capabilities,
            "frame.acquire": FIXTURE.frame,
            "scene.query": FIXTURE.scene,
            "control.intent": FIXTURE.control,
            "target.observation": FIXTURE.target,
            "playback.status": FIXTURE.playback,
        }
        handler = dispatch.get(kind)
        if handler is None:
            self._send(FIXTURE.envelope("error", request, status="rejected",
                                        error={"code": "INVALID_PAYLOAD",
                                               "message": f"替身不支持：{kind}",
                                               "retryable": False}), 400)
            return
        self._send(handler(request))

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] != "/interaction/events":
            self._send({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._chunk(b"retry: 3000\n\n")
        if FIXTURE.args.emit_guidance:
            event = {
                "schema_version": SCHEMA,
                "type": "guidance.event",
                "message_id": "msg-fixture-danger-001",
                "reply_to": None,
                "request_id": None,
                "session_id": None,
                "producer": {"role": "test_fixture", "instance_id": FIXTURE.instance_id},
                "clock_id": CLOCK_ID,
                "timestamp_mono_ms": now_ms(),
                "evidence_mode": "synthetic",
                "data": {
                    "event_id": "fixture-danger-001",
                    "sequence": 101,
                    "priority": "emergency",
                    "issued_mono_ms": now_ms(),
                    "valid_until_mono_ms": now_ms() + 8000,
                    "state_id": "fixture-state-999",
                    "text": "【替身样例】请先停下，前方空间无法确认。",
                    "interrupt_policy": "preempt_ordinary",
                    "supersedes_event_id": None,
                },
            }
            self._chunk(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
        try:
            while True:
                time.sleep(5)
                if self.wfile.closed:
                    break
                self._chunk(b": ping\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--instance-id", default="main-fixture-1")
    parser.add_argument("--frame", default=str(ROOT / "samples" / "scene-corridor.jpg"))
    parser.add_argument("--frame-age-ms", type=int, default=100)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--scene-mode", choices=("valid", "invalid", "stale"), default="valid")
    parser.add_argument("--control-mode", choices=("completed", "pending", "rejected"), default="completed")
    parser.add_argument("--emit-guidance", action="store_true")
    args = parser.parse_args()
    global FIXTURE
    FIXTURE = Fixture(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[fixture] listening http://{args.host}:{args.port} "
          f"instance={args.instance_id} frame={FIXTURE.frame_asset.name} scene={args.scene_mode} "
          f"emit_guidance={args.emit_guidance}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
