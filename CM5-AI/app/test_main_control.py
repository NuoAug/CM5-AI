# -*- coding: utf-8 -*-
"""interaction.v0.1 适配器与替身回归：握手、取图校验、时效、来源与取消。"""

import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import server
from main_control import MainControlAdapter
from output import PRIORITY_ANSWER, PRIORITY_GUIDE, PRIORITY_DANGER

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tools" / "main_control_fixture.py"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FixtureProcess:
    def __init__(self, *args):
        self.port = free_port()
        self.proc = subprocess.Popen(
            [sys.executable, str(FIXTURE), "--port", str(self.port), *args],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.url = f"http://127.0.0.1:{self.port}"
        self._wait()

    def _wait(self):
        import requests
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                requests.get(self.url + "/interaction/events", timeout=0.5, stream=True).close()
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        raise RuntimeError("fixture 未启动")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class AdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = FixtureProcess()
        cls.adapter = MainControlAdapter(cls.fx.url, allow_fixture=True)

    @classmethod
    def tearDownClass(cls):
        cls.adapter.stop()
        cls.fx.close()

    def test_handshake_records_instance_and_clock(self):
        result = self.adapter.handshake()
        self.assertEqual(result["status"], "ok")
        info = self.adapter.status()
        self.assertTrue(info["connected"])
        self.assertEqual(info["master_instance_id"], "main-fixture-1")
        self.assertEqual(info["clock_id"], "fixture-clock-1")
        self.assertEqual(info["evidence_mode"], "synthetic")

    def test_frame_asset_hash_verified(self):
        frame = self.adapter.acquire_frame(mode="latest", purpose="vqa", max_side_px=512)
        self.assertEqual(frame["frame_id"], "fixture-frame-001")
        self.assertEqual(frame["source_kind"], "fixture")
        self.assertEqual(frame["image"].size[0], frame["width_px"])

    def test_scene_valid_and_answer_has_no_fake_zero(self):
        scene = self.adapter.query_scene()
        self.assertFalse(scene["_stale"])
        text = __import__("main_control").format_scene_answer(scene)
        self.assertIn("1.2 米", text)
        self.assertNotIn("0 米", text)

    def test_control_completed_only_when_mode_matches(self):
        result = self.adapter.control(
            "start_guidance", utterance_id="u-1", final_transcript="开始引导",
            session_id="s-1",
        )
        self.assertEqual(result["operation_state"], "completed")
        self.assertEqual(result["actual_mode"], "walking")

    def test_target_observation_unmatched(self):
        result = self.adapter.target_observation(
            visual_target_id="v-1", frame_id="fixture-frame-001", roi_xyxy_px=[0, 0, 10, 10],
            label="水杯", text="", source="ocr", user_selected=True,
        )
        self.assertEqual(result["match"], "unmatched")
        self.assertIsNone(result["track_id"])

    def test_fixture_rejected_in_production_mode(self):
        prod = MainControlAdapter(self.fx.url, allow_fixture=False)
        result = prod.handshake()
        self.assertNotEqual(result["status"], "ok")
        self.assertIn("生产者角色", (result.get("error") or {}).get("message", ""))

    def test_event_rejects_user_source_and_expired(self):
        base = {
            "schema_version": "interaction.v0.1", "type": "guidance.event",
            "producer": {"role": "user", "instance_id": "x"},
            "evidence_mode": "synthetic", "clock_id": "fixture-clock-1",
            "data": {"event_id": "e1", "sequence": 1, "priority": "emergency",
                     "text": "假危险", "valid_until_mono_ms": int(time.monotonic() * 1000) + 1000},
        }
        got = []
        self.adapter._on_event = got.append
        self.adapter._handle_event(base)
        self.assertEqual(got, [])
        base["producer"] = {"role": "test_fixture", "instance_id": "main-fixture-1"}
        base["data"]["valid_until_mono_ms"] = int(time.monotonic() * 1000) - 1
        self.adapter._handle_event(base)
        self.assertEqual(got, [])
        base["data"]["valid_until_mono_ms"] = int(time.monotonic() * 1000) + 1000
        self.adapter._handle_event(base)
        self.adapter._handle_event(base)
        self.assertEqual(len(got), 1)


class StaleSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = FixtureProcess("--scene-mode", "stale")
        cls.adapter = MainControlAdapter(cls.fx.url, allow_fixture=True)
        cls.adapter.handshake()

    @classmethod
    def tearDownClass(cls):
        cls.adapter.stop()
        cls.fx.close()

    def test_stale_scene_rejected_for_walking_answer(self):
        scene = self.adapter.query_scene()
        self.assertTrue(scene["_stale"])
        text = __import__("main_control").format_scene_answer(scene)
        self.assertIn("已过期", text)
        self.assertNotIn("可以走", text)


class InteractionCancelEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_ordinary_cancel_is_acknowledged_without_fake_compute_stop(self):
        body = {
            "schema_version": "interaction.v0.1", "type": "output.cancel",
            "producer": {"role": "main_control", "instance_id": "mc-1"},
            "data": {"scope": "ordinary_turn", "target_request_id": "r-1"},
        }
        resp = self.client.post("/api/interaction/cancel", json=body)
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["type"], "output.cancel.result")
        self.assertFalse(data["data"]["generation_cancelled"])

    def test_user_source_cannot_cancel(self):
        body = {
            "schema_version": "interaction.v0.1", "type": "output.cancel",
            "producer": {"role": "user", "instance_id": "u-1"},
            "data": {"scope": "ordinary_turn"},
        }
        resp = self.client.post("/api/interaction/cancel", json=body)
        self.assertEqual(resp.status_code, 403)


class OutputPriorityTests(unittest.TestCase):
    def test_guidance_does_not_override_emergency(self):
        from output import OutputBus
        bus = OutputBus()
        nav = bus.push("导航提示", priority=PRIORITY_GUIDE, source="main_control",
                       meta={"event_id": "nav"})
        danger = bus.push("危险", priority=PRIORITY_DANGER, source="main_control",
                          meta={"event_id": "danger"})
        self.assertIsNotNone(nav)
        self.assertIsNotNone(danger)
        self.assertEqual(bus.stats()["last_event"]["priority"], PRIORITY_DANGER)
        ordinary = bus.push("普通回答", priority=PRIORITY_ANSWER, source="chat")
        self.assertTrue(ordinary["preempt"] is False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
