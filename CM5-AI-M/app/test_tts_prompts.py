# -*- coding: utf-8 -*-
"""预制提示音频：清单校验、事件 ID 加载、损坏回退。"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

from device_io import DeviceSink
from tts_prompts import PromptStore


def make_wav(seconds=0.6, rate=16000, channels=1, sampwidth=2):
    frames = int(rate * seconds)
    silent = b"\x00" * (frames * channels * sampwidth)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(silent)
    return buf.getvalue()


class PromptStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = make_wav()
        (self.dir / "wake_ack.wav").write_bytes(self.wav)
        self.sha = hashlib.sha256(self.wav).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, entries):
        manifest = {"version": 1, "sample_rate": 16000, "entries": entries}
        (self.dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def entry(self, **overrides):
        spec = {"file": "wake_ack.wav", "sha256": self.sha,
                "duration_s": 0.6, "text": "我在，请说",
                "min_duration_s": 0.2, "max_duration_s": 5.0}
        spec.update(overrides)
        return spec

    def test_loads_valid_entry_and_returns_wav(self):
        self.write_manifest({"wake_ack": self.entry()})
        store = PromptStore(self.dir).load()
        self.assertEqual(store.status()["prompts"], ["wake_ack"])
        self.assertEqual(store.status()["errors"], {})
        self.assertEqual(store.read("wake_ack"), self.wav)
        self.assertEqual(store.read("missing"), None)

    def test_sha_mismatch_is_rejected(self):
        self.write_manifest({"wake_ack": self.entry(sha256="0" * 64)})
        store = PromptStore(self.dir).load()
        self.assertNotIn("wake_ack", store.status()["prompts"])
        self.assertIn("sha256", store.status()["errors"]["wake_ack"])
        self.assertIsNone(store.read("wake_ack"))

    def test_missing_file_is_rejected(self):
        self.write_manifest({"wake_ack": self.entry(file="nope.wav")})
        store = PromptStore(self.dir).load()
        self.assertIn("不存在", store.status()["errors"]["wake_ack"])

    def test_wrong_sample_rate_is_rejected(self):
        bad = make_wav(rate=22050)
        (self.dir / "wake_ack.wav").write_bytes(bad)
        self.write_manifest({"wake_ack": self.entry(sha256=hashlib.sha256(bad).hexdigest())})
        store = PromptStore(self.dir).load()
        self.assertIn("采样率", store.status()["errors"]["wake_ack"])

    def test_duration_out_of_range_is_rejected(self):
        self.write_manifest({"wake_ack": self.entry(min_duration_s=1.0, max_duration_s=2.0)})
        store = PromptStore(self.dir).load()
        self.assertIn("时长", store.status()["errors"]["wake_ack"])

    def test_missing_manifest_falls_back(self):
        store = PromptStore(self.dir).load()
        self.assertEqual(store.read("wake_ack"), None)
        self.assertIn("manifest", store.status()["errors"])

    def test_missing_text_is_rejected(self):
        self.write_manifest({"wake_ack": self.entry(text="")})
        store = PromptStore(self.dir).load()
        self.assertIn("text", store.status()["errors"]["wake_ack"])

    def test_path_outside_directory_is_rejected(self):
        outside = Path(self.tmp.name).parent / "wake_ack_outside.wav"
        outside.write_bytes(self.wav)
        try:
            self.write_manifest({"wake_ack": self.entry(file="../wake_ack_outside.wav")})
            store = PromptStore(self.dir).load()
            self.assertIn("目录内", store.status()["errors"]["wake_ack"])
        finally:
            outside.unlink(missing_ok=True)


class FakePlayer:
    available = lambda self: True
    def status(self):
        return {}

    def stop(self):
        return None


class DeviceSinkPromptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = make_wav()
        (self.dir / "wake_ack.wav").write_bytes(self.wav)
        manifest = {"version": 1, "sample_rate": 16000, "entries": {"wake_ack": {
            "file": "wake_ack.wav", "sha256": hashlib.sha256(self.wav).hexdigest(),
            "text": "我在，请说"}}}
        (self.dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.store = PromptStore(self.dir).load()

    def tearDown(self):
        self.tmp.cleanup()

    def sink(self, prompts=None, synth=None):
        return DeviceSink(synth=synth or (lambda text: b"piper-" + text.encode()),
                          player=FakePlayer(), prompts=prompts)

    def test_prompt_audio_used_for_wake_ack(self):
        import threading
        sink = self.sink(self.store)
        payload = {"text": "我在，请说", "priority": 2,
                   "meta": {"prompt_id": "wake_ack"}}
        self.assertEqual(sink.prepare(payload, threading.Event()), self.wav)

    def test_prompt_id_direct_field_is_supported(self):
        import threading
        sink = self.sink(self.store)
        payload = {"text": "我在，请说", "priority": 2, "prompt_id": "wake_ack"}
        self.assertEqual(sink.prepare(payload, threading.Event()), self.wav)

    def test_unknown_prompt_falls_back_to_piper(self):
        import threading
        sink = self.sink(self.store)
        payload = {"text": "普通回答", "priority": 3, "meta": {"prompt_id": "nope"}}
        self.assertEqual(sink.prepare(payload, threading.Event()), b"piper-" + "普通回答".encode())

    def test_missing_store_falls_back_to_piper(self):
        import threading
        sink = self.sink(None)
        payload = {"text": "我在，请说", "priority": 2, "meta": {"prompt_id": "wake_ack"}}
        self.assertEqual(sink.prepare(payload, threading.Event()), b"piper-" + "我在，请说".encode())

    def test_cancelled_prepare_returns_empty(self):
        import threading
        sink = self.sink(self.store)
        event = threading.Event()
        event.set()
        self.assertEqual(sink.prepare({"text": "我在，请说", "meta": {"prompt_id": "wake_ack"}}, event), b"")


if __name__ == "__main__":
    unittest.main()
