# -*- coding: utf-8 -*-
"""KWS 离线验证与环形缓冲测试。

真实唤醒词「小冰小冰」需要真人录音；本测试只使用模型自带的真人测试音频，
不使用 TTS 合成音频冒充唤醒样本。
"""

import unittest
import wave
from pathlib import Path

import numpy as np

import kws

BASE = Path(kws.MODEL_DIR)
HAS_MODEL = BASE.is_dir() and (BASE / "tokens.txt").is_file()


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        n2 = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data), n2, endpoint=False), np.arange(len(data)), data)
    return data.astype(np.float32)


@unittest.skipUnless(HAS_MODEL, "KWS 模型未下载")
class KwsOfflineTests(unittest.TestCase):
    def test_official_human_wav_detects(self):
        engine = kws.KwsEngine()
        engine.keywords = BASE / "test_wavs" / "test_keywords.txt"
        self.assertTrue(engine.load())
        audio = read_wav(BASE / "test_wavs" / "3.wav")
        hits = engine.detect(audio)
        self.assertIn("文森特卡索", hits)

    def test_negative_wav_no_hit(self):
        engine = kws.KwsEngine()
        engine.keywords = BASE / "test_wavs" / "test_keywords.txt"
        self.assertTrue(engine.load())
        audio = read_wav(BASE / "test_wavs" / "0.wav")
        self.assertEqual(engine.detect(audio), [])

    def test_custom_keyword_file_exists(self):
        self.assertTrue(kws.KEYWORDS_FILE.is_file())
        text = kws.KEYWORDS_FILE.read_text(encoding="utf-8")
        self.assertIn("小冰小冰", text)


class RingBufferTests(unittest.TestCase):
    def test_trim_keeps_latest(self):
        ring = kws.AudioRingBuffer(seconds=0.5)
        for i in range(20):
            ring.append(np.full(1600, i, dtype=np.float32))
        data = ring.snapshot()
        self.assertLessEqual(len(data), 8000 + 1600)
        self.assertGreater(data[-1], 15)

    def test_snapshot_seconds(self):
        ring = kws.AudioRingBuffer(seconds=1.0)
        ring.append(np.full(16000, 1.0, dtype=np.float32))
        self.assertEqual(len(ring.snapshot(0.2)), 3200)

    def test_clear(self):
        ring = kws.AudioRingBuffer()
        ring.append(np.ones(1000, dtype=np.float32))
        ring.clear()
        self.assertEqual(ring.snapshot().size, 0)


class FakeStream:
    pass


class FakeKws:
    def __init__(self, hit_on_first=True):
        self.hit_on_first = hit_on_first
        self.calls = 0

    def new_stream(self):
        return FakeStream()

    def feed(self, stream, samples):
        self.calls += 1
        if self.hit_on_first and self.calls == 1:
            return ["小冰小冰"]
        return []


class WakeGateTests(unittest.TestCase):
    def test_wake_then_command_after_silence(self):
        gate = kws.WakeGate(engine_obj=FakeKws(), armed_seconds=5.0)
        chunk = 1600
        result = gate.feed(np.full(chunk, 0.02, dtype=np.float32))
        self.assertTrue(result["wake"])
        # 说一句指令，然后 0.8 秒静音
        for _ in range(5):
            result = gate.feed(np.full(chunk, 0.02, dtype=np.float32))
        command = None
        for _ in range(8):
            result = gate.feed(np.zeros(chunk, dtype=np.float32))
            if command is None:
                command = result["command"]
        self.assertIsNotNone(command)
        self.assertGreater(len(command), 0)

    def test_no_wake_no_command(self):
        gate = kws.WakeGate(engine_obj=FakeKws(hit_on_first=False), armed_seconds=5.0)
        result = gate.feed(np.full(1600, 0.02, dtype=np.float32))
        self.assertFalse(result["wake"])
        self.assertIsNone(result["command"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
