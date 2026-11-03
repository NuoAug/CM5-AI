# -*- coding: utf-8 -*-
"""设备播报分组测试：同一 utterance 只暂停一次麦克风，片段写入连续通道。"""

import threading
import unittest

import device_io


class FakePlayer:
    def __init__(self):
        self.fed = []
        self.ended = 0
        self.stopped = 0

    def feed(self, wav, cancel):
        self.fed.append(len(wav))
        return True

    def end_stream(self):
        self.ended += 1

    def stop(self):
        self.stopped += 1


class DeviceSinkTest(unittest.TestCase):
    def setUp(self):
        self.player = FakePlayer()
        self.events = []
        self.sink = device_io.DeviceSink(
            synth=lambda text: b"x" * len(text),
            player=self.player,
            before_speak=lambda: self.events.append("pause"),
            after_speak=lambda: self.events.append("resume"),
        )

    def test_same_utterance_one_pause(self):
        self.sink.speak(
            {"text": "第一句。", "utterance_id": "u1", "is_final": False},
            threading.Event(),
        )
        self.sink.speak(
            {"text": "第二句。", "utterance_id": "u1", "is_final": True},
            threading.Event(),
        )
        self.assertEqual(self.player.fed, [4, 4])
        self.assertEqual(self.events, ["pause", "resume"])
        self.assertEqual(self.player.ended, 1)

    def test_new_utterance_resumes_before_next(self):
        self.sink.speak(
            {"text": "旧的。", "utterance_id": "u1", "is_final": False},
            threading.Event(),
        )
        self.sink.speak(
            {"text": "新的。", "utterance_id": "u2", "is_final": True},
            threading.Event(),
        )
        self.assertEqual(self.events, ["pause", "resume", "pause", "resume"])
        self.assertEqual(self.player.ended, 2)

    def test_stop_resumes_microphone(self):
        self.sink.speak(
            {"text": "没播完。", "utterance_id": "u1", "is_final": False},
            threading.Event(),
        )
        self.sink.stop()
        self.assertEqual(self.player.stopped, 1)
        self.assertEqual(self.events, ["pause", "resume"])


if __name__ == "__main__":
    unittest.main()
