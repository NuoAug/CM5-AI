"""常开流式设备语音的构造回放；不代表实机AEC或真人插话通过。"""
from collections import deque
from pathlib import Path
import threading
import time
import unittest
from unittest import mock

import numpy as np
import device_voice
import kws
from streaming_asr import ASREvent, UtteranceLimitError


class FakeCapture:
    running = False
    def __init__(self):
        self.started = self.stopped = 0
    def available(self):
        return True
    def start(self):
        self.started += 1
        self.running = True
        return True
    def read(self):
        return None
    def stop(self):
        self.stopped += 1
        self.running = False


class FakeGate:
    ready = True
    threshold = 0.012
    armed_until = 0
    armed_seconds = 15
    def __init__(self, events=()):
        self.events = deque(events)
        self.feed_calls = 0
        self.resets = 0
    def reset(self):
        self.resets += 1
    def feed(self, samples):
        self.feed_calls += 1
        return self.events.popleft() if self.events else {"armed": True, "rms": 0.1}


class FakeASRSession:
    def __init__(self):
        self.results = deque()
        self.blocks = []
        self.resets = 0
    def feed(self, samples):
        self.blocks.append(len(samples))
        result = self.results.popleft() if self.results else []
        if isinstance(result, Exception):
            raise result
        return result
    def reset(self):
        self.resets += 1


class FakeASR:
    def __init__(self):
        self.loaded = 0
        self.closed = 0
        self.session = FakeASRSession()
    def load(self):
        self.loaded += 1
        return self
    def create_session(self):
        return self.session
    def status(self):
        return {"ready": self.loaded > self.closed}
    def close(self):
        self.closed += 1


class Response:
    def __init__(self, data=None, ok=True, status=200):
        self.data = data or {}
        self.ok = ok
        self.content = b"json"
        self.status_code = status
    def json(self):
        return self.data


def event(text, final=False, uid="speech-1"):
    return ASREvent(text, final, uid, 1, 0.1, 1.0, "endpoint" if final else "incremental")


class DeviceStreamingTests(unittest.TestCase):
    def setUp(self):
        self.capture = FakeCapture()
        self.gate = FakeGate()
        self.asr = FakeASR()
        self.received = []
        self.requested = threading.Event()
        self.release = threading.Event()
        self.output_patch = mock.patch.object(device_voice, "OUTPUT")
        self.output = self.output_patch.start()
        def post(url, **kwargs):
            self.received.append((url, kwargs["json"]))
            self.requested.set()
            return Response()
        self.post = post
        self.voice = device_voice.DeviceVoice(capture=self.capture, gate=self.gate,
            asr=self.asr, backend="streaming-paraformer", http_post=post, aec_verified=False)
        self.block = np.full(1600, 0.1, dtype=np.float32)

    def tearDown(self):
        self.release.set()
        self.voice.close()
        self.output_patch.stop()

    def start(self):
        self.assertTrue(self.voice.start())

    def wait_idle(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.voice._request_queue.unfinished_tasks == 0:
                return
            time.sleep(0.005)
        self.fail("request workers did not finish")

    def test_explicit_load_on_start_and_only_stop_stops_capture(self):
        self.assertEqual(self.asr.loaded, 0)
        self.start()
        self.assertEqual(self.asr.loaded, 1)
        self.voice.pause()
        resets = self.asr.session.resets
        self.voice.resume()
        self.assertEqual(self.asr.session.resets, resets)
        self.assertEqual(self.capture.stopped, 0)
        self.assertEqual(self.capture.started, 1)
        self.assertTrue(self.voice.status()["capture"])
        self.voice.stop()
        self.assertEqual(self.capture.stopped, 1)
        self.assertEqual(self.asr.closed, 1)

    def test_unverified_output_audio_is_not_sent_to_kws_or_asr(self):
        self.start()
        self.voice.pause()
        self.asr.session.results.append([event("开始引导", True)])
        self.assertEqual(self.voice.process_chunk(self.block), [])
        self.assertEqual(self.gate.feed_calls, 0)
        self.assertEqual(self.asr.session.blocks, [])
        self.assertFalse(self.requested.is_set())
        status = self.voice.status()
        self.assertTrue(status["capture"])
        self.assertIn("AEC_NOT_VERIFIED", status["degraded_reason"])
        self.assertFalse(status["human_barge_in_tested"])

    def test_only_final_transcript_goes_to_final_endpoint(self):
        self.start()
        self.asr.session.results.extend(([event("开始")], [event("不要开始引导", True)]))
        self.voice.process_chunk(self.block)
        self.assertFalse(self.requested.is_set())
        self.voice.process_chunk(self.block)
        self.assertTrue(self.requested.wait(1))
        self.wait_idle()
        url, body = self.received[0]
        self.assertTrue(url.endswith("/api/voice/final"))
        self.assertEqual(body["final_transcript"], "不要开始引导")
        self.assertTrue(body["is_final"])
        self.assertTrue(body["stream_speech"])
        self.assertEqual(body["request_id"], "speech-1")
        self.assertNotIn("audio", body)
        self.assertEqual(len(self.received), 1)

    def test_stream_does_not_execute_legacy_gate_command(self):
        self.start()
        self.gate.events.append({"armed": True, "rms": 0.1, "command": self.block})
        self.voice.process_chunk(self.block)
        self.assertFalse(self.requested.is_set())

    def test_in_progress_utterance_survives_armed_window_expiry(self):
        self.start()
        self.gate.events.extend(({"armed": False, "rms": 0.1}, {"armed": True, "rms": 0.1},
                                 {"armed": False, "rms": 0.0}))
        self.asr.session.results.extend(([event("我要")], [event("我要读这个", True)]))
        self.voice.process_chunk(self.block)
        self.assertEqual(self.asr.session.blocks, [])
        self.voice.process_chunk(self.block)
        self.voice.process_chunk(self.block)
        self.assertTrue(self.requested.wait(1))
        self.assertEqual(self.received[0][1]["final_transcript"], "我要读这个")

    def test_verified_barge_in_cancels_once_and_preserves_new_utterance(self):
        self.voice.aec_verified = True
        self.start()
        self.voice.pause()
        baseline_resets = self.asr.session.resets
        self.output.cancel.side_effect = lambda **kwargs: self.voice.resume()
        self.asr.session.results.extend(([event("请")], [event("请停一下")], [event("请停一下", True)]))
        self.voice.process_chunk(self.block)
        self.output.cancel.assert_not_called()
        self.voice.process_chunk(self.block)
        self.output.cancel.assert_called_once_with(source="barge_in")
        self.assertEqual(self.asr.session.resets, baseline_resets)
        self.assertFalse(self.voice._paused)
        self.voice.process_chunk(self.block)
        self.assertTrue(self.requested.wait(1))
        self.assertEqual(self.received[0][1]["final_transcript"], "请停一下")
        self.assertFalse(self.voice.status()["human_barge_in_tested"])

    def test_output_final_cannot_bypass_barge_confirmation(self):
        self.voice.aec_verified = True
        self.start()
        self.voice.pause()
        self.asr.session.results.append([event("开始引导", True)])
        self.voice.process_chunk(self.block)
        self.output.cancel.assert_not_called()
        self.assertFalse(self.requested.is_set())

    def test_overlong_utterance_is_not_sent_as_final(self):
        self.start()
        self.asr.session.results.append(UtteranceLimitError("too long"))
        self.voice.process_chunk(self.block)
        self.assertFalse(self.requested.is_set())
        self.assertEqual(self.gate.armed_until, 0)
        self.assertIn("未提交截断命令", self.voice.status()["error"])

    def test_cancelled_backend_response_is_silent(self):
        self.voice._post = lambda *args, **kwargs: Response({"code": "CANCELLED"}, False, 499)
        self.start()
        self.voice._enqueue_final("你好", "u-cancel")
        self.wait_idle()
        self.output.push.assert_not_called()
        self.assertEqual(self.voice.status()["stats"]["ignored_cancelled"], 1)

    def test_late_old_request_failure_does_not_speak_in_new_turn(self):
        old_started = threading.Event()
        new_done = threading.Event()
        def post(url, **kwargs):
            if kwargs["json"]["request_id"] == "old":
                old_started.set()
                self.release.wait(2)
                raise RuntimeError("old failure")
            new_done.set()
            return Response()
        self.voice._post = post
        self.start()
        self.voice._enqueue_final("旧话语", "old")
        self.assertTrue(old_started.wait(1))
        self.voice._enqueue_final("新话语", "new")
        self.assertTrue(new_done.wait(1))
        self.release.set()
        self.wait_idle()
        self.output.push.assert_not_called()
        self.assertEqual(self.voice.status()["stats"]["request_errors"], 0)

    def test_slow_http_keeps_workers_and_pending_queue_bounded(self):
        def post(*args, **kwargs):
            self.requested.set()
            self.release.wait(2)
            return Response()
        self.voice._post = post
        self.start()
        self.voice._enqueue_final("一", "first")
        self.assertTrue(self.requested.wait(1))
        for index in range(25):
            self.voice._enqueue_final("后续", "u-" + str(index))
        status = self.voice.status()
        self.assertLessEqual(status["request_queue"], 2)
        self.assertEqual(status["request_workers"], 2)
        self.assertGreater(status["stats"]["dropped_requests"], 0)
        self.release.set()
        self.wait_idle()

    def test_missing_stream_model_does_not_fallback_or_start_capture(self):
        self.asr.load = mock.Mock(side_effect=RuntimeError("model missing"))
        self.assertFalse(self.voice.start())
        self.assertEqual(self.capture.started, 0)
        self.assertEqual(self.voice.backend, "streaming-paraformer")

    def test_explicit_offline_baseline_uses_wav_and_no_stream_model(self):
        self.voice.backend = "offline-paraformer"
        self.start()
        self.assertEqual(self.asr.loaded, 0)
        self.gate.events.append({"armed": True, "rms": 0.1, "command": self.block})
        self.voice.process_chunk(self.block)
        self.assertTrue(self.requested.wait(1))
        self.assertEqual(self.received[0][0], self.voice.url)
        self.assertIn("audio", self.received[0][1])

    def test_preroll_block_includes_current_audio_only_once(self):
        self.start()
        self.gate.events.append({"armed": True, "rms": 0.1, "wake": True,
                                 "preroll": np.zeros(3200, dtype=np.float32)})
        self.voice.process_chunk(self.block)
        self.assertEqual(self.asr.session.blocks, [3200])
        self.output.push.assert_not_called()


class StreamingWakeGateTests(unittest.TestCase):
    def test_streaming_gate_never_builds_duplicate_command_recording(self):
        class Spotter:
            first = True
            def new_stream(self):
                return object()
            def feed(self, stream, samples):
                if self.first:
                    self.first = False
                    return ["小冰"]
                return []
        gate = kws.WakeGate(engine_obj=Spotter(), buffer_commands=False)
        for _ in range(90):
            result = gate.feed(np.ones(1600, dtype=np.float32) * 0.1)
            self.assertIsNone(result["command"])
        self.assertEqual(gate.command, [])
        self.assertLessEqual(len(gate.ring.snapshot()), 48000)


if __name__ == "__main__":
    unittest.main()
