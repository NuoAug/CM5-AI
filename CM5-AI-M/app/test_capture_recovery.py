"""故障注入检查 arecord 采集与恢复状态机，不访问设备。"""
from collections import deque
import queue
import subprocess
import threading
import time
import unittest
from unittest import mock

import numpy as np
import device_voice
import kws
from device_io import AlsaCapture, AlsaCaptureError
from test_device_voice_streaming import FakeASR, FakeGate, Response, event


class Pipe:
    def __init__(self, blocks):
        self.blocks = deque(blocks)
        self.closed = False

    def read(self, count):
        if not self.blocks:
            return b""
        block = self.blocks.popleft()
        if len(block) > count:
            self.blocks.appendleft(block[count:])
        return block[:count]

    def close(self):
        self.closed = True


class Process:
    def __init__(self, blocks=(), returncode=None):
        self.stdout = Pipe(blocks)
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode


class CapturePipeTests(unittest.TestCase):
    def setUp(self):
        self.capture = AlsaCapture(chunk_frames=4)
        self.addCleanup(self.capture.stop)
        reader = mock.patch.object(AlsaCapture, "_read_pipe", lambda self, proc, size, deadline: proc.stdout.read(size))
        reader.start()
        self.addCleanup(reader.stop)

    def test_short_reads_accumulate_even_across_sample_boundary(self):
        pcm = np.array([-32768, 0, 8192, 16384], dtype=np.int16).tobytes()
        self.capture._proc = Process([pcm[:1], pcm[1:3], pcm[3:]])
        np.testing.assert_array_equal(self.capture.read(), [-1, 0, .25, .5])
        self.assertTrue(self.capture.running)

    def test_exit_survives_status_check_and_reports_returncode(self):
        proc = Process(returncode=19)
        self.capture._proc = proc
        self.assertFalse(self.capture.running)
        with self.assertRaises(AlsaCaptureError) as caught:
            self.capture.read()
        self.assertEqual(caught.exception.returncode, 19)
        self.assertTrue(proc.stdout.closed)
        self.assertIs(self.capture.last_error, caught.exception)

    def test_eof_discards_incomplete_pcm_and_closes_owned_process(self):
        proc = Process([b"\0\0"])
        self.capture._proc = proc
        with self.assertRaisesRegex(AlsaCaptureError, "输出已结束"):
            self.capture.read()
        self.assertIsNone(self.capture.last_error.returncode)
        self.assertFalse(self.capture.running)
        self.assertTrue(proc.stdout.closed)

    def test_read_deadline_failure_is_visible(self):
        self.capture._proc = Process()
        with mock.patch.object(self.capture, "_read_pipe", side_effect=TimeoutError("PCM stalled")):
            with self.assertRaisesRegex(AlsaCaptureError, "PCM stalled"):
                self.capture.read()
        self.assertFalse(self.capture.running)

    def test_stubborn_retired_capture_blocks_second_open(self):
        proc = Process()
        proc.terminate = mock.Mock()
        proc.kill = mock.Mock()
        proc.wait = mock.Mock(side_effect=subprocess.TimeoutExpired('arecord', 1))
        self.capture._proc = proc
        self.capture.stop()
        with mock.patch.object(self.capture, 'available', return_value=True), mock.patch('device_io.subprocess.Popen') as spawn:
            self.assertFalse(self.capture.start())
        spawn.assert_not_called()
        self.assertIn('尚未退出', str(self.capture.last_error))
        proc.returncode = -9
        proc.wait = mock.Mock(return_value=-9)

    def test_stopped_old_read_does_not_clear_new_process(self):
        entered, release = threading.Event(), threading.Event()
        old, new = Process(), Process()
        self.capture._proc = old
        result = []
        def late_read(proc, size, deadline):
            entered.set()
            release.wait(1)
            return b""
        with mock.patch.object(self.capture, "_read_pipe", side_effect=late_read):
            thread = threading.Thread(target=lambda: result.append(self.capture.read()))
            thread.start()
            self.assertTrue(entered.wait(1))
            self.capture.stop()
            self.capture._proc = new
            release.set()
            thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertTrue(self.capture.running)
        self.assertIsNone(self.capture.last_error)
        self.assertIsNone(new.returncode)


class CaptureFixture:
    def __init__(self):
        self.running = False
        self.last_error = None
        self.starts = 0
        self.stops = 0
        self.outcomes = deque()
        self.chunks = queue.Queue()

    def available(self):
        return True

    def start(self):
        if self.running:
            raise AssertionError("second arecord owner")
        self.starts += 1
        self.running = self.outcomes.popleft() if self.outcomes else True
        self.last_error = None if self.running else AlsaCaptureError("设备仍不可用")
        return self.running

    def read(self):
        try:
            item = self.chunks.get(timeout=.01)
        except queue.Empty:
            return None
        if isinstance(item, AlsaCaptureError):
            self.running = False
            self.last_error = item
            raise item
        return item

    def stop(self):
        self.stops += 1
        self.running = False


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.capture = CaptureFixture()
        self.gate = FakeGate()
        self.gate.stream = None
        self.gate.ring = kws.AudioRingBuffer()
        self.asr = FakeASR()
        self.voice = device_voice.DeviceVoice(capture=self.capture, gate=self.gate, asr=self.asr,
                                               http_post=lambda *args, **kwargs: Response())
        self.addCleanup(self.voice.close)
        output = mock.patch.object(device_voice, "OUTPUT")
        self.output = output.start()
        self.addCleanup(output.stop)
        delays = mock.patch.object(device_voice, "CAPTURE_RETRY_DELAYS", (.02, .03, .04))
        delays.start()
        self.addCleanup(delays.stop)
        self.assertTrue(self.voice.start())
        self.block = np.zeros(1600, dtype=np.float32)

    def wait(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("capture transition not observed: " + str(self.voice.status()))

    def fail_capture(self):
        self.capture.chunks.put(AlsaCaptureError("arecord lost", 19))

    def test_recovery_waits_for_pcm_and_drops_old_asr_and_kws_context(self):
        self.gate.stream = object()
        self.gate.armed_until = time.monotonic() + 15
        self.gate.ring.append(self.block)
        self.asr.session.results.append([event("未说完")])
        self.voice.process_chunk(self.block)
        resets = self.asr.session.resets
        self.fail_capture()
        self.wait(lambda: self.capture.starts == 2)
        status = self.voice.status()
        self.assertEqual(status["capture_state"], "recovering")
        self.assertFalse(status["listening"])
        self.assertIn("arecord lost", status["error"])
        self.assertIsNone(self.gate.stream)
        self.assertEqual(self.gate.ring.snapshot().size, 0)
        self.assertEqual(self.gate.armed_until, 0)
        self.assertGreater(self.asr.session.resets, resets)
        self.assertEqual(status["last_partial"], "")
        self.assertTrue(self.voice.start())
        self.assertEqual(self.capture.starts, 2)
        self.capture.chunks.put(self.block)
        self.wait(lambda: self.voice.status()["listening"])
        status = self.voice.status()
        self.assertEqual(status["stats"]["capture_recoveries"], 1)
        self.assertIsNone(status["error"])
        self.assertEqual(status["capture_exit_code"], 19)
        self.assertIsNotNone(status["last_pcm_age_ms"])
        self.assertEqual(self.asr.loaded, 1)

    def test_stop_interrupts_backoff_and_cannot_reopen_capture(self):
        with mock.patch.object(device_voice, "CAPTURE_RETRY_DELAYS", (2.0,)):
            self.fail_capture()
            self.wait(lambda: self.voice.status()["capture_state"] == "recovering")
            started = time.monotonic()
            self.voice.stop()
            self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(self.capture.starts, 1)
        self.assertEqual(self.voice.status()["capture_state"], "stopped")
        self.assertFalse(self.voice.status()["running"])

    def test_retry_budget_exhaustion_is_visible_and_manual_start_works(self):
        self.capture.outcomes.extend((False, False, False))
        self.fail_capture()
        self.wait(lambda: self.voice.status()["capture_state"] == "failed")
        self.wait(lambda: not self.voice._thread.is_alive())
        status = self.voice.status()
        self.assertFalse(status["running"])
        self.assertFalse(status["capture"])
        self.assertEqual(status["stats"]["capture_retries"], 3)
        self.assertEqual(status["capture_exit_code"], 19)
        self.assertIn("次数已用尽", status["error"])
        self.assertEqual(self.asr.closed, 1)
        self.assertTrue(self.voice.start())
        self.assertTrue(self.voice.status()["listening"])
        self.assertEqual(self.capture.starts, 5)

    def test_process_open_followed_by_eof_cannot_reset_retry_budget(self):
        self.fail_capture()
        for starts in (2, 3, 4):
            self.wait(lambda: self.capture.starts == starts)
            self.fail_capture()
        self.wait(lambda: self.voice.status()["capture_state"] == "failed")
        self.assertEqual(self.capture.starts, 4)
        self.assertEqual(self.voice.status()["stats"]["capture_recoveries"], 0)

    def test_failure_drops_unsent_final_and_suppresses_old_http_failure(self):
        released = threading.Event()
        self.addCleanup(released.set)
        posted = queue.Queue()
        def post(url, **kwargs):
            posted.put(kwargs["json"]["request_id"])
            released.wait(1)
            raise RuntimeError("old HTTP failure")
        self.voice._post = post
        self.assertTrue(self.voice._enqueue_final("第一句", "old-1"))
        self.assertEqual(posted.get(timeout=1), "old-1")
        self.assertTrue(self.voice._enqueue_final("第二句", "old-2"))
        self.assertEqual(posted.get(timeout=1), "old-2")
        self.assertTrue(self.voice._enqueue_final("过期排队话语", "old-pending"))
        self.fail_capture()
        self.wait(lambda: self.voice.status()["stats"]["dropped_requests"] >= 1)
        released.set()
        self.wait(lambda: self.voice.status()["stats"]["active_requests"] == 0)
        self.assertTrue(posted.empty())
        self.output.push.assert_not_called()

    def test_reconnect_preserves_unverified_aec_playback_guard(self):
        self.voice.pause()
        self.fail_capture()
        self.wait(lambda: self.capture.starts == 2)
        self.capture.chunks.put(self.block)
        self.wait(lambda: self.voice.status()["listening"])
        self.assertTrue(self.voice.status()["paused"])
        self.assertEqual(self.gate.feed_calls, 0)


if __name__ == "__main__":
    unittest.main()
