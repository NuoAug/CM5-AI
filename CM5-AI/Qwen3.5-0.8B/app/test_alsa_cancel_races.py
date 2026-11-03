"""无硬件屏障回归：播放器启动/写入/排空期间的取消窗口。"""
import io
import threading
import unittest
import wave
from unittest import mock

import device_io


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x01" * 6000)
    return buffer.getvalue()


class Pipe:
    def __init__(self):
        self.writes = []
        self.closed = False
    def write(self, data):
        if self.closed:
            raise BrokenPipeError()
        self.writes.append(bytes(data))
        return len(data)
    def flush(self):
        if self.closed:
            raise BrokenPipeError()
    def close(self):
        self.closed = True


class Process:
    def __init__(self):
        self.stdin = Pipe()
        self.killed = threading.Event()
        self.waiting = threading.Event()
        self.release_wait = None
        self.returncode = None
    def poll(self):
        return self.returncode
    def kill(self):
        self.returncode = -9
        self.killed.set()
    def wait(self, timeout=None):
        self.waiting.set()
        if not self.killed.wait(2):
            raise TimeoutError("test process was not stopped")
        if self.release_wait is not None:
            self.release_wait.wait(2)
        return self.returncode


class AlsaCancelRaceTests(unittest.TestCase):
    def setUp(self):
        self.player = device_io.AlsaPlayer()
        self.available = mock.patch.object(self.player, "available", return_value=True)
        self.available.start()
        self.tools = mock.patch.object(device_io, "audio_tools", return_value={"aplay": "fake-aplay"})
        self.tools.start()
        self.threads = []
        self.releases = []
    def tearDown(self):
        for event in self.releases:
            event.set()
        self.player.stop()
        for thread in self.threads:
            thread.join(2)
        self.tools.stop()
        self.available.stop()
    def spawn(self, function):
        thread = threading.Thread(target=function, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread
    def install_stream(self, process):
        self.player._stream_proc = process
        self.player._stream_rate = 16000
        self.player._stream_channels = 1

    def test_cancel_while_popen_blocked_never_writes_late_pcm(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        process = Process()
        def popen(*args, **kwargs):
            entered.set()
            release.wait(2)
            return process
        results = []
        cancel = threading.Event()
        with mock.patch.object(device_io.subprocess, "Popen", side_effect=popen):
            thread = self.spawn(lambda: results.append(self.player.feed(wav_bytes(), cancel)))
            self.assertTrue(entered.wait(1))
            cancel.set()
            self.player.stop()
            release.set()
            thread.join(1)
        self.assertEqual(results, [False])
        self.assertTrue(process.killed.is_set())
        self.assertEqual(process.stdin.writes, [])
        self.assertIsNone(self.player._stream_proc)

    def test_cancel_after_open_does_not_kill_new_generation_stream(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        old, new = Process(), Process()
        original_open = self.player._open_stream
        def blocked_open(*args, **kwargs):
            result = original_open(*args, **kwargs)
            entered.set()
            release.wait(2)
            return result
        cancel = threading.Event()
        results = []
        with mock.patch.object(device_io.subprocess, "Popen", return_value=old), \
                mock.patch.object(self.player, "_open_stream", side_effect=blocked_open):
            thread = self.spawn(lambda: results.append(self.player.feed(wav_bytes(), cancel)))
            self.assertTrue(entered.wait(1))
            cancel.set()
            self.player.stop()
            # 在旧feed继续前，已产生一个真正属于新代次的播放器。
            self.install_stream(new)
            release.set()
            thread.join(1)
        self.assertEqual(results, [False])
        self.assertIs(self.player._stream_proc, new)
        self.assertFalse(new.killed.is_set())
        self.assertEqual(old.stdin.writes, [])

    def test_stop_finds_process_during_audio_drain(self):
        process = Process()
        self.install_stream(process)
        drain = self.spawn(self.player.end_stream)
        self.assertTrue(process.waiting.wait(1))
        stopped = threading.Event()
        self.spawn(lambda: (self.player.stop(), stopped.set()))
        self.assertTrue(stopped.wait(0.5), "stop waited for old audio to drain")
        drain.join(1)
        self.assertFalse(drain.is_alive())
        self.assertTrue(process.killed.is_set())

    def test_device_sink_stop_is_not_blocked_by_drain_lock(self):
        process = Process()
        self.install_stream(process)
        sink = device_io.DeviceSink(synth=lambda text: wav_bytes(), player=self.player)
        speaking = self.spawn(lambda: sink.speak({"text": "原回答", "utterance_id": "old", "is_final": True}, threading.Event()))
        self.assertTrue(process.waiting.wait(1))
        stopped = threading.Event()
        self.spawn(lambda: (sink.stop(), stopped.set()))
        self.assertTrue(stopped.wait(0.5), "sink.stop blocked behind end_stream")
        speaking.join(1)
        self.assertFalse(speaking.is_alive())
        self.assertTrue(process.killed.is_set())

    def test_old_drain_cleanup_does_not_clear_new_process(self):
        old, new = Process(), Process()
        release = threading.Event()
        self.releases.append(release)
        old.release_wait = release
        self.install_stream(old)
        drain = self.spawn(self.player.end_stream)
        self.assertTrue(old.waiting.wait(1))
        self.player.stop()
        self.install_stream(new)
        release.set()
        drain.join(1)
        self.assertIs(self.player._stream_proc, new)
        self.assertFalse(new.killed.is_set())

    def test_cancel_between_pcm_chunks_discards_remaining_audio(self):
        process = Process()
        self.install_stream(process)
        cancel = threading.Event()
        original = process.stdin.write
        def write(data):
            written = original(data)
            cancel.set()
            return written
        process.stdin.write = write
        self.assertFalse(self.player.feed(wav_bytes(), cancel))
        self.assertEqual(len(process.stdin.writes), 1)
        self.assertEqual(len(process.stdin.writes[0]), 4096)
        self.assertTrue(process.killed.is_set())

    def test_cancelled_file_play_does_not_create_process(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(device_io.subprocess, "Popen") as popen:
            self.assertFalse(self.player.play(wav_bytes(), cancel))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
