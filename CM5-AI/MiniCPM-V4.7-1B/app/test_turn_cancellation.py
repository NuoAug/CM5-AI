"""请求取消链与合成抢占回归；不冒充真实设备首声测试。"""
import json
import threading
import time
import unittest
from unittest import mock
from turns import TurnManager, TurnCancelled
from output import OutputBus, PRIORITY_DANGER
from device_io import DeviceSink
import server


class TurnCancellationTests(unittest.TestCase):
    def test_replacement_cancels_only_its_session(self):
        manager = TurnManager(max_sessions=2)
        old = manager.begin('a', 'one')
        other = manager.begin('b', 'two')
        new = manager.begin('a', 'three')
        self.assertTrue(old.cancelled.is_set())
        self.assertFalse(other.cancelled.is_set())
        self.assertGreater(new.generation, old.generation)
        self.assertFalse(manager.is_current(old.identity()))

    def test_eviction_cannot_resurrect_old_generation(self):
        manager = TurnManager(max_sessions=1)
        old = manager.begin('a', 'one')
        manager.begin('b', 'two')
        new = manager.begin('a', 'one')
        self.assertTrue(old.cancelled.is_set())
        self.assertNotEqual(old.generation, new.generation)

    def test_late_segment_rejected_after_cancel(self):
        bus = OutputBus()
        token = bus.begin_turn('s', 'r')
        bus.cancel()
        self.assertIsNone(bus.push_segment('晚到声音。', turn=token))
        self.assertFalse(token.compute_finished.is_set())
        token.finish_compute()
        self.assertTrue(token.compute_finished.is_set())

    def test_reconnect_history_cannot_replay_previous_turn(self):
        bus = OutputBus()
        old = bus.begin_turn('s', 'old')
        bus.push_segment('旧回答。', turn=old)
        bus.begin_turn('s', 'new')
        self.assertEqual(bus.history_since(0), [])

    def test_new_turn_blocks_old_completion_and_preserves_new(self):
        bus = OutputBus()
        old = bus.begin_turn('s', 'old')
        new = bus.begin_turn('s', 'new')
        self.assertIsNone(bus.push_segment('旧回答。', turn=old))
        self.assertEqual(bus.push_segment('新回答。', turn=new)['turn'], new.identity())

    def test_urgent_audio_does_not_wait_for_ordinary_synthesis(self):
        started, release, urgent = threading.Event(), threading.Event(), threading.Event()
        writes = []
        def synth(text):
            started.set()
            release.wait(2)
            return b'old'
        class Player:
            def feed(self, wav, cancel):
                if not cancel.is_set():
                    writes.append(wav)
                    if wav == b'urgent': urgent.set()
                return True
            def stop(self): pass
            def end_stream(self): pass
        bus = OutputBus()
        bus.attach_sink(DeviceSink(synth, Player(), emergency_synth=lambda _: b'urgent'))
        old = bus.begin_turn('s', 'old')
        try:
            bus.push_segment('旧回答。', turn=old)
            self.assertTrue(started.wait(.5))
            bus.push('请先停下。', priority=PRIORITY_DANGER)
            self.assertTrue(urgent.wait(.75), '危险提示不应等待旧TTS')
            release.set()
            time.sleep(.05)
            self.assertNotIn(b'old', writes)
        finally:
            release.set()
            bus.detach_sink()

    def test_expired_segment_never_reaches_sink(self):
        bus = OutputBus()
        writes = []
        class Sink:
            def speak(self, payload, cancel): writes.append(payload)
            def stop(self): pass
        bus.attach_sink(Sink())
        try:
            bus.push('过期。', ttl=-1)
            time.sleep(.05)
            self.assertEqual(writes, [])
        finally: bus.detach_sink()

    def test_stream_cancel_blocks_late_tokens_and_history(self):
        session = 'cancel-stream-regression'
        def stream(*args, **kwargs):
            yield '这是一个完整的新回答段落。'
            server.OUTPUT.cancel()
            yield '不应复活的旧片段。'
        client = server.app.test_client()
        with mock.patch.object(server, 'vlm_health', return_value=True), mock.patch.object(server, 'stream_vlm', side_effect=stream):
            response = client.post('/api/talk', json={'task': 'chat', 'question': '聊聊今天的压力', 'session_id': session}, buffered=True)
        text = response.get_data(as_text=True)
        self.assertIn('"cancelled": true', text)
        self.assertNotIn('不应复活', text)
        self.assertFalse(any('不应复活' in x['a'] for x in server.get_session(session)['history']))
