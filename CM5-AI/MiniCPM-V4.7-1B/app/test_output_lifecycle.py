"""连接关闭和播放器重连的有界生命周期屏障测试；无真实HTTP/设备。"""
import threading
import time
import unittest

from turns import CloseDispatcher, TurnManager, TurnCancelled
from output import OutputBus


class CloseLifecycleTests(unittest.TestCase):
    def test_blocked_closes_use_fixed_workers_and_report_queue_saturation(self):
        dispatcher = CloseDispatcher(workers=1, max_pending=1)
        manager = TurnManager(max_sessions=2, close_dispatcher=dispatcher)
        entered, release = threading.Event(), threading.Event()
        def close():
            entered.set()
            release.wait(2)
        try:
            first = manager.begin('s', 'one')
            first.register_close(close)
            first.cancel()
            self.assertTrue(entered.wait(1))
            queued = manager.begin('s', 'two')
            queued.register_close(close)
            queued.cancel()
            rejected = manager.begin('s', 'three')
            rejected.register_close(close)
            rejected.cancel()
            status = manager.close_status()
            self.assertEqual(status['workers'], 1)
            self.assertEqual(status['active'], 1)
            self.assertEqual(status['pending'], 1)
            self.assertEqual(status['rejected'], 1)
            self.assertEqual(rejected.close_status()['connection_close_rejected'], 1)
            self.assertFalse(first.compute_finished.is_set())
            self.assertFalse(queued.compute_finished.is_set())
            self.assertFalse(rejected.compute_finished.is_set())
            self.assertIsNone(rejected.backend_exit_confirmed)
            release.set()
            deadline = time.monotonic() + 1
            while dispatcher.status()['completed'] < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(dispatcher.status()['completed'], 2)
        finally:
            release.set()
            dispatcher.shutdown()

    def test_registered_closers_are_also_bounded_and_deduplicated(self):
        manager = TurnManager()
        token = manager.begin('s', 'one')
        close = lambda: None
        for _ in range(20):
            self.assertTrue(token.register_close(close))
        self.assertEqual(len(token._closers), 1)
        for _ in range(20):
            token.register_close(lambda: None)
        self.assertEqual(len(token._closers), 8)
        self.assertGreater(token.close_status()['connection_close_rejected'], 0)
        token.finish_compute()
        self.assertEqual(token._closers, [])

    def test_fact_commit_guard_serializes_replacement_and_rejects_stale_write(self):
        manager = TurnManager()
        old = manager.begin('s', 'old')
        held, release, replaced = threading.Event(), threading.Event(), threading.Event()
        writes = []
        def commit():
            with old.guard_context():
                held.set()
                release.wait(2)
                writes.append('old-before-replacement')
        writer = threading.Thread(target=commit, daemon=True)
        replacement = threading.Thread(target=lambda: (manager.begin('s', 'new'), replaced.set()), daemon=True)
        try:
            writer.start()
            self.assertTrue(held.wait(1))
            replacement.start()
            self.assertFalse(replaced.wait(.05))
            release.set()
            self.assertTrue(replaced.wait(1))
            with self.assertRaises(TurnCancelled), old.guard_context():
                writes.append('stale')
            self.assertEqual(writes, ['old-before-replacement'])
        finally:
            release.set()
            writer.join(1)
            replacement.join(1)

    def test_fact_commit_guard_serializes_cancel(self):
        manager = TurnManager()
        token = manager.begin('s', 'one')
        held, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        def commit():
            with token.guard_context():
                held.set()
                release.wait(2)
        writer = threading.Thread(target=commit, daemon=True)
        cancellation = threading.Thread(target=lambda: (manager.cancel('s'), cancelled.set()), daemon=True)
        try:
            writer.start()
            self.assertTrue(held.wait(1))
            cancellation.start()
            self.assertFalse(cancelled.wait(.05))
            release.set()
            self.assertTrue(cancelled.wait(1))
            with self.assertRaises(TurnCancelled), token.guard_context():
                pass
        finally:
            release.set()
            writer.join(1)
            cancellation.join(1)


class Sink:
    def __init__(self, *, slow=False):
        self.slow = slow
        self.started = threading.Event()
        self.release = threading.Event()
        self.played = threading.Event()
        self.writes = []
    def prepare(self, payload, cancel):
        if self.slow:
            self.started.set()
            self.release.wait(2)
        return payload['text'].encode('utf-8')
    def speak(self, payload, cancel):
        if not cancel.is_set():
            self.writes.append(payload.get('_prepared_wav'))
            self.played.set()
    def stop(self):
        pass


class OutputAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.bus = OutputBus()
        self.sinks = []
    def tearDown(self):
        for sink in self.sinks:
            sink.release.set()
        threads = [t for t in self.bus._prepare_threads if t is not None]
        if self.bus._sink_thread:
            threads.append(self.bus._sink_thread)
        self.bus.detach_sink()
        for thread in threads:
            thread.join(.6)
    def sink(self, slow=False):
        sink = Sink(slow=slow)
        self.sinks.append(sink)
        return sink

    def test_duplicate_attach_is_idempotent(self):
        sink = self.sink()
        self.bus.attach_sink(sink)
        workers = tuple(self.bus._prepare_threads)
        player = self.bus._sink_thread
        generation = self.bus._attachment_generation
        for _ in range(20):
            self.bus.attach_sink(sink)
        self.assertEqual(tuple(self.bus._prepare_threads), workers)
        self.assertIs(self.bus._sink_thread, player)
        self.assertEqual(self.bus._attachment_generation, generation)
        self.assertEqual(self.bus.stats()['prepare_workers'], 2)

    def test_old_prepared_pcm_cannot_reach_reconnected_sink(self):
        old, new = self.sink(slow=True), self.sink()
        self.bus.attach_sink(old)
        token = self.bus.begin_turn('s', 'old')
        self.bus.push_segment('旧PCM', turn=token, ttl=10)
        self.assertTrue(old.started.wait(1))
        old_stop = self.bus._sink_stop
        self.bus.detach_sink()
        self.bus.attach_sink(new)
        self.assertTrue(old_stop.is_set())
        self.assertIsNot(old_stop, self.bus._sink_stop)
        current = self.bus.begin_turn('s', 'new')
        self.bus.push_segment('新PCM', turn=current, ttl=10)
        old.release.set()
        self.assertTrue(new.played.wait(1))
        self.assertEqual(new.writes, ['新PCM'.encode('utf-8')])
        self.assertEqual(old.writes, [])
        self.assertLessEqual(self.bus.stats()['prepare_workers'], 2)

    def test_many_reconnections_with_blocked_synth_do_not_spawn_unbounded_workers(self):
        old = self.sink(slow=True)
        self.bus.attach_sink(old)
        self.bus.push('阻塞旧合成', ttl=10)
        self.assertTrue(old.started.wait(1))
        ordinary = self.bus._prepare_threads[0]
        for _ in range(15):
            self.bus.detach_sink()
            self.bus.attach_sink(self.sink())
            self.assertIs(self.bus._prepare_threads[0], ordinary)
            self.assertLessEqual(self.bus.stats()['prepare_workers'], 2)
        old.release.set()
        self.assertFalse(any(sink.writes for sink in self.sinks))

    def test_detach_cancels_old_turn_and_clears_pending_history(self):
        old = self.sink(slow=True)
        self.bus.attach_sink(old)
        token = self.bus.begin_turn('s', 'old')
        self.bus.push_segment('旧声音', turn=token, ttl=10)
        self.assertTrue(old.started.wait(1))
        self.bus.finish_utterance('old', token)
        self.bus.detach_sink()
        self.assertTrue(token.cancelled.is_set())
        self.assertEqual(self.bus.history_since(0), [])
        self.assertEqual(self.bus._sink_queue.qsize(), 0)
        self.assertTrue(all(q.qsize() == 0 for q in self.bus._prepare_queues.values()))


if __name__ == '__main__':
    unittest.main()
