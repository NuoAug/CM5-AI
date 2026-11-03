"""流式ASR控制边界；假引擎只证明适配行为，不代表识别精度。"""
from pathlib import Path
import unittest
import numpy as np

from streaming_asr import StreamingASR, StreamingASRConfig, StreamingASRError, UtteranceLimitError


class FakeStream:
    def __init__(self):
        self.blocks = []
        self.pending = 0
        self.decoded = 0
        self.finished = False
        self.endpoint = False
        self.text = ""

    def accept_waveform(self, sample_rate, samples):
        self.blocks.append((sample_rate, len(samples)))
        self.pending += 1

    def input_finished(self):
        self.finished = True


class FakeEngine:
    def __init__(self):
        self.streams = []

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return stream.pending > 0

    def decode_stream(self, stream):
        stream.pending -= 1
        stream.decoded += 1
        stream.text = "开始" if stream.decoded == 1 else "不要开始引导"

    def get_result(self, stream):
        return stream.text

    def is_endpoint(self, stream):
        return stream.endpoint


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.owner = StreamingASR(StreamingASRConfig(Path("not-loaded")), engine=self.engine)
        self.session = self.owner.create_session()
        self.block = np.zeros(1600, dtype=np.float32)

    def tearDown(self):
        self.owner.close()

    def test_partial_revision_is_not_final_action(self):
        first = self.session.feed(self.block)
        second = self.session.feed(self.block)
        self.assertEqual(first[0].text, "开始")
        self.assertEqual(second[0].text, "不要开始引导")
        self.assertFalse(first[0].is_final)
        self.assertFalse(second[0].is_final)
        final = self.session.finish()
        self.assertEqual(final[0].text, "不要开始引导")
        self.assertTrue(final[0].is_final)
        self.assertEqual(first[0].utterance_id, final[0].utterance_id)
        self.assertEqual(final[0].audio_time_s, 0.2)
        self.assertIsNone(final[0].to_dict()["word_timestamps"])

    def test_each_audio_block_is_passed_once_to_same_stream(self):
        for _ in range(4):
            self.session.feed(self.block)
        stream = self.engine.streams[0]
        self.assertEqual(stream.blocks, [(16000, 1600)] * 4)
        self.assertEqual(stream.decoded, 4)
        self.assertEqual(len(self.engine.streams), 1)
        self.assertFalse(hasattr(self.session, "audio"))

    def test_endpoint_final_releases_old_stream_and_new_identity(self):
        self.session.feed(self.block)
        old = self.session.utterance_id
        self.engine.streams[-1].endpoint = True
        events = self.session.feed(self.block)
        self.assertTrue(events[-1].is_final)
        self.assertEqual(events[-1].utterance_id, old)
        self.assertNotEqual(self.session.utterance_id, old)
        self.assertEqual(len(self.engine.streams), 2)
        self.assertEqual(self.engine.streams[-1].blocks, [])

    def test_changed_partial_only_and_empty_chunk_no_decode(self):
        self.assertEqual(self.session.feed(np.array([], dtype=np.float32)), [])
        self.session.feed(self.block)
        self.session.feed(self.block)
        self.assertEqual(self.session.feed(self.block), [])

    def test_finish_idempotent_and_requires_reset_before_feed(self):
        self.session.feed(self.block)
        self.assertEqual(len(self.session.finish()), 1)
        self.assertEqual(self.session.finish(), [])
        with self.assertRaises(StreamingASRError):
            self.session.feed(self.block)
        self.session.reset()
        self.assertFalse(self.session.feed(self.block)[0].is_final)

    def test_empty_finish_never_pads_unspoken_input(self):
        stream = self.engine.streams[0]
        self.assertEqual(self.session.finish(), [])
        self.assertEqual(stream.blocks, [])

    def test_invalid_audio_never_reaches_engine(self):
        bad = (np.zeros(10, dtype=np.int16), np.zeros((2, 10), dtype=np.float32),
               np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32),
               np.array([1.01], dtype=np.float32), np.zeros(16001, dtype=np.float32))
        for block in bad:
            with self.subTest(shape=block.shape), self.assertRaises(ValueError):
                self.session.feed(block)
        with self.assertRaises(ValueError):
            self.session.feed(self.block, sample_rate=8000)
        self.assertEqual(self.engine.streams[0].blocks, [])

    def test_overflow_does_not_dispatch_truncated_final(self):
        owner = StreamingASR(StreamingASRConfig(Path("none"), max_utterance_seconds=0.15), engine=FakeEngine())
        stream = owner.create_session()
        stream.feed(self.block)
        with self.assertRaises(UtteranceLimitError):
            stream.feed(self.block)
        self.assertIsNone(stream._stream)
        with self.assertRaises(UtteranceLimitError):
            stream.finish()
        with self.assertRaises(UtteranceLimitError):
            stream.feed(self.block)
        stream.reset()
        self.assertFalse(stream.feed(self.block)[0].is_final)
        owner.close()

    def test_sessions_are_independent_and_bounded(self):
        other = self.owner.create_session()
        with self.assertRaises(StreamingASRError):
            self.owner.create_session()
        self.session.feed(self.block)
        self.assertEqual(self.engine.streams[1].blocks, [])
        other.close()
        replacement = self.owner.create_session()
        replacement.close()
        self.assertEqual(self.owner.status()["active_sessions"], 1)

    def test_closed_sessions_reject_all_mutations(self):
        self.session.close()
        for call in (lambda: self.session.feed(self.block), self.session.finish, self.session.reset):
            with self.assertRaises(StreamingASRError):
                call()
        self.assertEqual(self.owner.status()["active_sessions"], 0)

    def test_decoder_must_consume_ready_frames(self):
        self.engine.is_ready = lambda stream: True
        with self.assertRaises(StreamingASRError):
            self.session.feed(self.block)


class ConfigurationTests(unittest.TestCase):
    def test_default_does_not_load_or_require_models(self):
        owner = StreamingASR(StreamingASRConfig(Path("missing")))
        self.assertFalse(owner.status()["ready"])
        with self.assertRaises(StreamingASRError):
            owner.create_session()
        with self.assertRaises(StreamingASRError):
            owner.load()

    def test_invalid_limits(self):
        for kwargs in ({"num_threads": True}, {"max_sessions": 0}, {"sample_rate": 8000},
                       {"max_utterance_seconds": float("nan")}, {"max_chunk_samples": 0},
                       {"tail_padding_seconds": 99}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StreamingASRConfig(Path("none"), **kwargs)


if __name__ == "__main__":
    unittest.main()
