# -*- coding: utf-8 -*-
"""请求时间线与汇总测试。"""

import tempfile
import unittest
from pathlib import Path

from trace import TraceStore


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = TraceStore(Path(self.dir.name) / "traces.jsonl")

    def tearDown(self):
        self.dir.cleanup()

    def test_metrics(self):
        t = self.store.start("t1", "voice")
        t.mark("speech_end", wall_ms=1000)
        t.mark("vad_end", wall_ms=1700)
        t.mark("asr_end", wall_ms=2200)
        t.mark("intent_end", wall_ms=2250)
        t.mark("first_segment", wall_ms=5000)
        t.mark("tts_ready", wall_ms=5600, value_ms=300)
        t.mark("play_start", wall_ms=5800)
        t.mark("answer", wall_ms=5200)
        self.store.finish("t1")
        row = self.store.records(1)[0]
        m = row["metrics"]
        self.assertEqual(m["vad_tail_ms"], 700)
        self.assertEqual(m["speech_to_asr_ms"], 1200)
        self.assertEqual(m["speech_to_first_segment_ms"], 4000)
        self.assertEqual(m["speech_to_tts_ready_ms"], 4600)
        self.assertEqual(m["speech_to_play_start_ms"], 4800)
        self.assertEqual(m["tts_gen_ms"], 300)

    def test_mark_after_finish_updates_record(self):
        t = self.store.start("t2", "talk")
        t.mark("request_start", wall_ms=1000)
        t.mark("answer", wall_ms=3000)
        self.store.finish("t2")
        self.store.mark("t2", "play_start", wall_ms=3500)
        row = self.store.records(1)[0]
        self.assertTrue(any(x["stage"] == "play_start" for x in row["marks"]))

    def test_first_play_start_wins(self):
        t = self.store.start("t3", "talk")
        t.mark("speech_end", wall_ms=1000)
        t.mark("play_start", wall_ms=5800, segment_id="s1")
        t.mark("play_start", wall_ms=9000, segment_id="s2")
        self.store.finish("t3")
        self.assertEqual(self.store.records(1)[0]["metrics"]["speech_to_play_start_ms"], 4800)

    def test_summary_percentiles(self):
        for i in range(5):
            t = self.store.start(f"t{i}", "eyes")
            t.mark("request_start", wall_ms=1000)
            t.mark("answer", wall_ms=1000 + (i + 1) * 100)
            self.store.finish(f"t{i}")
        summary = self.store.summary()["eyes"]
        self.assertEqual(summary["n"], 5)
        self.assertEqual(summary["metrics"]["request_to_answer_ms"]["p50"], 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
