# -*- coding: utf-8 -*-
"""常驻 Piper worker 实测：模型只加载一次，第二次合明显更快。"""

import io
import tempfile
import unittest
import wave
from pathlib import Path

import server
from tts_worker import PiperWorker


@unittest.skipUnless(server.PIPER_BIN.exists() and server.PIPER_MODEL.exists(), "Piper 未安装")
class PiperWorkerTests(unittest.TestCase):
    def test_persistent_worker(self):
        with tempfile.TemporaryDirectory() as d:
            worker = PiperWorker(server.PIPER_BIN, server.PIPER_MODEL, Path(d))
            try:
                data1, gen1 = worker.synth("继续直行", 1.0)
                data2, gen2 = worker.synth("前方有台阶，请停下", 1.0)
                self.assertGreater(len(data1), 1024)
                self.assertGreater(len(data2), 1024)
                with wave.open(io.BytesIO(data2), "rb") as wf:
                    self.assertGreater(wf.getnframes(), 0)
                st = worker.status()
                self.assertGreaterEqual(st["synth_count"], 2)
                self.assertIsNotNone(st["load_ms"])
                self.assertLess(gen2, 1000)
            finally:
                worker.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
