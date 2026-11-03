# -*- coding: utf-8 -*-
"""聊天流式 SSE 到输出总线的回归：段先推，done 不重播全文。"""

import json
import time
import unittest
from unittest import mock

import server
from output import OUTPUT


def parse_sse(resp):
    events = []
    buf = ""
    for chunk in resp.response:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", "ignore")
        buf += chunk
        while "\n\n" in buf:
            raw, buf = buf.split("\n\n", 1)
            for line in raw.splitlines():
                if line.startswith("data:"):
                    try:
                        events.append(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        pass
    return events


class TalkStreamTests(unittest.TestCase):
    def test_segments_before_done(self):
        def fake_stream(prompt, img_b64=None, max_tokens=160, temperature=0.3):
            for part in ("前方近处是黄", "色盲道，往远", "是灰色路面。", "左边是建", "筑物。"):
                yield part
                time.sleep(0.01)

        client = server.app.test_client()
        q = OUTPUT.subscribe()
        with mock.patch.object(server, "vlm_health", return_value=True), \
                mock.patch.object(server, "stream_vlm", side_effect=fake_stream):
            resp = client.post(
                "/api/talk",
                json={"task": "chat", "question": "前面有什么", "session_id": "talk-test", "request_id": "talk-1"},
                buffered=False,
            )
            self.assertEqual(resp.status_code, 200)
            events = parse_sse(resp)
        OUTPUT.unsubscribe(q)
        segments = [e["segment"] for e in events if e.get("segment")]
        self.assertGreaterEqual(len(segments), 2, events)
        self.assertTrue(events[-1].get("done"))
        self.assertNotIn("sentence", events[-1])
        # done 之后不能再推一次完整回答
        speech = []
        while not q.empty():
            speech.append(q.get_nowait())
        self.assertTrue(all(s.get("is_final") is not None for s in speech if s.get("type") == "speech"))
        self.assertEqual(sum(1 for s in speech if s.get("is_final")), 1)
        self.assertEqual(events[0].get("trace_id"), "talk-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
