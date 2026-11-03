# -*- coding: utf-8 -*-
"""流式语义分段测试。"""

import unittest

from stream_text import SemanticSegmenter, normalize_for_tts, segment_text


class SegmentTests(unittest.TestCase):
    def test_first_short_later_longer(self):
        text = "前方近处是黄色盲道，往远是灰色路面，左边是盲道，右边是路面，前方是盲道，中间是路面，远处也是路面。"
        pieces = segment_text(text)
        self.assertGreaterEqual(len(pieces), 2)
        self.assertLessEqual(len(pieces[0]), 25)
        self.assertGreaterEqual(len(pieces[0]), 12)

    def test_does_not_split_date_or_unit(self):
        text = "生产日期是2026年3月15日，保质期12个月，距离约3.5米。"
        pieces = segment_text(text)
        joined = "".join(pieces)
        self.assertIn("2026年3月15日", joined)
        for piece in pieces:
            self.assertNotEqual(piece.strip(), "2026年")
            self.assertNotEqual(piece.strip(), "3.5")

    def test_negation_stays_with_clause(self):
        seg = SemanticSegmenter(first_min=12, first_max=25, later_min=25, later_max=50)
        pieces = seg.feed("前方不")
        pieces += seg.feed("能通行，请向左绕行。")
        pieces += seg.flush()
        joined = "".join(pieces)
        self.assertIn("不能通行", joined)

    def test_streaming_emits_before_end(self):
        seg = SemanticSegmenter()
        out = []
        for ch in "前方近处是黄色盲道，往远是灰色路面，左边是盲道，右边是路面。":
            out.extend(seg.feed(ch))
        out.extend(seg.flush())
        self.assertGreaterEqual(len(out), 2)
        self.assertTrue(out[-1].endswith("。"))

    def test_modal_walk_claim_not_split(self):
        # 切点正好落在“可以|走”之间时，不能先播“可以”，再补“走”。
        pieces = segment_text("甲" * 23 + "可以走，但是后面不能通行。")
        self.assertFalse(pieces[0].endswith("可以"))
        self.assertIn("可以走", "".join(pieces[1:]))

    def test_final_flush_short_answer(self):
        self.assertEqual(segment_text("前方有台阶，请停下。"), ["前方有台阶，请停下。"])

    def test_empty(self):
        self.assertEqual(segment_text(""), [])

    def test_markdown_removed_numbers_kept(self):
        s = normalize_for_tts("**注意**：前方 3.5 米处有 `1` 个障碍。")
        self.assertNotIn("**", s)
        self.assertIn("3.5", s)
        self.assertIn("1", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
