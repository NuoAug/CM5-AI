# -*- coding: utf-8 -*-
"""统一输出仲裁测试：优先级、去重、取消、有界队列。"""

import unittest

from output import (
    OutputBus,
    PRIORITY_ANSWER,
    PRIORITY_CONFIRM,
    PRIORITY_DANGER,
    PRIORITY_GUIDE,
)


class OutputBusTests(unittest.TestCase):
    def setUp(self):
        self.bus = OutputBus()
        self.q = self.bus.subscribe()

    def test_danger_preempts_normal(self):
        self.bus.push("普通回答", priority=PRIORITY_ANSWER, source="chat")
        self.bus.push("前方有台阶，请停下", priority=PRIORITY_DANGER, source="guide", reason="stairs")
        items = []
        while not self.q.empty():
            items.append(self.q.get_nowait())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["priority"], PRIORITY_DANGER)
        self.assertEqual(items[0]["text"], "前方有台阶，请停下")
        self.assertTrue(items[0]["preempt"])

    def test_dedupe_same_key(self):
        first = self.bus.push("继续直行", priority=PRIORITY_GUIDE, source="guide", dedupe_key="g")
        second = self.bus.push("继续直行", priority=PRIORITY_GUIDE, source="guide", dedupe_key="g")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(self.bus.stats()["deduped"], 1)

    def test_cancel_increments_epoch_and_drops_normal(self):
        self.bus.push("一长段回答", priority=PRIORITY_ANSWER, source="chat")
        event = self.bus.cancel(min_priority=PRIORITY_GUIDE, source="voice")
        self.assertEqual(event["type"], "cancel")
        self.assertEqual(self.bus.stats()["epoch"], 1)
        items = []
        while not self.q.empty():
            items.append(self.q.get_nowait())
        self.assertEqual([i["type"] for i in items], ["cancel"])

    def test_cancel_keeps_pending_danger(self):
        self.bus.push("前方有台阶，请停下", priority=PRIORITY_DANGER, source="guide")
        self.bus.cancel(min_priority=PRIORITY_GUIDE, source="voice")
        items = []
        while not self.q.empty():
            items.append(self.q.get_nowait())
        self.assertEqual([i["type"] for i in items], ["speech", "cancel"])

    def test_queue_bounded(self):
        for i in range(30):
            self.bus.push(f"句子{i}", priority=PRIORITY_ANSWER, source="chat", dedupe_key=f"k{i}")
        self.assertLessEqual(self.q.qsize(), 16)
        self.assertGreaterEqual(self.bus.stats()["dropped"], 0)

    def test_empty_text_ignored(self):
        self.assertIsNone(self.bus.push("", priority=PRIORITY_ANSWER))
        self.assertIsNone(self.bus.push("   ", priority=PRIORITY_ANSWER))

    def test_priority_guide_preempt_flag(self):
        event = self.bus.push("向左一点", priority=PRIORITY_GUIDE, source="guide")
        self.assertTrue(event["preempt"])
        event = self.bus.push("确认", priority=PRIORITY_CONFIRM, source="voice")
        self.assertFalse(event["preempt"])

    def test_unsubscribe(self):
        self.bus.unsubscribe(self.q)
        self.assertEqual(self.bus.stats()["subscribers"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
