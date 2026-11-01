# -*- coding: utf-8 -*-
"""意图规则测试集：加载 testsets/intents.jsonl，逐条核对意图、槽位与动作。"""

import json
import unittest
from pathlib import Path

import intent

CASES = Path(__file__).resolve().parent.parent / "testsets" / "intents.jsonl"


def load_cases():
    rows = []
    with CASES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class GoldenIntentTests(unittest.TestCase):
    def test_golden_cases(self):
        failures = []
        for case in load_cases():
            task = intent.parse(
                case["text"],
                has_context=case.get("has_context", False),
                guide_active=case.get("guide_active", False),
                speech_active=case.get("speech_active", False),
            )
            expect = case["expect"]
            ok = task.intent == expect["intent"]
            for key, value in expect.get("slots", {}).items():
                ok = ok and task.slots.get(key) == value
            for action in expect.get("actions", []):
                ok = ok and action in task.actions
            if not ok:
                failures.append(
                    f"{case['text']!r}: got {task.intent} {task.slots} {task.actions}, "
                    f"want {expect}"
                )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_case_count(self):
        self.assertGreaterEqual(len(load_cases()), 50)


class ControlIntentTests(unittest.TestCase):
    def test_negated_stop_is_not_stop(self):
        for text in ("不要停止引导", "别停", "别停播", "不用关闭引导"):
            task = intent.parse(text, guide_active=True, speech_active=True)
            self.assertEqual(task.intent, "ack", text)

    def test_stop_speech_does_not_stop_guide(self):
        task = intent.parse("别念了", guide_active=True, speech_active=True)
        self.assertEqual(task.intent, "speech_stop")
        self.assertEqual(task.actions, ["cancel_speech"])
        self.assertNotIn("stop_guide", task.actions)

    def test_stop_guide_does_not_cancel_speech(self):
        task = intent.parse("关闭引导", guide_active=True, speech_active=True)
        self.assertEqual(task.intent, "guide_stop")
        self.assertEqual(task.actions, ["stop_guide"])
        self.assertNotIn("cancel_speech", task.actions)

    def test_ambiguous_stop_with_guide_only(self):
        task = intent.parse("停一下", guide_active=True, speech_active=False)
        self.assertEqual(task.intent, "guide_stop")
        self.assertIn("stop_guide", task.actions)

    def test_ambiguous_stop_with_speech_only(self):
        task = intent.parse("停一下", guide_active=False, speech_active=True)
        self.assertEqual(task.intent, "speech_stop")

    def test_ambiguous_stop_with_both(self):
        task = intent.parse("停一下", guide_active=True, speech_active=True)
        self.assertEqual(task.intent, "clarify")
        self.assertIn("停止播报", task.clarify)

    def test_ambiguous_stop_with_none(self):
        task = intent.parse("停一下")
        self.assertEqual(task.intent, "clarify")
        self.assertIn("没有", task.clarify)


class TaskShapeTests(unittest.TestCase):
    def test_task_has_required_fields(self):
        task = intent.parse("前面有什么", session_id="s1")
        for field in ("request_id", "session_id", "needs_image", "priority", "deadline_s", "cancel_key"):
            self.assertTrue(hasattr(task, field))
        self.assertTrue(task.request_id)
        self.assertTrue(task.needs_image)
        self.assertEqual(task.session_id, "s1")

    def test_unknown_does_not_need_image(self):
        task = intent.parse("那个东西呢")
        self.assertEqual(task.intent, "unknown")
        self.assertFalse(task.needs_image)
        self.assertIn("没听清", task.clarify)

    def test_missing_item_asks_back(self):
        for text in ("找一下", "帮我找一下", "有没有"):
            task = intent.parse(text)
            self.assertEqual(task.intent, "clarify", text)

    def test_image_intents_need_image(self):
        for text in ("前面有什么", "读一下", "保质期到什么时候", "水杯在哪", "眼前这个是什么"):
            task = intent.parse(text)
            self.assertTrue(task.needs_image, text)

    def test_confidence_is_rule_strength(self):
        task = intent.parse("停止引导")
        self.assertGreaterEqual(task.confidence, 0.9)
        self.assertEqual(task.source, "rule")


if __name__ == "__main__":
    unittest.main(verbosity=2)
