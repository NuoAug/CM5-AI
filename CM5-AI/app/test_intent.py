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
                speech_active=case.get("speech_active", False),
            )
            expect = case["expect"]
            ok = task.intent == expect["intent"]
            for key, value in expect.get("slots", {}).items():
                ok = ok and task.slots.get(key) == value
            if "actions" in expect:
                ok = ok and task.actions == expect["actions"]
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
        for text in ("不要停止播报", "别停", "别停播", "不用关闭播报"):
            task = intent.parse(text, speech_active=True)
            self.assertEqual(task.intent, "ack", text)
            self.assertEqual(task.actions, [], text)

    def test_explicit_stop_only_cancels_speech(self):
        task = intent.parse("别念了", speech_active=True)
        self.assertEqual(task.intent, "speech_stop")
        self.assertEqual(task.actions, ["cancel_speech"])

    def test_removed_navigation_commands_return_capability_message(self):
        for text in ("开始引导", "关闭引导", "打开导航", "不用导航", "不要停止引导",
                     "引导状态", "带我走", "导航到卫生间", "请导航"):
            with self.subTest(text=text):
                task = intent.parse(text, has_context=True, speech_active=True)
                self.assertEqual(task.intent, "unsupported")
                self.assertEqual(task.clarify, intent.UNSUPPORTED_NAVIGATION_REPLY)
                self.assertFalse(task.needs_image)
                self.assertEqual(task.slots, {})
                self.assertEqual(task.actions, [])

    def test_passage_questions_do_not_become_visual_followups(self):
        for text in ("现在能走吗", "前面能走吗", "前面通不通", "可以直行吗",
                     "能不能走楼梯", "前面能不能走楼梯", "可以过马路吗", "这里可以通过吗"):
            with self.subTest(text=text):
                task = intent.parse(text, has_context=True)
                self.assertEqual(task.intent, "unsupported")
                self.assertFalse(task.needs_image)
                self.assertEqual(task.actions, [])

    def test_non_navigation_questions_keep_regular_routing(self):
        for text, expected in (("今天可以过生日吗", "chat"), ("能不能过生日", "unknown"),
                               ("能不能通过考试", "unknown"), ("带我去聊聊音乐", "chat")):
            with self.subTest(text=text):
                task = intent.parse(text)
                self.assertEqual(task.intent, expected)
                self.assertEqual(task.actions, [])

    def test_camera_off_only_closes_camera(self):
        task = intent.parse("关闭摄像头", speech_active=True)
        self.assertEqual(task.intent, "camera_off")
        self.assertEqual(task.actions, ["close_camera"])

    def test_topic_switch_has_no_device_actions(self):
        task = intent.parse("只聊天", speech_active=True)
        self.assertEqual(task.intent, "chat")
        self.assertEqual(task.actions, [])

    def test_ambiguous_stop_with_speech(self):
        task = intent.parse("停一下", speech_active=True)
        self.assertEqual(task.intent, "speech_stop")
        self.assertEqual(task.actions, ["cancel_speech"])

    def test_ambiguous_stop_without_speech(self):
        task = intent.parse("停一下")
        self.assertEqual(task.intent, "clarify")
        self.assertEqual(task.clarify, "现在没有正在进行的播报。")
        self.assertEqual(task.actions, [])

    def test_navigation_definition_remains_chat(self):
        task = intent.parse("导航是什么意思")
        self.assertEqual(task.intent, "chat")


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
        task = intent.parse("停止播报")
        self.assertGreaterEqual(task.confidence, 0.9)
        self.assertEqual(task.source, "rule")


if __name__ == "__main__":
    unittest.main(verbosity=2)
