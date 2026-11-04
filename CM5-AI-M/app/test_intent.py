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

    def test_guidance_commands_only_submit_main_control_intents(self):
        cases = (("开始引导", "start_guidance"), ("暂停引导", "pause_guidance"),
                 ("继续引导", "resume_guidance"), ("关闭引导", "end_guidance"),
                 ("重复引导", "repeat_guidance"))
        for text, action in cases:
            with self.subTest(text=text):
                task = intent.parse(text)
                self.assertEqual(task.intent, "control_intent")
                self.assertEqual(task.slots, {"action": action})
                self.assertFalse(task.needs_image)
                self.assertEqual(task.actions, [])

    def test_negation_revision_and_partial_do_not_submit_control(self):
        for text in ("不要开始引导", "不要停止引导", "不是让你结束引导", "开始引导改成暂停引导"):
            task = intent.parse(text)
            self.assertEqual(task.intent, "clarify", text)
            self.assertEqual(task.actions, [])
        self.assertEqual(intent.parse("开始引导", is_final=False).intent, "clarify")

    def test_passage_questions_only_query_main_control(self):
        for text in ("现在能走吗", "前面能走吗", "前面通不通", "可以直行吗",
                     "能不能走楼梯", "前面能不能走楼梯", "可以过马路吗", "这里可以通过吗",
                     "离我多远", "前方距离多少米", "应该怎么绕", "引导状态",
                     "我该左转还是右转", "这条路怎么走", "这门过得去吗", "往前走会撞到吗"):
            with self.subTest(text=text):
                task = intent.parse(text, has_context=True)
                self.assertEqual(task.intent, "scene_query")
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


class SessionRoutingTests(unittest.TestCase):
    def test_chat_continues_even_when_a_photo_is_cached(self):
        context = intent.SessionContext(topic="chat", has_frame=True)
        for text in ("那你觉得呢", "为什么", "为什么我总是在意别人的评价", "那该怎么办",
                     "然后呢", "可以详细说说吗", "你刚才说的前面有什么问题", "我比较难过"):
            with self.subTest(text=text):
                task = intent.parse(text, context=context)
                self.assertEqual(task.intent, "chat")
                self.assertFalse(task.needs_image)

    def test_visual_followup_depends_on_current_topic_and_frame(self):
        text = "那你觉得呢"
        self.assertEqual(intent.parse(text, context=intent.SessionContext("vision", True)).intent, "followup")
        task = intent.parse(text, context=intent.SessionContext("vision", False))
        self.assertEqual(task.intent, "clarify")
        self.assertFalse(task.needs_image)
        # An explicit question about the visible scene can change topic.
        task = intent.parse("眼前这个是什么", context=intent.SessionContext("chat", False))
        self.assertEqual(task.intent, "followup")
        self.assertTrue(task.needs_image)

    def test_abstract_comparison_and_general_cost_are_chat(self):
        context = intent.SessionContext("chat", True)
        for text in ("这两种解决方式有什么区别", "比较一下这两个方案", "保质期和到期日有什么区别",
                     "学费多少钱", "工资和房租的价格对比", "为什么和朋友有距离感"):
            with self.subTest(text=text):
                task = intent.parse(text, context=context)
                self.assertEqual(task.intent, "chat")
                self.assertFalse(task.needs_image)

    def test_natural_visible_comparison_is_not_scene_description(self):
        for text in ("这两个东西有什么区别", "这俩有什么差别", "比较一下这两本天气手册",
                     "左边和右边这两个一样吗"):
            with self.subTest(text=text):
                task = intent.parse(text)
                self.assertEqual(task.intent, "compare")
                self.assertTrue(task.needs_image)
                self.assertTrue(task.slots["item"])
                self.assertNotIn("comparison", task.slots)

    def test_price_routes_distinguish_one_label_and_two_products(self):
        for text, target in (("这个多少钱", "这个"), ("帮我看看右边那包的价签", "右边那包的"),
                             ("请告诉我这瓶多少块", "这瓶")):
            with self.subTest(text=text):
                task = intent.parse(text)
                self.assertEqual(task.intent, "read_price")
                self.assertEqual(task.slots["item"], target)
                self.assertTrue(task.needs_image)
        for text in ("这两个哪个更便宜", "这两个分别多少钱", "哪款更划算", "两包的价格一样吗"):
            with self.subTest(text=text):
                task = intent.parse(text)
                self.assertEqual(task.intent, "compare")
                self.assertEqual(task.slots["comparison"], "price")
                self.assertTrue(task.needs_image)
        self.assertEqual(intent.parse("这几块砖是什么颜色").intent, "followup")

    def test_natural_passage_queries_keep_main_control_authority(self):
        for text in ("请告诉我能不能从椅子旁边穿过去", "我能从门框边上过吗",
                     "能不能从桌椅缝里钻过去", "我从书架旁挤得过去吗", "这个缝够我侧身过吗",
                     "能从它们中间走过去不", "请看前方我可以迈过去吗"):
            with self.subTest(text=text):
                task = intent.parse(text, context=intent.SessionContext("chat", True))
                self.assertEqual(task.intent, "scene_query")
                self.assertFalse(task.needs_image)
                self.assertEqual(task.actions, [])
                self.assertNotIn("action", task.slots)


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
