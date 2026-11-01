"""逻辑回归测试：不加载模型，只测纯函数、路由与提示词。"""

import time
import unittest
from datetime import datetime, timedelta, timezone

import server


class SettleTests(unittest.TestCase):
    def test_found_with_comma_not_negated(self):
        got = server.settle_eyes_answer("item", "水杯在右边，没有被挡住", "画面里没看到水杯", "水杯")
        self.assertEqual(got, "水杯在右边，没有被挡住")

    def test_explicit_missing_converges(self):
        got = server.settle_eyes_answer("item", "画面里没看到水杯", "画面里没看到水杯", "水杯")
        self.assertEqual(got, "画面里没看到水杯")

    def test_negation_before_target(self):
        got = server.settle_eyes_answer("toilet", "没有卫生间标识", "画面里没看到卫生间标识")
        self.assertEqual(got, "画面里没看到卫生间标识")

    def test_found_toilet_kept(self):
        got = server.settle_eyes_answer("toilet", "卫生间标识在右边，很近", "兜底")
        self.assertEqual(got, "卫生间标识在右边，很近")

    def test_empty_uses_fallback(self):
        self.assertEqual(server.settle_eyes_answer("exit", "", "画面里没看到出口标识"), "画面里没看到出口标识")


class IntentTests(unittest.TestCase):
    def test_toilet(self):
        self.assertEqual(server.parse_voice_intent("卫生间在哪"), ("toilet", ""))

    def test_date(self):
        self.assertEqual(server.parse_voice_intent("保质期到什么时候"), ("date", ""))

    def test_text(self):
        self.assertEqual(server.parse_voice_intent("读一下上面的字"), ("text", ""))

    def test_chat(self):
        self.assertEqual(server.parse_voice_intent("今天有点烦陪我聊聊"), ("chat", ""))

    def test_news_routes_chat(self):
        self.assertEqual(server.parse_voice_intent("今天有什么新闻"), ("chat", ""))

    def test_item(self):
        intent, item = server.parse_voice_intent("帮我找一下水杯")
        self.assertEqual(intent, "item")
        self.assertEqual(item, "水杯")

    def test_compare(self):
        intent, item = server.parse_voice_intent("比较一下这两个杯子")
        self.assertEqual(intent, "compare")
        self.assertIn("两个杯子", item)

    def test_detail(self):
        self.assertEqual(server.parse_voice_intent("详细说说前面"), ("detail", ""))

    def test_scene(self):
        self.assertEqual(server.parse_voice_intent("前面有什么"), ("scene", ""))

    def test_camera_off_goes_chat(self):
        intent, _ = server.parse_voice_intent("不要打开摄像头")
        self.assertEqual(intent, "chat")

    def test_topic_switch_goes_chat(self):
        intent, _ = server.parse_voice_intent("换个话题，不看东西")
        self.assertEqual(intent, "chat")

    def test_definition_question_goes_chat(self):
        intent, _ = server.parse_voice_intent("这个是什么意思")
        self.assertEqual(intent, "chat")

    def test_definition_with_physical_target_keeps_vision(self):
        intent, _ = server.parse_voice_intent("眼前这个是什么意思")
        self.assertNotEqual(intent, "chat")

    def test_scene_front(self):
        self.assertEqual(server.parse_voice_intent("看前方"), ("scene", ""))


class WakeTests(unittest.TestCase):
    def test_default_wake_homophones(self):
        pat = server.wake_pattern("小V")
        for text in ("小V 前面有什么", "小微前面有什么", "小维你看一下", "小威", "晓V你好", "小 v"):
            self.assertRegex(text, pat)

    def test_default_wake_ignores_plain_speech(self):
        pat = server.wake_pattern("小V")
        self.assertIsNone(pat.search("前面有什么"))

    def test_custom_wake_literal(self):
        pat = server.wake_pattern("你好眼睛")
        self.assertIsNone(pat.search("你好 现在"))
        self.assertRegex("你好眼睛，读一下", pat)

    def test_ack_is_wake_only(self):
        self.assertTrue(server.ACK_ONLY.fullmatch("你好"))
        self.assertFalse(server.ACK_ONLY.fullmatch("看前方"))
        self.assertFalse(server.ACK_ONLY.fullmatch("读一下"))


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class DateTests(unittest.TestCase):
    def test_normal_fields(self):
        lines = [
            {"text": "生产日期：2026-03-15", "score": 0.95, "box": _box(100, 100, 300, 120)},
            {"text": "保质期至：2027-03-14", "score": 0.93, "box": _box(100, 130, 300, 150)},
            {"text": "批号：A2031", "score": 0.91, "box": _box(100, 160, 300, 180)},
        ]
        parsed = server.parse_date_label(lines)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["fields"]["生产日期"], "2026年3月15日")
        self.assertEqual(parsed["fields"]["到期日"], "2027年3月14日")
        self.assertEqual(parsed["conflicts"], [])

    def test_placeholder_rejected(self):
        lines = [{"text": "生产日期：见瓶底", "score": 0.95, "box": _box(100, 100, 300, 120)}]
        parsed = server.parse_date_label(lines)
        self.assertIsNone(parsed)

    def test_invalid_date_rejected(self):
        lines = [{"text": "生产日期：2026-19-45", "score": 0.95, "box": _box(100, 100, 300, 120)}]
        self.assertIsNone(server.parse_date_label(lines))

    def test_missing_score_not_trusted(self):
        lines = [{"text": "生产日期：2026-03-15", "score": None, "box": _box(100, 100, 300, 120)}]
        self.assertIsNone(server.parse_date_label(lines))

    def test_conflict_rejected(self):
        lines = [
            {"text": "生产日期：2026-03-15", "score": 0.95, "box": _box(100, 100, 300, 120)},
            {"text": "生产日期：2027-01-01", "score": 0.95, "box": _box(100, 160, 300, 180)},
        ]
        parsed = server.parse_date_label(lines)
        self.assertIsNotNone(parsed)
        self.assertNotIn("生产日期", parsed["fields"])
        self.assertIn("生产日期", parsed["conflicts"])

    def test_package_date_not_production(self):
        lines = [{"text": "包装日期：2026-03-15", "score": 0.95, "box": _box(100, 100, 300, 120)}]
        parsed = server.parse_date_label(lines)
        self.assertIn("包装日期", parsed["fields"])
        self.assertNotIn("生产日期", parsed["fields"])

    def test_shelf_life_duration_kept(self):
        lines = [{"text": "保质期：12个月", "score": 0.95, "box": _box(100, 100, 300, 120)}]
        parsed = server.parse_date_label(lines)
        self.assertEqual(parsed["fields"]["保质期"], "12个月")

    def test_lowercase_mfg(self):
        lines = [{"text": "mfg 2026-03-15", "score": 0.95, "box": _box(100, 100, 300, 120)}]
        parsed = server.parse_date_label(lines)
        self.assertEqual(parsed["fields"]["生产日期"], "2026年3月15日")

    def test_neighbor_by_box_not_order(self):
        # 值行在列表前面，但空间上紧贴关键词行下方
        lines = [
            {"text": "2026-03-15", "score": 0.9, "box": _box(100, 125, 300, 145)},
            {"text": "生产日期", "score": 0.9, "box": _box(100, 100, 300, 120)},
        ]
        parsed = server.parse_date_label(lines)
        self.assertEqual(parsed["fields"]["生产日期"], "2026年3月15日")

    def test_no_keyword_returns_none(self):
        lines = [{"text": "安全出口 EXIT", "score": 0.99, "box": _box(100, 100, 200, 120)}]
        self.assertIsNone(server.parse_date_label(lines))


class SentenceTests(unittest.TestCase):
    def test_no_silent_drop(self):
        text = "。".join([f"第{i}句" for i in range(20)]) + "。"
        got = server.split_sentences(text, max_len=30)
        self.assertGreater(len(got), 1)
        joined = "".join(got)
        for i in range(20):
            self.assertIn(f"第{i}句", joined)

    def test_question_mark_kept(self):
        got = server.split_sentences("你还好吗？我在这里。")
        self.assertIn("？", "".join(got))

    def test_long_no_punct_hard_split(self):
        text = "这是一个没有任何标点符号的超长句子需要被强制拆分成多段才能继续播报" * 4
        got = server.split_sentences(text, max_len=40)
        self.assertGreater(len(got), 1)
        self.assertTrue(all(len(s) <= 40 for s in got))


class SessionTests(unittest.TestCase):
    def setUp(self):
        with server.SESSION_LOCK:
            server.SESSIONS.clear()

    def test_sessions_isolated(self):
        server.session_begin("s1", "f1")
        server.session_remember("s1", "问题一", "回答一")
        self.assertEqual(server.history_text("s2"), "")

    def test_frame_switch_keeps_chat_and_marks_stale(self):
        server.session_begin("s1", "f1")
        server.session_remember("s1", "左边是什么", "一只猫", visual=True, frame_id="f1")
        server.session_begin("s1", "f2")
        text = server.history_text("s1")
        self.assertIn("左边是什么", text)
        self.assertIn("旧画面已失效", text)

    def test_turn_limit(self):
        server.session_begin("s1", "f1")
        for i in range(6):
            server.session_remember("s1", f"问题{i}", "回答" * 20)
        self.assertLessEqual(len(server.get_session("s1")["history"]), server.MAX_HISTORY_TURNS)

    def test_session_cap(self):
        for i in range(server.MAX_SESSIONS + 2):
            server.get_session(f"s{i}")
        self.assertLessEqual(len(server.SESSIONS), server.MAX_SESSIONS)

    def test_ttl_expired(self):
        sess = server.get_session("old")
        with server.SESSION_LOCK:
            sess["updated"] = time.time() - server.SESSION_TTL - 10
        server.get_session("new")
        self.assertNotIn("old", server.SESSIONS)


class PromptTests(unittest.TestCase):
    def test_chat_prompt_has_question_and_history(self):
        prompt = server.prompt_for("chat", context="问：你好 答：我在", question="今天有点烦")
        self.assertIn("今天有点烦", prompt)
        self.assertIn("问：你好 答：我在", prompt)
        self.assertNotIn("1到2句", prompt)

    def test_followup_prompt_allows_image(self):
        prompt = server.build_eyes_prompt("followup", context="左边有一只猫", question="它是什么颜色")
        self.assertIn("给了画面", prompt)
        self.assertIn("它是什么颜色", prompt)

    def test_detail_uses_detail_rule(self):
        prompt = server.build_eyes_prompt("detail")
        self.assertIn("三到五句", prompt)
        self.assertNotIn("1到2句", prompt)

    def test_scene_uses_short_rule(self):
        self.assertIn("1到2句", server.build_eyes_prompt("scene"))


class FrameAgeTests(unittest.TestCase):
    def test_recent_frame(self):
        now = datetime.now(timezone.utc).isoformat()
        self.assertLess(server.frame_age_seconds(now), 5)

    def test_empty_returns_none(self):
        self.assertIsNone(server.frame_age_seconds(""))

    def test_bad_format_returns_none(self):
        self.assertIsNone(server.frame_age_seconds("not-a-date"))

    def test_old_frame_detected(self):
        old = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        self.assertGreater(server.frame_age_seconds(old), server.FRAME_MAX_AGE)


class ResourceConfigTests(unittest.TestCase):
    def test_temperature_split(self):
        self.assertEqual(server.temperature_for("chat"), 0.4)
        self.assertEqual(server.temperature_for("scene"), 0.1)

    def test_debug_snapshot_off_by_default(self):
        self.assertFalse(server.DEBUG_SNAPSHOT)
        self.assertIsNone(server.save_debug_image("x.png", None))

    def test_thread_config_positive(self):
        self.assertGreaterEqual(server.OCR_INTRA_THREADS, 1)
        self.assertGreaterEqual(server.OCR_INTER_THREADS, 1)
        self.assertGreaterEqual(server.TTS_THREADS, 1)
        self.assertGreaterEqual(server.ASR_THREADS, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
