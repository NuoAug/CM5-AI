"""构造照片/协议的图片会话回归，不冒充真实视觉模型或设备验收。"""
from copy import deepcopy
import hashlib
import json
import unittest

from visual_context import VisualContextError, VisualContextStore


class Clock:
    def __init__(self):
        self.value = 1000

    def __call__(self):
        return self.value


def provider_frame(frame_id="f1", content=b"picture", captured=900, lease=2000):
    return {"schema_version": "interaction.v0.1", "type": "frame.result", "status": "ok",
            "producer": {"role": "test_fixture", "instance_id": "fixture-main"},
            "evidence_mode": "synthetic", "clock_id": "fixture-clock",
            "data": {"frame_id": frame_id, "captured_mono_ms": captured,
                     "source_kind": "fixture", "is_historical": False,
                     "width_px": 100, "height_px": 80, "color_space": "RGB",
                     "parent_frame_id": None, "transform_to_parent": None,
                     "quality": {"sharpness": "unknown"},
                     "asset": {"asset_id": "asset-" + frame_id, "transport": "file",
                               "path": "not-a-real-file.jpg", "encoding": "jpeg",
                               "sha256": hashlib.sha256(content).hexdigest(),
                               "available_until_mono_ms": lease}}}


class VisualContextTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = VisualContextStore(now_ms=self.clock)

    def upload(self, frame_id="front", content=b"front", **kwargs):
        return self.store.remember_upload("s1", frame_id, content, width_px=100, height_px=80, **kwargs)

    def assert_code(self, code, func, *args, **kwargs):
        with self.assertRaises(VisualContextError) as caught:
            func(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_followup_keeps_frame_content_and_unknown_capture_time(self):
        original = self.upload()
        self.clock.value += 100
        repeated = self.upload()
        self.assertEqual(original["frame_id"], repeated["frame_id"])
        self.assertIsNone(repeated["captured_mono_ms"])
        self.assertTrue(repeated["is_historical"])
        self.assertEqual(self.store.image_bytes("s1", "front"), b"front")

    def test_same_frame_id_rejects_changed_content(self):
        self.upload()
        self.assert_code("INVALID_FRAME_REF", self.upload, content=b"different")
        self.assertEqual(self.store.image_bytes("s1", "front"), b"front")

    def test_same_frame_id_rejects_refreshed_capture_time(self):
        self.upload(captured_mono_ms=500)
        self.assert_code("INVALID_FRAME_REF", self.upload, captured_mono_ms=700)
        self.assertEqual(self.store.current_frame("s1")["captured_mono_ms"], 500)

    def test_provider_digest_uses_actual_asset_bytes(self):
        result = provider_frame(content=b"correct")
        self.assert_code("INVALID_FRAME_REF", self.store.remember_frame, "s1", result, asset_bytes=b"wrong")

    def test_expired_asset_not_accepted(self):
        self.assert_code("STALE_FRAME", self.store.remember_frame, "s1", provider_frame(lease=999), asset_bytes=b"picture")

    def test_metadata_pin_checks_lease_without_refreshing_capture(self):
        result = provider_frame()
        self.store.remember_frame("s1", result, asset_bytes=b"picture")
        self.clock.value = 2000
        self.assert_code("STALE_FRAME", self.store.current_frame, "s1")
        historical_metadata = self.store.current_frame("s1", require_asset=False)
        self.assertEqual(historical_metadata["captured_mono_ms"], 900)
        self.assertFalse(historical_metadata["asset_available"])
        self.assert_code("NO_FRAME", self.store.image_bytes, "s1", "f1")

    def test_explicit_retained_copy_survives_external_lease_as_original_photo(self):
        self.store.remember_frame("s1", provider_frame(), asset_bytes=b"picture", retain_copy=True)
        self.clock.value = 2001
        frame = self.store.current_frame("s1")
        self.assertTrue(frame["retained_copy"])
        self.assertEqual(frame["captured_mono_ms"], 900)
        self.assertTrue(frame["historical_reference"])
        self.assertEqual(frame["asset"]["available_until_mono_ms"], 2000)
        self.assertEqual(self.store.image_bytes("s1", "f1"), b"picture")

    def test_reissued_asset_lease_does_not_change_capture_identity(self):
        result = provider_frame()
        self.store.remember_frame("s1", result, asset_bytes=b"picture")
        result["data"]["asset"]["available_until_mono_ms"] = 3000
        self.clock.value = 1500
        renewed = self.store.remember_frame("s1", result, asset_bytes=b"picture")
        self.assertEqual(renewed["captured_mono_ms"], 900)
        self.assertEqual(renewed["asset"]["available_until_mono_ms"], 3000)

    def test_fixture_cannot_be_relabelled_live(self):
        result = provider_frame()
        result["evidence_mode"] = "live"
        self.assert_code("INVALID_PAYLOAD", self.store.remember_frame, "s1", result, asset_bytes=b"picture")

    def test_missing_clock_rejected(self):
        result = provider_frame()
        result["clock_id"] = None
        self.assert_code("CLOCK_UNVERIFIED", self.store.remember_frame, "s1", result, asset_bytes=b"picture")

    def test_missing_or_malformed_frame_data_returns_structured_error(self):
        for data in (None, [], {"bad": "shape"}):
            result = provider_frame()
            result["data"] = data
            self.assert_code("INVALID_PAYLOAD" if data is None or data == [] else "INVALID_FRAME_REF",
                             self.store.remember_frame, "s1", result, asset_bytes=b"picture")

    def test_sessions_are_isolated_and_no_default_session_is_created(self):
        self.upload()
        self.assert_code("NO_FRAME", self.store.frame, "s2", "front")
        for sid in (None, "", [], "x" * 129):
            self.assert_code("INVALID_PAYLOAD", self.store.snapshot, sid)

    def test_old_session_expiry_releases_owned_bytes(self):
        store = VisualContextStore(now_ms=self.clock, session_ttl_ms=100)
        store.remember_upload("old", "f", b"bytes", width_px=2, height_px=2)
        self.clock.value += 100
        snap = store.snapshot("new")
        self.assertEqual(snap["cached_bytes"], 0)
        self.assert_code("NO_FRAME", store.frame, "old", "f")

    def test_max_sessions_evicts_least_recently_used_content(self):
        store = VisualContextStore(now_ms=self.clock, max_sessions=1)
        store.remember_upload("one", "f", b"one", width_px=2, height_px=2)
        store.remember_upload("two", "f", b"two", width_px=2, height_px=2)
        self.assertEqual(store.image_bytes("two", "f"), b"two")
        self.assert_code("NO_FRAME", store.frame, "one", "f")

    def test_frame_eviction_removes_only_its_ocr_and_object_views(self):
        self.store = VisualContextStore(now_ms=self.clock, max_frames_per_session=2)
        self.upload("one", b"one")
        self.store.bind_object("s1", "one", [0, 0, 50, 50], object_id="old")
        self.store.add_ocr("s1", "one", [{"text": "旧文字", "roi_xyxy_px": [1, 1, 20, 20], "score": .9}], object_id="old")
        self.upload("two", b"two")
        self.upload("three", b"three")
        self.assert_code("NO_FRAME", self.store.frame, "s1", "one")
        self.assertEqual(self.store.ocr_for("s1", "three"), [])
        self.assertEqual(self.store.snapshot("s1")["objects"], [])

    def test_comparison_pins_prevent_silent_eviction(self):
        self.store = VisualContextStore(now_ms=self.clock, max_frames_per_session=2)
        self.upload("one", b"one", slot="A")
        self.upload("two", b"two", slot="B")
        self.assert_code("BUSY", self.upload, "three", b"three")
        self.assertEqual(self.store.frame_for_slot("s1", "A")["frame_id"], "one")
        self.assertEqual(self.store.frame_for_slot("s1", "B")["frame_id"], "two")

    def test_byte_budget_is_hard_limit_and_release_recovers_capacity(self):
        self.store = VisualContextStore(now_ms=self.clock, max_cached_bytes=10)
        self.upload("one", b"123456")
        self.assert_code("BUSY", self.upload, "two", b"abcdef")
        self.assertEqual(self.store.snapshot("s1")["cached_bytes"], 6)
        self.store.release_frame("s1", "one")
        self.upload("two", b"abcdef")
        self.assertEqual(self.store.snapshot("s1")["cached_bytes"], 6)

    def test_flip_links_explicit_object_without_migrating_ocr(self):
        self.upload()
        self.store.bind_object("s1", "front", [1, 1, 50, 50], object_id="package")
        self.store.add_ocr("s1", "front", [{"text": "商品A", "roi_xyxy_px": [1, 1, 40, 10], "score": .9}], object_id="package")
        self.upload("back", b"back", relation="flip", object_id="package")
        self.assertEqual(self.store.snapshot("s1")["current_object_id"], "package")
        self.assertEqual(self.store.ocr_for("s1", "back", object_id="package"), [])
        self.assertEqual(self.store.ocr_for("s1", "front", object_id="package")[0]["raw_text"], "商品A")
        self.assert_code("INVALID_TARGET", self.upload, "other", b"other", relation="flip")

    def test_correction_moves_reference_not_ocr_or_geometry(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 40, 80], object_id="left")
        self.store.bind_object("s1", "front", [60, 0, 100, 80], object_id="right")
        self.store.add_ocr("s1", "front", [{"text": "左边的字", "roi_xyxy_px": [0, 0, 20, 10], "score": .9}], object_id="left")
        self.store.correct_object("s1", "right")
        self.assertEqual(self.store.snapshot("s1")["current_object_id"], "right")
        self.assertEqual(self.store.ocr_for("s1", "front", object_id="right"), [])
        self.assert_code("INVALID_TARGET", self.store.bind_object, "s1", "front", [10, 0, 50, 80], object_id="left")

    def test_swapping_A_B_keeps_their_own_frame_and_ocr(self):
        self.upload("one", b"one", slot="A")
        self.upload("two", b"two", slot="B")
        self.store.add_ocr("s1", "one", [{"text": "一", "roi_xyxy_px": [0, 0, 10, 10]}])
        self.store.swap_slots("s1")
        self.assertEqual(self.store.frame_for_slot("s1", "A")["frame_id"], "two")
        self.assertEqual(self.store.frame_for_slot("s1", "B")["frame_id"], "one")
        self.assertEqual(self.store.ocr_for("s1", "two"), [])

    def test_ocr_rejects_invalid_roi_score_and_changed_source_atomically(self):
        self.upload()
        rows = [{"text": "valid", "roi_xyxy_px": [0, 0, 10, 10], "score": .9}]
        for patch in ({"roi_xyxy_px": [0, 0, 101, 80]}, {"score": float("nan")}, {"frame_id": "different"}, {"roi_xyxy_px": [1, 1, 0, 0]}):
            bad = dict(rows[0], **patch)
            with self.assertRaises(VisualContextError):
                self.store.add_ocr("s1", "front", rows + [bad])
            self.assertEqual(self.store.ocr_for("s1", "front"), [])

    def test_ocr_cannot_bind_an_object_from_a_different_photo(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 10, 10], object_id="first")
        self.upload("back", b"back")
        self.assert_code("INVALID_TARGET", self.store.add_ocr, "s1", "back", [], object_id="first")

    def test_object_ocr_outside_selected_roi_is_rejected(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 40, 80], object_id="left")
        self.assert_code("INVALID_TARGET", self.store.add_ocr, "s1", "front",
                         [{"text": "右物品价签", "roi_xyxy_px": [60, 0, 90, 10], "score": .9}], object_id="left")
        self.assertEqual(self.store.ocr_for("s1", "front", object_id="left"), [])

    def test_old_picture_must_be_selected_not_reused_as_new_flip(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 40, 80], object_id="item")
        self.upload("back", b"back", relation="flip", object_id="item")
        self.assert_code("INVALID_FRAME_REF", self.upload, "front", b"front", relation="flip", object_id="item")
        self.store.select_frame("s1", "front")
        self.assertEqual(self.store.current_frame("s1")["frame_id"], "front")

    def test_ocr_replace_only_replaces_matching_recognizer_and_object(self):
        self.store = VisualContextStore(now_ms=self.clock, max_ocr_per_frame=3)
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 50, 80], object_id="left")
        row = lambda text: [{"text": text, "roi_xyxy_px": [0, 0, 20, 10], "score": .9}]
        self.store.add_ocr("s1", "front", row("保留的对象证据"), object_id="left", recognizer="RapidOCR")
        self.store.add_ocr("s1", "front", row("其他运行器"), recognizer="OtherOCR")
        for i in range(10):
            self.store.add_ocr("s1", "front", row("本次" + str(i)), recognizer="RapidOCR", replace=True)
        unbound = self.store.ocr_for("s1", "front")
        self.assertEqual([r["raw_text"] for r in unbound], ["其他运行器", "本次9"])
        self.assertEqual(self.store.ocr_for("s1", "front", object_id="left")[0]["raw_text"], "保留的对象证据")

    def test_invalid_ocr_replacement_preserves_previous_evidence(self):
        self.upload()
        self.store.add_ocr("s1", "front", [{"text": "有效旧证据", "roi_xyxy_px": [0, 0, 10, 10], "score": .9}], recognizer="RapidOCR")
        before = self.store.ocr_for("s1", "front")
        self.assert_code("INVALID_FRAME_REF", self.store.add_ocr, "s1", "front", [{"text": "无效新框", "roi_xyxy_px": [0, 0, 101, 10]}], recognizer="RapidOCR", replace=True)
        self.assertEqual(self.store.ocr_for("s1", "front"), before)

    def test_explicit_ocr_association_only_moves_unbound_rows_inside_roi(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 40, 80], object_id="left")
        self.store.bind_object("s1", "front", [60, 0, 100, 80], object_id="right")
        added = self.store.add_ocr("s1", "front", [
            {"text": "3元/斤", "roi_xyxy_px": [0, 0, 20, 10], "score": .9},
            {"text": "5元/盒", "roi_xyxy_px": [60, 0, 90, 10], "score": .9}])
        left = self.store.associate_ocr("s1", "front", "left")
        self.assertEqual(left["associated_count"], 1)
        self.assertEqual(left["associated_row_ids"], [added[0]["row_id"]])
        self.assertEqual(left["evidence"]["fields"]["price"][0]["unit"], "斤")
        self.assertEqual(len(self.store.ocr_for("s1", "front")), 1)
        self.assertEqual(self.store.associate_ocr("s1", "front", "left")["associated_count"], 0)
        right = self.store.associate_ocr("s1", "front", "right", row_ids=[added[1]["row_id"]])
        self.assertEqual(right["evidence"]["fields"]["price"][0]["value"], "5")
        self.assert_code("INVALID_TARGET", self.store.associate_ocr, "s1", "front", "right", row_ids=[added[0]["row_id"]])

    def test_bad_explicit_association_list_is_atomic_and_cannot_cross_frame(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 40, 80], object_id="left")
        rows = self.store.add_ocr("s1", "front", [
            {"text": "inside", "roi_xyxy_px": [0, 0, 20, 10]},
            {"text": "outside", "roi_xyxy_px": [60, 0, 90, 10]}])
        self.assert_code("INVALID_TARGET", self.store.associate_ocr, "s1", "front", "left", row_ids=[r["row_id"] for r in rows])
        self.assertEqual(len(self.store.ocr_for("s1", "front")), 2)
        self.upload("new", b"new")
        self.store.bind_object("s1", "new", [0, 0, 100, 80], object_id="new-item")
        self.assert_code("INVALID_FRAME_REF", self.store.associate_ocr, "s1", "new", "new-item", row_ids=[rows[0]["row_id"]])

    def test_snapshot_is_json_metadata_without_pixel_payload(self):
        self.upload(content=b"sensitive-picture")
        snap = self.store.snapshot("s1")
        self.assertNotIn("sensitive-picture", json.dumps(snap))
        snap["frames"][0]["asset"]["sha256"] = "tampered"
        self.assertEqual(self.store.image_bytes("s1", "front"), b"sensitive-picture")
        self.assertNotEqual(self.store.current_frame("s1")["asset"]["sha256"], "tampered")

    def test_current_context_has_no_spatial_authorization(self):
        self.upload()
        self.store.bind_object("s1", "front", [0, 0, 10, 10])
        snapshot = json.dumps(self.store.snapshot("s1"))
        for forbidden in ("track_id", "distance_m", "navigation", "can_pass"):
            self.assertNotIn(forbidden, snapshot)


class StructuredOCRTests(unittest.TestCase):
    def setUp(self):
        self.store = VisualContextStore()
        self.store.remember_upload("s", "f", b"sample", width_px=100, height_px=100)
        self.store.bind_object("s", "f", [0, 0, 100, 100], object_id="item")

    def fields(self, texts, *, score=.95, object_id="item"):
        rows = [{"text": text, "roi_xyxy_px": [0, i, 100, i + 1], "score": score} for i, text in enumerate(texts)]
        self.store.add_ocr("s", "f", rows, object_id=object_id)
        return self.store.structured_fields("s", "f", object_id=object_id)

    def test_unlabelled_number_is_not_a_production_date(self):
        result = self.fields(["2026-11-02", "20261102"])
        self.assertEqual(result["fields"]["production_date"], [])
        self.assertEqual(len(result["numeric_candidates"]), 2)

    def test_date_labels_shelf_life_and_batch_stay_distinct(self):
        result = self.fields(["生产日期：2026年11月2日", "有效期至 2027/11/02", "保质期：12个月", "批号 LOT-A7"])
        self.assertEqual(result["fields"]["production_date"][0]["value"], "2026-11-02")
        self.assertEqual(result["fields"]["expiry_date"][0]["value"], "2027-11-02")
        self.assertEqual(result["fields"]["shelf_life"][0]["value"], 12)
        self.assertEqual(result["fields"]["batch_id"][0]["value"], "LOT-A7")
        self.assertEqual(result["fields"]["production_date"][0]["source"]["frame_id"], "f")

    def test_missing_production_date_does_not_take_later_expiry_number(self):
        result = self.fields(["生产日期见瓶底，有效期至2027-11-02"])
        self.assertEqual(result["fields"]["production_date"], [])
        self.assertEqual(result["fields"]["expiry_date"][0]["value"], "2027-11-02")

    def test_impossible_and_low_confidence_numbers_are_candidates_only(self):
        result = self.fields(["生产日期2026-02-30", "2026-11-02"])
        self.assertEqual(result["fields"]["production_date"], [])
        self.assertEqual(len(result["numeric_candidates"]), 2)
        self.store = VisualContextStore()
        self.store.remember_upload("s", "f", b"sample", width_px=100, height_px=100)
        self.store.bind_object("s", "f", [0, 0, 100, 100], object_id="item")
        result = self.fields(["生产日期2026-11-02"], score=.2)
        self.assertEqual(result["fields"]["production_date"], [])
        self.assertEqual(result["numeric_candidates"][0]["reason"], "LOW_CONFIDENCE")

    def test_price_keeps_unit_membership_and_raw_condition(self):
        result = self.fields(["会员价：3.50元/斤，限购2斤"])
        price = result["fields"]["price"][0]
        self.assertEqual(price["value"], "3.50")
        self.assertEqual(price["unit"], "斤")
        self.assertEqual(price["conditions"], ["会员", "限购"])
        self.assertIn("限购2斤", price["conditions_raw"])
        self.assertEqual(result["association"], "confirmed")

    def test_unbound_price_tag_is_ambiguous_and_missing_unit_stays_null(self):
        result = self.fields(["￥9.99"], object_id=None)
        self.assertEqual(result["association"], "ambiguous")
        self.assertIsNone(result["fields"]["price"][0]["unit"])
        self.assertIsNone(result["fields"]["price"][0]["source"]["object_id"])

    def test_multiple_prices_do_not_share_an_invented_unit(self):
        result = self.fields(["苹果3元/斤，盒装5元/盒"])
        self.assertEqual(len(result["fields"]["price"]), 2)
        self.assertTrue(all(p["unit"] is None and p["unit_ambiguous"] for p in result["fields"]["price"]))

    def test_unknown_confidence_does_not_silently_confirm_digits(self):
        result = self.fields(["会员价9元/袋"], score=None)
        self.assertEqual(result["fields"]["price"], [])
        self.assertEqual(result["raw_observations"][0]["raw_text"], "会员价9元/袋")


if __name__ == "__main__":
    unittest.main()
