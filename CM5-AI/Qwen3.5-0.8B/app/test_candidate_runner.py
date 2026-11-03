"""候选模型身份、流式API与中断边界；不代表实际推理质量。"""
import io
import json
import threading
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import requests
from model_runner import ModelRunner, validate_response
import model_runner


def response(events):
    r = requests.Response()
    r.status_code = 200
    r.raw = io.BytesIO(("".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events) + "data: [DONE]\n\n").encode())
    r.headers["Content-Type"] = "text/event-stream"
    return r


class CandidateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = ModelRunner("http://127.0.0.1:18101")
        self.models = mock.Mock()
        self.models.status_code = 200
        self.models.json.return_value = {"data":[{"id":"qwen3.5-0.8b"}]}

    def test_requires_explicit_local_runner(self):
        for url in ("https://example.com", "http://192.0.2.10", "", "https://127.0.0.1"):
            with self.subTest(url=url), self.assertRaises(ValueError): ModelRunner(url)

    def test_wrong_model_identity_never_generates(self):
        self.models.json.return_value = {"data":[{"id":"minicpm-v"}]}
        with mock.patch("model_runner.requests.get",return_value=self.models), mock.patch("model_runner.requests.post") as post:
            with self.assertRaises(RuntimeError): self.runner.complete([])
            post.assert_not_called()

    def test_utf8_complete_and_usage(self):
        data = response([{"choices":[{"delta":{"content":"包装日期，不能猜。"}}]}, {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":8}}])
        with mock.patch("model_runner.requests.get",return_value=self.models), mock.patch("model_runner.requests.post",return_value=data):
            result = self.runner.complete([{"role":"user","content":"读一下"}])
        self.assertEqual(result["answer"],"包装日期，不能猜。")
        self.assertEqual(result["finish_reason"],"stop")
        self.assertFalse(result["cancelled"])

    def test_cancelled_before_request_never_connects(self):
        cancel=threading.Event();cancel.set()
        with mock.patch("model_runner.requests.get") as get, mock.patch("model_runner.requests.post") as post:
            result=self.runner.complete([],cancel=cancel)
        self.assertTrue(result["cancelled"])
        get.assert_not_called();post.assert_not_called()

    def test_incomplete_stream_is_not_success(self):
        with mock.patch("model_runner.requests.get",return_value=self.models), mock.patch("model_runner.requests.post",return_value=response([])):
            with self.assertRaises(RuntimeError):self.runner.complete([])

    def test_remote_image_is_rejected_before_any_request(self):
        messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}]}]
        with mock.patch("model_runner.requests.get") as get, mock.patch("model_runner.requests.post") as post:
            with self.assertRaises(ValueError):self.runner.complete(messages)
        get.assert_not_called();post.assert_not_called()

    def test_two_images_remain_distinct_and_thinking_disabled(self):
        messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,A"}},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,B"}}]}]
        data=response([{"choices":[{"delta":{"content":"不同对象"},"finish_reason":"stop"}]}])
        with mock.patch("model_runner.requests.get",return_value=self.models), mock.patch("model_runner.requests.post",return_value=data) as post:
            self.runner.complete(messages)
        payload=post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"],messages)
        self.assertEqual(payload["model"],"qwen3.5-0.8b")
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])


    def test_profile_model_cannot_be_replaced_by_environment_model_id(self):
        with mock.patch("model_runner.requests.get") as get:
            with self.assertRaises(ValueError):
                ModelRunner("http://127.0.0.1:18101", "minicpm-v")
        get.assert_not_called()

    def test_profile_paths_must_remain_inside_candidate(self):
        profile = model_runner.load_profile()
        for key in ("model_file", "mmproj_file", "runner_file", "runner_file_posix"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config").mkdir()
                altered = dict(profile)
                altered[key] = "../MiniCPM-V4.7-1B/models/other.gguf"
                (root / "config/model-profile.json").write_text(json.dumps(altered), encoding="utf-8")
                with mock.patch.object(model_runner, "ROOT", root), self.assertRaises(ValueError):
                    ModelRunner("http://127.0.0.1:18101")

    def test_endpoint_rejects_hidden_credentials_paths_and_redirect_parameters(self):
        for url in ("http://user@localhost:18101", "http://127.0.0.1:18101/forward",
                    "http://127.0.0.1:18101?url=https://example.com", "http://localhost:bad",
                    "http://localhost:18101#fragment", " http://127.0.0.1:18101"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ModelRunner(url)

    def test_probe_redirect_never_starts_generation(self):
        redirect = requests.Response()
        redirect.status_code = 302
        redirect.raw = io.BytesIO(b"")
        redirect.headers["Location"] = "https://example.com/v1/models"
        with mock.patch("model_runner.requests.get", return_value=redirect) as get, mock.patch("model_runner.requests.post") as post:
            with self.assertRaises(RuntimeError):
                self.runner.complete([])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        post.assert_not_called()

    def test_generation_redirect_is_rejected_without_following(self):
        redirect = response([])
        redirect.status_code = 307
        redirect.headers["Location"] = "http://127.0.0.1:8091/v1/chat/completions"
        with mock.patch("model_runner.requests.get", return_value=self.models), mock.patch("model_runner.requests.post", return_value=redirect) as post:
            with self.assertRaises(RuntimeError):
                self.runner.complete([])
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_count, 1)

    def test_shared_response_validator_rejects_all_redirect_statuses(self):
        for status in (301, 302, 303, 304, 307, 308):
            with self.subTest(status=status):
                reply = requests.Response()
                reply.status_code = status
                with self.assertRaises(RuntimeError):
                    validate_response(reply)

    def test_cancel_during_probe_does_not_start_generation(self):
        cancel = threading.Event()
        def probe():
            cancel.set()
            return {"ready": True}
        with mock.patch.object(self.runner, "probe", side_effect=probe), mock.patch("model_runner.requests.post") as post:
            result = self.runner.complete([], cancel=cancel)
        self.assertTrue(result["cancelled"])
        post.assert_not_called()

if __name__ == "__main__":unittest.main()
