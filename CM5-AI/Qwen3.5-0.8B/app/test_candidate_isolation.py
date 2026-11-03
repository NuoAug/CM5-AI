"""Candidate path isolation, fail-closed readiness and request-boundary regression."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import candidate_paths
import precheck
import server


class CandidateIsolationTests(unittest.TestCase):
    def test_relative_escape_rejected(self):
        with self.assertRaises(ValueError):
            candidate_paths.resolve_local("../MiniCPM-V4.7-1B/data")

    def test_every_path_environment_rejected(self):
        for name in candidate_paths.PATH_ENVIRONMENTS:
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: str(candidate_paths.ROOT.parent / "MiniCPM-V4.7-1B/data")}):
                with self.assertRaises(ValueError):
                    candidate_paths.validate_environment()

    def test_direct_server_import_rejects_external_cache_before_write(self):
        env = dict(os.environ)
        env["DATA_DIR"] = str(candidate_paths.ROOT.parent / "MiniCPM-V4.7-1B/data")
        proc = subprocess.run([sys.executable, "-c", "import server"], cwd=Path(__file__).parent,
                              env=env, capture_output=True, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"candidate path must stay inside", proc.stderr)

    def test_empty_vlm_assets_are_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {"model_file": "empty.gguf", "mmproj_file": "empty-projector.gguf",
                       "runner_file": "empty-runner.exe", "runner_file_posix": "empty-runner"}
            for value in profile.values(): (root / value).touch()
            self.assertFalse(any(precheck.collect_checks(profile, vlm_only=True, root=root).values()))

    def test_full_readiness_requires_flag_and_real_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "VERSION").write_text("test-candidate", encoding="utf-8")
            profile = {"profile_id": "test", "model_file": "model.gguf", "mmproj_file": "projector.gguf",
                       "runner_file": "runner.exe", "runner_file_posix": "runner",
                       "complete_interaction_assets_installed": False}
            for key in ("model_file", "mmproj_file", "runner_file", "runner_file_posix"):
                (root / profile[key]).write_bytes(b"not a real model")
            checks = precheck.collect_checks(profile, root=root)
            self.assertFalse(checks["profile_interaction_installed"])
            self.assertFalse(checks["interaction_real_validation_receipt"])
            self.assertFalse(checks["receipt_covers_current_assets"])
            for name in ("asr_tokens", "tts_config", "piper_binary", "ocr_cls", "kws_encoder", "kws_decoder", "kws_joiner"):
                self.assertFalse(checks[name])

    def test_real_call_rejects_remote_url_without_network(self):
        with mock.patch.object(server, "VLM_URL", "http://192.0.2.10:18101"), mock.patch.object(server.requests, "get") as get, mock.patch.object(server.requests, "post") as post:
            with self.assertRaises(ValueError): server.call_vlm("", "private prompt")
            get.assert_not_called()
            post.assert_not_called()

    def test_real_call_rejects_wrong_alias_without_generation(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"data": [{"id": "minicpm-v"}]}
        with mock.patch.object(server, "VLM_URL", "http://127.0.0.1:18101"), mock.patch.object(server.requests, "get", return_value=response) as get, mock.patch.object(server.requests, "post") as post:
            with self.assertRaises(RuntimeError): server.call_vlm("", "private prompt")
            post.assert_not_called()
            self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_real_call_rejects_generation_redirect(self):
        response = mock.Mock(status_code=302)
        with mock.patch.object(server, "VLM_URL", "http://127.0.0.1:18101"), mock.patch.object(server.ModelRunner, "probe", return_value={"ready": True}), mock.patch.object(server.requests, "post", return_value=response) as post:
            with self.assertRaises(RuntimeError): server.call_vlm("", "private prompt")
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            response.json.assert_not_called()

    def test_real_stream_rejects_remote_url_without_network(self):
        with mock.patch.object(server, "VLM_URL", "http://192.0.2.10:18101"), mock.patch.object(server.requests, "get") as get, mock.patch.object(server.requests, "post") as post:
            with self.assertRaises(ValueError): list(server.stream_vlm("private prompt"))
            get.assert_not_called()
            post.assert_not_called()

    def test_real_stream_rejects_generation_redirect(self):
        response = mock.MagicMock(status_code=302)
        response.__enter__.return_value = response
        with mock.patch.object(server, "VLM_URL", "http://127.0.0.1:18101"), mock.patch.object(server.ModelRunner, "probe", return_value={"ready": True}), mock.patch.object(server.requests, "post", return_value=response) as post:
            with self.assertRaises(RuntimeError): list(server.stream_vlm("private prompt"))
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            response.iter_lines.assert_not_called()


if __name__ == "__main__":
    unittest.main()
