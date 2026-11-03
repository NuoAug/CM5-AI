"""独立候选的本地图文运行器；不承担主控事实或导航权限。"""
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import requests

ROOT = Path(__file__).resolve().parent.parent


def load_profile():
    """只接受本候选目录内的模型与运行器，不沿用外部工程路径。"""
    try:
        profile = json.loads((ROOT / "config/model-profile.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("候选profile不可读，禁止回退其他模型") from exc
    if not isinstance(profile, dict) or not isinstance(profile.get("model_id"), str) or not profile["model_id"].strip():
        raise ValueError("候选profile缺少model_id")
    base = ROOT.resolve()
    for key in ("model_file", "mmproj_file", "runner_file", "runner_file_posix"):
        value = profile.get(key)
        if key == "runner_file_posix" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError("候选profile缺少路径: " + key)
        path = Path(value)
        if path.is_absolute() or not (base / path).resolve().is_relative_to(base):
            raise ValueError("候选profile路径越出本工程: " + key)
    return profile


def validate_endpoint(url):
    if not isinstance(url, str) or not url or url.strip() != url or any(ord(c) < 32 for c in url):
        raise ValueError("候选需要显式本机HTTP运行器地址")
    try:
        parsed = urlparse(url)
        port = parsed.port
        valid = (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
                 and parsed.username is None and parsed.password is None
                 and parsed.path in ("", "/") and not parsed.query and not parsed.fragment
                 and (port is None or 1 <= port <= 65535))
    except ValueError as exc:
        raise ValueError("候选运行器地址无效") from exc
    if not valid:
        raise ValueError("候选仅允许本机HTTP根地址，不允许凭据、路径或转发参数")
    return url.rstrip("/")


def validate_response(response):
    """生成与身份查询共用：3xx不能跳到别的模型或外部地址。"""
    if 300 <= response.status_code < 400:
        raise RuntimeError("候选运行器返回重定向，拒绝跟随其他地址")
    response.raise_for_status()
    return response


class ModelRunner:
    def __init__(self, url, model_id="qwen3.5-0.8b", api_key="", timeout=120):
        self.url = validate_endpoint(url)
        profile = load_profile()
        if model_id != profile["model_id"]:
            raise ValueError("请求模型身份与本候选profile不一致")
        self.model_id, self.timeout = model_id, timeout
        self.headers = {"Authorization": "Bearer " + api_key} if api_key else {}

    def probe(self):
        # 仅health成功不能证明连到正确候选，必须校验实际模型别名。
        response = requests.get(self.url + "/v1/models", headers=self.headers,
                                timeout=(3, 5), allow_redirects=False)
        try:
            validate_response(response)
            payload = response.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(models, list) or not any(isinstance(row, dict) and row.get("id") == self.model_id for row in models):
                raise RuntimeError("运行器模型身份不匹配，拒绝回退或误连其他模型")
            return {"ready": True, "model_id": self.model_id}
        finally:
            response.close()

    def stream(self, messages, *, max_tokens=256, temperature=0.2, cancel=None):
        if not isinstance(messages, list):
            raise ValueError("messages必须是列表")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("消息必须是对象")
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        raise ValueError("内容片段必须是对象")
                    if item.get("type") == "image_url":
                        descriptor = item.get("image_url")
                        url = descriptor.get("url", "") if isinstance(descriptor, dict) else ""
                        if not isinstance(url, str) or not re.match(r"^data:image/(?:jpeg|png|webp);base64,", url):
                            raise ValueError("离线图像只接受调用方提供的本地编码字节，禁止远端URL")
        if cancel is not None and cancel.is_set():
            return
        self.probe()
        if cancel is not None and cancel.is_set():
            return
        payload = {"model": self.model_id, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": 0.9, "cache_prompt": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        with requests.post(self.url + "/v1/chat/completions", json=payload,
                           headers=self.headers, stream=True, timeout=(5, self.timeout),
                           allow_redirects=False) as response:
            validate_response(response)
            for raw in response.iter_lines(chunk_size=1):
                if cancel is not None and cancel.is_set():
                    response.close()
                    return
                if not raw or not raw.startswith(b"data:"):
                    continue
                text = raw[5:].strip()
                if text == b"[DONE]":
                    return
                yield json.loads(text.decode("utf-8"))

    def complete(self, messages, **kwargs):
        answer, finish, usage = "", None, None
        for event in self.stream(messages, **kwargs):
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                answer += choice.get("delta", {}).get("content") or ""
                finish = choice.get("finish_reason") or finish
        cancelled = kwargs.get("cancel") is not None and kwargs["cancel"].is_set()
        if finish is None and not cancelled:
            raise RuntimeError("运行器流未完成，不能声明回答完成")
        return {"answer": answer, "finish_reason": finish, "usage": usage, "cancelled": cancelled}
