"""M0/M3 固定少量同题输入的运行器基线，不代表语音或目标板验收。"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import time

import requests
from PIL import Image


def run(url, headers, messages, max_tokens):
    started = time.perf_counter()
    result = {"answer": "", "ttft_content_ms": None, "finish_reason": None}
    payload = dict(messages=messages, model="local", stream=True,
                   stream_options={"include_usage": True}, max_tokens=max_tokens,
                   temperature=0.2, top_p=0.9, cache_prompt=False)
    with requests.post(url.rstrip("/") + "/v1/chat/completions", headers=headers,
                       json=payload, stream=True, timeout=(5, 120)) as response:
        response.raise_for_status()
        # 按字节拆行后解码，避免 Latin-1 将中文中的字节误判为分隔符。
        for raw in response.iter_lines(chunk_size=1):
            if not raw or not raw.startswith(b"data:"):
                continue
            text = raw[5:].strip()
            if text == b"[DONE]":
                break
            event = json.loads(text.decode("utf-8"))
            for choice in event.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                if piece and result["ttft_content_ms"] is None:
                    result["ttft_content_ms"] = (time.perf_counter() - started) * 1000
                result["answer"] += piece
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
            if event.get("usage"):
                result["usage"] = event["usage"]
    result["total_ms"] = (time.perf_counter() - started) * 1000
    if result["finish_reason"] is None:
        raise RuntimeError("SSE 未返回完成状态，不能将中断连接算作通过")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="显式指定本地模型运行器URL，与主控无关")
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    cases = json.loads((root / "docs/m0-runner-cases-20261103.json").read_text(encoding="utf-8"))
    token = os.environ.get(args.api_key_env, "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    results = {"evidence": "runner_only", "cases": []}
    last_answer = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases["cases"]:
        messages = case["messages"]
        if case["case_id"] == "chat-multiturn":
            # 追问必须使用本次运行的新答案，不沿用报告中的旧答案。
            messages = [dict(m) for m in messages]
            messages[2]["content"] = last_answer
        if case["case_id"] == "vision-cats":
            image = Image.open(root / "CM5-AI/Qwen3.5-0.8B/samples/cats.jpg").convert("RGB")
            image.thumbnail((512, 512))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            messages = [cases["cases"][0]["messages"][0], {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": "请描述图里有什么，具体说说数量、姿态和主要颜色。看不清就说看不清。"}]}]
        result = run(args.url, headers, messages, case["max_tokens"])
        result["case_id"] = case["case_id"]
        if case["case_id"] == "chat-natural":
            last_answer = result["answer"]
        results["cases"].append(result)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
