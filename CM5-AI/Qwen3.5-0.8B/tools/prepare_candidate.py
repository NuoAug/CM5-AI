"""下载固定版本的候选资产，仅写本工程models/runtime，支持校验后安装。"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import threading
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "unsloth/Qwen3.5-0.8B-GGUF"
REVISION = "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
FILES = ("Qwen3.5-0.8B-Q4_K_M.gguf", "mmproj-F16.gguf")
RUNNER_TAG = "b12066"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ranged_download(url, path, size, expected):
    if path.exists() and path.stat().st_size == size and digest(path) == expected:
        return
    partial = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    step = 4 * 1024 * 1024
    with partial.open("wb") as f:
        f.truncate(size)
        def part(start):
            end = min(start + step, size) - 1
            last = None
            for attempt in range(3):
                try:
                    separator = "&" if "?" in url else "?"
                    chunk_url = url + separator + "download=true&chunk_offset=" + str(start)
                    with requests.get(chunk_url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(8, 20)) as r:
                        r.raise_for_status()
                        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise RuntimeError("官方文件Range响应与请求不匹配")
                        data = bytearray()
                        began = time.monotonic()
                        for block in r.iter_content(65536):
                            data.extend(block)
                            if len(data) > end - start + 1 or time.monotonic() - began > 60:
                                raise RuntimeError("下载块超过范围或时间限制")
                        if len(data) != end - start + 1:
                            raise RuntimeError("下载块不完整")
                    with lock:
                        f.seek(start); f.write(data)
                    return
                except (requests.RequestException, RuntimeError) as exc:
                    last = exc
            raise RuntimeError("下载块失败 offset=" + str(start)) from last
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(part, range(0, size, step))):
                if i % 16 == 0: print(path.name, "verified blocks", i + 1, flush=True)
    if digest(partial) != expected:
        raise RuntimeError("完整资产SHA256不匹配，未安装")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    if not args.models and not args.runner:
        parser.error("显式选择 --models 和/或 --runner")
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError("可用磁盘不足2GiB，停止下载")
    if args.models:
        response = requests.get(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true", timeout=20)
        response.raise_for_status()
        items = {x["path"]: x for x in response.json() if x.get("path") in FILES}
        records = []
        for name in FILES:
            info = items[name]
            expected = info["lfs"]["oid"]
            path = ROOT / "models/vlm/Qwen3.5-0.8B" / name
            ranged_download(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", path, info["size"], expected)
            records.append({"file": str(path.relative_to(ROOT)), "bytes": info["size"], "sha256": expected, "source": REPO, "revision": REVISION})
            print("Installed and verified", name, flush=True)
        (ROOT / "models/assets-installed.json").write_text(json.dumps({"status":"downloaded_not_runtime_validated", "model_files":records}, indent=2), encoding="utf-8")
    if args.runner:
        if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError("本预编译运行器仅Windows x64；目标系统请按固定tag独立构建并验证")
        response = requests.get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{RUNNER_TAG}", timeout=20)
        response.raise_for_status()
        asset = next(a for a in response.json()["assets"] if a["name"] == f"llama-{RUNNER_TAG}-bin-win-cpu-x64.zip")
        path = ROOT / "runtime" / asset["name"]
        path.parent.mkdir(exist_ok=True)
        # GitHub发布资产用普通流，整体字节与官方SHA必须一致。
        with requests.get(asset["browser_download_url"], stream=True, timeout=(8, 30)) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_content(65536): f.write(chunk)
        expected = asset["digest"].split(":",1)[1]
        if path.stat().st_size != asset["size"] or digest(path) != expected:
            raise RuntimeError("运行器发布包校验失败")
        target = ROOT / ("runtime/llama.cpp-" + RUNNER_TAG)
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            if any(not (target / i.filename).resolve().is_relative_to(target.resolve()) for i in archive.infolist()):
                raise RuntimeError("ZIP资产路径越界")
            archive.extractall(target)
        (ROOT / "runtime/installed.json").write_text(json.dumps({"tag":RUNNER_TAG,"source":"ggml-org/llama.cpp","archive_sha256":expected,"archive_bytes":asset["size"],"status":"downloaded_not_runtime_validated"},indent=2),encoding="utf-8")
        print("Installed and verified official CPU runner", RUNNER_TAG, flush=True)


if __name__ == "__main__": main()
