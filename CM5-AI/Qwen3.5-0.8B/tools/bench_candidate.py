"""仅本机候选运行器冒烟：显式端口、独立PID、真实文字与单/双图，结束即清理。"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time

import psutil
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from model_runner import ModelRunner


def data_url(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535: parser.error("port范围无效")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT / "test-runs").resolve()):
        parser.error("输出只能放在本候选test-runs下，避免覆盖源码")
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = json.loads((ROOT / "config/model-profile.json").read_text(encoding="utf-8"))
    runner, model, mmproj = [ROOT / profile[k] for k in ("runner_file", "model_file", "mmproj_file")]
    if not all(p.is_file() for p in (runner, model, mmproj)): raise RuntimeError("候选资产未安装")
    token = secrets.token_hex(24)
    url = "http://127.0.0.1:" + str(args.port)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    samples, results = [], {"evidence":"development_runner_only", "configuration":{"threads":4,"ctx":2048,"parallel":1,"gpu_layers":0,"image_max_side":512,"jpeg_quality":85,"enable_thinking":False}, "cases":[]}
    stop = threading.Event()
    process = None
    with output.with_suffix(".runner.log").open("wb") as log:
        try:
            started = time.perf_counter()
            process = subprocess.Popen([str(runner),"-m",str(model),"--mmproj",str(mmproj),"--no-mmproj-offload","-c","2048","-t","4","-ngl","0","--parallel","1","--host","127.0.0.1","--port",str(args.port),"--alias",profile["model_id"],"--api-key",token],cwd=runner.parent,stdout=log,stderr=log,creationflags=flags)
            owned = psutil.Process(process.pid)
            def sample():
                while not stop.wait(.1):
                    try:
                        info=owned.memory_info();samples.append({"private":getattr(info,"private",None),"working_set":info.rss})
                    except psutil.Error:return
            thread=threading.Thread(target=sample,daemon=True);thread.start()
            client=ModelRunner(url, profile["model_id"],token)
            for _ in range(180):
                if process.poll() is not None:raise RuntimeError("运行器提前退出")
                try:client.probe();break
                except (requests.RequestException,RuntimeError):time.sleep(.5)
            else:raise RuntimeError("候选未在90s内就绪")
            results["process_start_to_ready_ms"]=(time.perf_counter()-started)*1000
            system={"role":"system","content":"你是离线中文交流助手。先回答核心问题，按需要展开。理解上下文，区分用户和助手，不编造数字。不根据照片判断行走安全。"}
            natural=[system,{"role":"user","content":"我今天工作总出错，越着急越做不好，觉得自己挺没用的。你能陪我聊聊吗？"}]
            def run(case_id,messages,limit):
                t=time.perf_counter();answer="";first=None;finish=None;usage=None
                for event in client.stream(messages,max_tokens=limit):
                    usage=event.get("usage") or usage
                    for choice in event.get("choices",[]):
                        piece=choice.get("delta",{}).get("content") or ""
                        if piece and first is None:first=(time.perf_counter()-t)*1000
                        answer+=piece;finish=choice.get("finish_reason") or finish
                row={"case_id":case_id,"answer":answer,"first_content_ms":first,"total_ms":(time.perf_counter()-t)*1000,"finish_reason":finish,"max_tokens":limit,"usage":usage}
                if finish is None:raise RuntimeError("回答流未完成")
                results["cases"].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
                return answer
            answer=run("chat-natural",natural,192)
            run("chat-multiturn",natural+[{"role":"assistant","content":answer},{"role":"user","content":"你刚才说的我能理解，不过我最怕的是同事觉得我拖后腿。你能接着刚才的话说说吗？"}],192)
            image_a=data_url(ROOT/"samples/cats.jpg")
            run("single-image",[system,{"role":"user","content":[{"type":"image_url","image_url":{"url":image_a}},{"type":"text","text":"图里有几只猫？请说主要颜色，看不清就说明。"}]}],160)
            image_b=data_url(ROOT/"samples/scene-desk.jpg")
            run("two-images",[system,{"role":"user","content":[{"type":"text","text":"第一张是A，第二张是B。分别描述主要物品，再说两图有什么明显不同，不要混淆。"},{"type":"image_url","image_url":{"url":image_a}},{"type":"image_url","image_url":{"url":image_b}}]}],256)
            results["status"]="completed"
        except Exception as exc:
            results["status"]="failed";results["error"]=str(exc);print("FAILED "+str(exc),flush=True)
        finally:
            stop.set()
            if process is not None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                results["owned_process_stopped"]=process.poll() is not None
            results["observed_model_process_private_peak_bytes"]=max((x["private"] for x in samples if x["private"] is not None),default=None)
            results["observed_model_process_working_set_peak_bytes"]=max((x["working_set"] for x in samples),default=None)
            results["memory_note"]="Single owned runner,100ms sampling; overlapping metrics must not be added; not whole interaction RAM"
            results["not_validated"]=["quality acceptance","TTS first sound","physical audio","target board","runner compute cancellation","battery/thermal"]
            output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if results["status"]=="completed" else 1


if __name__=="__main__":raise SystemExit(main())
