"""固定同题、同运行器参数的轮流加载对照；仅runner-only，不代表整机或首声。"""
import argparse
import base64
import hashlib
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"app"))
from model_runner import ModelRunner


def image_url(path):
    image=Image.open(path).convert("RGB");image.thumbnail((512,512))
    buffer=io.BytesIO();image.save(buffer,format="JPEG",quality=85)
    return "data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root",required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    baseline=Path(args.baseline_root).resolve();output=Path(args.output).resolve()
    if not output.is_relative_to((ROOT/"test-runs").resolve()):parser.error("结果只能写本候选test-runs")
    if not 1024<=args.port<=65535:parser.error("端口无效")
    output.parent.mkdir(parents=True,exist_ok=True)
    corpus=ROOT.parents[1]/"docs/05-测试与验证/m0-runner-cases-20261103.json"
    cases=json.loads(corpus.read_text(encoding="utf-8"))["cases"]
    runner=ROOT/"runtime/llama.cpp-b12066/llama-server.exe"
    variants=[("minicpm-v",baseline/"models/vlm/MiniCPM-V 4.7 1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf",baseline/"models/vlm/MiniCPM-V 4.7 1B/mmproj-model-f16.gguf"),("qwen3.5-0.8b",ROOT/"models/vlm/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf",ROOT/"models/vlm/Qwen3.5-0.8B/mmproj-F16.gguf")]
    result={"evidence":"paired_runner_only_small_sample", "configuration":{"runner_tag":"b12066","threads":4,"context":2048,"parallel":1,"gpu_layers":0,"temperature":0.2,"top_p":0.9,"cache_prompt":False,"enable_thinking":False,"image_max_side":512,"jpeg_quality":85,"sampling_ms":100},"corpus_sha256":hashlib.sha256(corpus.read_bytes()).hexdigest(),"history_policy":"Identical frozen messages including prior assistant answer for each model; not a fresh generated dialogue chain","variants":[]}
    prepared=[]
    for case in cases:
        messages=case["messages"]
        if case["case_id"]=="vision-cats":
            messages=[cases[0]["messages"][0],{"role":"user","content":[{"type":"image_url","image_url":{"url":image_url(ROOT/"samples/cats.jpg")}},{"type":"text","text":"请描述图里有什么，具体说说数量、姿态和主要颜色。看不清就说看不清。"}]}]
        prepared.append((case,messages,hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
    for alias,model,projector in variants:
        if not all(p.is_file() for p in (runner,model,projector)):raise RuntimeError("对照组件缺失")
        with socket.socket() as sock:sock.bind(("127.0.0.1",args.port))
        row={"model_id":alias,"cases":[]};samples=[];stop=threading.Event();process=None
        result["variants"].append(row)
        with output.with_name(alias+".runner.log").open("wb") as log:
            try:
                token=secrets.token_hex(24);url="http://127.0.0.1:"+str(args.port)
                start=time.perf_counter()
                process=subprocess.Popen([str(runner),"-m",str(model),"--mmproj",str(projector),"--no-mmproj-offload","-c","2048","-t","4","-ngl","0","--parallel","1","--host","127.0.0.1","--port",str(args.port),"--alias",alias,"--api-key",token],cwd=runner.parent,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                owned=psutil.Process(process.pid)
                def sampler():
                    while not stop.wait(.1):
                        try:
                            info=owned.memory_info();samples.append({"private":getattr(info,"private",None),"working_set":info.rss})
                        except psutil.Error:return
                thread=threading.Thread(target=sampler,daemon=True);thread.start()
                client=ModelRunner(url,alias,token)
                for _ in range(180):
                    if process.poll() is not None:raise RuntimeError("运行器提前退出")
                    try:client.probe();break
                    except (requests.RequestException,RuntimeError):time.sleep(.5)
                else:raise RuntimeError("加载超时")
                row["process_start_to_ready_ms"]=(time.perf_counter()-start)*1000
                print("READY "+alias,flush=True)
                for case,messages,input_hash in prepared:
                    start=time.perf_counter();first=None;answer="";finish=None;usage=None
                    for event in client.stream(messages,max_tokens=case["max_tokens"],temperature=.2):
                        usage=event.get("usage") or usage
                        for choice in event.get("choices",[]):
                            delta=choice.get("delta",{}).get("content") or ""
                            if delta and first is None:first=(time.perf_counter()-start)*1000
                            answer+=delta;finish=choice.get("finish_reason") or finish
                    case_row={"case_id":case["case_id"],"input_sha256":input_hash,"max_tokens":case["max_tokens"],"first_content_ms":first,"total_ms":(time.perf_counter()-start)*1000,"answer":answer,"finish_reason":finish,"usage":usage}
                    if finish is None:raise RuntimeError("不完整回答流")
                    row["cases"].append(case_row)
                    print(json.dumps({"model_id":alias,**case_row},ensure_ascii=False),flush=True)
                row["status"]="completed"
            except Exception as exc:row["status"]="failed";row["error"]=str(exc);print("FAILED "+alias+": "+str(exc),flush=True)
            finally:
                stop.set()
                if process is not None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                    row["owned_process_stopped"]=process.poll() is not None
                row["observed_private_peak_bytes"]=max((s["private"] for s in samples if s["private"] is not None),default=None)
                row["observed_working_set_peak_bytes"]=max((s["working_set"] for s in samples),default=None)
                row["memory_basis"]="Only owned Windows runner process,100ms sample; private and working-set overlap; not subsystem RAM"
                output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        time.sleep(1)
    result["all_input_hashes_match"] = len(result["variants"])==2 and [x["input_sha256"] for x in result["variants"][0]["cases"]]==[x["input_sha256"] for x in result["variants"][1]["cases"]]
    result["limits"]=["Each case once; no P95","No ASR/TTS or effective first sound","No human quality grading","No target board, thermal or battery result","OS file cache not controlled; sequential order fixed MiniCPM then Qwen"]
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x.get("status")=="completed" for x in result["variants"]) and result["all_input_hashes_match"] else 1


if __name__=="__main__":raise SystemExit(main())
