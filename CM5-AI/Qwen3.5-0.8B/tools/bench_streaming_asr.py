"""原生流式 ASR 文件增量回放；不采集设备，不将回放计作真人插话验收。"""
import argparse
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from streaming_asr import StreamingASR, StreamingASRConfig


def replay(engine, path, chunk_ms, realtime):
    row = {"input": str(path), "evidence_mode": "replay", "events": [], "status": "ok"}
    start = time.perf_counter()
    compute_seconds = 0
    input_samples = 0
    session = engine.create_session()
    try:
        with wave.open(str(path), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
                raise ValueError("回放只接受 16kHz/单声道/PCM16 WAV；预处理须另行记录")
            frames = round(chunk_ms * 16)
            while raw := source.readframes(frames):
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                input_samples += len(samples)
                before = time.perf_counter()
                events = session.feed(samples)
                compute_seconds += time.perf_counter() - before
                for event in events:
                    row["events"].append(dict(event.to_dict(), elapsed_ms=(time.perf_counter() - start) * 1000))
                if realtime:
                    remaining = input_samples / 16000 - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
            before = time.perf_counter()
            events = session.finish()
            compute_seconds += time.perf_counter() - before
            for event in events:
                row["events"].append(dict(event.to_dict(), elapsed_ms=(time.perf_counter() - start) * 1000))
        row["final_transcripts"] = [event["text"] for event in row["events"] if event["is_final"]]
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        session.close()
    row.update(audio_seconds=input_samples / 16000, processing_seconds=compute_seconds,
               total_wall_seconds=time.perf_counter() - start)
    row["processing_rtf"] = compute_seconds / row["audio_seconds"] if input_samples else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--realtime", action="store_true", help="按音频时长节奏送块；默认尽快增量回放")
    args = parser.parse_args()
    if not 10 <= args.chunk_ms <= 1000:
        parser.error("chunk-ms 必须在 10—1000 范围")
    config = StreamingASRConfig(args.model_dir, num_threads=args.threads)
    engine = StreamingASR(config)
    load_start = time.perf_counter()
    engine.load()
    report = {"evidence_mode": "replay", "backend": "native_online_paraformer_int8",
              "model_load_ms": (time.perf_counter() - load_start) * 1000,
              "threads": args.threads, "chunk_ms": args.chunk_ms, "paced": args.realtime,
              "word_timestamps_supported": False, "cases": [],
              "limits": ["No live microphone or playback", "No target-board evidence", "Small sample; no quality or P95 claim"]}
    try:
        for path in args.wav:
            row = replay(engine, path, args.chunk_ms, args.realtime)
            report["cases"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: row.get(key) for key in ("input", "status", "final_transcripts", "processing_rtf")}, ensure_ascii=False), flush=True)
    finally:
        engine.close()
    raise SystemExit(any(row["status"] != "ok" for row in report["cases"]))


if __name__ == "__main__":
    main()
