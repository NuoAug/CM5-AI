# -*- coding: utf-8 -*-
"""设备常开采集与原生流式识别，HTTP只提交最终话语。

AEC_REFERENCE_VERIFIED 是部署方的显式配置声明，不是实机验收结果。默认在
输出期间持续读取但不识别回声音频；启用声明后才开放待标定的真人插话候选链。
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from candidate_paths import resolve_local, validate_environment
import queue
import threading
import time
import uuid
from typing import Optional

import numpy as np
import requests
from PIL import Image

import kws
from device_io import AlsaCapture, DeviceCamera, encode_wav
from output import OUTPUT, PRIORITY_CONFIRM
from streaming_asr import StreamingASR, StreamingASRConfig, UtteranceLimitError

VOICE_URL = "http://127.0.0.1:18100/api/voice"
ROOT = Path(__file__).resolve().parent.parent


def jpeg_b64(img: Image.Image, max_side: int = 1600, quality: int = 88) -> str:
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DeviceVoice:
    def __init__(self, camera: Optional[DeviceCamera] = None,
                 capture: Optional[AlsaCapture] = None, url: str = VOICE_URL,
                 session_id: str = "device", *, asr=None, gate=None,
                 backend: Optional[str] = None, final_url: Optional[str] = None,
                 aec_verified: Optional[bool] = None, http_post=None):
        self.camera = camera
        self.capture = capture or AlsaCapture()
        self.url = url
        self.final_url = final_url or url.rstrip("/") + "/final"
        self.session_id = session_id
        self.backend = (backend or os.environ.get("ASR_BACKEND", "streaming-paraformer")).strip().lower()
        self.aec_verified = (os.environ.get("AEC_REFERENCE_VERIFIED", "0") == "1") if aec_verified is None else bool(aec_verified)
        self._gate_injected = gate is not None
        self.gate = gate if gate is not None else kws.WakeGate(
            armed_seconds=15.0, buffer_commands=self.backend == "offline-paraformer")
        self._asr = asr
        self._asr_session = None
        self._asr_active = False
        self._asr_open = False
        self._conversation_active = False
        self._partial = ""
        self._post = http_post or requests.post
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        self._audio_lock = threading.RLock()
        self._paused = False  # 只表示输出活动，不再停止采集。
        self._sink_present = False
        self._last_error = None
        self._last_command_at = 0.0
        self._seq = 0
        self._output_voiced_s = 0.0
        self._barge_confirmed = False
        self._barge_min_seconds = float(os.environ.get("BARGE_IN_MIN_VOICED_MS", "200")) / 1000
        if not 0.05 <= self._barge_min_seconds <= 2:
            raise ValueError("BARGE_IN_MIN_VOICED_MS 必须在50—2000范围")
        self._request_queue = queue.Queue(maxsize=2)
        self._workers = []
        self._worker_stop = threading.Event()
        self._dispatch_epoch = 0
        self._stats = {"partial_results": 0, "final_results": 0, "barge_ins": 0,
                       "dropped_requests": 0, "request_errors": 0, "stream_errors": 0,
                       "ignored_cancelled": 0, "active_requests": 0}

    def _kws_status(self):
        if self._gate_injected:
            return {"ready": bool(getattr(self.gate, "ready", True)), "injected": True}
        return kws.status()

    def available(self):
        return self.capture.available() and self._kws_status().get("ready", False)

    def status(self):
        with self._lock:
            data = {"available": self.available(), "running": self._running,
                    "paused": self._paused and not self.aec_verified,
                    "output_active": self._paused, "sink_present": self._sink_present,
                    "capture": self.capture.running, "kws": self._kws_status(),
                    "backend": self.backend, "aec_reference_verified": self.aec_verified,
                    "aec_verification_basis": "configuration_declaration",
                    "human_barge_in_tested": False,
                    "degraded_reason": None if self.aec_verified else "AEC_NOT_VERIFIED_OUTPUT_ASR_GUARDED",
                    "error": self._last_error, "last_partial": self._partial,
                    "request_queue": self._request_queue.qsize(), "request_queue_limit": 2,
                    "request_workers": sum(t.is_alive() for t in self._workers),
                    "stats": dict(self._stats),
                    "last_command_age_s": round(time.monotonic() - self._last_command_at, 1) if self._last_command_at else None}
        data["asr"] = self._asr.status() if self._asr is not None else {"ready": False}
        return data

    def set_sink_present(self, present):
        self._sink_present = bool(present)

    def _ensure_workers(self):
        self._workers = [t for t in self._workers if t.is_alive()]
        while len(self._workers) < 2:
            thread = threading.Thread(target=self._request_worker, name="device-request", daemon=True)
            self._workers.append(thread)
            thread.start()

    def start(self):
        with self._lock:
            if self._running:
                return True
            if self._thread is not None and self._thread.is_alive():
                self._last_error = "旧采集线程尚未退出"
                return False
        if self.backend not in ("streaming-paraformer", "offline-paraformer"):
            self._last_error = "ASR_BACKEND 不支持；没有自动回退"
            return False
        if not self.capture.available() or not self._kws_status().get("ready", False):
            self._last_error = "麦克风工具或KWS未就绪"
            return False
        try:
            if self.backend == "streaming-paraformer":
                if self._asr is None:
                    model_dir = resolve_local(os.environ.get("ASR_STREAMING_MODEL_DIR") or
                                     ROOT / "models/asr-streaming/sherpa-onnx-streaming-paraformer-bilingual-zh-en")
                    self._asr = StreamingASR(StreamingASRConfig(
                        model_dir, num_threads=int(os.environ.get("ASR_STREAMING_THREADS", "1"))))
                self._asr.load()
                self._asr_session = self._asr.create_session()
            if not self.capture.start():
                raise RuntimeError("麦克风采集启动失败")
        except Exception as exc:
            self._last_error = str(exc)
            if self._asr is not None:
                self._asr.close()
            self._asr_session = None
            return False
        with self._audio_lock:
            self._paused = False
            self._last_error = None
            self.gate.reset()
            if hasattr(self.gate, "armed_until"):
                self.gate.armed_until = 0.0
            self._asr_active = self._asr_open = False
            self._conversation_active = False
            self._running = True
            self._ensure_workers()
            self._thread = threading.Thread(target=self._loop, name="device-voice", daemon=True)
            self._thread.start()
        return True

    def _loop(self):
        while self._running:
            chunk = self.capture.read()
            if chunk is None:
                time.sleep(0.02)
                continue
            try:
                self.process_chunk(chunk)
            except Exception as exc:
                self._last_error = "语音输入异常：" + str(exc)
                self._stats["stream_errors"] += 1
                self._reset_recognition(disarm=True)

    def _reset_recognition(self, disarm=False):
        with self._audio_lock:
            if self._asr_session is not None:
                self._asr_session.reset()
            self._asr_active = self._asr_open = False
            self._partial = ""
            self.gate.reset()
            if disarm and hasattr(self.gate, "armed_until"):
                self.gate.armed_until = 0.0
            if disarm:
                self._conversation_active = False

    def process_chunk(self, samples):
        """设备循环与可注入回放共同入口；此函数不发起阻塞HTTP。"""
        with self._audio_lock:
            if not self._running:
                return []
            if self._paused and not self.aec_verified:
                # 仍持续消耗采集块，播放回声不送KWS/ASR，更不执行控制。
                return []
            if self._paused and self.aec_verified and self._conversation_active and hasattr(self.gate, "armed_until"):
                self.gate.armed_until = time.monotonic() + getattr(self.gate, "armed_seconds", 15.0)
            event = self.gate.feed(samples)
            if event.get("wake"):
                self._conversation_active = True
                if self.backend == "offline-paraformer":
                    OUTPUT.push("我在，请说", priority=PRIORITY_CONFIRM, source="device",
                                reason="wake", ttl=2.0, dedupe_key="device-wake")
            if self.backend == "offline-paraformer":
                command = event.get("command")
                if command is not None and len(command):
                    self._handle_command(command)
                return []
            armed = bool(event.get("armed"))
            if not armed and not self._asr_active:
                if self._asr_open:
                    self._reset_recognition()
                return []
            rms = float(event.get("rms", 0.0))
            voiced = rms >= float(getattr(self.gate, "threshold", 0.012))
            self._asr_active = self._asr_active or voiced
            self._asr_open = True
            if self._paused:
                self._output_voiced_s = self._output_voiced_s + len(samples) / 16000 if voiced else 0.0
            source = event.get("preroll") if event.get("wake") else None
            source = samples if source is None or not len(source) else source
            try:
                results = self._asr_session.feed(source)
            except UtteranceLimitError:
                self._last_error = "话语太长，请重新唤醒后分段说；未提交截断命令"
                self._stats["stream_errors"] += 1
                self._reset_recognition(disarm=True)
                return []
            for result in results:
                if result.text and not result.is_final:
                    self._asr_active = True
                    self._partial = result.text[:512]
                    self._stats["partial_results"] += 1
                    if self._paused and self.aec_verified and voiced and not self._barge_confirmed and self._output_voiced_s >= self._barge_min_seconds:
                        self._barge_confirmed = True
                        self._stats["barge_ins"] += 1
                        self._invalidate_pending()
                        # sink.stop 可能同步调用 resume；resume不能重置这句新话语。
                        OUTPUT.cancel(source="barge_in")
                if result.is_final:
                    self._stats["final_results"] += 1
                    if result.text.strip() and (not self._paused or self._barge_confirmed):
                        self._conversation_active = True
                        self._enqueue_final(result.text, result.utterance_id)
                    self._asr_active = self._asr_open = False
                    self._partial = ""
                    self._barge_confirmed = False
                    self._output_voiced_s = 0.0
            return results

    def _drain_pending(self):
        count = 0
        while True:
            try:
                self._request_queue.get_nowait()
                self._request_queue.task_done()
                count += 1
            except queue.Empty:
                return count

    def _invalidate_pending(self):
        with self._lock:
            self._dispatch_epoch += 1
            self._stats["dropped_requests"] += self._drain_pending()

    def _enqueue(self, url, body):
        with self._lock:
            if not self._running:
                return False
            self._dispatch_epoch += 1
            epoch = self._dispatch_epoch
            self._stats["dropped_requests"] += self._drain_pending()
            try:
                self._request_queue.put_nowait((epoch, url, body))
            except queue.Full:
                self._stats["dropped_requests"] += 1
                return False
            self._last_command_at = time.monotonic()
            return True

    def _enqueue_final(self, text, utterance_id):
        return self._enqueue(self.final_url, {
            "final_transcript": text, "is_final": True, "wake_detected": True,
            "session_id": self.session_id, "utterance_id": utterance_id,
            "request_id": utterance_id, "stream_speech": True})

    def _handle_command(self, samples):
        """显式旧基线；不自动从流式错误回退到此路径。"""
        body = {"audio": "data:audio/wav;base64," + base64.b64encode(encode_wav(samples)).decode("ascii"),
                "session_id": self.session_id, "wake_detected": True,
                "request_id": "device-offline-" + uuid.uuid4().hex}
        if self.camera is not None and self.camera.status().get("running"):
            still = self.camera.capture_still()
            if still is not None:
                self._seq += 1
                body.update(image="data:image/jpeg;base64," + jpeg_b64(still),
                            image_vlm="data:image/jpeg;base64," + jpeg_b64(still, 800),
                            frame_id=f"dev-{self._seq}",
                            captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            image_source="device_camera")
        return self._enqueue(self.url, body)

    def _request_worker(self):
        while not self._worker_stop.is_set():
            try:
                epoch, url, body = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    if not self._running or epoch != self._dispatch_epoch:
                        self._stats["dropped_requests"] += 1
                        continue
                    self._stats["active_requests"] += 1
                try:
                    response = self._post(url, json=body, timeout=(3, 120))
                    data = response.json() if response.content else {}
                    code = data.get("code")
                    if not code and isinstance(data.get("error"), dict):
                        code = data["error"].get("code")
                    if code == "CANCELLED":
                        with self._lock:
                            self._stats["ignored_cancelled"] += 1
                        continue
                    if not response.ok:
                        raise RuntimeError(str(data.get("error") or f"HTTP {response.status_code}"))
                except Exception as exc:
                    with self._lock:
                        if self._running and epoch == self._dispatch_epoch:
                            self._last_error = str(exc)
                            self._stats["request_errors"] += 1
                            OUTPUT.push("设备语音请求失败，请稍后再试", priority=PRIORITY_CONFIRM,
                                        source="device", reason="voice_error", ttl=4.0)
                finally:
                    with self._lock:
                        self._stats["active_requests"] -= 1
            finally:
                self._request_queue.task_done()

    def pause(self):
        """输出开始回调：清旧识别状态，采集持续运行。"""
        with self._audio_lock:
            if self._paused:
                return
            self._paused = True
            self._output_voiced_s = 0.0
            self._barge_confirmed = False
            self._reset_recognition()

    def resume(self):
        """输出结束/取消回调；保留正在形成的插话内容。"""
        with self._lock:
            self._paused = False

    def stop(self):
        self._running = False
        self._invalidate_pending()
        self.capture.stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is None or not thread.is_alive():
            self._thread = None
            if self._asr is not None:
                self._asr.close()
            self._asr_session = None
        else:
            self._last_error = "采集线程尚未退出；没有重新启动第二线程"
        self._paused = False

    def close(self):
        self.stop()
        self._worker_stop.set()
        for thread in self._workers:
            thread.join(timeout=0.2)
