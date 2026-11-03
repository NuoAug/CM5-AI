# -*- coding: utf-8 -*-
"""常驻 Piper 合成 worker。

Piper CLI 每次启动都会重新加载音色；这里用 `--json-input` 保持一个进程，
按行提交文本和输出文件，模型只在第一次合成时加载。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional


class TTSInterrupted(RuntimeError):
    pass


class PiperWorker:
    def __init__(
        self,
        piper_bin: Path,
        model: Path,
        tmp_dir: Path,
        timeout_s: float = 90.0,
        idle_seconds: float = 0.0,
    ):
        self.piper_bin = Path(piper_bin).resolve()
        self.model = Path(model).resolve()
        self.tmp_dir = Path(tmp_dir).resolve()
        self.timeout_s = max(5.0, float(timeout_s))
        self.idle_seconds = max(0.0, float(idle_seconds))
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._seq = 0
        self._load_ms = None
        self._synth_count = 0
        self._last_gen_ms = None
        self._last_used = 0.0
        self._stop = threading.Event()
        self._reaper: Optional[threading.Thread] = None
        self._tmp_dir_ready = False

    def _ensure_tmp(self) -> None:
        if not self._tmp_dir_ready:
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            self._tmp_dir_ready = True

    def _spawn(self) -> None:
        self._ensure_tmp()
        if not self.piper_bin.exists():
            raise RuntimeError(f"Piper 可执行文件不存在：{self.piper_bin}")
        if not self.model.exists():
            raise RuntimeError(f"Piper 模型不存在：{self.model}")
        self._proc = subprocess.Popen(
            [str(self.piper_bin), "-m", str(self.model), "--json-input", "--quiet"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self.piper_bin.parent),
        )
        if self._reaper is None and self.idle_seconds > 0:
            self._reaper = threading.Thread(target=self._idle_loop, name="piper-idle", daemon=True)
            self._reaper.start()

    def _idle_loop(self) -> None:
        while not self._stop.wait(5.0):
            with self._lock:
                proc = self._proc
                if proc is None or self.idle_seconds <= 0:
                    continue
                if time.monotonic() - self._last_used < self.idle_seconds:
                    continue
                self._proc = None
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def _wait_file(self, path: Path) -> None:
        deadline = time.monotonic() + self.timeout_s
        last_size = -1
        stable = 0
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise TTSInterrupted("TTS worker stopped")
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"Piper 进程退出，code={proc.returncode}")
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            if size > 1024:
                if size == last_size:
                    stable += 1
                    if stable >= 3:
                        return
                else:
                    stable = 0
                    last_size = size
            time.sleep(0.01)
        raise RuntimeError(f"Piper 合成超时（>{self.timeout_s:.0f}s）")

    def synth(self, text: str, length_scale: float = 1.0) -> tuple[bytes, int]:
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("TTS 文本为空")
        with self._lock:
            if self._stop.is_set():
                raise TTSInterrupted("TTS worker stopped")
            cold = self._proc is None
            if self._proc is None:
                self._spawn()
            assert self._proc is not None and self._proc.stdin is not None
            self._seq += 1
            out = self.tmp_dir / f"piper-{os.getpid()}-{self._seq}.wav"
            payload = {
                "text": text,
                "output_file": str(out),
                "length_scale": float(length_scale),
            }
            t0 = time.perf_counter()
            try:
                self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
            except Exception as exc:  # noqa: BLE001
                self._proc = None
                raise RuntimeError(f"Piper 写入失败：{exc}")
            try:
                self._wait_file(out)
                data = out.read_bytes()
                if len(data) < 1024:
                    raise RuntimeError("Piper 输出过小")
            except Exception:
                # 进程可能卡在上一句；杀掉后下次重新加载，避免队列持续错位。
                proc = self._proc
                self._proc = None
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                raise
            finally:
                try:
                    out.unlink()
                except OSError:
                    pass
            gen_ms = int((time.perf_counter() - t0) * 1000)
            if cold and self._load_ms is None:
                self._load_ms = gen_ms
            self._synth_count += 1
            self._last_gen_ms = gen_ms
            self._last_used = time.monotonic()
            return data, gen_ms

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def status(self) -> dict:
        with self._lock:
            return {
                "engine": "piper-persistent",
                "alive": self._proc is not None and self._proc.poll() is None,
                "load_ms": self._load_ms,
                "synth_count": self._synth_count,
                "last_gen_ms": self._last_gen_ms,
                "idle_seconds": self.idle_seconds,
                "model": str(self.model),
            }
