# -*- coding: utf-8 -*-
"""常驻 Piper；取消仅结束该代次，下一句可重新加载健康进程。"""
from __future__ import annotations
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


class TTSInterrupted(RuntimeError):
    pass


class PiperWorker:
    def __init__(self, piper_bin: Path, model: Path, tmp_dir: Path,
                 timeout_s: float = 90.0, idle_seconds: float = 0.0):
        self.piper_bin = Path(piper_bin).resolve()
        self.model = Path(model).resolve()
        self.tmp_dir = Path(tmp_dir).resolve()
        self.timeout_s = max(5.0, float(timeout_s))
        self.idle_seconds = max(0.0, float(idle_seconds))
        self._lock = threading.RLock()  # one synthesis at a time
        self._state_lock = threading.RLock()  # cancellation never waits for synthesis
        self._proc: Optional[subprocess.Popen] = None
        self._generation = 0
        self._seq = 0
        self._spawn_count = 0
        self._load_ms = None
        self._synth_count = 0
        self._last_gen_ms = None
        self._last_used = 0.0
        self._stop = threading.Event()
        self._reaper = None
        self._tmp_dir_ready = False

    def _ensure_tmp(self):
        if not self._tmp_dir_ready:
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            self._tmp_dir_ready = True

    @staticmethod
    def _dispose(proc):
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass

    def _spawn(self):
        self._ensure_tmp()
        if not self.piper_bin.exists():
            raise RuntimeError(f"Piper 可执行文件不存在：{self.piper_bin}")
        if not self.model.exists():
            raise RuntimeError(f"Piper 模型不存在：{self.model}")
        # Publication is guarded in synth; interrupt during Popen invalidates this spawn.
        return subprocess.Popen(
            [str(self.piper_bin), "-m", str(self.model), "--json-input", "--quiet"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(self.piper_bin.parent))

    def _check(self, generation, proc=None):
        with self._state_lock:
            if self._stop.is_set() or generation != self._generation:
                raise TTSInterrupted("TTS synthesis cancelled")
            if proc is not None and self._proc is not proc:
                raise TTSInterrupted("TTS worker replaced")

    def _idle_loop(self):
        while not self._stop.wait(5.0):
            with self._lock:
                with self._state_lock:
                    if self.idle_seconds <= 0 or time.monotonic() - self._last_used < self.idle_seconds:
                        continue
                    proc = self._proc
                    self._proc = None
                self._dispose(proc)

    def _wait_file(self, path, proc, generation):
        deadline = time.monotonic() + self.timeout_s
        last_size, stable = -1, 0
        while time.monotonic() < deadline:
            self._check(generation, proc)
            if proc.poll() is not None:
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
                    stable, last_size = 0, size
            time.sleep(0.01)
        raise RuntimeError(f"Piper 合成超时（>{self.timeout_s:.0f}s）")

    def synth(self, text, length_scale=1.0):
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("TTS 文本为空")
        with self._lock:
            with self._state_lock:
                generation = self._generation
                self._check(generation)
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    self._proc = None
                    stale, proc = proc, None
                else:
                    stale = None
            self._dispose(stale)
            cold = proc is None
            if cold:
                proc = self._spawn()
                try:
                    with self._state_lock:
                        self._check(generation)
                        self._proc = proc
                        self._spawn_count += 1
                        if self._reaper is None and self.idle_seconds > 0:
                            self._reaper = threading.Thread(target=self._idle_loop, name="piper-idle", daemon=True)
                            self._reaper.start()
                except Exception:
                    self._dispose(proc)
                    raise
            assert proc is not None and proc.stdin is not None
            self._seq += 1
            out = self.tmp_dir / f"piper-{os.getpid()}-{self._seq}.wav"
            payload = {"text": text, "output_file": str(out), "length_scale": float(length_scale)}
            started = time.perf_counter()
            try:
                self._check(generation, proc)
                try:
                    proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    proc.stdin.flush()
                except Exception as exc:
                    self._check(generation, proc)
                    raise RuntimeError(f"Piper 写入失败：{exc}") from exc
                self._wait_file(out, proc, generation)
                data = out.read_bytes()
                if len(data) < 1024:
                    raise RuntimeError("Piper 输出过小")
                gen_ms = int((time.perf_counter() - started) * 1000)
                with self._state_lock:
                    self._check(generation, proc)
                    if cold and self._load_ms is None:
                        self._load_ms = gen_ms
                    self._synth_count += 1
                    self._last_gen_ms = gen_ms
                    self._last_used = time.monotonic()
                return data, gen_ms
            except Exception:
                with self._state_lock:
                    if self._proc is proc:
                        self._proc = None
                self._dispose(proc)
                raise
            finally:
                try:
                    out.unlink()
                except OSError:
                    pass

    def _interrupt_process(self):
        with self._state_lock:
            self._generation += 1
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
        return proc

    def interrupt(self):
        # Detach even an idle warm process. The next synthesis never reuses a killed handle.
        # Do not close its pipe here: an old write may still hold the BufferedWriter lock.
        return self._interrupt_process() is not None

    def stop(self):
        self._stop.set()
        proc = self._interrupt_process()
        with self._lock:
            self._dispose(proc)

    def status(self):
        # Status must remain available while long synthesis owns the serial lock.
        with self._state_lock:
            return {"engine": "piper-persistent", "alive": self._proc is not None and self._proc.poll() is None,
                    "load_ms": self._load_ms, "synth_count": self._synth_count,
                    "last_gen_ms": self._last_gen_ms, "idle_seconds": self.idle_seconds,
                    "model": str(self.model), "generation": self._generation,
                    "spawn_count": self._spawn_count}
