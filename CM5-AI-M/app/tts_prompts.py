"""本地预制语音提示：启动校验、按事件 ID 加载、损坏时回退。

预制音频只用于固定提示（例如唤醒确认「我在，请说」）。播放仍走唯一输出总线，
优先级、取消与主控危险提示不变；文件缺失、摘要不符或格式不符时返回 None，
由调用方回退到本地 Piper。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import wave
from pathlib import Path

MAX_AUDIO_BYTES = 8 * 1024 * 1024
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPWIDTH = 2


class PromptStore:
    """读取并校验 models/tts/prompts 下的固定提示 WAV。"""

    def __init__(self, directory: Path | str | None, *, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 max_bytes: int = MAX_AUDIO_BYTES, manifest_name: str = "manifest.json"):
        self.directory = Path(directory) if directory else None
        self.sample_rate = int(sample_rate)
        self.max_bytes = int(max_bytes)
        self.manifest_name = manifest_name
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._cache: dict[str, bytes] = {}
        self._errors: dict[str, str] = {}
        self._loaded = False
        self._manifest_path: Path | None = None

    @property
    def manifest_path(self) -> Path | None:
        if self._manifest_path is not None:
            return self._manifest_path
        if self.directory is None:
            return None
        return self.directory / self.manifest_name

    def load(self) -> "PromptStore":
        """读取清单并校验每个预制音频；单个条目失败只记录，不影响其他条目。"""
        with self._lock:
            self._entries = {}
            self._cache = {}
            self._errors = {}
            self._loaded = True
            manifest = self.manifest_path
            if self.directory is None or manifest is None or not manifest.is_file():
                self._errors["manifest"] = "清单不存在"
                return self
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._errors["manifest"] = f"清单读取失败：{exc}"
                return self
            if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
                self._errors["manifest"] = "清单缺少 entries 对象"
                return self
            expected_rate = int(data.get("sample_rate") or self.sample_rate)
            if expected_rate != self.sample_rate:
                self._errors["manifest"] = f"清单采样率 {expected_rate} 与播放约定 {self.sample_rate} 不一致"
                return self
            for prompt_id, spec in data["entries"].items():
                try:
                    entry = self._validate_entry(prompt_id, spec)
                except (OSError, ValueError, KeyError) as exc:
                    self._errors[str(prompt_id)] = str(exc)
                    continue
                self._entries[str(prompt_id)] = entry
            return self

    def _validate_entry(self, prompt_id: str, spec) -> dict:
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("事件 ID 必须是非空字符串")
        if not isinstance(spec, dict):
            raise ValueError("条目必须是对象")
        name = spec.get("file")
        if not isinstance(name, str) or not name:
            raise ValueError("缺少 file")
        path = (self.directory / name).resolve()
        if self.directory is not None and self.directory.resolve() not in path.parents and path.parent != self.directory.resolve():
            raise ValueError("file 必须位于预制目录内")
        if not path.is_file():
            raise ValueError(f"文件不存在：{name}")
        size = path.stat().st_size
        if size > self.max_bytes:
            raise ValueError(f"文件超过 {self.max_bytes} 字节上限")
        raw = path.read_bytes()
        expected_sha = str(spec.get("sha256") or "").strip().lower()
        if not expected_sha:
            raise ValueError("缺少 sha256")
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("sha256 不匹配")
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.getnframes()
                duration = frames / rate if rate else 0.0
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"不是有效 WAV：{exc}") from exc
        if (channels, sampwidth) != (DEFAULT_CHANNELS, DEFAULT_SAMPWIDTH):
            raise ValueError("必须是单声道 16 位 PCM")
        if rate != self.sample_rate:
            raise ValueError(f"采样率 {rate} 与约定 {self.sample_rate} 不一致")
        min_duration = float(spec.get("min_duration_s", 0.2))
        max_duration = float(spec.get("max_duration_s", 8.0))
        if min_duration <= 0 or max_duration < min_duration:
            raise ValueError("时长范围不合法")
        if not (min_duration <= duration <= max_duration):
            raise ValueError(f"时长 {duration:.3f}s 不在 {min_duration}-{max_duration}s 范围")
        text = spec.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("缺少 text")
        return {"prompt_id": prompt_id, "file": name, "sha256": actual_sha,
                "duration_s": round(duration, 4), "sample_rate": rate, "bytes": size,
                "text": text}

    def read(self, prompt_id: str) -> bytes | None:
        """返回可播放 WAV；未知或损坏条目返回 None，由调用方回退。"""
        name = str(prompt_id or "")
        if not name:
            return None
        with self._lock:
            if not self._loaded:
                self.load()
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            entry = self._entries.get(name)
            if entry is None:
                return None
            try:
                raw = (self.directory / entry["file"]).read_bytes()
            except OSError:
                return None
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                self._errors[name] = "读取时 sha256 不匹配"
                return None
            self._cache[name] = raw
            return raw

    def status(self) -> dict:
        with self._lock:
            return {"loaded": self._loaded, "directory": str(self.directory) if self.directory else None,
                    "prompts": sorted(self._entries), "errors": dict(self._errors)}
