"""Fail-closed path boundaries for every candidate launch route."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent.parent
PATH_ENVIRONMENTS = ("DATA_DIR", "PIPER_BIN", "PIPER_MODEL", "KWS_MODEL_DIR",
                     "KWS_KEYWORDS", "ASR_STREAMING_MODEL_DIR", "STREAMING_ASR_MODEL_DIR")

def resolve_local(value, *, root=ROOT):
    root = Path(root).resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("candidate path must stay inside this independent project") from exc
    return path


def validate_environment():
    for name in PATH_ENVIRONMENTS:
        if os.environ.get(name):
            resolve_local(os.environ[name])


def validate_profile(profile=None):
    if profile is None:
        profile = json.loads((ROOT / "config/model-profile.json").read_text(encoding="utf-8"))
    for key, value in profile.items():
        if key.endswith(("_file", "_file_posix", "_dir")):
            if not isinstance(value, str) or not value:
                raise ValueError("invalid candidate profile local path: " + key)
            resolve_local(value)
    return profile
