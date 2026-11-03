"""设备扬声器音量，固定操作 ALSA 播放控制，不调节录音增益。"""
import os
import re
import shutil
import subprocess


def _mixer():
    exe = shutil.which("amixer")
    if not exe:
        raise RuntimeError("树莓派未安装 amixer")
    device = os.environ.get("ALSA_OUTPUT", "plughw:CARD=Array,DEV=0")
    match = re.search(r"CARD=([^,]+)", device)
    card = os.environ.get("ALSA_MIXER_CARD") or (match.group(1) if match else "0")
    control = os.environ.get("ALSA_MIXER_CONTROL", "PCM,0")
    return [exe, "-c", card], card, control


def _run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("读取或设置树莓派音量失败") from exc
    if result.returncode:
        raise RuntimeError("树莓派扬声器音量控制不可用")
    return result.stdout


def get_volume():
    args, card, control = _mixer()
    output = _run(args + ["sget", control])
    if "pvolume" not in output:
        raise RuntimeError("配置的 ALSA 控制不支持播放音量")
    values = [int(x) for x in re.findall(r"Playback[^\n]*\[(\d+)%\]", output)]
    switches = re.findall(r"Playback[^\n]*\[(on|off)\]", output)
    if not values:
        raise RuntimeError("无法读取扬声器音量百分比")
    return {"volume_percent": round(sum(values) / len(values)),
            "muted": bool(switches) and all(x == "off" for x in switches),
            "mute_supported": bool(switches), "card": card, "control": control,
            "output": "device"}


def set_volume(volume_percent=None, muted=None):
    if volume_percent is None and muted is None:
        raise ValueError("请提供 volume_percent 或 muted")
    if volume_percent is not None and (type(volume_percent) is not int or not 0 <= volume_percent <= 100):
        raise ValueError("volume_percent 必须是 0 到 100 的整数")
    if muted is not None and type(muted) is not bool:
        raise ValueError("muted 必须为布尔值")
    args, _, control = _mixer()
    state = get_volume()
    if muted is not None and not state["mute_supported"]:
        raise ValueError("该扬声器控制不支持静音开关")
    changes = []
    if volume_percent is not None:
        changes.append(str(volume_percent) + "%")
    if muted is not None:
        changes.append("mute" if muted else "unmute")
    _run(args + ["sset", control] + changes)
    return get_volume()
