# -*- coding: utf-8 -*-
"""规则意图解析：先规则、再状态消歧、最后低置信度澄清。

不把意图判断交给 VLM，也不为了路由常驻分类器。confidence 是规则置信度，
不是校准过的概率；没有训练数据前不宣称概率可靠。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

ASR_ALIASES = (
    ("眼轻", "眼前"),
    ("眼钱", "眼前"),
    ("前边", "前面"),
    ("卫生见", "卫生间"),
    ("出扣", "出口"),
    ("出抠", "出口"),
    ("保质其", "保质期"),
)

ACK_ONLY = re.compile(
    r"^(?:喂|哎|嗯|嗯嗯|你好|您好|在吗|在不在|听得到吗|能听到吗|"
    r"对|对对|对对对|好的|好|行|可以|知道了|谢谢)$"
)

WAKE_HEAD = "小晓筱"
WAKE_TAIL = "冰兵并病丙饼秉炳柄禀斌宾滨彬缤濒"
WAKE_SEP = r"[\s，,。.、！!？?·]*"
WAKE_PATTERN = re.compile(rf"[{WAKE_HEAD}]{WAKE_SEP}[{WAKE_TAIL}]", re.I)

NEGATED_STOP = re.compile(
    r"(?:不要|别|不用|不想|没必要)(?:再)?(?:停止|停掉|关掉|关闭|停(?!止))(?:播报|语音|引导|导航)?"
)
SPEECH_STOP = re.compile(
    r"停止播报|停止语音|停止说话|别播报|别念了|别说了|不要播报|不要念|关闭播报|关掉播报|"
    r"停播|闭嘴|安静|声音停|停止播放|别出声"
)
GUIDE_STOP = re.compile(
    r"停止引导|停止导航|关闭引导|关闭导航|关掉引导|关掉导航|结束引导|结束导航|"
    r"退出引导|退出导航|不用引导|不用导航|别引导|别导航|不要引导|不要导航|"
    r"停止带路|别带路"
)
GUIDE_START = re.compile(
    r"开始引导|开始导航|打开引导|开启引导|打开导航|继续引导|继续导航|帮我导航|给我导航|"
    r"带我走|开始带路|开始引路|开始走路引导"
)
CAMERA_OFF = re.compile(
    r"(?:不要|别|不用)(?:再)?(?:打开|开启|用)?(?:摄像头|相机|镜头)"
    r"|(?:关闭|关掉|停用)(?:摄像头|相机|镜头)|(?:摄像头|相机)(?:关掉|关闭|停用)"
    r"|别看了|不看东西"
)
TOPIC_SWITCH = re.compile(r"换个话题|聊点别的|只聊天|继续聊天|不看东西|不想看图")
DEFINITION = re.compile(r"是什么意思|什么意思|怎么理解|如何定义")
GUIDE_QUERY = re.compile(
    r"现在能走吗|前面能走吗|可以直行吗|现在安全吗|引导状态|前面通不通|"
    r"能不能走|可以走吗|现在可以走吗|前方能走吗|还能走吗"
)
AMBIGUOUS_STOP = re.compile(r"^(?:停一下|停下来|暂停|先停|停停|等等|等一下)$")

DATE_WORDS = ("日期", "生产", "保质期", "过期", "批号", "到期", "有效期")
TEXT_WORDS = ("读", "念", "什么字", "写的什么", "文字", "上面的字", "标签上", "写的啥")
DETAIL_WORDS = ("详细", "仔细说", "具体说", "多说说", "多讲", "描述一下")
COMPARE_WORDS = ("比较", "对比", "哪个更", "有什么不同")
CHAT_WORDS = ("聊", "烦", "难受", "难过", "开心", "累", "陪我", "新闻", "天气", "心情", "无聊", "孤单")
SCENE_WORDS = ("有什么", "看看", "前面", "前方", "周围", "什么情况", "前面是", "这边是")
FOLLOW_MARKERS = re.compile(r"这|那|它|他|她|刚才|刚刚|为什么|怎么|多少|哪个|哪边|是不是|吗|呢|还|再")

TOILET_WORDS = ("卫生间", "洗手间", "厕所", "WC", "wc", "toilet", "Toilet")
EXIT_WORDS = ("出口", "安全出口", "EXIT", "exit")

VISION_INTENTS = {"scene", "read_text", "read_date", "find_object", "followup", "compare", "detail"}


@dataclass
class Task:
    intent: str
    slots: dict = field(default_factory=dict)
    request_id: str = ""
    session_id: str = ""
    needs_image: bool = False
    priority: int = 3
    deadline_s: float = 30.0
    cancel_key: str = ""
    confidence: float = 0.0
    source: str = "rule"
    clarify: str = ""
    actions: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "slots": self.slots,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "needs_image": self.needs_image,
            "priority": self.priority,
            "deadline_s": self.deadline_s,
            "cancel_key": self.cancel_key,
            "confidence": self.confidence,
            "source": self.source,
            "clarify": self.clarify,
            "actions": self.actions,
            "note": self.note,
        }


def normalize_text(text: str) -> str:
    out = (text or "").strip()
    for src, dst in ASR_ALIASES:
        out = out.replace(src, dst)
    return out


def clean_command(text: str) -> str:
    """去掉唤醒词、标点和空白，保留汉字/字母/数字。"""
    out = WAKE_PATTERN.sub("", normalize_text(text))
    return re.sub(r"[，。！？、,.!?\s]", "", out)


def contains_wake(text: str) -> bool:
    return bool(WAKE_PATTERN.search(text or ""))


def extract_item(cmd: str) -> str:
    for pattern in (
        r"帮我找(?:一下|一找)?(.+)",
        r"找(?:一下|一找)?(.+)",
        r"有没有(.+)",
        r"(.+?)(?:在哪儿|在哪里|在哪|呢)$",
    ):
        m = re.search(pattern, cmd)
        if m:
            name = re.sub(r"[吗呢啊吧，。]", "", m.group(1)).strip()
            name = re.sub(r"^(一下|一找)", "", name).strip()
            if name and name not in ("东西", "什么"):
                return name
    return ""


def extract_compare(cmd: str) -> str:
    m = re.search(r"(?:比较|对比)(?:一下)?(.+)", cmd)
    if m:
        name = re.sub(r"[吗呢啊吧，。]", "", m.group(1)).strip()
        if name in ("一下", "一找"):
            return ""
        return name
    return ""


def _task(
    intent: str,
    slots: Optional[dict] = None,
    *,
    needs_image: bool = False,
    priority: int = 3,
    confidence: float = 0.9,
    session_id: str = "",
    clarify: str = "",
    actions: Optional[list] = None,
    note: str = "",
    deadline_s: float = 30.0,
) -> Task:
    return Task(
        intent=intent,
        slots=slots or {},
        request_id=uuid.uuid4().hex[:12],
        session_id=session_id,
        needs_image=needs_image,
        priority=priority,
        deadline_s=deadline_s,
        cancel_key=f"{session_id}:{intent}:{uuid.uuid4().hex[:6]}",
        confidence=confidence,
        clarify=clarify,
        actions=actions or [],
        note=note,
    )


def parse(
    text: str,
    *,
    has_context: bool = False,
    guide_active: bool = False,
    speech_active: bool = False,
    session_id: str = "",
) -> Task:
    """解析一条已去掉唤醒词的语音指令。"""
    normalized = normalize_text(text)
    cmd = clean_command(normalized)
    if not cmd:
        return _task("silent", confidence=1.0, session_id=session_id, note="空文本")
    if ACK_ONLY.fullmatch(cmd):
        return _task("ack", confidence=1.0, session_id=session_id)

    if not WAKE_PATTERN.search(normalized):
        pass  # 调用方已处理唤醒；这里保留原文用于日志

    # 否定“停止”优先：不要停止引导 = 不执行停止
    if NEGATED_STOP.search(cmd) and not re.search(r"不要(引导|导航)", cmd):
        return _task(
            "ack",
            confidence=0.9,
            session_id=session_id,
            note="否定停止",
        )

    # 明确停止播报
    if SPEECH_STOP.search(cmd):
        return _task(
            "speech_stop",
            priority=2,
            confidence=0.98,
            session_id=session_id,
            actions=["cancel_speech"],
            deadline_s=5.0,
        )

    # 明确关闭引导/导航
    if GUIDE_STOP.search(cmd):
        return _task(
            "guide_stop",
            priority=2,
            confidence=0.98,
            session_id=session_id,
            actions=["stop_guide"],
            deadline_s=5.0,
        )

    # 明确开启引导
    if GUIDE_START.search(cmd):
        return _task(
            "guide_start",
            priority=2,
            confidence=0.98,
            session_id=session_id,
            actions=["start_guide"],
            deadline_s=5.0,
        )

    # 换话题/只聊天优先于“不看东西”这类关闭摄像头词
    if TOPIC_SWITCH.search(cmd):
        return _task(
            "chat",
            priority=3,
            confidence=0.9,
            session_id=session_id,
            actions=["stop_guide"] if guide_active else [],
        )

    # 关闭摄像头：在引导中同时停止引导，避免继续输出旧方向
    if CAMERA_OFF.search(cmd):
        return _task(
            "camera_off",
            priority=2,
            confidence=0.95,
            session_id=session_id,
            actions=["stop_guide", "close_camera"],
            deadline_s=5.0,
        )

    # 只说“停一下/暂停”：结合当前状态消歧
    if AMBIGUOUS_STOP.match(cmd) or cmd in ("停", "停止", "停下"):
        if guide_active and not speech_active:
            return _task(
                "guide_stop", priority=2, confidence=0.75, session_id=session_id,
                actions=["stop_guide"], deadline_s=5.0,
                note="按当前状态解释为停止引导",
            )
        if speech_active and not guide_active:
            return _task(
                "speech_stop", priority=2, confidence=0.75, session_id=session_id,
                actions=["cancel_speech"], deadline_s=5.0,
                note="按当前状态解释为停止播报",
            )
        if guide_active and speech_active:
            return _task(
                "clarify",
                priority=2,
                confidence=0.4,
                session_id=session_id,
                clarify="你是要停止播报，还是关闭引导？",
                note="两种控制都在进行，不能猜",
            )
        return _task(
            "clarify",
            priority=2,
            confidence=0.4,
            session_id=session_id,
            clarify="现在没有正在进行的播报或引导。",
            note="无状态可消歧",
        )

    # 引导状态查询走快路径，不调用 VLM；语义只复用当前引导事件。
    # 询问楼梯/扶梯/过马路时不是“当前能否直行”，继续走视觉/安全兜底。
    if GUIDE_QUERY.search(cmd) and not re.search(r"楼梯|扶梯|电梯|马路|斑马线|过街", cmd):
        return _task(
            "guide_status",
            priority=2,
            confidence=0.95,
            session_id=session_id,
            deadline_s=5.0,
        )

    # 视觉任务：卫生间隔间/出口归入 find_object 的特殊目标
    if any(w in cmd for w in TOILET_WORDS):
        return _task(
            "find_object",
            {"target": "toilet", "item": "卫生间标识"},
            needs_image=True, confidence=0.95, session_id=session_id,
        )
    if any(w in cmd for w in EXIT_WORDS):
        return _task(
            "find_object",
            {"target": "exit", "item": "出口标识"},
            needs_image=True, confidence=0.95, session_id=session_id,
        )
    if any(w in cmd for w in DATE_WORDS):
        return _task(
            "read_date",
            needs_image=True, confidence=0.95, session_id=session_id,
        )
    if any(w in cmd for w in TEXT_WORDS):
        return _task(
            "read_text",
            needs_image=True, confidence=0.9, session_id=session_id,
        )

    if any(w in cmd for w in COMPARE_WORDS):
        item = extract_compare(cmd)
        if not item:
            return _task(
                "clarify", priority=2, confidence=0.5, session_id=session_id,
                clarify="你要比较哪两样东西？",
            )
        return _task(
            "compare", {"item": item}, needs_image=True, confidence=0.85,
            session_id=session_id,
        )

    if any(w in cmd for w in DETAIL_WORDS):
        return _task("detail", needs_image=True, confidence=0.85, session_id=session_id)

    if "找" in cmd or "有没有" in cmd:
        item = extract_item(cmd)
        if not item:
            return _task(
                "clarify", priority=2, confidence=0.5, session_id=session_id,
                clarify="你要找什么？",
            )
        return _task(
            "find_object", {"item": item}, needs_image=True, confidence=0.85,
            session_id=session_id,
        )

    m = re.search(r"(.+?)(?:在哪儿|在哪里|在哪)$", cmd)
    if m:
        item = re.sub(r"[吗呢啊吧，。]", "", m.group(1)).strip()
        if item and item not in ("它", "他", "她", "这个", "那个", "东西"):
            return _task(
                "find_object", {"item": item}, needs_image=True, confidence=0.8,
                session_id=session_id,
            )

    # 聊天与实时信息优先于“有什么”这类泛场景词
    if DEFINITION.search(cmd) and "眼前" not in cmd and "手里" not in cmd:
        return _task("chat", {"question": cmd}, confidence=0.85, session_id=session_id)
    if any(w in cmd for w in CHAT_WORDS):
        return _task("chat", {"question": cmd}, confidence=0.8, session_id=session_id)

    # “眼前这个是什么”优先按追问；“眼前有什么/手里有什么”仍按场景描述
    if ("眼前" in cmd or "手里" in cmd) and not any(w in cmd for w in ("有什么", "看看", "周围", "什么情况")):
        return _task(
            "followup", {"question": cmd}, needs_image=True, confidence=0.8,
            session_id=session_id,
        )
    if any(w in cmd for w in SCENE_WORDS):
        return _task("scene", needs_image=True, confidence=0.85, session_id=session_id)

    if has_context and (FOLLOW_MARKERS.search(cmd) or len(cmd) >= 6):
        return _task(
            "followup", {"question": cmd}, needs_image=True, confidence=0.6,
            session_id=session_id,
        )

    if len(cmd) >= 8:
        return _task("chat", {"question": cmd}, confidence=0.65, session_id=session_id)

    return _task(
        "unknown",
        priority=2,
        confidence=0.3,
        session_id=session_id,
        clarify="我没听清，请再说一遍。",
        note="短文本未命中规则",
    )


# 兼容旧调用：返回 (intent, item)，不携带结构化字段。
def parse_legacy(cmd: str, has_context: bool = False) -> tuple:
    task = parse(cmd, has_context=has_context)
    item = task.slots.get("item", "")
    mapping = {
        "read_text": "text",
        "read_date": "date",
        "find_object": task.slots.get("target") or "item",
        "clarify": "unknown",
    }
    if task.slots.get("target") in ("toilet", "exit"):
        item = ""
    return mapping.get(task.intent, task.intent), item
