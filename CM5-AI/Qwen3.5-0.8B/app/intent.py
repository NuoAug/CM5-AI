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
    r"(?:不要|别|不用|不想|没必要)(?:再)?(?:停止|停掉|关掉|关闭|停(?!止))(?:播报|语音)?"
)
SPEECH_STOP = re.compile(
    r"停止播报|停止语音|停止说话|别播报|别念了|别说了|不要播报|不要念|关闭播报|关掉播报|"
    r"停播|闭嘴|安静|声音停|停止播放|别出声"
)
# 控制只认明确的完整话语；否定、改口和目的地请求交由澄清。
NAVIGATION_WORDS = re.compile(r"引导|导航|带路|引路")
CONTROL_ACTIONS = (
    (r"(?:开始|打开|开启)(?:引导|导航|带路|引路)|开始走路引导", "start_guidance"),
    (r"暂停(?:引导|导航|带路|引路)", "pause_guidance"),
    (r"(?:继续|恢复)(?:引导|导航|带路|引路)", "resume_guidance"),
    (r"(?:停止|关闭|关掉|结束|退出)(?:引导|导航|带路|引路)", "end_guidance"),
    (r"(?:重复|再说一遍|重播)(?:刚才的)?(?:引导|导航|带路|引路)(?:提示)?", "repeat_guidance"),
)
NAVIGATION_NEGATION = re.compile(r"不要|不用|别|不是|不想|并非|没让|改成|改为|而是")
NAVIGATION_STATUS = re.compile(r"(?:引导|导航|带路|引路)(?:的)?(?:状态|情况)|当前.*(?:引导|导航)")
CAMERA_OFF = re.compile(
    r"(?:不要|别|不用)(?:再)?(?:打开|开启|用)?(?:摄像头|相机|镜头)"
    r"|(?:关闭|关掉|停用)(?:摄像头|相机|镜头)|(?:摄像头|相机)(?:关掉|关闭|停用)"
    r"|别看了|不看东西"
)
TOPIC_SWITCH = re.compile(r"换个话题|聊点别的|只聊天|继续聊天|不看东西|不想看图")
DEFINITION = re.compile(r"是什么意思|什么意思|怎么理解|如何定义")
PASSAGE_QUERY = re.compile(
    r"能不能(?:走|直行|通行|过马路|过街)|(?:能|可以)(?:直行|走|通行|过马路|过街).{0,12}(?:吗|么|不)|"
    r"(?:能不能|能|可以)通过(?:这|那|路口|通道|马路|街道|楼梯|走廊|大门)|"
    r"(?:前面|前方|这里|那里|路口|通道).{0,8}(?:能不能|能|可以)通过|^(?:能不能|能|可以)通过(?:吗|么)?$|"
    r"现在安全吗|前面通不通|前方通不通"
)
SPATIAL_QUERY = re.compile(
    r"距离|多远|多少米|几米|怎么绕|如何绕|怎样绕|往哪(?:边|里)?走|该走哪|"
    r"能否(?:走|通行|通过)|(?:通道|道路|路面|前方|前面|这里).{0,8}(?:安全|宽度|够宽|能过)|"
    r"(?:宽度|高度).{0,8}(?:通过|通行|够不够)|"
    r"(?:这|那|前面|前方).{0,8}(?:路|门|通道).{0,8}(?:怎么走|过得去|走得通)|"
    r"(?:我|现在|这里|前面|前方).{0,8}(?:左转|右转|直走|往前走|绕过去)|"
    r"^(?:左转|右转|直走|往前走)(?:吗|还是|会|能|$)|往前走.{0,8}(?:撞|危险)"
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
    speech_active: bool = False,
    session_id: str = "",
    is_final: bool = True,
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

    # 概念解释可聊天；同句中的当前行动/测量问题不能被“什么意思”豁免。
    concrete_spatial_question = PASSAGE_QUERY.search(cmd) or re.search(
        r"多远|多少(?:米|厘米)|几米|怎么绕|如何绕|怎样绕|过得去|走得通|会撞|左转还是右转", cmd
    )
    definition_only = (DEFINITION.search(cmd) and not concrete_spatial_question
                       and re.search(r"(?:是什么意思|什么意思|怎么理解|如何定义)$", cmd))
    if not definition_only:
        if NAVIGATION_WORDS.search(cmd) or cmd == "带我走":
            if NAVIGATION_NEGATION.search(cmd):
                return _task(
                    "clarify", priority=2, session_id=session_id,
                    clarify="我还没有更改引导。请明确说开始、暂停、继续或结束引导。",
                    note="否定或改口不直接提交控制意图",
                )
            for pattern, action in CONTROL_ACTIONS:
                if re.fullmatch(r"(?:请|请你|帮我)?(?:" + pattern + r")(?:一下|吧)?", cmd):
                    if is_final is not True:
                        return _task("clarify", priority=2, session_id=session_id,
                                     clarify="等待你说完后再确认引导操作。", note="临时ASR不执行控制")
                    return _task("control_intent", {"action": action}, priority=2,
                                 session_id=session_id, deadline_s=5.0,
                                 note="仅向主控提交意图，不在本地执行引导")
            if NAVIGATION_STATUS.search(cmd):
                return _task("scene_query", priority=2, session_id=session_id, deadline_s=5.0)
            return _task("clarify", priority=2, session_id=session_id,
                         clarify="请明确说开始、暂停、继续、结束或重复引导。目的地关联还需主控接入。")
        if PASSAGE_QUERY.search(cmd) or SPATIAL_QUERY.search(cmd):
            return _task("scene_query", priority=2, session_id=session_id, deadline_s=5.0,
                         note="空间事实只读取主控，不交给图文模型判断")

    # 否定“停止”优先，不取消正在进行的播报。
    if NEGATED_STOP.search(cmd):
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

    # 换话题/只聊天优先于“不看东西”这类关闭摄像头词
    if TOPIC_SWITCH.search(cmd):
        return _task(
            "chat",
            priority=3,
            confidence=0.9,
            session_id=session_id,
        )

    # 关闭摄像头只控制相机。
    if CAMERA_OFF.search(cmd):
        return _task(
            "camera_off",
            priority=2,
            confidence=0.95,
            session_id=session_id,
            actions=["close_camera"],
            deadline_s=5.0,
        )

    # “停一下/暂停”只按当前播报状态解释。
    if AMBIGUOUS_STOP.match(cmd) or cmd in ("停", "停止", "停下"):
        if speech_active:
            return _task(
                "speech_stop", priority=2, confidence=0.75, session_id=session_id,
                actions=["cancel_speech"], deadline_s=5.0,
                note="按当前状态解释为停止播报",
            )
        return _task(
            "clarify", priority=2, confidence=0.4, session_id=session_id,
            clarify="现在没有正在进行的播报。", note="无播报可停止",
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
