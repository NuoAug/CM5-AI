# -*- coding: utf-8 -*-
"""规则意图解析：先规则、再状态消歧、最后低置信度澄清。

不把意图判断交给 VLM，也不为了路由常驻分类器。confidence 是规则置信度，
不是校准过的概率；没有训练数据前不宣称概率可靠。
"""

from __future__ import annotations

import re
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
    r"(?:能不能|可不可以|能否|可否|能|可以).{0,20}?(?:穿|钻|挤|跨|迈|走|绕)?(?:过去|过来|通过|通行)(?:吗|么|不|呢|吧|$)|"
    r"(?:穿|钻|挤|跨|迈|走|过|绕)得(?:过去|过来|过|去|通|了)(?:吗|么|不|呢|$)|"
    r"(?:从|沿着|顺着|贴着).{1,16}(?:边|旁|间|口|缝|侧).{0,4}(?:通过|通行|过|走过去)(?:吗|么|不|呢)|"
    r"(?:够不够|够).{0,10}(?:我|人|身体|侧身|转身).{0,6}(?:通过|通行|过去|过|走)(?:吗|么|不|$)|"
    r"现在安全吗|前面通不通|前方通不通"
)
SPATIAL_QUERY = re.compile(
    r"多远|多少(?:米|厘米)|几米|^(?:距离|测距)(?:呢|吗|多少)?$|"
    r"(?:前方|前面|目标|障碍|这个|那个|离我).{0,8}距离|距离.{0,4}(?:多少|多大)|"
    r"怎么绕|如何绕|怎样绕|往哪(?:边|里)?走|该走哪|"
    r"能否(?:走|通行|通过)|(?:通道|道路|路面|前方|前面|这里).{0,8}(?:安全|宽度|够宽|能过)|"
    r"(?:宽度|高度).{0,8}(?:通过|通行|够不够)|"
    r"(?:这|那|前面|前方).{0,8}(?:路|门|通道).{0,8}(?:怎么走|过得去|走得通)|"
    r"(?:我|现在|这里|前面|前方).{0,8}(?:左转|右转|直走|往前走|绕过去)|"
    r"^(?:左转|右转|直走|往前走)(?:吗|还是|会|能|$)|往前走.{0,8}(?:撞|危险)"
)

AMBIGUOUS_STOP = re.compile(r"^(?:停一下|停下来|暂停|先停|停停|等等|等一下)$")

DATE_WORDS = ("日期", "保质期", "过期", "批号", "到期", "有效期")
TEXT_QUERY = re.compile(
    r"什么字|写的什么|写了什么|写的啥|上面的字|标签上|文字|"
    r"(?:^|帮我|请|请你|给我|你能)(?:读|念)(?:一下|一读|一念|出|这|那|上面|标签|包装|路牌|字|$)"
)
DETAIL_WORDS = ("详细", "仔细说", "具体说", "多说说", "多讲", "描述一下")
CHAT_WORDS = ("聊", "烦", "难受", "难过", "开心", "累", "陪我", "新闻", "天气", "心情", "无聊", "孤单")
CHAT_TOPIC = re.compile(
    r"情绪|感受|评价|意见|观点|想法|看法|建议|办法|方案|做法|方式|心理|压力(?!锅|表|计|罐|阀)|"
    r"烦恼|担心|在意|介意|焦虑|心态|同事|人际|关系|工作中|学习上|读书"
)
DISCOURSE_REF = re.compile(
    r"(?:刚才|刚刚|之前|前面|上次)(?:我|你|我们)?(?:说|讲|提|回答|解释)"
)
VISUAL_SOURCE = re.compile(r"眼前|手里|画面|图[里中上]|图片|照片|镜头|这张图|那张图|标签上|包装上|价签")
VISUAL_LOCATION = re.compile(r"前面|前方|周围|附近|这边|这里")
SCENE_QUERY = re.compile(
    r"(?:眼前|前面|前方|周围|附近|这边|这里|手里|画面(?:里|中)?)(?:现在)?(?:有什么|什么情况|是什么)|"
    r"(?:看|看看|描述)(?:一下|下)?(?:眼前|前面|前方|周围|画面|这里)|^(?:有什么|看看|什么情况)$"
)
OBJECT_REF = re.compile(r"(?:这|那)(?:(?:几|两|三|四|五|[1-9])?(?:个|件|只|袋|瓶|款|样|盒|张|杯|包|本|块|根|片)|两|些|是(?=什么|哪))|它|(?:左|右)(?:边|侧)")
VISUAL_ATTRIBUTE = re.compile(r"颜色|长什么样|什么形状|什么材质|是什么东西|几只|几件|挡住|有没有破损")
FOLLOW_MARKERS = re.compile(r"为什么|为何|为啥|怎么|多少|哪个|哪边|是不是|然后|继续|接着|展开|这|那|它|他|她|刚才|刚刚|吗|呢|还|再")
PAIR_REF = re.compile(r"这俩|那俩|(?:两|二|2|俩)(?:个|件|瓶|袋|包|盒|样|张|款|只|杯|罐|条|种|份)|左右|左边.{0,12}(?:和|与|比)右边|右边.{0,12}(?:和|与|比)左边")
COMPARE_REQUEST = re.compile(r"^(?:请|请你|帮我|给我|想|要|能不能|可不可以)*(?:比较|对比)(?:一下)?")
COMPARE_RELATION = re.compile(r"区别|不同|差别|一样吗|(?:哪个|哪款|哪种|哪边|哪一个|哪瓶|哪包|谁)(?:更|好|差|贵|便宜|划算)")
PRICE_QUERY = re.compile(
    r"多少钱|(?:多少|几)(?:块钱|元(?:钱)?|块(?=吗|呢|么|$))|什么价|卖多少|售价|标价|价签|价钱|价格|单价"
)
PRICE_COMPARE = re.compile(r"更(?:贵|便宜|划算)|(?:哪个|哪款|哪种|哪包|哪瓶|哪边|谁)(?:贵|便宜|划算)|哪个价格|哪款价格")
COMPARE_ADJECTIVE = re.compile(r"^(?:有点|很|挺|更|不)?(?:烦|累|开心(?!果)|难受|难过|孤单|无聊)")
NON_VISUAL_PRICE = re.compile(r"汇率|利率|房租|工资|学费|租金|月费|预算|机票|市场行情|物价")

TOILET_WORDS = ("卫生间", "洗手间", "厕所", "WC", "wc", "toilet", "Toilet")
EXIT_WORDS = ("出口", "安全出口", "EXIT", "exit")

VISION_INTENTS = {"scene", "read_text", "read_date", "read_price", "find_object", "followup", "compare", "detail"}


@dataclass(frozen=True)
class SessionContext:
    """上一轮话题与是否有可用画面独立传入，不把任意历史当成看图。"""

    topic: str = "unknown"  # chat / vision / unknown
    has_frame: bool = False


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
    context: Optional[SessionContext] = None,
    speech_active: bool = False,
    session_id: str = "",
    is_final: bool = True,
) -> Task:
    """解析最终话语或明确文本请求。新调用方传入可区分话题和图像的上下文。"""
    normalized = normalize_text(text)
    cmd = clean_command(normalized)
    if not cmd:
        return _task("silent", confidence=1.0, session_id=session_id, note="空文本")
    if ACK_ONLY.fullmatch(cmd):
        return _task("ack", confidence=1.0, session_id=session_id)

    if context is None:
        # 旧二元接口只用于兼容；正常业务入口传明确的会话信号。
        context = SessionContext("vision", True) if has_context else SessionContext()
    visual_source = bool(VISUAL_SOURCE.search(cmd))
    object_ref = bool(OBJECT_REF.search(cmd))
    scene_request = bool(SCENE_QUERY.search(cmd))
    conversational_topic = not visual_source and (
        bool(CHAT_TOPIC.search(cmd)) or (context.topic == "chat" and bool(DISCOURSE_REF.search(cmd)))
    )

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
    if definition_only and not visual_source:
        return _task("chat", {"question": cmd}, confidence=0.85, session_id=session_id)

    pair_ref = bool(PAIR_REF.search(cmd))
    price_query = bool(PRICE_QUERY.search(cmd))
    price_compare = bool(PRICE_COMPARE.search(cmd))
    compare_request = COMPARE_REQUEST.match(cmd)
    compare_relation = bool(COMPARE_RELATION.search(cmd))
    implicit_pair = (compare_relation and (pair_ref or object_ref or context.topic == "vision" or price_compare)) or (
        pair_ref and price_query
    )
    non_visual_price = price_query and not visual_source and NON_VISUAL_PRICE.search(cmd)
    if not conversational_topic and not non_visual_price and (compare_request or implicit_pair):
        item = extract_compare(cmd) if compare_request else cmd
        if item and (pair_ref or visual_source or not COMPARE_ADJECTIVE.match(item)):
            slots = {"item": item, "question": cmd}
            if price_query or price_compare:
                slots["comparison"] = "price"
            return _task("compare", slots, needs_image=True, confidence=0.85, session_id=session_id)
        if not item and context.topic != "chat":
            return _task("clarify", priority=2, confidence=0.5, session_id=session_id,
                         clarify="你要比较哪两样东西？")

    if price_query and not conversational_topic and not non_visual_price:
        prefix = PRICE_QUERY.split(cmd, maxsplit=1)[0]
        item = re.sub(r"^(?:请你|请|能不能|可以|帮我|给我|告诉我|看看|看一下|看下|读一下|念一下|看|读|念)+", "", prefix)
        return _task("read_price", {"question": cmd, "item": item},
                     needs_image=True, confidence=0.9, session_id=session_id)

    concept_comparison = compare_relation and not (pair_ref or object_ref or visual_source or context.topic == "vision")
    if conversational_topic or non_visual_price or concept_comparison or any(w in cmd for w in CHAT_WORDS):
        return _task("chat", {"question": cmd}, confidence=0.8, session_id=session_id)
    if any(w in cmd for w in DATE_WORDS):
        return _task("read_date", needs_image=True, confidence=0.95, session_id=session_id)
    if TEXT_QUERY.search(cmd):
        return _task("read_text", needs_image=True, confidence=0.9, session_id=session_id)

    if any(w in cmd for w in DETAIL_WORDS):
        if visual_source or scene_request or VISUAL_LOCATION.search(cmd) or context.topic == "vision":
            return _task("detail", needs_image=True, confidence=0.85, session_id=session_id)
        return _task("chat", {"question": cmd}, confidence=0.8, session_id=session_id)

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

    if scene_request:
        return _task("scene", needs_image=True, confidence=0.85, session_id=session_id)

    has_visual_attribute = bool(VISUAL_ATTRIBUTE.search(cmd))
    explicit_visual = visual_source or (object_ref and has_visual_attribute)
    if explicit_visual:
        return _task("followup", {"question": cmd}, needs_image=True, confidence=0.8, session_id=session_id)

    if FOLLOW_MARKERS.search(cmd):
        if context.topic == "vision":
            if context.has_frame:
                return _task("followup", {"question": cmd}, needs_image=True, confidence=0.7, session_id=session_id)
            return _task("clarify", priority=2, confidence=0.5, session_id=session_id,
                         clarify="前一张画面现在不可用。你要重新看一下，还是继续聊天？")
        if context.has_frame and context.topic == "unknown" and (object_ref or has_visual_attribute):
            return _task("followup", {"question": cmd}, needs_image=True, confidence=0.7, session_id=session_id)
        if context.topic == "chat":
            return _task("chat", {"question": cmd}, confidence=0.8, session_id=session_id)

    if context.topic == "chat" and len(cmd) >= 3:
        return _task("chat", {"question": cmd}, confidence=0.7, session_id=session_id)

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
