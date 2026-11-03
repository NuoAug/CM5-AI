# -*- coding: utf-8 -*-
"""流式语义分段与朗读文本规范化。

目标：首段短而完整，后续段适当加长；不拆断否定、数字单位、日期、左右关系。
"""

from __future__ import annotations

import re

TTS_NORM_VERSION = "tts-norm-v1"
STRONG = "。！？；!?;\n"
CLAUSE = "，、：,:"
NEGATIONS = "不没无未别勿莫"
DIRECTIONS = "左右前后上下内外"
DIRECTION_TAILS = "边侧面方部"
MODAL_TAILS = ("可以", "能够", "不能", "无法", "不可", "能", "会", "要")
WALK_HEADS = "走通前过"

DATE_RE = re.compile(r"20\d{2}\s*[年\-/.]\s*\d{1,2}\s*[月\-/.]\s*\d{1,2}\s*[日号]?")
NUMBER_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:个月|月|年|天|小时|分钟|秒|米|厘米|毫米|公里|%|％|度|岁|元|块|号|点|时|分|kg|g|ml|L|毫安|伏)"
)
LATIN_RE = re.compile(r"[A-Za-z0-9_.+-]+")


def normalize_for_tts(text: str) -> str:
    """清理 Markdown/代码/链接，保留数字、单位、日期和事实。"""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"```[\s\S]*?```", "。", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"https?://\S+", "链接", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"^[ \t]*[#>*\-+]+[ \t]*", "", s, flags=re.M)
    s = re.sub(r"[\[\]{}<>]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "。", s)
    s = re.sub(r"。{2,}", "。", s)
    s = re.sub(r"。，", "。", s)
    s = s.strip(" ，、；：")
    if s and s[-1] not in "。！？；!?;":
        s += "。"
    return s


def _safe_boundary(text: str, pos: int, lo: int, hi: int) -> int:
    """把切点移到不会被拆断数字/日期/单位/否定/方向词的位置。"""
    pos = max(lo, min(pos, hi))
    # 日期或数字单位跨越切点：先尝试移到单位之后，移不过去就退到数字之前。
    for pattern in (DATE_RE, NUMBER_UNIT_RE):
        for match in pattern.finditer(text):
            if match.start() < pos < match.end():
                if match.end() <= hi + 2:
                    return match.end()
                return max(lo, match.start())
    # 否定词不能留在上一段末尾。
    back = pos
    while back > lo and text[back - 1] in NEGATIONS:
        back -= 1
    # 左右/前后/上下 不能和 边/侧/面/方 拆开。
    if back < len(text) and text[back] in DIRECTION_TAILS and text[back - 1] in DIRECTIONS:
        back -= 1
    # “可以走”“不能通行”这类通行结论不能被切成两段。
    for tail in MODAL_TAILS:
        if text[:back].endswith(tail) and back < len(text) and text[back] in WALK_HEADS:
            back = max(lo, back - len(tail))
            break
    # 英文/数字 token 不拆开。
    if back < len(text) and back > 0:
        m = LATIN_RE.match(text, back)
        if m and m.start() == back and m.end() > back:
            back = max(lo, back - 1)
        m2 = LATIN_RE.search(text, 0, back)
        if m2 and m2.end() == back:
            back = max(lo, m2.start())
    return max(lo, min(back, hi))


class SemanticSegmenter:
    def __init__(self, first_min: int = 12, first_max: int = 25, later_min: int = 25, later_max: int = 50):
        self.first_min = max(4, int(first_min))
        self.first_max = max(self.first_min, int(first_max))
        self.later_min = max(8, int(later_min))
        self.later_max = max(self.later_min, int(later_max))
        self.buffer = ""
        self.sequence = 0
        self._first = True

    def _limits(self) -> tuple[int, int]:
        return (self.first_min, self.first_max) if self._first else (self.later_min, self.later_max)

    def _emit_at(self, pos: int) -> str:
        pos = _safe_boundary(self.buffer, pos, *self._limits())
        if pos <= 0:
            return ""
        piece = self.buffer[:pos].strip()
        self.buffer = self.buffer[pos:].lstrip()
        if piece:
            self.sequence += 1
            self._first = False
        return piece

    def _try_emit(self, final: bool) -> list[str]:
        out = []
        lo, hi = self._limits()
        while self.buffer:
            # 强句末：达到最短长度就优先切；不足最短则等下一句，避免“好。”单独成段。
            strong = [i for i, ch in enumerate(self.buffer) if ch in STRONG]
            strong = [i + 1 for i in strong if i + 1 >= lo]
            if final and not strong:
                pass
            if strong:
                cut = min(strong[0], len(self.buffer))
                if cut <= hi or final:
                    piece = self._emit_at(cut)
                    if piece:
                        out.append(piece)
                    continue
            # 子句标点：在窗口内选最后一个，避免把一句话拆得太碎。
            clauses = [i + 1 for i, ch in enumerate(self.buffer[:hi]) if ch in CLAUSE and i + 1 >= lo]
            if clauses:
                piece = self._emit_at(clauses[-1])
                if piece:
                    out.append(piece)
                    continue
            # 超长仍无标点：在安全边界硬切。
            if len(self.buffer) > hi:
                piece = self._emit_at(hi)
                if piece:
                    out.append(piece)
                    continue
            if not final:
                break
            piece = self._emit_at(len(self.buffer))
            if piece:
                out.append(piece)
            break
        return out

    def feed(self, delta: str) -> list[str]:
        if delta:
            self.buffer += delta
        return self._try_emit(final=False)

    def flush(self) -> list[str]:
        return self._try_emit(final=True)


def segment_text(text: str, first_min: int = 12, first_max: int = 25, later_min: int = 25, later_max: int = 50) -> list[str]:
    seg = SemanticSegmenter(first_min, first_max, later_min, later_max)
    pieces = seg.feed(normalize_for_tts(text))
    pieces.extend(seg.flush())
    return [p for p in pieces if p]
