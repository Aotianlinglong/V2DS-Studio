# coding=utf-8
"""slot 引擎（新构筑，方案 B）：全局 uid 的"格子"模型。

程序意义上的"格子" = 唯一携带 uid 的标注单元：
- zh / yue：一个字符 = 一格
- en：一个空格分隔的词 = 一格
- ja：分词器(MeCab + 完整 UniDic，与 tifa.cpp 一致)自动分组 = 一格（UI 逐字只是显示，无独立 uid）

方案 B（分词器=工具，内容实例锚定）：
- 编辑时用 diff 对齐，内容未变的格子沿用旧 uid；只有内容/边界变了才重发新 uid
- 日语整段重新分词只由"日语分词"按钮显式触发（auto_group），编辑本身不整段重分

手动分组（日语专用）：
- merge(b, e)：把 [b,e) 内相邻格子合并成一格 → 新格发新 uid
- split(pos)：把一个格子从 pos 拆成两格 → 各自发新 uid
合并发音的清空是 Mark 层的操作（overrides），本模块只负责格子/uid 结构。
"""
from __future__ import annotations
import difflib

# 英文词内字符
_LATIN_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'’-")
# 排除的标点（不含空格）
_PUNCT = set("，。！？；：、（）《》【】「」『』…—·.,;:!?()[]{}<>\"“”‘’\\/|@#$%^&*~+=_")


def is_word_char(ch: str) -> bool:
    return ch in _LATIN_OK


def is_punct(ch: str) -> bool:
    return ch in _PUNCT


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7AF or 0x3400 <= o <= 0x4DBF)


class UidAllocator:
    """全局 uid 分配器：整个工程共享一个计数器，永不回收、永不重复。"""

    def __init__(self, next_uid: int = 0):
        self._next = next_uid

    def alloc(self) -> int:
        u = self._next
        self._next += 1
        return u

    @property
    def next_uid(self) -> int:
        return self._next

    def to_dict(self) -> dict:
        return {"next": self._next}

    @classmethod
    def from_dict(cls, d: dict | None) -> "UidAllocator":
        if not d:
            return cls()
        return cls(int(d.get("next", 0)))


# ---- 全局分配器单例：整个工程共享，永不回收、永不重复 ----
_ALLOC = UidAllocator()


def get_alloc() -> UidAllocator:
    return _ALLOC


def set_alloc_next(n: int):
    """加载工程时把全局分配器推进到 n（保证后续分配不撞旧 uid）。"""
    if n > _ALLOC._next:
        _ALLOC._next = n


def _kind_of(ch: str) -> str:
    """非空格、非标点的单字符标注单元（zh/yue/en 的 cjk/latin 之外的兜底）。"""
    if ch.isspace():
        return "space"
    if is_punct(ch):
        return "punct"
    return "slot"


def split_units(text: str, lang: str):
    """按语言切分成 (start, end, kind) 列表。kind: 'slot'/'space'/'punct'。

    zh/yue/en 共用启发式：连续英文块=一词；CJK 单字=一格；空格/标点无格。
    ja 用 pyopenjtalk 分词：每个分词组=一格；分词器未覆盖的字符按单字兜底。
    """
    units = []
    i, n = 0, len(text)
    if lang == "ja":
        return _ja_split(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            units.append((i, i + 1, "space")); i += 1
        elif is_punct(ch):
            units.append((i, i + 1, "punct")); i += 1
        elif is_word_char(ch):
            j = i
            while j < n and is_word_char(text[j]):
                j += 1
            units.append((i, j, "slot")); i = j
        else:
            units.append((i, i + 1, "slot")); i += 1
    return units


def _ja_split(text: str):
    """日语分词：fugashi(MeCab + 完整 UniDic，与 tifa.cpp 一致) 自动分组为一格；
    失败或未覆盖字符逐字兜底。"""
    from g2p import ja_word_seg
    groups = []
    try:
        for m in ja_word_seg(text):
            b, e = m["begin"], m["end"]
            if 0 <= b < e <= len(text):
                groups.append((b, e))
    except Exception:
        groups = []
    units = []
    covered = [False] * len(text)
    for (b, e) in groups:
        for k in range(b, min(e, len(text))):
            covered[k] = True
    i, n = 0, len(text)
    for (b, e) in groups:
        while i < b:                       # 分词器未覆盖的前导/间隔字符
            units.append((i, i + 1, _kind_of(text[i]))); i += 1
        gtext = text[b:e]
        # 纯标点/空格的分词组 → 分隔符（不占格），与空格同级
        if gtext and all(is_punct(c) or c.isspace() for c in gtext):
            kind = "space" if gtext.isspace() else "punct"
        else:
            kind = "slot"
        units.append((b, e, kind)); i = e
    while i < n:                           # 尾部未覆盖
        units.append((i, i + 1, _kind_of(text[i]))); i += 1
    return units


class SlotMap:
    """文本 ↔ 格子(uid) ↔ 标记 的底层引擎。UID 全局唯一（共享 UidAllocator）。"""

    def __init__(self, allocator: UidAllocator, text: str = "", lang: str = "zh"):
        self._alloc = allocator
        self._lang = lang
        self._text = ""
        self._uids: list[int] = []          # 与 text 等长；格子内字符=该格 uid，空格/标点=-1
        self._pos_of_uid: dict[int, list[int]] = {}
        self._uid_content: dict[int, str] = {}
        self.last_changed: list[int] = []   # 最近一次 sync 中内容变化的 uid
        if text:
            self.reset(text, lang)

    # ---------- 切分与分配 ----------
    def reset(self, text: str, lang: str | None = None):
        if lang:
            self._lang = lang
        self._text = text
        uids = []
        content = {}
        for (b, e, kind) in split_units(text, self._lang):
            if kind == "slot":
                u = self._alloc.alloc()
                for k in range(b, e):
                    uids.append(u)
                content[u] = text[b:e]
            else:
                for k in range(b, e):
                    uids.append(-1)
        self._uids = uids
        self._uid_content = content
        self.last_changed = []
        self._rebuild()

    def _rebuild(self):
        m = {}
        for pos, u in enumerate(self._uids):
            if u >= 0:
                m.setdefault(u, []).append(pos)
        self._pos_of_uid = m

    # ---------- 编辑同步（方案 B：内容实例锚定） ----------
    def sync(self, new_text: str, lang: str | None = None):
        if new_text == self._text:
            return
        if lang:
            self._lang = lang
        if self._lang == "ja":
            self._sync_ja(new_text)
        else:
            self._sync_heuristic(new_text)

    def _sync_heuristic(self, new_text: str):
        """zh/yue/en：diff 对齐，保留沿用旧 uid，新增发新 uid，英文词内容变→重发。"""
        old, old_uids = self._text, self._uids
        old_content = dict(self._uid_content)
        new_uids = [-1] * len(new_text)
        sm = difflib.SequenceMatcher(None, old, new_text)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    new_uids[j1 + k] = old_uids[i1 + k]
        for (b, e, kind) in split_units(new_text, self._lang):
            if kind != "slot":
                for k in range(b, e):
                    new_uids[k] = -1
                continue
            # 连续词（en）或单字（zh/yue）：取已继承的首个 uid
            u = None
            for k in range(b, e):
                if new_uids[k] >= 0:
                    u = new_uids[k]; break
            if u is None:
                u = self._alloc.alloc()
            elif e - b > 1:   # 词内容变（增删字母）→ 重发新 ID
                old_content2 = self._uid_content.get(u)
                if old_content2 is not None and old_content2 != new_text[b:e]:
                    u = self._alloc.alloc()
            for k in range(b, e):
                new_uids[k] = u
        self._text = new_text
        self._uids = new_uids
        self._rebuild()
        self._refresh_content(new_text)
        self._mark_changed(old_content)

    def _sync_ja(self, new_text: str):
        """ja：分词器给出候选组，但用 diff 锚定内容——未变的组沿用旧 uid，变的才重发。

        对每个候选组：若其全部字符都是 diff"equal"继承过来的，且映射回旧文本恰是
        同一个旧组（文本相同）→ 沿用该组 uid；否则（含新增/改动/边界吞并）→ 发新 uid。
        """
        old, old_uids = self._text, self._uids
        old_content = dict(self._uid_content)
        n = len(old)
        new_uids = [-1] * len(new_text)
        sm = difflib.SequenceMatcher(None, old, new_text)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    new_uids[j1 + k] = old_uids[i1 + k]
        # 对每个 ja 候选组：全部继承且文本与旧组一致 → 沿用，否则重发
        groups = [t for t in split_units(new_text, "ja") if t[2] == "slot"]
        for (b, e, _) in groups:
            old_ids = set()
            ok = True
            for p in range(b, e):
                if not (0 <= p < n and new_uids[p] >= 0):
                    ok = False
                    break
                old_ids.add(new_uids[p])
            if ok and len(old_ids) == 1:
                u = old_ids.pop()
                if self._uid_content.get(u) == new_text[b:e]:
                    continue  # 沿用 u
            # 重发
            u = self._alloc.alloc()
            for k in range(b, e):
                new_uids[k] = u
        self._text = new_text
        self._uids = new_uids
        self._rebuild()
        self._refresh_content(new_text)
        self._mark_changed(old_content)

    def _refresh_content(self, text: str):
        self._uid_content = {}
        for u, poses in self._pos_of_uid.items():
            self._uid_content[u] = "".join(text[p] for p in sorted(poses))

    def _mark_changed(self, old_content: dict):
        """只标记"内容变了"的 uid（含新增 uid）；位置移动但内容未变不标记。"""
        changed = []
        for u, poses in self._pos_of_uid.items():
            new_c = "".join(self._text[p] for p in sorted(poses))
            if old_content.get(u) != new_c:
                changed.append(u)
        self.last_changed = sorted(changed)

    # ---------- 日语手动分组 ----------
    def auto_group(self):
        """强制整段重新分词（"日语分词"按钮）：按当前文本全量重分，全部发新 uid。"""
        self._uids = [-1] * len(self._text)
        content = {}
        for (b, e, kind) in split_units(self._text, "ja"):
            if kind == "slot":
                u = self._alloc.alloc()
                for k in range(b, e):
                    self._uids[k] = u
                content[u] = self._text[b:e]
            else:
                for k in range(b, e):
                    self._uids[k] = -1
        self._uid_content = content
        self.last_changed = list(content.keys())
        self._rebuild()

    def merge(self, b: int, e: int) -> int | None:
        """把 [b,e) 内相邻格子合并成一格，返回新格 uid；范围无效返回 None。"""
        if b >= e:
            return None
        self._uids = list(self._uids)
        # 只合并 [b,e) 内非空格/标点的格子
        old_ids = {self._uids[k] for k in range(b, e) if self._uids[k] >= 0}
        if not old_ids:
            return None
        u = self._alloc.alloc()
        for k in range(b, e):
            if self._uids[k] >= 0:
                self._uids[k] = u
        self._rebuild()
        self._refresh_content(self._text)
        self.last_changed = [u]
        return u

    def split(self, pos: int) -> tuple[int, int] | None:
        """把 pos 处的一格拆成两格，返回 (左格 uid, 右格 uid)；pos 不在格内返回 None。"""
        if not (0 < pos < len(self._uids)):
            return None
        u = self._uids[pos]
        if u < 0 or self._uids[pos - 1] != u:
            return None
        # 找该格完整范围 [sb, se)
        sb = pos - 1
        while sb >= 0 and self._uids[sb] == u:
            sb -= 1
        sb += 1
        se = pos
        while se < len(self._uids) and self._uids[se] == u:
            se += 1
        self._uids = list(self._uids)
        ul = self._alloc.alloc()
        ur = self._alloc.alloc()
        for k in range(sb, pos):
            self._uids[k] = ul
        for k in range(pos, se):
            self._uids[k] = ur
        self._rebuild()
        self._refresh_content(self._text)
        self.last_changed = [ul, ur]
        return (ul, ur)

    def ungroup(self, pos: int) -> bool:
        """把 pos 所在的格子拆成逐字格（取消分词）。返回 True 若确实发生拆分。"""
        if not (0 <= pos < len(self._uids)):
            return False
        u = self._uids[pos]
        if u < 0:
            return False
        sb = pos
        while sb >= 0 and self._uids[sb] == u:
            sb -= 1
        sb += 1
        se = pos
        while se < len(self._uids) and self._uids[se] == u:
            se += 1
        if se - sb <= 1:
            return False  # 单字格无需拆
        self._uids = list(self._uids)
        new_uids = []
        for k in range(sb, se):
            nu = self._alloc.alloc()
            self._uids[k] = nu
            new_uids.append(nu)
        self._rebuild()
        self._refresh_content(self._text)
        self.last_changed = new_uids
        return True

    # ---------- 查询 ----------
    @property
    def text(self) -> str: return self._text
    @property
    def lang(self) -> str: return self._lang
    @property
    def uids(self) -> list[int]: return self._uids

    def uid_at(self, pos: int) -> int:
        if 0 <= pos < len(self._uids):
            return self._uids[pos]
        return -1

    def is_alive(self, uid: int) -> bool:
        return uid in self._pos_of_uid

    def pos_of(self, uid: int) -> int:
        ps = self._pos_of_uid.get(uid)
        return ps[0] if ps else -1

    def unit_of_pos(self, pos: int):
        if not (0 <= pos < len(self._uids)):
            return None
        u = self._uids[pos]
        if u < 0:
            return (pos, pos + 1)
        return (min(self._pos_of_uid[u]), max(self._pos_of_uid[u]) + 1)

    def slot_spans(self) -> list[tuple[int, int]]:
        """按位置顺序返回每个格子的跨度 [(b,e),...]（供 UI 画格子边界/分组）。"""
        spans = [(min(p), max(p) + 1) for p in self._pos_of_uid.values()]
        spans.sort()
        return spans

    def range_to_pos(self, su: int, eu=None):
        """uid 区间 [su,eu) → 位置区间 [lo, hi+1)，覆盖区间内所有存活格子的最小~最大位置。

        空格/标点无 uid，不会被包含 → 发音/词跨度不会被标点拉长或位移。
        eu=None 表示一直到末尾，但只到最后一个存活格子（不含尾部空格/标点）。
        """
        if su not in self._pos_of_uid:
            return None
        su_poses = self._pos_of_uid[su]
        lo, hi = min(su_poses), max(su_poses)
        if eu is not None:
            if eu not in self._pos_of_uid:
                return None
            target = min(self._pos_of_uid[eu])
            for poses in self._pos_of_uid.values():
                p0, p1 = min(poses), max(poses)
                # 只包含与 [lo, target) 有位置交集的格子
                if p0 < target and p1 >= lo:
                    if p0 < lo:
                        lo = p0
                    if p1 > hi:
                        hi = p1
        else:
            for poses in self._pos_of_uid.values():
                p0, p1 = min(poses), max(poses)
                if p0 >= lo or p1 >= lo:
                    if p0 < lo:
                        lo = p0
                    if p1 > hi:
                        hi = p1
        return (lo, hi + 1)

    # ---------- 复制 ----------
    def clone(self) -> "SlotMap":
        """深拷贝格子结构（uid/内容），但共享全局分配器（不重新发号）。"""
        sm = SlotMap(self._alloc, text="", lang=self._lang)
        sm._text = self._text
        sm._uids = list(self._uids)
        sm._rebuild()
        sm._refresh_content(self._text)
        return sm

    # ---------- PFML 标注映射 ----------
    def _eu_of(self, end):
        """位置 end 之后第一个有 uid 的格子 uid；到末尾返回 None。"""
        if end is None:
            return None
        i = end
        n = len(self._uids)
        while i < n:
            u = self._uids[i]
            if u >= 0:
                return u
            i += 1
        return None

    def _last_alive_uid(self) -> int:
        """最后一个存活格子的 uid；无格子返回 -1。"""
        for k in range(len(self._uids) - 1, -1, -1):
            if self._uids[k] >= 0:
                return self._uids[k]
        return -1

    def to_uid_pfml(self, pfml_pos: dict) -> dict:
        """位置版 pfml（begin/end）→ uid 版（su/eu），用于加载旧工程/合并后重锚。

        插入音素（begin==end）锚定规则：
        - 位置版带 side="after" 提示 → 锚到 begin 之前最后一个存活格，side="after"
        - 位置落在某格子上 → su=eu=该格 uid，side=before（缺省，不写 side 字段）
        - 位置落在末尾/空格/标点 → su=eu=前一个存活格 uid，side="after"（显式写出）
        """
        out = {"words": [], "overrides": [], "spans": []}
        for w in (pfml_pos or {}).get("words", []):
            try:
                b, e = int(w["begin"]), int(w["end"])
            except Exception:
                continue
            su = self.uid_at(b)
            if su < 0:
                continue
            out["words"].append({"su": su, "eu": self._eu_of(e)})
        for o in (pfml_pos or {}).get("overrides", []):
            try:
                b, e = int(o["begin"]), int(o["end"])
            except Exception:
                continue
            item = dict(o)
            item.pop("begin", None)
            item.pop("end", None)
            item.pop("key", None)
            side_hint = item.pop("side", None)
            if b == e:
                if side_hint == "after":
                    su = -1
                    for k in range(min(b, len(self._uids)) - 1, -1, -1):
                        if self._uids[k] >= 0:
                            su = self._uids[k]
                            break
                    if su < 0:
                        continue
                    item["su"] = item["eu"] = su
                    item["side"] = "after"
                else:
                    su = self.uid_at(b)
                    if su >= 0:
                        item["su"] = item["eu"] = su
                    else:
                        su = self._last_alive_uid()
                        if su < 0:
                            continue
                        item["su"] = item["eu"] = su
                        item["side"] = "after"
            else:
                su = self.uid_at(b)
                if su < 0:
                    continue
                item["su"] = su
                item["eu"] = self._eu_of(e)
            out["overrides"].append(item)
        for s in (pfml_pos or {}).get("spans", []):
            try:
                b, e = int(s["begin"]), int(s["end"])
            except Exception:
                continue
            su = self.uid_at(b)
            if su < 0:
                continue
            out["spans"].append({"su": su, "eu": self._eu_of(e),
                                 "language": s.get("language", "zh")})
        return out

    def to_pos_pfml(self, pfml: dict) -> dict:
        """uid 版 pfml（su/eu）→ 位置版（begin/end），供渲染/导出。"""
        out = {"words": [], "overrides": [], "spans": []}
        for w in (pfml or {}).get("words", []):
            try:
                r = self.range_to_pos(int(w["su"]), w.get("eu"))
            except Exception:
                continue
            if r:
                out["words"].append({"begin": r[0], "end": r[1]})
        for o in (pfml or {}).get("overrides", []):
            try:
                su, eu = int(o["su"]), o.get("eu")
            except Exception:
                continue
            item = dict(o)
            if su == eu:
                # 插入音素/点标注：side=before → 该格起始位置；side=after → 该格结束位置
                ps = self._pos_of_uid.get(su)
                if not ps:
                    continue
                p = max(ps) + 1 if o.get("side") == "after" else min(ps)
                item["begin"] = item["end"] = p
            else:
                r = self.range_to_pos(su, eu)
                if not r:
                    continue
                item["begin"], item["end"] = r[0], r[1]
            item.pop("su", None)
            item.pop("eu", None)
            # side 保留在位置版，供 to_uid_pfml 重锚时识别后置插入
            out["overrides"].append(item)
        for s in (pfml or {}).get("spans", []):
            try:
                r = self.range_to_pos(int(s["su"]), s.get("eu"))
            except Exception:
                continue
            if r:
                out["spans"].append({"begin": r[0], "end": r[1],
                                     "language": s.get("language", "zh")})
        return out

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {"text": self._text, "lang": self._lang, "uids": self._uids}

    @classmethod
    def from_dict(cls, d: dict, allocator: UidAllocator | None = None) -> "SlotMap":
        text = str(d.get("text", ""))
        lang = str(d.get("lang", "zh") or "zh")
        alloc = allocator or get_alloc()
        sm = cls(alloc, text="", lang=lang)
        sm._text = text
        uids = d.get("uids", [])
        if len(uids) != len(text):
            sm.reset(text, lang)
            return sm
        sm._uids = [int(u) for u in uids]
        sm._rebuild()
        sm._refresh_content(text)
        # 把共享分配器推进到现存最大 uid 之上，避免后续分配撞车
        if sm._uids:
            top = max(sm._uids)
            if top >= alloc.next_uid:
                alloc._next = top + 1
        return sm
