# coding=utf-8
"""字符永久唯一 ID 管理器（CharIdMap），单元粒度版本。

给文本里的"标注单元"分配永久身份证号（uid），永不重复、永不回收。
标注（发音 / 插入音素 / 词 / 语言）绑定 uid，而不是易变的编号。

单元粒度（用户确认）：
- 中文 / 日文：一个字 → 一个 uid
- 英文：一个词（空格分隔的连续英文块，含缩写撇号 '）→ 一个 uid
- 空格：词的边界，不分配 uid、不参与标注（UI 布局上仍留间距隔开方块）
- 标点符号：排除，不分配 uid、不标发音、不标音素（UI 布局上仍留间距隔开方块）

内部实现：
- uids 与 text 等长，每个"单元代表字符"填该单元的 uid，英文词内所有字母共享同一个词 uid；
  空格 / 标点位置填 -1（无身份，不可标注）
- 编辑同步 sync()：用逐字 diff，保留的字符沿用旧 uid；新增字符按单元归属继承或发新 uid；
  被删单元作废
- 查询 uid_at / pos_of / is_alive / range_to_pos（uid 区间 → 当前文本位置区间）

序列化：to_dict / from_dict（存 next + uids，重开工程能还原身份）。
"""
from __future__ import annotations
import difflib
from typing import Optional

# 英文词内允许的字符：字母数字 + 缩写撇号 + 连字符 + 全角撇号
_LATIN_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'’-")
# 视为标点排除的字符（中文/日文标点 + 常见西文标点；不含空格）
_PUNCT = set("，。！？；：、（）《》【】「」『』…—·.,;:!?()[]{}<>\"“”‘’\\/|@#$%^&*~+=_")


def is_word_char(ch: str) -> bool:
    """是否属于英文词内字符（词 = 连续英文块）。"""
    return ch in _LATIN_OK


def is_punct(ch: str) -> bool:
    """是否标点（应排除出标注单元）。"""
    return ch in _PUNCT


class CharIdMap:
    def __init__(self, text: str = ""):
        self._uids: list[int] = []      # 与 text 等长；单元代表字符=uid，词内共享，空格/标点=-1
        self._text: str = ""
        self._next: int = 0             # 下一个可分配的 uid（永不回收）
        self._pos_of_uid: dict[int, list[int]] = {}  # uid -> 该单元占据的位置列表
        self._uid_content: dict[int, str] = {}  # uid -> 该单元当前文本内容（供防错检测）
        self.last_changed: list[int] = []  # 最近一次 sync 中"内容发生变化"的 uid
        if text:
            self.reset(text)

    # ---------- 单元划分 ----------
    @staticmethod
    def _split_units(text: str):
        """把 text 切成 (start, end, kind) 列表。kind: 'latin'(英文词)/'cjk'(单字)/'space'/'punct'。"""
        units = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                units.append((i, i + 1, "space"))
                i += 1
            elif is_punct(ch):
                units.append((i, i + 1, "punct"))
                i += 1
            elif is_word_char(ch):
                j = i
                while j < n and is_word_char(text[j]):
                    j += 1
                units.append((i, j, "latin"))
                i = j
            else:
                # CJK 单字（汉字/假名/韩文等非标点非空格）
                units.append((i, i + 1, "cjk"))
                i += 1
        return units

    # ---------- 分配 ----------
    def reset(self, text: str):
        """从零为整段文本按单元分配 uid（用于新建/加载）。"""
        self._text = text
        uids = []
        self._uid_content = {}
        self.last_changed = []
        for (b, e, kind) in self._split_units(text):
            if kind == "latin":
                u = self._next
                self._next += 1
                for _ in range(e - b):
                    uids.append(u)
                self._uid_content[u] = text[b:e]
            elif kind == "cjk":
                u = self._next
                self._next += 1
                uids.append(u)
                self._uid_content[u] = text[b]
            else:  # space / punct
                for _ in range(e - b):
                    uids.append(-1)
        self._uids = uids
        self._rebuild_index()

    def _rebuild_index(self):
        m = {}
        for pos, u in enumerate(self._uids):
            if u >= 0:
                m.setdefault(u, []).append(pos)
        self._pos_of_uid = m

    # ---------- 编辑同步 ----------
    def sync(self, new_text: str):
        """文本从当前内容变为 new_text：保留沿用旧 uid，新增发新 uid，删除作废。"""
        if new_text == self._text:
            return
        old = self._text
        old_uids = self._uids
        new_uids = [-1] * len(new_text)
        # 第一遍：diff 的 equal 段直接继承（位置一一对应）
        sm = difflib.SequenceMatcher(None, old, new_text)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    new_uids[j1 + k] = old_uids[i1 + k]
        # 第二遍：按新文本单元划分，补齐新增位置的 uid
        for (b, e, kind) in self._split_units(new_text):
            if kind in ("space", "punct"):
                for k in range(b, e):
                    new_uids[k] = -1
                continue
            if kind == "latin":
                # 英文词：先取词内已继承的首个 uid
                u = None
                for k in range(b, e):
                    if new_uids[k] >= 0:
                        u = new_uids[k]
                        break
                if u is None:
                    # 全新增的词
                    u = self._next
                    self._next += 1
                else:
                    # 词的内容若已变化（增删字母）→ 这个空格作废，重发新 ID
                    old_content = self._uid_content.get(u)
                    if old_content is not None and old_content != new_text[b:e]:
                        u = self._next
                        self._next += 1
                for k in range(b, e):
                    new_uids[k] = u
            else:  # cjk
                if new_uids[b] < 0:
                    new_uids[b] = self._next
                    self._next += 1
        self._text = new_text
        self._uids = new_uids
        self._rebuild_index()
        # 重建内容快照（供下次 sync 判断英文词内容是否变化）
        self._uid_content = {}
        for u, poses in self._pos_of_uid.items():
            self._uid_content[u] = "".join(new_text[p] for p in sorted(poses))

    # ---------- 查询 ----------
    @property
    def text(self) -> str:
        return self._text

    def uid_at(self, pos: int) -> int:
        if 0 <= pos < len(self._uids):
            return self._uids[pos]
        return -1

    def pos_of(self, uid: int) -> int:
        """返回 uid 代表的首个位置；无效返回 -1。"""
        ps = self._pos_of_uid.get(uid)
        if ps:
            return ps[0]
        return -1

    def is_alive(self, uid: int) -> bool:
        return uid in self._pos_of_uid

    def unit_of_pos(self, pos: int) -> Optional[tuple]:
        """返回 pos 所在标注单元的首个位置与末尾位置 (unit_start, unit_end)。
        若 pos 落在空格/标点（无单元）返回 None。"""
        if not (0 <= pos < len(self._uids)):
            return None
        u = self._uids[pos]
        if u < 0:
            return None
        poses = self._pos_of_uid.get(u)
        if poses:
            return (poses[0], poses[-1] + 1)
        return None

    # ---------- 区间映射 ----------
    def range_to_pos(self, su: int, eu: Optional[int]) -> Optional[tuple]:
        """uid 区间 [su,eu) → 当前文本位置区间 [ps,pe)。返回 None 表示 su 已失效（标注应丢弃）。

        语义：包含所有"uid ∈ [su,eu) 且存活"单元所占位置的最小到最大范围；
        区间内新插入的单元（夹在中间）会被一并包含，因为物理跨度包含它。
        起始 uid 失效则返回 None。
        """
        ps = self._pos_of_uid.get(su)
        if not ps:
            return None
        lo = hi = ps[0]
        for u, poses in self._pos_of_uid.items():
            # 只在 [su, eu) 内的单元（eu=None 表示一直到末尾，但仍须 u>=su）
            if u < su:
                continue
            if eu is not None and u >= eu:
                continue
            for p in poses:
                if p < lo:
                    lo = p
                if p > hi:
                    hi = p
        return (lo, hi + 1)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {"next": self._next, "uids": self._uids}

    @classmethod
    def from_dict(cls, text: str, d: dict | None) -> "CharIdMap":
        cm = cls()
        cm._text = text
        if not d:
            cm.reset(text)
            return cm
        cm._next = int(d.get("next", 0))
        uids = d.get("uids", [])
        if len(uids) != len(text):
            cm.reset(text)
            return cm
        cm._uids = [int(u) for u in uids]
        cm._rebuild_index()
        return cm

    # ---------- PFML 标注专用映射 ----------
    def to_uid_pfml(self, pfml_pos: dict) -> dict:
        """把位置版 pfml（begin/end）转成 uid 版（su/eu），用于加载旧工程。

        位置版来自旧 .labproj：begin/end 是字符位置编号。
        - su = begin 位置所在单元的 uid
        - eu = end 位置之后第一个有 uid 单元的 uid；若到末尾则为 None
        插入音素（begin==end）：绑定"该位置后边的字符"，su=该位置单元 uid。
        标点/空格无 uid，跳过。
        """
        def eu_of(end: int):
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

        out = {"words": [], "overrides": [], "spans": []}
        for w in (pfml_pos or {}).get("words", []):
            try:
                b, e = int(w["begin"]), int(w["end"])
            except Exception:
                continue
            su = self.uid_at(b)
            if su < 0:
                continue
            out["words"].append({"su": su, "eu": eu_of(e)})
        for o in (pfml_pos or {}).get("overrides", []):
            try:
                b, e = int(o["begin"]), int(o["end"])
            except Exception:
                continue
            su = self.uid_at(b)
            if su < 0:
                continue
            item = dict(o)
            item.pop("begin", None)
            item.pop("end", None)
            item.pop("key", None)  # 旧 key 锚已废弃
            if b == e:
                # 插入音素：绑定后边字符 uid
                item["su"] = su
                item["eu"] = su
            else:
                item["su"] = su
                item["eu"] = eu_of(e)
            out["overrides"].append(item)
        for s in (pfml_pos or {}).get("spans", []):
            try:
                b, e = int(s["begin"]), int(s["end"])
            except Exception:
                continue
            su = self.uid_at(b)
            if su < 0:
                continue
            out["spans"].append({"su": su, "eu": eu_of(e),
                                "language": s.get("language", "zh")})
        return out

    def to_pos_pfml(self, pfml: dict) -> dict:
        """把 uid 版 pfml 映射成位置版 pfml（供 pfml_strip 渲染 / to_pfml 导出）。

        输入（uid 版）：
          words:    [{su, eu}]                     （uid 区间）
          overrides:[{su, eu, phonemes...}]         （发音）
          spans:    [{su, eu, language}]            （语言）
          插入音素：overrides 中 su==eu 的项（绑定后边字符 uid）
        输出（位置版，pfml_strip/to_pfml 可读）：
          begin/end 为当前位置。
        """
        out = {"words": [], "overrides": [], "spans": []}
        for w in (pfml or {}).get("words", []):
            r = self.range_to_pos(int(w["su"]), w.get("eu"))
            if r:
                out["words"].append({"begin": r[0], "end": r[1]})
        for o in (pfml or {}).get("overrides", []):
            su, eu = int(o["su"]), o.get("eu")
            item = dict(o)
            if su == eu:
                # 插入音素：su==eu 表示"绑定后边字符的 uid"，映射到该 uid 位置
                p = self.pos_of(su)
                if p < 0:
                    continue
                item["begin"] = item["end"] = p
            else:
                r = self.range_to_pos(su, eu)
                if not r:
                    continue
                item["begin"], item["end"] = r[0], r[1]
            item.pop("su", None)
            item.pop("eu", None)
            out["overrides"].append(item)
        for s in (pfml or {}).get("spans", []):
            r = self.range_to_pos(int(s["su"]), s.get("eu"))
            if r:
                out["spans"].append({"begin": r[0], "end": r[1],
                                     "language": s.get("language", "zh")})
        return out
