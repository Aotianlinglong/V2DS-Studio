# coding=utf-8
"""CharIdMap —— 文本 ↔ 空格(标注单元)/ID ↔ 标记 的底层映射引擎。

核心设计（与用户确认的规则完全一致）：
- 中文/日文/粤语：一个字符 = 一个空格 = 一个 ID。
- 英文：不打空格时，连续字母聚合成一个空格 = 一个 ID；打空格才发下一个空格/ID。
- 空格/标点：不分配 ID（无身份），但在 UI 里作为断句间距存在。
- 删一个字符/词 → 该 ID 作废，空格消失，其标记（发音/插入音素）自动消失。
- 改一个词的内容（love→lov）→ 该空格作废，重发新 ID，标记随旧 ID 消失。
- 插入新字符/词 → 新空格新 ID，原来的空格与标记原封不动。

这套引擎只在"文本变化时"维护 ID 表；标记（uid 版）不随文本现算，
而是天然挂在 ID 上，导出时才映射回字符串位置。
"""

import difflib

# 英文缩写内的撇号（如 don't）应视为词内字符
_WORD_PUNCT = set("'’`-")


def is_word_char(ch: str) -> bool:
    """英文词内字符：字母、数字、撇号、连字符（不含空格）。"""
    return (ch.isalnum() and not is_cjk(ch)) or ch in _WORD_PUNCT


def is_cjk(ch: str) -> bool:
    """是否为汉字/假名/谚文等 CJK 单字（英文词分隔的基础判断）。"""
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7AF or 0x3400 <= o <= 0x4DBF)


def is_punct(ch: str) -> bool:
    """标点符号（不含空格）。"""
    if ch.isspace():
        return False
    if is_cjk(ch) or ch.isalnum():
        return False
    if ch in _WORD_PUNCT:
        return False
    return True


class CharIdMap:
    """文本的字符永久 ID 引擎（永不回收，只增发）。"""

    def __init__(self, text: str = ""):
        self._uids: list[int] = []      # 与 text 等长；单元代表字符=uid，词内共享，空格/标点=-1
        self._text: str = ""
        self._next: int = 0             # 下一个可分配的 uid（永不回收）
        self._pos_of_uid: dict[int, list[int]] = {}  # uid -> 该单元占据的位置列表
        self._uid_content: dict[int, str] = {}  # uid -> 该单元当前文本内容（供英文词判断是否变化）
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
        """文本从当前内容变为 new_text。

        规则：
        - 内容未变的字符/词：沿用原 uid（位置可移动）。
        - 新增的字符/词：发新 uid。
        - 删除的字符/词：uid 作废（其标记随 uid 消失）。
        - 英文词内容一变（增删字母）→ 该词作废，重发新 ID（当作新空格）。
        """
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
                # 取词内已继承的首个 uid
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
                    # 词的内容若已变化 → 这个空格作废，重发新 ID
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
    def uids(self) -> list[int]:
        return self._uids

    @property
    def text(self) -> str:
        return self._text

    def uid_at(self, pos: int) -> int:
        """pos 位置所在的 uid；无身份（空格/标点）返回 -1。"""
        if 0 <= pos < len(self._uids):
            return self._uids[pos]
        return -1

    def unit_of_pos(self, pos: int):
        """pos 所在单元的位置区间 [b, e)；空格/标点返回 (pos, pos+1)。"""
        if not (0 <= pos < len(self._uids)):
            return None
        u = self._uids[pos]
        if u < 0:
            return (pos, pos + 1)
        return (min(self._pos_of_uid[u]), max(self._pos_of_uid[u]) + 1)

    def pos_of(self, uid: int) -> int:
        """uid 的首个位置；uid 作废返回 -1。"""
        if uid in self._pos_of_uid:
            return min(self._pos_of_uid[uid])
        return -1

    def range_to_pos(self, su: int, eu=None):
        """uid 区间 [su, eu) → 位置区间 [b, e)。eu 为 None 表示到文本末尾。
        起始 su 失效返回 None。"""
        if su not in self._pos_of_uid:
            return None
        b = min(self._pos_of_uid[su])
        if eu is None:
            return (b, len(self._text))
        if eu not in self._pos_of_uid:
            return None
        e = min(self._pos_of_uid[eu])
        return (b, e)

    # ---------- PFML 标注专用映射 ----------
    def to_uid_pfml(self, pfml_pos: dict) -> dict:
        """位置版 pfml（begin/end）→ uid 版（su/eu），用于加载旧工程。
        标点/空格无 uid，跳过。"""
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
            item.pop("key", None)
            if b == e:
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
        """uid 版 pfml（su/eu）→ 位置版（begin/end），供渲染/导出。"""
        out = {"words": [], "overrides": [], "spans": []}
        for w in (pfml or {}).get("words", []):
            r = self.range_to_pos(int(w["su"]), w.get("eu"))
            if r:
                out["words"].append({"begin": r[0], "end": r[1]})
        for o in (pfml or {}).get("overrides", []):
            su, eu = int(o["su"]), o.get("eu")
            item = dict(o)
            if su == eu:
                # 插入音素：su==eu 表示绑定"该位置单元 uid"，映射回该 uid 位置
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

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {"text": self._text, "uids": self._uids, "next": self._next,
                "content": self._uid_content}

    @classmethod
    def from_dict(cls, d: dict):
        cm = cls.__new__(cls)
        cm._text = d.get("text", "")
        cm._uids = [int(u) for u in d.get("uids", [])]
        cm._next = int(d.get("next", 0))
        cm._uid_content = {int(k): v for k, v in d.get("content", {}).items()}
        cm._rebuild_index()
        return cm
