# coding=utf-8
"""PFML 数据模型与序列化。对齐 MaxLabel 的对象模型。

存储单位：字符索引（Python str 的位置）。序列化时转 UTF-8 字节偏移。

三层标注（都作用在 text 的字符区间上）：
- words:    词边界 [(b,e),...]          → <word>text</word>
- overrides: 发音覆盖 [(b,e,script,phonemes)] → <word phonemes="...">text</word>
             begin==end 表示插入音素        → <phoneme symbol="..."/>
- spans:    嵌套语言区间 [(b,e,lang)]    → <scope language="lang">...</scope>
主语言 = Mark.lang，整个 text 默认包一层 <scope>。
"""
from __future__ import annotations
import difflib
import xml.sax.saxutils as saxutils


def esc(s: str) -> str:
    return saxutils.escape(s, {'"': "&quot;"})


class PfmlData:
    """一条标记的 PFML 标注。主语言来自 Mark.lang。"""

    def __init__(self):
        self.words: list[dict] = []      # [{"begin":i,"end":j}]
        self.overrides: list[dict] = []  # [{"begin":i,"end":j,"script":"","phonemes":[...]}]
        self.spans: list[dict] = []      # [{"begin":i,"end":j,"language":"en"}]

    def to_dict(self):
        return {"words": self.words, "overrides": self.overrides, "spans": self.spans}

    @classmethod
    def from_dict(cls, d):
        p = cls()
        if not d:
            return p
        p.words = list(d.get("words", []))
        p.overrides = list(d.get("overrides", []))
        p.spans = list(d.get("spans", []))
        return p

    def is_empty(self) -> bool:
        return not self.words and not self.overrides and not self.spans


def _char_to_byte(text: str, char_pos: int) -> int:
    """字符位置 → UTF-8 字节位置。"""
    return text[:char_pos].encode("utf-8").__len__()


def remap_pfml(old_text: str, new_text: str, pfml: dict) -> dict:
    """文本编辑后，把 pfml 的 begin/end 编号按 diff 迁移到新文本位置。

    每个标注绑定的是字符的"实例位置"，靠 SequenceMatcher 算旧→新位置映射，
    不看字符内容 → 相同字符也不会错位。
    - 区间 [b,e)：取区间内存活字符的新位置范围
    - 插入音素（begin==end）：绑定其前方字符（若被删则向后找最近存活）
    - 文字被删（区间内无存活字符）→ 丢弃该标注
    """
    if not old_text or old_text == new_text:
        return {k: list(v) for k, v in (pfml or {}).items()}

    # 旧文本每个字符索引 → 新文本位置（-1 = 该字符被删除）
    n = len(old_text)
    pos_map = [-1] * n
    sm = difflib.SequenceMatcher(None, old_text, new_text)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pos_map[i1 + k] = j1 + k

    def alive(idx: int) -> bool:
        return 0 <= idx < n and pos_map[idx] >= 0

    out = {"words": [], "overrides": [], "spans": []}
    if not isinstance(pfml, dict):
        return out

    # 区间标注
    for kind in ("words", "overrides", "spans"):
        for it in pfml.get(kind, []):
            try:
                b, e = int(it["begin"]), int(it["end"])
            except Exception:
                continue
            if b == e:
                continue  # 插入音素下面单独处理
            b, e = max(0, b), min(n, e)
            live = [pos_map[k] for k in range(b, e) if alive(k)]
            if not live:
                continue  # 文字被删，丢弃
            item = dict(it)
            item["begin"], item["end"] = min(live), max(live) + 1
            out[kind].append(item)

    # 插入音素（begin==end）：绑定其后边那个字符（索引 pos 的字符）。
    # 后边字符被删（或 pos 落在末尾）才回退向前找最近存活；周围全删则丢弃。
    for it in pfml.get("overrides", []):
        try:
            pos = int(it["begin"])
        except Exception:
            continue
        if it.get("end") != it["begin"]:
            continue
        # 优先绑定 pos 索引的那个字符（插入音素插在它前面，宿主是它）
        if 0 <= pos < n and alive(pos):
            nb = pos_map[pos]
        else:
            # 后边字符被删 / pos 在末尾：向前找最近存活
            nb = -1
            for k in range(pos - 1, -1, -1):
                if alive(k):
                    nb = pos_map[k]
                    break
            # 前方也全删，再向后找
            if nb < 0:
                for k in range(pos, n):
                    if alive(k):
                        nb = pos_map[k]
                        break
        if nb < 0:
            continue  # 周围文字全被删，插入音素丢弃
        item = dict(it)
        item["begin"] = item["end"] = nb
        out["overrides"].append(item)

    return out




# 内置特殊标签（写死）：词汇表中是裸符号，带语言前缀反而会 OOV
BARE_PHONEME_SYMS = frozenset({"SP", "AP", "EP", "GS"})

# 内置音素中的带语言特例：面板显示 ja/cl，导出为 language="ja" 的 cl
# （DiffSinger 惯例写法）。格式：{显示名: (language, 实际symbol)}
BUILTIN_PHONEME_EXPORT = {"ja/cl": ("ja", "cl")}


def _lang_at(data: PfmlData, pos: int, main_lang: str) -> str:
    """位置 pos 的有效语言：副标签区覆盖 → 该区语言，否则主语言。"""
    for sp in data.spans:
        if sp["begin"] <= pos < sp["end"] or (pos == sp["end"] and sp["begin"] < sp["end"]):
            return sp["language"]
    return main_lang


def to_pfml(text: str, main_lang: str, data: PfmlData | dict | None) -> str:
    """把 text + 标注序列化成 PFML 片段。

    简化规则（覆盖常见情况）：
    - 无标注 → <scope language="main">text</scope>
    - 有 words/overrides：把 text 按词边界切开，词包 <word>，override 写 phonemes
    - 有 insert（begin==end 的 override）：在字符位置插 <phoneme symbol="..."/>
    - 有 spans：按嵌套语言切 <scope language=...>
    """
    if isinstance(data, dict):
        data = PfmlData.from_dict(data)
    if data is None:
        data = PfmlData()
    if data.is_empty():
        return f'<scope language="{esc(main_lang)}">{esc(text)}</scope>'

    # 收集切分点：所有 word/override 的 begin/end，以及 span 边界
    cuts = set([0, len(text)])
    for w in data.words:
        cuts.add(w["begin"]); cuts.add(w["end"])
    for o in data.overrides:
        cuts.add(o["begin"]); cuts.add(o["end"])
    for s in data.spans:
        cuts.add(s["begin"]); cuts.add(s["end"])
    cuts = sorted(c for c in cuts if 0 <= c <= len(text))

    # 建段：[ (start,end, lang_override, word?, phonemes?) ]
    segs = []
    for i in range(len(cuts) - 1):
        b, e = cuts[i], cuts[i + 1]
        if b >= e:
            continue
        segs.append({"b": b, "e": e, "lang": None, "word": False, "ph": None})

    # 标记哪些段是 word
    for w in data.words:
        for s in segs:
            if s["b"] >= w["begin"] and s["e"] <= w["end"]:
                s["word"] = True
    # 标记 override（发音覆盖，优先于 word）
    for o in data.overrides:
        if o["begin"] == o["end"]:
            continue  # 插入音素单独处理
        for s in segs:
            if s["b"] >= o["begin"] and s["e"] <= o["end"]:
                s["ph"] = o.get("phonemes", [])
                s["word"] = True
    # 标记嵌套语言
    for sp in data.spans:
        for s in segs:
            if s["b"] >= sp["begin"] and s["e"] <= sp["end"]:
                s["lang"] = sp["language"]

    # 序列化段；part_ranges 与 out_parts 平行记录每段的 (b,e)，供插入音素定位
    out_parts = []
    part_ranges = []
    for s in segs:
        seg_text = text[s["b"]:s["e"]]
        if not seg_text:
            continue
        lang = s["lang"] or main_lang
        # 纯空白段跳过（合并产生的段间空格），避免空 scope；
        # 但若空白属于 word/ph 段则保留（词内空白不能丢，否则词长对不上）
        if not seg_text.strip() and not s["word"] and not s["ph"]:
            continue
        if s["ph"]:
            ph_attr = " ".join(s["ph"])
            out_parts.append(
                f'<scope language="{esc(lang)}">'
                f'<word phonemes="{esc(ph_attr)}">{esc(seg_text)}</word>'
                f'</scope>')
        elif s["word"]:
            out_parts.append(
                f'<scope language="{esc(lang)}"><word>{esc(seg_text)}</word></scope>')
        else:
            out_parts.append(f'<scope language="{esc(lang)}">{esc(seg_text)}</scope>')
        part_ranges.append((s["b"], s["e"]))

    # 插入音素（begin==end 的 override）：独立隐式词格，按 side 精确落位。
    # before → 插开始于 pos 的段之前；after → 插结束于 pos 的段之后（含末尾）。
    # 语言规则：SP/AP/EP/GS 为词汇表裸符号不写语言；ja/cl 固定导出 ja/cl；
    # 其余自定义音素用 ins["lang"] 或位置语言。
    inserts = sorted([o for o in data.overrides if o["begin"] == o["end"]],
                     key=lambda o: o["begin"])
    for ins in inserts:
        pos = ins["begin"]
        sym = " ".join(ins.get("phonemes", [])).strip()
        if not sym:
            continue
        if sym in BARE_PHONEME_SYMS:
            tag = f'<phoneme symbol="{esc(sym)}" />'
        elif sym in BUILTIN_PHONEME_EXPORT:
            bl, bs = BUILTIN_PHONEME_EXPORT[sym]
            tag = f'<phoneme language="{bl}" symbol="{esc(bs)}" />'
        else:
            lang = ins.get("lang") or _lang_at(data, pos, main_lang)
            tag = f'<phoneme language="{esc(lang)}" symbol="{esc(sym)}" />'
        idx = None
        if ins.get("side") == "after":
            # 取最后一个结束于 pos 的位置（同位置多插入保持顺序）
            for i, (b, e) in enumerate(part_ranges):
                if e == pos:
                    idx = i + 1
        else:
            # 找开始于 pos 的真实段（b<e，排除已插入的占位），插其前
            for i, (b, e) in enumerate(part_ranges):
                if b == pos and e > b:
                    idx = i
                    break
            if idx is None:
                # 退化：位置在段中间或段落被跳过，找结束于 pos 的段插其后
                for i, (b, e) in enumerate(part_ranges):
                    if e == pos:
                        idx = i + 1
        if idx is None:
            idx = len(out_parts)
        out_parts.insert(idx, tag)
        part_ranges.insert(idx, (pos, pos))  # 占位，保持后续插入定位

    return "".join(out_parts)
