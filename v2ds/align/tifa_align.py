# coding=utf-8
"""TIFA 对齐（ONNX 运行时）：切片 → 对齐 → 解析 TextGrid → 匹配 g2p 候选。

用途：LS 的「运行 TIFA」独立步骤。对每个标记：把 [start,end) 音频切片，
用 tifa_onnx_runner 的 TIFA ONNX 流水线对齐（输出 texts/phones 两层 TextGrid），
解析每个词/字区间对应的「对齐音素」（模型根据音频帧概率选出的读音），
再与我们的 g2pflow 候选比对，命中哪个候选就是音频判定最可能的读音。
"""
from __future__ import annotations
import re
from pathlib import Path

from v2ds.paths import ROOT_DIR


def align_available() -> bool:
    """TIFA ONNX 对齐引擎是否就位。"""
    try:
        from v2ds.align import tifa_onnx_runner as onnxr
        return onnxr.tifa_available()
    except Exception:
        return False


# 成功对齐的段数（供 UI 报告）
ALIGN_COUNT = 0


def backend_report() -> str:
    """返回本次会话的对齐后端摘要。"""
    return f"ONNX {ALIGN_COUNT} 段" if ALIGN_COUNT else "未跑任何对齐"


def slice_wav(wav_path, start, end, out_path) -> Path:
    """把 [start,end) 秒切片写成 wav（PCM16）。用 soundfile（捆绑 Python 自带）。"""
    import soundfile as sf
    data, sr = sf.read(str(wav_path), always_2d=False, dtype="float32")
    i0, i1 = int(round(start * sr)), int(round(end * sr))
    i0, i1 = max(0, i0), min(len(data), i1)
    if i1 <= i0:
        raise ValueError(f"切片为空 [{start:.3f},{end:.3f})")
    sf.write(str(out_path), data[i0:i1], sr, subtype="PCM_16")
    return out_path


def parse_textgrid(path) -> dict[str, list[tuple[float, float, str]]]:
    """解析 ooTextFile TextGrid：{tier名: [(xmin,xmax,text), ...]}。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tiers: dict[str, list] = {}
    tier_name, in_interval, iv = None, False, []
    for raw in lines:
        s = raw.strip()
        if s.startswith('name = "'):
            tier_name = s.split('name = "')[1].rsplit('"', 1)[0]
            tiers.setdefault(tier_name, [])
            in_interval = False
        elif s.startswith("intervals ["):
            in_interval, iv = True, []
        elif in_interval:
            if s.startswith("xmin = "):
                iv.append(float(s.split("= ")[1]))
            elif s.startswith("xmax = "):
                iv.append(float(s.split("= ")[1]))
            elif s.startswith("text = "):
                t = s.split("text = ", 1)[1].strip('"')
                iv.append(t)
            if len(iv) >= 3:
                tiers[tier_name].append((iv[0], iv[1], iv[2]))
                in_interval = False
    return tiers


def _overlap(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def words_with_phones(textgrid_path) -> list[dict]:
    """每个非空词/字区间 + 该区间内（时间重叠）的对齐音素。

    返回 [{label, t0, t1, phones:[...]}]
    """
    tiers = parse_textgrid(textgrid_path)
    texts = tiers.get("texts", [])
    phones = tiers.get("phones", [])
    out = []
    for (t0, t1, label) in texts:
        if not label.strip():
            continue
        ph = [p for (pt0, pt1, p) in phones
              if p.strip() and _overlap(t0, t1, pt0, pt1)]
        out.append({"label": label, "t0": t0, "t1": t1, "phones": ph})
    return out


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def span_regions(mark) -> list[tuple[int, int, str]]:
    """从 mark 的语言副标签(spans)切出 [(b,e,lang)]：裁剪到文本内、按 begin 排序、
    重叠区段截断，忽略与主语言相同的标签。"""
    try:
        spans = (mark.pos_pfml() or {}).get("spans", [])
    except Exception:
        spans = []
    n = len(mark.text)
    main = mark.lang or "zh"
    regs = []
    for s in spans:
        try:
            b, e = int(s["begin"]), int(s["end"])
        except Exception:
            continue
        lang = s.get("language") or ""
        b, e = max(0, b), min(n, e)
        if b < e and lang and lang != main:
            regs.append((b, e, lang))
    regs.sort()
    out = []
    last = 0
    for b, e, lang in regs:
        if b < last:
            if e <= last:
                continue
            b = last
        out.append((b, e, lang))
        last = e
    return out


def scoped_text(mark) -> str:
    """标记 → 完整 PFML 文本，作为 TIFA 推理输入（与导出文件夹同源）。

    统一走 pfml.to_pfml：语言副标签 → <scope>，分词 → <word>，读音覆盖 →
    <word phonemes>，插入音素 → 独立 <phoneme/>（AP/EP/GS 裸符号，自定义带语言）。
    tifa.cpp / PT 服务端均自动识别 PFML。
    """
    from v2ds.core.pfml import to_pfml
    return to_pfml(mark.text or "", mark.lang or "zh", mark.pos_pfml())


# 未标语言检测：只采用「窄而强」的字符集特征，避免误报
# 拉丁词：≥2 个连续字母，内部允许 ' ’ - （don't / co-operate）
_RE_LATIN_WORD = re.compile(r"[A-Za-z](?:['’\-]?[A-Za-z])+")
# 平假名/片假名：日语独有，中文不含
_RE_KANA = re.compile(r"[぀-ゟ゠-ヿ]+")
# 汉字（中日共享，仅在主语言为 en 时作为疑似中文信号）
_RE_HAN = re.compile(r"[一-鿿㐀-䶿]+")
# 无语义间隔符（空白/数字/标点），用于把相邻命中合并成一个提示片段
_RE_GAP_OK = re.compile(r"^[^A-Za-z一-鿿㐀-䶿぀-ゟ゠-ヿ]*$")


def _foreign_hits(seg: str, main_lang: str, off: int) -> list[tuple[int, int, str]]:
    """对一段主语言文本做窄特征检测，返回全局偏移的疑似外语命中。"""
    hits = []
    if main_lang in ("zh", "yue", "ja"):
        for mt in _RE_LATIN_WORD.finditer(seg):
            hits.append((off + mt.start(), off + mt.end(), "en"))
    if main_lang in ("zh", "yue"):
        for mt in _RE_KANA.finditer(seg):
            hits.append((off + mt.start(), off + mt.end(), "ja"))
    if main_lang == "en":
        for mt in _RE_KANA.finditer(seg):
            hits.append((off + mt.start(), off + mt.end(), "ja"))
        for mt in _RE_HAN.finditer(seg):
            hits.append((off + mt.start(), off + mt.end(), "zh"))
    return hits


def _coalesce_hits(hits: list[tuple[int, int, str]], text: str) -> list[tuple[int, int, str]]:
    """同语言相邻命中，间隔只含空白/数字/标点时合并为一个片段（不跨语言标签区）。"""
    out: list[tuple[int, int, str]] = []
    for b, e, lang in sorted(hits):
        if out and out[-1][2] == lang and _RE_GAP_OK.match(text[out[-1][1]:b] or ""):
            out[-1] = (out[-1][0], e, lang)
        else:
            out.append((b, e, lang))
    return out


def detect_unlabeled_languages(mark) -> list[tuple[int, int, str]]:
    """检测标记中「属于其他语言、但没有语言副标签覆盖」的文本片段。

    只检查 spans 之外的主语言区；命中条件用窄而强的字符特征：
    - 主语言 zh/yue：≥2 字母拉丁词→en，平/片假名→ja
    - 主语言 ja：拉丁词→en（日语文本本身含汉字，不判汉字）
    - 主语言 en：假名→ja，汉字→zh
    返回 [(begin, end, lang)]（片段已合并）。纯数字/标点不触发。
    """
    spans = span_regions(mark)
    text = mark.text
    main = mark.lang or "zh"
    zones = []
    cur = 0
    for b, e, _lang in spans:
        if cur < b:
            zones.append((cur, b))
        cur = e
    if cur < len(text):
        zones.append((cur, len(text)))
    hits = []
    for b0, b1 in zones:
        hits += _foreign_hits(text[b0:b1], main, b0)
    return _coalesce_hits(hits, text)


def tifa_units(mark) -> list[tuple[int, int, str]]:
    """发音单位 [(b,e,lang)]，与 TIFA 按嵌套 scope G2P 后的词粒度对齐。

    主语言区：沿用 slot 格子（含用户手动分词结果）；
    语言副标签区：slot 是按主语言切的（如 zh 逐字），与该语言的 G2P 粒度
    不一致（ja=MeCab 词组），故用该区语言重新切分。
    """
    from v2ds.core.slots import split_units
    text = mark.text
    main = mark.lang or "zh"
    spans = span_regions(mark)
    if not spans:
        return [(b, e, main) for (b, e) in mark.slots.slot_spans()]
    slot_spans = mark.slots.slot_spans()

    def main_units(b0, b1):
        return [(b, e, main) for (b, e) in slot_spans if b >= b0 and e <= b1]

    units = []
    cur = 0
    n = len(text)
    for (b, e, lang) in spans:
        if cur < b:
            units += main_units(cur, b)
        for (sb, se, kind) in split_units(text[b:e], lang):
            if kind == "slot":
                units.append((b + sb, b + se, lang))
        cur = e
    if cur < n:
        units += main_units(cur, n)
    units.sort(key=lambda t: t[0])
    return units


def _pair_units(mark, words, word_label, word_ok, make_item):
    """按序配对发音单位与 TIFA 词表：词面不符跳过（不中断），避免一处错位拖垮全句。"""
    units = tifa_units(mark)
    out = []
    wi = 0
    nw = len(words)
    for (b, e, lang) in units:
        label = mark.text[b:e]
        j = wi
        while j < nw and (word_label(words[j]) or "").lower() != label.lower():
            j += 1
        if j >= nw:
            continue  # 该单位在 TIFA 词表里找不到（分词不一致/OOV），跳过
        wi = j + 1
        if not word_ok(words[j]):
            continue
        item = make_item(words[j], b, e, label, lang)
        if item:
            out.append(item)
    return out


def _strip_lang_prefix(phones: list[str], lang: str) -> list[str]:
    """剥掉音素的语言前缀后比对候选。

    TIFA 输出 TextGrid 时，主语言音素裸写、非主语言（如 en/d、zh/zh）带
    "<lang>/" 前缀；而 g2p 候选 list_candidates 始终是裸音素。若不剥前缀，
    副语言区的多音词/多音字永远匹配不上候选 → 选音结果写不回（Bug 实测：
    "Don't" 对齐得 en/d en/ow en/n en/t，候选为 d ow n t，严格相等失败）。
    """
    prefixes = tuple(f"{l}/" for l in ("zh", "yue", "ja", "en"))
    if lang:
        prefixes = (f"{lang}/",) + prefixes
    out = []
    for p in phones:
        for pfx in prefixes:
            if p.startswith(pfx):
                p = p[len(pfx):]
                break
        out.append(p)
    return out


def chosen_readings(mark, words: list[dict]) -> list[dict]:
    """按发音单位顺序把标记与 tifa 对齐词配对，比对候选。

    用 tifa_units（主语言=slot 格子，语言副标签区=该语言切分）当发音单位，
    避免文本匹配的重复字塌缩 / 漂移。只处理候选 >= 2 的多音字/多音词，且对齐
    音素须命中某个候选才写（否则跳过，不误标）。返回 [{b, e, label, script, phonemes}]。
    """
    from v2ds.g2p import list_candidates

    def word_ok(w):
        return bool(w["phones"])

    def make_item(w, b, e, label, lang):
        cands = list_candidates(label, lang)
        if len(cands) < 2:
            return None  # 单候选不算多音字
        aligned = _strip_lang_prefix(w["phones"], lang)
        chosen = next((c for c in cands if list(c["phonemes"]) == aligned), None)
        if chosen is None:
            return None  # 对齐音素与候选对不上，跳过（不误标）
        return {"b": b, "e": e, "label": label,
                "script": chosen["script"], "phonemes": chosen["phonemes"]}

    return _pair_units(mark, words,
                       word_label=lambda w: w["label"],
                       word_ok=word_ok, make_item=make_item)


def _align(wav_path, text, lang, out_dir,
           extra_langs=None) -> tuple[bool, Path | None, str, dict, list]:
    """TIFA ONNX 对齐入口（tifa_onnx_runner.run_align_onnx + 计数）。"""
    global ALIGN_COUNT
    from v2ds.align import tifa_onnx_runner as onnxr
    ok, tg, err, extra = onnxr.run_align_onnx(wav_path, text, lang, out_dir,
                                              extra_langs=extra_langs)
    if ok:
        ALIGN_COUNT += 1
    return ok, tg, err, extra.get("metrics", {}), extra.get("score_words", [])


def compute_for_mark(mark, wav_path, out_dir) -> tuple[int, list[dict]]:
    """对一个标记算「音频判读音」候选 override 列表。

    先把 [mark.start, mark.end) 音频切片——每个标记只对齐自己那一段。若不切片，
    单行歌词会被拿去对齐整首歌，音素被大幅拉伸对不上候选 → 无写入且慢。
    返回 (ok, [chosen...])；ok=0 成功、1 无tifa/失败、2 无多音字。
    """
    text = mark.text.strip()
    if not text:
        return 1, []
    lang = mark.lang or "zh"
    # 含语言副标签：传嵌套 scope 的 PFML 文本
    text = scoped_text(mark).strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        clip = out_dir / "mark.wav"
        slice_wav(wav_path, float(mark.start), float(mark.end), clip)
    except Exception:
        return 1, []
    ok, tg, err, _metrics, _sw = _align(clip, text, lang, out_dir)
    if not ok:
        return 1, []
    words = words_with_phones(tg)
    chosen = chosen_readings(mark, words)
    return (0, chosen) if chosen else (2, [])


# ---------------- 呼吸检测（FBL ONNX + TIFA 词边界） ----------------

def breath_for_mark(mark, wav_path, out_dir) -> tuple[int, list[dict]]:
    """对一个标记做呼吸检测（FBL ONNX 版，弃用 cpp breathe）。

    流程（与旧 FBL Python 流水线等价）：
      1. 切片 → 2. align 出第一遍 TG（texts 层词边界）
      → 3. FBL ONNX 直接推理切片 → AP 概率段
      → 4. 只保留中心落在词间空隙的段（等价旧流程"SP 中找 AP"）→ 映射到字符位置。
      尾部段（中心在最后词结束之后）→ EP。

    返回 (ok, [{b, sym, side}...])；ok=0 成功、1 无模型/失败、2 无检出。
    """
    from v2ds.align import tifa_onnx_runner as onnxr
    if not onnxr.fbl_available():
        return 1, []
    text = mark.text.strip()
    if not text:
        return 1, []
    lang = mark.lang or "zh"
    text = scoped_text(mark).strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        clip = out_dir / "breath_mark.wav"
        slice_wav(wav_path, float(mark.start), float(mark.end), clip)
    except Exception:
        return 1, []
    pass1 = out_dir / "pass1"
    ok, tg, _err, _m, _sw = _align(clip, text, lang, pass1)
    if not ok:
        return 1, []
    try:
        segments = onnxr.fbl_infer(clip)
    except Exception:
        return 1, []
    if not segments:
        return 2, []
    inserts = _breath_inserts_fbl(mark, tg, segments)
    return (0, inserts) if inserts else (2, [])


def _breath_inserts_fbl(mark, pass1_tg_path, ap_segments) -> list[dict]:
    """FBL AP 段 + 第一遍 TG 词边界 → 插入指令 [{b, sym, side}]。

    与旧 FBL 流程等价的 SP 约束：只保留中心落在词间空隙（前词结束~后词开始）
    的段；词内部检出的段丢弃（多为误检）。尾部段 → EP。
    """
    tiers = parse_textgrid(pass1_tg_path)
    texts = [(t0, t1, lab) for (t0, t1, lab) in tiers.get("texts", []) if lab.strip()]
    if not texts:
        return []
    # TG texts 层是词粒度；把每个词标签在标记文本中顺序定位（TIFA 已过滤标点，
    # 词标签拼接 ≈ 文本去掉标点），得 (文本b, 文本e, t0, t1)。呼吸只插词间，
    # 词粒度正合适。
    pairs = []  # (b, e, t0, t1)
    cursor = 0
    lower_text = mark.text.lower()
    for (t0, t1, label) in texts:
        p = lower_text.find((label or "").lower(), cursor)
        if p < 0:
            continue
        pairs.append((p, p + len(label), t0, t1))
        cursor = p + len(label)
    if not pairs:
        return []
    out = []
    seen = set()
    for (a0, a1) in ap_segments:
        center = (a0 + a1) / 2.0
        item = None
        if center < pairs[0][2]:
            # 首词之前的头部空隙：旧流程同样会检（SP 段），保留
            item = {"b": pairs[0][0], "sym": "AP", "side": "before"}
        elif center >= pairs[-1][3]:
            item = {"b": pairs[-1][1], "sym": "EP", "side": "after"}
        else:
            for i in range(len(pairs) - 1):
                g0, g1 = pairs[i][3], pairs[i + 1][2]
                if g0 < g1 and g0 <= center < g1:
                    # 词间呼吸挂在「后词第一格之前」（前置）：断句处常带空格，
                    # 挂前词词尾会被 uid 回退逻辑锚成前格后置，位置语义不对。
                    item = {"b": pairs[i + 1][0], "sym": "AP", "side": "before"}
                    break
        if item and (item["b"], item["sym"]) not in seen:
            seen.add((item["b"], item["sym"]))
            out.append(item)
    return out


if __name__ == "__main__":
    import sys
    print("available:", align_available())
    if len(sys.argv) >= 3:
        # 自测：python tifa_align.py <wav> <text> <lang>
        wav, txt, lang = sys.argv[1], sys.argv[2], sys.argv[3]
        from v2ds.core.project import Mark
        mk = Mark(0.0, 0.0, txt, lang)  # 仅用其 text/slots
        out = ROOT_DIR / "_tifa_test"
        ok, chosen = compute_for_mark(mk, wav, out)
        print("ok:", ok)
        for c in chosen:
            print(c)
