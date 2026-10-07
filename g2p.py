# coding=utf-8
"""G2P 引擎（以 openvpi/TIFA 的 g2pflow 为标准，即 tifa.cpp 的 G2P 同源实现）。

发音候选 list_candidates / 日语分词 ja_word_seg 走 g2pflow 管道：
- 转换器顺序与 TIFA configs/g2p.yaml 一致（chinese-pinyin → japanese-mecab →
  yue-jyutping → lstm/en → dictionary(zh/ja/yue) → passthrough）。
- 词典用 LS 自带 g2p/models/dictionaries（与 TIFA/tifa.cpp 完全相同）。
- 日语分词器用 MeCab + 完整 UniDic（与 TIFA 推理一致；若当前环境已安装完整
  unidic 则优先使用，缺则 unidic_lite 兜底）；与 TIFA 的 japanese-mecab 同源分词。
- 预处理与 TIFA 模型 inference.g2p 一致（无 lowercase）。
- 英文 LSTM：g2p/models/LstmG2p-Eng 放好模型（含 char.json/phonemes.json）即启用，
  否则用 dictionary(cmudict) 兜底。

PFML 校验 validate_pfml / 音素查表 check_phoneme 仍走 maxlabel_cli（模型词表权威）。
多音字消歧 auto_polyphonic 沿用 pinyin_engine（TIFA cpp-pinyin 移植）。
"""
from __future__ import annotations
import itertools
import subprocess
import sys
from pathlib import Path

_G2P_DIR = Path(__file__).resolve().parent / "g2p"
_CLI = _G2P_DIR / ("maxlabel_cli.exe" if sys.platform.startswith("win") else "maxlabel_cli")
_MODELS = _G2P_DIR / "models"
_DICTS = _MODELS / "dictionaries"
_LSTM = _MODELS / "LstmG2p-Eng"


def _run(args: list[str], input_text: str | None = None, timeout: float = 30.0):
    cmd = [str(_CLI)] + args + ["--models", str(_MODELS)]
    try:
        p = subprocess.run(
            cmd, input=input_text, capture_output=True, text=True,
            encoding="utf-8", timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return -1, "", "maxlabel_cli not found: " + str(_CLI)
    except Exception as e:
        return -2, "", str(e)


# ---------------------------------------------------------------------------
# g2pflow 管道（镜像 TIFA g2p.yaml，用 LS 词典 + 完整 UniDic）
# ---------------------------------------------------------------------------
_pipeline = None
_fugashi = None


def _ja_unidic_dir() -> str:
    """返回完整 UniDic 词典目录（与 TIFA 推理一致）。

    依赖完整版 unidic 包（安装后需执行一次 python -m unidic download 拉取词典）。
    """
    import unidic
    d = Path(unidic.DICDIR)
    if not (d / "sys.dic").is_file():
        raise RuntimeError("未找到完整 UniDic 词典，请执行: python -m unidic download")
    return str(d.resolve())


def _build_pipeline():
    from g2pflow import G2PPipelineConfig, build_pipeline_from_config
    converters = [
        {"id": "chinese-pinyin", "kwargs": {"dict_path": str(_DICTS / "ds-zh-pinyin-lite.txt")}},
        {"id": "japanese-mecab", "kwargs": {"dict_path": str(_DICTS / "japanese_dict_full.txt"),
                                            "nbest": 32, "double_written_sokuon": False,
                                            "unidic_dir": _ja_unidic_dir()}},
        {"id": "yue-jyutping", "kwargs": {"dict_path": str(_DICTS / "jyutping_dict.txt")}},
    ]
    if (_LSTM / "char.json").is_file():
        # 英文 LSTM 模型就位即启用（OOV 推理）；缺模型时用 cmudict 兜底
        converters.append({"id": "lstm", "language": "en", "kwargs": {
            "dict_path": str(_DICTS / "ds_cmudict-07b.txt"),
            "model_path": str(_LSTM), "beam_size": 16}})
    else:
        converters.append({"id": "dictionary", "language": "en",
                           "kwargs": {"dict_path": str(_DICTS / "ds_cmudict-07b.txt")}})
    converters += [
        {"id": "dictionary", "language": "zh", "kwargs": {"dict_path": str(_DICTS / "ds-zh-pinyin-lite.txt")}},
        {"id": "dictionary", "language": "ja", "kwargs": {"dict_path": str(_DICTS / "japanese_dict_full.txt")}},
        {"id": "dictionary", "language": "yue", "kwargs": {"dict_path": str(_DICTS / "jyutping_dict.txt")}},
        {"id": "passthrough"},
    ]
    # 预处理与 TIFA 模型推理配置一致：无 lowercase（TIFA inference.g2p 里被注释掉）
    cfg = G2PPipelineConfig(
        preprocessors=[{"id": "filter-punctuation"}, {"id": "strip-whitespace"}],
        converters=converters,
    )
    return build_pipeline_from_config(cfg, root_path=str(_DICTS))


def _g2pflow_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def list_candidates(text: str, lang: str) -> list[dict]:
    """G2P 候选（g2pflow，与 tifa.cpp 同源）：[{'script','phonemes'}, ...]。

    覆盖选中整个发音单位：zh/yue 各字、en 词、ja 整词组的读音候选，逐词
    路径求笛卡尔积（上限 MAX_C 条），保默认读音优先。
    """
    MAX_C = 32
    try:
        ws = _g2pflow_pipeline().convert(text, languages=[lang])
    except Exception:
        return []
    per = []
    for w in ws:
        paths = []
        for r in w.readings:
            for p in r.paths:
                if p:
                    paths.append(p)
        if not paths:
            return []          # 某词无完整读音路径 → 无法组合
        per.append(paths)
    res = []
    seen = set()
    for combo in itertools.product(*per):
        scripts, ph = [], []
        for path in combo:
            for g in path:
                scripts.append(g.script)
                ph.extend(g.phonemes)
        if not ph:
            continue
        key = tuple(ph)
        if key in seen:
            continue
        seen.add(key)
        res.append({"script": " ".join(scripts), "phonemes": ph})
        if len(res) >= MAX_C:
            break
    return res


# ---------------------------------------------------------------------------
# 日语分词（MeCab + 完整 UniDic，与 tifa.cpp 一致）
# ---------------------------------------------------------------------------
def _get_fugashi():
    global _fugashi
    if _fugashi is None:
        import fugashi
        ud = _ja_unidic_dir()
        _fugashi = fugashi.Tagger('-r "{0}\\mecabrc" -d "{0}"'.format(ud))
    return _fugashi


def ja_word_seg(text: str) -> list[dict]:
    """日语切词：fugashi(MeCab + 完整 UniDic) 返回词边界和读音。

    与 pyopenjtalk(ipadic) 不同，这里与 tifa.cpp 的 MeCab+UniDic 分词一致。
    返回 [{begin, end, string, read, pron, pos}]，begin/end 为字符偏移。
    """
    try:
        t = _get_fugashi()
        res = []
        pos = 0
        for node in t(text):
            s = node.surface
            n = len(s)
            if n == 0:
                continue
            b, e = pos, pos + n
            pos = e
            pron = getattr(node.feature, "pron", "") or ""
            pos_ = getattr(node.feature, "pos1", "") or ""
            res.append({"begin": b, "end": e, "string": s,
                        "read": pron, "pron": pron, "pos": pos_})
        return res
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 多音字自动消歧（词典式，TIFA cpp-pinyin 移植）
# ---------------------------------------------------------------------------
_pinyin_engines = {}


def _pinyin_engine(lang: str):
    """普通话(zh)/粤语(yue) 词典消歧引擎（惰性加载，缓存）。"""
    if lang not in _pinyin_engines:
        from pinyin_engine import PinyinEngine, STYLE_NORMAL
        d = Path(__file__).resolve().parent / "pinyin_engine" / "dicts" / (
            "mandarin" if lang == "zh" else "cantonese")
        _pinyin_engines[lang] = (PinyinEngine(d), STYLE_NORMAL)
    return _pinyin_engines[lang]


def auto_polyphonic(text: str, lang: str) -> dict[int, str]:
    """词典消歧多音字读音。返回 {字符偏移: 消歧后的拼音(无调)}，只含多音字。

    lang 仅支持 zh/yue；其余语言返回空。
    """
    if lang not in ("zh", "yue"):
        return {}
    eng, style = _pinyin_engine(lang)
    chars = list(text)
    simp = eng.simplify(chars)
    readings = eng.query_raw(simp, style=style)  # 每字一个消歧读音
    out = {}
    for i, sc in enumerate(simp):
        # "多音字" = 多于一个【不同基础音节】。仅声调不同的不算（如好 hǎo/hào
        # 音素都是 h+ao，对唱歌标音无歧义；游/事/来/要 等也只有声调变体）。
        if len(set(eng.readings(sc, style=style))) > 1:
            out[i] = readings[i][0]
    return out


def check_phoneme(symbol: str, langs: str = "zh,en,ja,yue") -> bool:
    """校验音素是否在词表里。"""
    rc, out, err = _run(["phoneme", symbol, "-l", langs])
    return rc == 0 and "resolves" in out


def validate_pfml(pfml_text: str) -> tuple[bool, str]:
    """校验 PFML 片段。返回 (ok, error)。validate 要文件参数，写临时文件。"""
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".pfml", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(pfml_text)
        cmd = [str(_CLI), "validate", path]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if p.returncode == 0:
            return True, ""
        return False, ((p.stdout or "") + " " + (p.stderr or "")).strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


if __name__ == "__main__":
    for lang, txt in [("zh", "重"), ("en", "love"), ("ja", "日本"), ("yue", "可遇")]:
        print(f"{lang} {txt!r} -> {list_candidates(txt, lang)}")
    print("ja seg:", ja_word_seg("東京にいる"))
    print("poly:", auto_polyphonic("银行", "zh"))
    print("check n:", check_phoneme("n", "zh"))
    print("validate:", validate_pfml('<scope language="zh">你好</scope>'))
