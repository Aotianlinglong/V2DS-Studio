# coding=utf-8
"""cpp-pinyin 词典消歧 G2P 引擎（纯 Python，自带普通话/粤语词典）。

从 TIFA-main 移植：`PinyinEngine.query_raw()` 用滑动窗口短语匹配，
按上下文消歧多音字读音。用于 V2DS Studio 的"自动标音"。
"""
from .engine import PinyinEngine
from .tones import apply_tone, STYLE_NORMAL, STYLE_TONE3

__all__ = ["PinyinEngine", "apply_tone", "STYLE_NORMAL", "STYLE_TONE3"]
