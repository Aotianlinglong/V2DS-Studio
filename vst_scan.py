# coding=utf-8
"""VST3 扫描器：扫描系统默认目录 + 用户自定义目录，列出所有 .vst3 插件。"""
from __future__ import annotations
import os

DEFAULT_DIRS = [
    r"C:\Program Files\Common Files\VST3",
    r"C:\Program Files\VST3",
    r"C:\Program Files\Common Files\Steinberg\VST3",
]


def scan(dirs=None):
    """返回 [(名称, 完整路径)]，去重、仅 .vst3。"""
    seen = {}
    for d in (dirs or DEFAULT_DIRS):
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".vst3"):
                    p = os.path.join(root, f)
                    if p not in seen:
                        seen[p] = os.path.splitext(f)[0]
    return sorted(((name, p) for p, name in seen.items()), key=lambda t: t[0].lower())
