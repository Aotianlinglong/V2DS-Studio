# coding=utf-8
"""全局配置管理：config.json。

结构（用户已确认）：
{
  "onnx_asr_model_path": "C:/path/to/Qwen3-ASR-1.7B-DML",
  "default_lang": "zh",
  "window_width": 1200,
  "window_height": 800,
  "wave_spec_divider_height": 400
}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# 软件根目录（v2ds/core/config.py → 上溯两级）：数据目录仍在软件根
from v2ds.paths import ROOT_DIR as APP_DIR
# 默认配置存放位置（软件目录内）
CONFIG_PATH = APP_DIR / "config.json"

# 默认模型路径：优先软件目录内 asr_model/（整合包布局），其次旧的 Vocal2Midi 位置
V2M_DIR = APP_DIR.parent / "Vocal2Midi"
DEFAULT_MODEL_CANDIDATES = [
    APP_DIR / "asr_model",
    V2M_DIR / "experiments" / "Qwen3-ASR-1.7B-dml",
    V2M_DIR / "models" / "Qwen3-ASR-1.7B-dml",
]

LANG_MAP = {"zh": "普通话", "ja": "日文", "en": "英文", "yue": "粤语"}
LANG_CODES = ["zh", "ja", "en", "yue"]

# 内置特殊音素（写死定义）：SP/AP/EP/GS 是词汇表中的无语言裸符号，导出永远不带
# language 属性；ja/cl 例外——贴合 DiffSinger 惯例，固定导出为 language="ja" 的 cl。
# side 固定：before=插在被点格之前，after=紧随其后。
# symbol 字段：存储/导出用的实际符号（缺省=键名），用于「同符号不同 side」的条目。
BUILTIN_PHONEMES = {
    "SP": {"side": "before"},
    "AP": {"side": "before"},
    "EP": {"side": "after"},
    "GS": {"side": "before"},
    "AP（后置）": {"side": "after", "symbol": "AP"},
    "ja/cl": {"side": "before", "lang": "ja"},
}

DEFAULT_CONFIG = {
    "onnx_asr_model_path": "",
    "default_lang": "zh",
    "window_width": 1280,
    "window_height": 820,
    "wave_spec_divider_height": 380,
    "recent_projects": [],
    "enabled_vsts": [],
    # 内置特殊音素在右键菜单中的可见性（默认可用 AP/EP/AP（后置）/ja/cl；SP/GS 默认隐藏）
    "builtin_phoneme_visibility": {"SP": False, "AP": True, "EP": True,
                                   "GS": False, "AP（后置）": True, "ja/cl": True},
    # 呼吸检测：句尾（最后词结束之后）检出的呼吸是否自动插入 EP
    "breath_insert_ep": True,
    # 自定义快捷音素：{symbol, side}；不允许与内置音素重名。
    # 导出时带语言属性（默认标记主语言，可在音素格上右键改）。
    "quick_phonemes": [
        {"symbol": "cl", "side": "before"},
        {"symbol": "n", "side": "before"},
    ],
    "show_tooltips": True,
}


def normalize_quick_phonemes(raw) -> list[dict]:
    """自定义快捷音素配置归一化为 [{"symbol": str, "side": "before"|"after"}, ...]。

    旧配置迁移：纯字符串列表 → 全部视为 before；内置音素（SP/AP/EP/GS/AP（后置）/ja/cl）剔除
    （它们已写死为内置，由 builtin_phoneme_visibility 控制可见性）。
    """
    out = []
    if not isinstance(raw, list):
        return out
    for it in raw:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append({"symbol": s, "side": "before"})
        elif isinstance(it, dict):
            s = str(it.get("symbol", "")).strip()
            if not s:
                continue
            side = str(it.get("side", "before")).strip().lower()
            out.append({"symbol": s, "side": "after" if side == "after" else "before"})
    return [it for it in out if it["symbol"] not in BUILTIN_PHONEMES]


def normalize_builtin_visibility(raw) -> dict:
    """内置音素可见性归一化：补全缺省键、剔除未知键。默认 SP/GS 隐藏。"""
    default = {"SP": False, "AP": True, "EP": True, "GS": False,
               "AP（后置）": True, "ja/cl": True}
    if not isinstance(raw, dict):
        return default
    return {k: bool(raw.get(k, default[k])) for k in BUILTIN_PHONEMES}


def is_valid_model_dir(path: str) -> bool:
    """目录存在且包含可用模型文件（.gguf 或 encoder*.onnx + embed_tokens.bin）。"""
    if not path:
        return False
    p = Path(path).expanduser()
    if not p.is_dir():
        return False
    return (
        any(p.glob("*.gguf"))
        or (p / "embed_tokens.bin").is_file()
        or any(p.glob("encoder*.onnx"))
    )


def find_default_model() -> str:
    """返回一个可用的默认模型路径（存在模型标记文件），否则空串。"""
    for cand in DEFAULT_MODEL_CANDIDATES:
        if is_valid_model_dir(cand):
            return str(cand.resolve())
    return ""


class Config:
    """封装 config.json 的读写与内存状态。"""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else CONFIG_PATH
        self.data = dict(DEFAULT_CONFIG)
        self.load()

    def load(self):
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    # 合并，保留默认键
                    merged = dict(DEFAULT_CONFIG)
                    merged.update(loaded)
                    self.data = merged
            except Exception:
                self.data = dict(DEFAULT_CONFIG)
        # 快捷音素归一化（旧字符串列表自动迁移为 {symbol, side}）
        self.data["quick_phonemes"] = normalize_quick_phonemes(
            self.data.get("quick_phonemes"))
        self.data["builtin_phoneme_visibility"] = normalize_builtin_visibility(
            self.data.get("builtin_phoneme_visibility"))
        # 模型路径：为空或已失效（目录改名/移动）时，回退到软件目录内自动探测
        if not is_valid_model_dir(self.data.get("onnx_asr_model_path", "")):
            self.data["onnx_asr_model_path"] = find_default_model()

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            raise OSError(f"配置保存失败: {e}")

    # ---- 便捷访问 ----
    def __getitem__(self, key):
        return self.data.get(key)

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)
