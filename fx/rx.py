# coding=utf-8
"""调用 iZotope RX VST3 插件（通过 Spotify pedalboard）离线处理音频。

默认插件目录为系统标准 VST3 路径，需要本机已安装 iZotope RX：
  C:\\Program Files\\Common Files\\VST3\\iZotope\\
    RX 10 Voice De-noise.vst3   降噪
    RX 10 Mouth De-click.vst3   去口水音
    RX 10 De-plosive.vst3       去喷麦
pedalboard 是 Python VST3 host（Spotify 开源，win x64 预编译 wheel），
离线喂 wav 处理。插件实例缓存复用，强度用 dry/wet 混合。
"""
from __future__ import annotations
import os
import numpy as np

VST_DIR = r"C:\Program Files\Common Files\VST3\iZotope"
PLUGINS = {
    "denoise": "RX 10 Voice De-noise.vst3",
    "declick": "RX 10 Mouth De-click.vst3",
    "deplosive": "RX 10 De-plosive.vst3",
}
_cache = {}


def _load(key: str):
    if key not in _cache:
        from pedalboard import load_plugin
        path = os.path.join(VST_DIR, PLUGINS[key])
        if not os.path.exists(path):
            raise RuntimeError(f"找不到 RX10 插件：{path}（请确认 iZotope RX10 已安装）")
        _cache[key] = load_plugin(path)
    return _cache[key]


def process(key: str, wave: np.ndarray, sr: int, strength: float = 100.0) -> np.ndarray:
    """调用对应 RX10 插件处理。strength: 0(原音)~100(插件输出全开)。"""
    x = np.asarray(wave, dtype=np.float32)
    if len(x) < 64:
        return x.copy()
    p = _load(key)
    wet = np.asarray(p.process(x, sr), dtype=np.float32).ravel()
    m = min(len(x), len(wet))
    w = float(np.clip(strength, 0.0, 100.0)) / 100.0
    out = wet[:m] * w + x[:m] * (1.0 - w)
    if len(out) < len(x):
        out = np.concatenate([out, x[len(out):]])
    return out.astype(np.float32)
