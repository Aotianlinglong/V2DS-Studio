# coding=utf-8
"""降噪 = iZotope RX10 Voice De-noise（VST3，经 pedalboard）。"""
from __future__ import annotations
import numpy as np
from . import rx


def process(wave: np.ndarray, sr: int, strength: float = 100.0) -> np.ndarray:
    return rx.process("denoise", wave, sr, strength=strength)
