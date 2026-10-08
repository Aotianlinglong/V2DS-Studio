# coding=utf-8
"""去口水音 = iZotope RX10 Mouth De-click（VST3，经 pedalboard）。"""
from __future__ import annotations
import numpy as np
from . import rx


def process(wave: np.ndarray, sr: int, strength: float = 100.0) -> np.ndarray:
    return rx.process("declick", wave, sr, strength=strength)
