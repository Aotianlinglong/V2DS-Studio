# coding=utf-8
# 本文件逐行改编自 openvpi/audio-slicer 的 slicer2.py
#   上游：https://github.com/openvpi/audio-slicer
#   Copyright (c) 2022 Team OpenVPI，MIT License
#   许可证全文：third_party/MIT.txt（get_rms 亦注明源自 librosa，ISC）
"""基于 RMS 能量的静音切分。

逐行复刻 openvpi/audio-slicer 的 slicer2.py（dataset-tools 里 AudioSlicer
的官方 Python 原版），保证切片效果与 dataset-tools 完全一致。

参数（单位同官方）：
  threshold_db   dB 阈值，低于此视为静音（默认 -40）
  min_length_ms  最短片段长度 ms（默认 5000）
  min_interval_ms 最短静音间隔 ms，静音不足不切（默认 300）
  hop_ms         帧长 ms（默认 20；dataset-tools GUI 常用 10）
  max_sil_kept_ms 切片两端最多保留的静音 ms（默认 5000）
"""
from __future__ import annotations

import numpy as np


def _get_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """librosa 风格的逐帧 RMS（stride 分帧，与官方一致）。"""
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode="constant")
    y = np.asarray(y)
    # 以窗口为单位分帧
    n_frames = 1 + (y.shape[0] - frame_length) // hop_length
    idx = (np.arange(frame_length)[None, :]
           + np.arange(n_frames)[:, None] * hop_length)
    frames = y[idx]
    power = np.mean(np.abs(frames) ** 2, axis=-1)
    return np.sqrt(power)


def slice_audio(samples: np.ndarray, sr: int,
                threshold_db: float = -40.0,
                min_length_ms: float = 5000.0,
                min_interval_ms: float = 300.0,
                hop_ms: float = 20.0,
                max_sil_kept_ms: float = 5000.0) -> list[tuple[float, float]]:
    """对单声道 float[-1,1] 波形做静音切分，返回 [(start_sec, end_sec), ...]。"""
    if samples is None or len(samples) == 0:
        return []
    x = np.asarray(samples, dtype=np.float32).reshape(-1)

    hop_size = round(sr * hop_ms / 1000)
    if hop_size < 1:
        hop_size = 1
    min_interval_samples = sr * min_interval_ms / 1000
    threshold = 10.0 ** (threshold_db / 20.0)
    win_size = min(round(min_interval_samples), 4 * hop_size)
    min_length = round(sr * min_length_ms / 1000 / hop_size)
    min_interval = round(min_interval_samples / hop_size)
    max_sil_kept = round(sr * max_sil_kept_ms / 1000 / hop_size)

    if (x.shape[0] + hop_size - 1) // hop_size <= min_length:
        # 整段过短，不切
        return [(0.0, x.shape[0] / sr)]

    rms_list = _get_rms(x, frame_length=win_size, hop_length=hop_size)

    sil_tags = []
    silence_start = None
    clip_start = 0
    for i, rms in enumerate(rms_list):
        # 静音帧：记录起点
        if rms < threshold:
            if silence_start is None:
                silence_start = i
            continue
        # 非静音帧，且尚未记录静音起点
        if silence_start is None:
            continue
        # 静音不够长 或 片段不够长 → 不切，清掉本次静音起点
        is_leading_silence = silence_start == 0 and i > max_sil_kept
        need_slice_middle = (i - silence_start >= min_interval
                             and i - clip_start >= min_length)
        if not is_leading_silence and not need_slice_middle:
            silence_start = None
            continue
        # 需要切：在静音区内找 RMS 最小的点作为切割位置
        if i - silence_start <= max_sil_kept:
            pos = rms_list[silence_start:i + 1].argmin() + silence_start
            if silence_start == 0:
                sil_tags.append((0, pos))
            else:
                sil_tags.append((pos, pos))
            clip_start = pos
        elif i - silence_start <= max_sil_kept * 2:
            pos = rms_list[i - max_sil_kept: silence_start + max_sil_kept + 1].argmin()
            pos += i - max_sil_kept
            pos_l = rms_list[silence_start: silence_start + max_sil_kept + 1].argmin() + silence_start
            pos_r = rms_list[i - max_sil_kept:i + 1].argmin() + i - max_sil_kept
            if silence_start == 0:
                sil_tags.append((0, pos_r))
                clip_start = pos_r
            else:
                sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                clip_start = max(pos_r, pos)
        else:
            pos_l = rms_list[silence_start: silence_start + max_sil_kept + 1].argmin() + silence_start
            pos_r = rms_list[i - max_sil_kept:i + 1].argmin() + i - max_sil_kept
            if silence_start == 0:
                sil_tags.append((0, pos_r))
            else:
                sil_tags.append((pos_l, pos_r))
            clip_start = pos_r
        silence_start = None

    # 尾部静音
    total_frames = rms_list.shape[0]
    if silence_start is not None and total_frames - silence_start >= min_interval:
        silence_end = min(total_frames, silence_start + max_sil_kept)
        pos = rms_list[silence_start:silence_end + 1].argmin() + silence_start
        sil_tags.append((pos, total_frames + 1))

    if not sil_tags:
        return [(0.0, x.shape[0] / sr)]

    # 由 sil_tags 拼出各切片（帧号 → 采样点 → 秒）
    chunks = []
    if sil_tags[0][0] > 0:
        chunks.append((0, sil_tags[0][0]))
    for i in range(len(sil_tags) - 1):
        chunks.append((sil_tags[i][1], sil_tags[i + 1][0]))
    if sil_tags[-1][1] < total_frames:
        chunks.append((sil_tags[-1][1], total_frames))

    out = []
    for b, e in chunks:
        s = b * hop_size
        en = min(x.shape[0], e * hop_size)
        if en > s:
            out.append((s / sr, en / sr))
    return out
