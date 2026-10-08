# coding=utf-8
"""后台音频加载线程：读 wav + 计算梅尔频谱，避免卡住 UI。"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class AudioLoader(QThread):
    loaded = pyqtSignal(str, object)   # (wav_name, (wave, sr, mel, mel_fps))
    failed = pyqtSignal(str)

    def __init__(self, path: str, name: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.name = name

    def run(self):
        try:
            import soundfile as sf
            wave, sr = sf.read(self.path, dtype="float32", always_2d=True)
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            wave = np.ascontiguousarray(wave, dtype=np.float32)
            mel, mel_fps = self._compute_mel(wave, sr)
            self.loaded.emit(self.name, (wave, int(sr), mel, float(mel_fps)))
        except Exception as e:
            self.failed.emit(str(e))

    @staticmethod
    def _compute_mel(wave, sr):
        import librosa
        # 全频段：不降采样，保留到奈奎斯特（口水音高频 8-16kHz 需可见）
        hop = 512
        n_fft = 2048 if sr >= 44100 else 1024
        mel = librosa.feature.melspectrogram(
            y=wave, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=96, power=1.0
        )
        mel = librosa.power_to_db(mel, ref=1.0)
        lo = float(mel.min())
        rng = float(mel.ptp())
        if rng > 1e-6:
            mel = (mel - lo) / rng
        else:
            mel = np.zeros_like(mel)
        mel = (mel * 255.0).astype(np.uint8)
        return mel, float(sr) / hop
