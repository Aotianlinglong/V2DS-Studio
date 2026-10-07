# coding=utf-8
"""基于 sounddevice 的播放器：流式播放 wav 片段，精确控制位置与暂停/恢复。

不依赖 Qt Multimedia 后端（在整合包 embeddable Python 下不可靠）。
sounddevice 直接向音频设备播放 numpy 数组。
"""
from __future__ import annotations

import time


class AudioPlayer:
    def __init__(self):
        import sounddevice  # 延迟 import，避免无设备时启动报错
        self._sd = sounddevice
        self._wave = None
        self._sr = 44100
        self._mark_start = 0.0
        self._mark_end = 0.0
        self._paused_at = 0.0
        self._wall_start = 0.0
        self._device = None
        self._latency = 0.02   # 设备输出缓冲延迟补偿(秒)，使播放头贴近实际听感
        self.playing = False
        self._stream = None
        self._vst_hook = None  # callback(block_float32) 实时发给 VST 子进程
        self._on_stop = None   # 停止/暂停时调用（发 flush 给 VST）

    def set_device(self, device_index):
        """设定输出设备（sounddevice 设备索引，None=系统默认）。"""
        self._device = device_index
        self._query_latency()
        self.stop()

    def _query_latency(self):
        """从设备信息读取默认输出缓冲延迟作为补偿量。"""
        try:
            info = self._sd.query_devices(self._device, "output")
            if isinstance(info, dict):
                lat = info.get("default_low_output_latency")
                if lat is not None:
                    self._latency = min(max(float(lat), 0.0), 0.2)
                    return
            self._latency = 0.02
        except Exception:
            self._latency = 0.02

    def load(self, wave, sr):
        """设定当前音频数据（加载新音频时调用，重置播放状态）。"""
        self._wave = wave
        self._sr = sr
        self.stop()

    def stop(self):
        try:
            self._sd.stop()
        except Exception:
            pass
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
            self._stream = None
        self.playing = False
        self._paused_at = 0.0
        if self._on_stop:
            try: self._on_stop()
            except Exception: pass

    def play_slice(self, start: float, end: float):
        """从头播放 [start, end] 片段。"""
        self._mark_start = float(start)
        self._mark_end = float(end)
        self._paused_at = 0.0
        self._start_from(0.0)

    def toggle(self):
        """播放/暂停切换。暂停后恢复从暂停位置继续。"""
        if self.playing:
            self.pause()
            return False
        self._start_from(self._paused_at)
        return True

    def pause(self):
        if not self.playing:
            return
        self._paused_at += time.perf_counter() - self._wall_start
        self.playing = False
        try:
            self._sd.stop()
        except Exception:
            pass
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
            self._stream = None
        if self._on_stop:
            try: self._on_stop()
            except Exception: pass

    def _start_from(self, offset_sec: float):
        if self._wave is None:
            return
        s = int((self._mark_start + offset_sec) * self._sr)
        e = int(self._mark_end * self._sr)
        e = min(len(self._wave), e)
        if s >= e:
            self.stop()
            return
        try:
            self._sd.stop()
            if self._stream is not None:
                try: self._stream.close()
                except Exception: pass

            import numpy as np
            block_size = 128
            pos = [s]
            end_pos = e
            hook = self._vst_hook

            def cb(outdata, frames, time_info, status):
                start = pos[0]
                n = min(frames, end_pos - start)
                if n <= 0:
                    outdata.fill(0)
                    return
                chunk = self._wave[start:start + n]
                if hook is not None:
                    # VST 模式：主程序不出声，只发 block 给插件，插件处理后输出
                    outdata.fill(0)
                    try: hook(np.ascontiguousarray(chunk, dtype=np.float32))
                    except Exception: pass
                else:
                    outdata[:n, 0] = chunk
                    outdata[n:, 0] = 0
                    outdata[:, 1] = outdata[:, 0]
                pos[0] = start + n

            self._stream = self._sd.OutputStream(
                samplerate=self._sr, channels=2, blocksize=block_size,
                device=self._device, callback=cb)
            self._stream.start()
        except Exception:
            self.stop()
            return
        self._paused_at = offset_sec
        self._wall_start = time.perf_counter()
        self.playing = True

    def current_pos(self) -> float:
        """当前播放位置（秒，片段内绝对时间），墙钟计时并补偿设备缓冲延迟。
        结果 clamp 到 [mark_start, mark_end]，避免播放刚开始时补偿导致负值使播放头消失。"""
        if self.playing:
            pos = self._mark_start + self._paused_at + (time.perf_counter() - self._wall_start) - self._latency
            return max(self._mark_start, min(pos, self._mark_end))
        return self._mark_start + self._paused_at

    @property
    def end(self) -> float:
        return self._mark_end

    @property
    def start(self) -> float:
        return self._mark_start
