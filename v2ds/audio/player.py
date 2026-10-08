# coding=utf-8
"""基于 sounddevice 的播放器：流式播放 wav 片段，精确控制位置与暂停/恢复。

不依赖 Qt Multimedia 后端（在整合包 embeddable Python 下不可靠）。
sounddevice 直接向音频设备播放 numpy 数组。

播放速度（0.5~2.0，保调）：用 librosa 相位声码器 time_stretch 仅为当前播放
片段惰性生成变速缓冲，以原采样率播放 → 音调不变、时长按 1/speed 缩放。
首次使用前在后台线程预热 numba JIT，避免第一次点调速时长时间卡顿。
"""
from __future__ import annotations

import threading
import time

SPEED_MIN = 0.5
SPEED_MAX = 2.0
SPEED_STEP = 0.1


def _warmup_stretch():
    """后台预热 librosa/numba（首次 time_stretch 会 JIT 编译，耗时数秒）。"""
    try:
        import numpy as np
        import librosa
        y = np.zeros(22050, dtype=np.float32)
        librosa.effects.time_stretch(y, rate=1.2)
    except Exception:
        pass


class AudioPlayer:
    def __init__(self):
        import sounddevice  # 延迟 import，避免无设备时启动报错
        self._sd = sounddevice
        self._wave = None
        self._sr = 44100
        self._stretched = None   # 变速缓冲（仅为当前播放片段生成）；None 表示未就绪/1.0x
        self._stretch_base = 0   # 变速缓冲对应的源起始采样下标（区域变速用）
        self._stretch_win = None # 变速缓冲对应的 (源起采样, 源止采样, 速度)，用于判定窗口是否仍匹配
        self._speed = 1.0
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
        threading.Thread(target=_warmup_stretch, daemon=True).start()

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
        """设定当前音频数据（加载新音频时调用，重置播放状态）。

        不在此处做变速：变速缓冲仅为"当前播放片段"惰性生成，
        避免每次换音频/改速度都对整段长音频做相位声码器（长音频会卡顿数秒）。
        """
        self._wave = wave
        self._sr = sr
        self._stretched = None
        self._stretch_base = 0
        self._stretch_win = None
        self.stop()

    # ---------------- 播放速度（保调，相位声码器） ----------------
    @property
    def speed(self) -> float:
        return self._speed

    def _segment_window(self):
        """当前播放片段 [mark_start, mark_end] 对应的源采样半开区间 [s0, s1)。

        片段过短（<16 样本）或非法时回退到整段（保底正确）。
        """
        n = len(self._wave)
        s0, s1 = 0, n
        if self._mark_end > self._mark_start:
            a = max(0, int(self._mark_start * self._sr))
            b = min(n, int(self._mark_end * self._sr))
            if b - a >= 16:
                s0, s1 = a, b
        return s0, s1

    def _ensure_buf(self):
        """按当前速度、只为当前播放片段 [mark_start, mark_end] 生成变速缓冲。

        只处理要播放的区间（通常是一两个语的片段，远小于整首歌），
        因此改速度几乎是瞬时的。窗口非法时退回整段（保底正确）。
        """
        self._stretched = None
        self._stretch_base = 0
        self._stretch_win = None
        if self._wave is None or abs(self._speed - 1.0) < 1e-9:
            return
        import numpy as np
        import librosa
        s0, s1 = self._segment_window()
        if s1 - s0 < 16:
            return
        seg = np.ascontiguousarray(self._wave[s0:s1], dtype=np.float32)
        out = librosa.effects.time_stretch(seg, rate=self._speed)
        self._stretched = np.ascontiguousarray(out, dtype=np.float32)
        self._stretch_base = s0
        self._stretch_win = (s0, s1, self._speed)

    def set_speed(self, rate: float) -> float:
        """设定播放速度（钳制 0.5~2.0）。

        只记录速度并让旧缓冲失效；真正的变速缓冲在下次播放（_start_from）
        时按当前片段惰性构建。若正在播放，则保持当前源时间位置立即重建续播。
        返回实际生效的速度。
        """
        rate = round(min(SPEED_MAX, max(SPEED_MIN, float(rate))), 3)
        if self._wave is None:
            self._speed = rate
            return rate
        was_playing = self.playing
        if was_playing:
            self.pause()
        src_pos = self.current_pos() if (was_playing or self._paused_at > 0) \
            else self._mark_start
        src_off = max(0.0, src_pos - self._mark_start)
        self._speed = rate
        self._stretched = None      # 失效旧速度缓冲，续播时按新速度重建
        self._stretch_base = 0
        self._stretch_win = None
        self._paused_at = src_off / rate
        if was_playing:
            self._start_from(self._paused_at)
        return rate

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
        rate = self._speed
        # 需要变速：缓冲缺失，或其窗口/速度与当前片段不符（如上个片段播完后切到
        # 更靠后的片段，旧缓冲不能复用）→ 按当前片段重新生成，否则索引会落到旧缓冲
        # 之外导致 s>=e 立即停止（表现为"点播放没声音"）。
        if abs(rate - 1.0) >= 1e-9:
            win = self._segment_window()
            if self._stretched is None or self._stretch_win != (win[0], win[1], rate):
                self._ensure_buf()
        if self._stretched is not None:
            # 变速缓冲下标 j 对应源采样 base + j*rate；
            # offset_sec 是"输出秒"，源位置 = mark_start + offset_sec*rate
            buf = self._stretched
            base = self._stretch_base
            s = int((self._mark_start * self._sr - base) / rate) + int(offset_sec * self._sr)
            e = int((self._mark_end * self._sr - base) / rate)
        else:
            buf = self._wave
            s = int(self._mark_start * self._sr) + int(offset_sec * self._sr)
            e = int(self._mark_end * self._sr)
        s = max(0, s)
        e = min(len(buf), e)
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
                chunk = buf[start:start + n]
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
        """当前播放位置（源音频秒，片段内绝对时间），墙钟计时并补偿设备缓冲延迟。

        计时沿"输出时间轴"累加（暂停点+墙钟增量），乘播放速度换算回源时间。
        结果 clamp 到 [mark_start, mark_end]，避免播放刚开始时补偿导致负值使播放头消失。"""
        if self.playing:
            out_off = self._paused_at + (time.perf_counter() - self._wall_start) - self._latency
            pos = self._mark_start + max(0.0, out_off) * self._speed
            return max(self._mark_start, min(pos, self._mark_end))
        return self._mark_start + max(0.0, self._paused_at) * self._speed

    @property
    def end(self) -> float:
        return self._mark_end

    @property
    def start(self) -> float:
        return self._mark_start
