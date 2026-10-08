# coding=utf-8
"""VST3 子进程（DawDreamer 宿主，Adaptive/Learn 正确）：
- process 模式：离线渲染 in.wav -> out.wav
- editor 模式：DawDreamer 加载音频 render 一次（让插件拿到音频数据，
  Adaptive/Learn 正确），open_editor 打开原生窗口。
  无实时声音输出（插件仅用于调参数），离线处理选区由主程序触发。
"""
from __future__ import annotations
import sys
import os
import threading

from pathlib import Path
# fx/vst_subprocess.py → parents[0]=fx, parents[1]=软件根目录
LAB_DIR = str(Path(__file__).resolve().parents[1])
STATE_DIR = os.path.join(LAB_DIR, ".vst_states")
os.makedirs(STATE_DIR, exist_ok=True)


def _state_path(plugin_path: str) -> str:
    import hashlib
    h = hashlib.md5(plugin_path.encode()).hexdigest()[:16]
    return os.path.join(STATE_DIR, h + ".vststate")


def _load_state(plug, plugin_path: str):
    sp = _state_path(plugin_path)
    if os.path.isfile(sp):
        try:
            with open(sp, "rb") as f:
                plug.load_state(f.read())
        except Exception:
            pass


def _save_state(plug, plugin_path: str):
    try:
        state = plug.save_state()
        if state:
            with open(_state_path(plugin_path), "wb") as f:
                f.write(state)
    except Exception:
        pass


def _process(plugin_path: str, in_wav: str, out_wav: str):
    import numpy as np
    import soundfile as sf
    from pedalboard import load_plugin
    x, sr = sf.read(in_wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    plugin = load_plugin(plugin_path)
    _load_state(plugin, plugin_path)
    out = plugin.process(x, sr)
    out = np.asarray(out, dtype="float32").ravel()
    sf.write(out_wav, out, sr)


def _patch_window():
    import time, ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    GWL_STYLE, GWL_EXSTYLE = -16, -20
    PID = os.getpid()
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    for _ in range(40):
        found = []
        def _find(hwnd, _):
            p = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
            if p.value != PID: return True
            if not user32.IsWindowVisible(hwnd): return True
            r = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            w, h = r.right - r.left, r.bottom - r.top
            if w < 50 or h < 50: return True
            found.append(hwnd)
            return True
        user32.EnumWindows(WNDENUMPROC(_find), 0)
        if found:
            hwnd = found[0]
            r = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            cw, ch = r.right - r.left, r.bottom - r.top
            style = user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
            new_style = (style & ~0x80000000) | 0x00CF0000
            user32.SetWindowLongPtrW(hwnd, GWL_STYLE, new_style)
            ex = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex | 0x00000008)
            adj = wintypes.RECT()
            adj.left, adj.top, adj.right, adj.bottom = 0, 0, cw, ch
            user32.AdjustWindowRect(ctypes.byref(adj), new_style, False)
            user32.SetWindowPos(hwnd, wintypes.HWND(-1), 100, 100,
                                 adj.right - adj.left, adj.bottom - adj.top,
                                 0x0020 | 0x0010 | 0x0008)
            return
        time.sleep(0.2)


def _editor(plugin_path: str, wav_path: str):
    import numpy as np
    import soundfile as sf
    import dawdreamer as daw
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim == 1:
        x = np.stack([x, x])
    elif x.shape[0] != 2:
        x = x.T
    dur = x.shape[1] / sr
    engine = daw.RenderEngine(sr, 512)
    pb = engine.make_playback_processor("pb", x)
    plug = engine.make_plugin_processor("fx", plugin_path)
    _load_state(plug, plugin_path)
    engine.load_graph([(pb, []), (plug, ["pb"])])
    engine.render(dur)  # 让插件拿到完整音频数据，Adaptive/Learn 正确
    threading.Thread(target=_patch_window, daemon=True).start()
    plug.open_editor()  # 阻塞到关窗
    _save_state(plug, plugin_path)


def main():
    mode = sys.argv[1]
    if mode == "process":
        _process(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "editor":
        _editor(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
