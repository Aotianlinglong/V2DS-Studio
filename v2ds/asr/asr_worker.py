# coding=utf-8
"""ASR 后台线程：通过独立子进程运行转录，避开 GUI 进程的 onnxruntime DLL 冲突。

GUI 进程加载 PyQt5 后，onnxruntime_pybind11_state 的 DLL 初始化会失败（已验证）。
因此把 ASR 放到一个不加载 PyQt5 的独立 Python 子进程里执行（asr_subprocess.py）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

_SUBPROC = Path(__file__).resolve().parent / "asr_subprocess.py"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ASRWorker(QThread):
    progress = pyqtSignal(int, int)     # 预留：子进程暂不逐段回报
    finished_ok = pyqtSignal(object)    # list[str] 每段文本
    failed = pyqtSignal(str)
    loaded_model = pyqtSignal()

    def __init__(self, model_path: str, wav_path: str, marks, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.wav_path = wav_path
        self.marks = marks
        self._proc = None

    def cancel(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self):
        tmpdir = tempfile.mkdtemp(prefix="lab_asr_")
        try:
            marks_file = os.path.join(tmpdir, "marks.json")
            out_file = os.path.join(tmpdir, "out.json")
            log_file = os.path.join(tmpdir, "asr.log")
            data = [{"start": m.start, "end": m.end,
                     "text": getattr(m, "text", ""), "lang": getattr(m, "lang", "zh") or "zh"}
                    for m in self.marks]
            with open(marks_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            self.loaded_model.emit()
            cmd = [sys.executable, "-u", str(_SUBPROC),
                   self.model_path, self.wav_path, marks_file, out_file]
            self._proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, "w", encoding="utf-8", errors="replace"),
                stderr=subprocess.STDOUT,
                creationflags=_CREATE_NO_WINDOW,
            )
            rc = self._proc.wait()
            self._proc = None
            if rc != 0:
                msg = ""
                try:
                    msg = Path(log_file).read_text(encoding="utf-8", errors="replace")[-2500:]
                except Exception:
                    pass
                self.failed.emit(f"ASR 子进程异常（退出码 {rc}）：\n{msg}")
                return
            texts = json.loads(Path(out_file).read_text(encoding="utf-8"))
            self.finished_ok.emit(texts)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
