# coding=utf-8
"""ASR 子进程入口：在独立 Python 进程中运行，避开 GUI 进程的 onnxruntime DLL 冲突。

用法: python asr_subprocess.py <model_path> <wav_path> <marks.json> <out.json>
  marks.json: [{"start":..,"end":..,"text":"","lang":"zh"}]
  成功后把逐段文本列表写入 out.json。
"""
from __future__ import annotations

import json
import os
import sys

# 让本目录(V2DS Studio)可被 import，从而引用 asr_engine/project
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _main():
    model_path, wav_path, marks_file, out_file = sys.argv[1:5]
    with open(marks_file, encoding="utf-8") as f:
        data = json.load(f)
    from project import Mark
    from asr_engine import ASREngine

    marks = [Mark(float(m["start"]), float(m["end"]), m.get("text", ""),
                  m.get("lang", "zh") or "zh") for m in data]
    if not marks:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        return
    eng = ASREngine(model_path, device="dml")
    texts = eng.transcribe_marks(wav_path, marks)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([t or "" for t in texts], f, ensure_ascii=False)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        _main()
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
