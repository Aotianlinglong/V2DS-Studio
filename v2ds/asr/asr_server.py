# coding=utf-8
"""常驻 ASR 服务进程：加载 Qwen3-ASR 模型后保持，多次转录复用，省去重复加载。

通过 stdin/stdout 的 JSON 行协议与主进程通信：
  请求  {"cmd":"load"}                 → 加载模型（保持加载状态）
        {"cmd":"transcribe","wav":...,"marks":[...]}  → 逐段转录
        {"cmd":"unload"}               → 释放模型并退出

响应  {"ok":true,"result":...} / {"ok":false,"error":...}
所有日志写 stderr（主进程可捕获查看）。
"""
from __future__ import annotations

import json
import os
import sys

# 作为脚本直接运行时（python v2ds/asr/asr_server.py），把软件根目录加入
# sys.path，使 v2ds 包与 inference 包可被 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # v2ds/asr → v2ds → 根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _redirect_prints_to_stderr():
    """把 print() 输出改道到 stderr。

    模型加载时 inference/asr_api 内部有多个 print()，若放任其写 stdout，
    会污染本进程与主进程约定的 JSON 行协议（主进程 readline 后直接 json.loads 而报错）。
    """
    import builtins

    _orig_print = builtins.print

    def _print_to_stderr(*args, **kwargs):
        kwargs["file"] = sys.stderr
        _orig_print(*args, **kwargs)

    builtins.print = _print_to_stderr


def _force_utf8_streams():
    """强制本进程 stdin/stdout/stderr 用 UTF-8，避免中文结果被按 GBK 写出。"""
    for _s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _main():
    _force_utf8_streams()
    _redirect_prints_to_stderr()
    from v2ds.asr.asr_engine import ASREngine, validate_model_dir

    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    engine = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _send({"ok": False, "error": f"bad request: {e}"})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "load":
                if engine is None:
                    mp = req.get("model") or model_path
                    if not mp:
                        raise ValueError("未指定模型路径")
                    engine = ASREngine(validate_model_dir(mp), device="dml")
                    engine.load()
                _send({"ok": True, "result": "loaded"})
            elif cmd == "transcribe":
                if engine is None:
                    raise RuntimeError("模型未加载，请先加载")
                wav = req.get("wav", "")
                marks = req.get("marks", [])
                from v2ds.core.project import Mark
                mlist = [Mark(float(m["start"]), float(m["end"]),
                             m.get("text", ""), m.get("lang", "zh") or "zh")
                         for m in marks]
                texts = engine.transcribe_marks(wav, mlist)
                _send({"ok": True, "result": texts})
            elif cmd == "unload":
                engine = None
                _send({"ok": True, "result": "unloaded"})
                return  # 释放后退出
            elif cmd == "ping":
                _send({"ok": True, "result": "alive",
                       "loaded": engine is not None})
            else:
                _send({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _send({"ok": False, "error": str(e)})


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    _main()
