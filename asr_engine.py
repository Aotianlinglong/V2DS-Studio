# coding=utf-8
"""ASR 引擎封装：复用整合包 Vocal2Midi 的 inference.API.asr_api（同一 Python 环境）。

- 用整合包 python 运行时，Vocal2Midi 的 inference 代码可直接 import。
- 模型只加载一次（进程内缓存），后续片段复用。
- 每个标记片段独立切片、独立指定语言，批量识别，返回每段文本。
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

# ---- 环境准备：inference 包与本文件同目录，加载 llama.cpp 动态库 ----
V2M_DIR = Path(__file__).resolve().parent
_BIN = V2M_DIR / "inference" / "qwen3asr_dml" / "bin"


def _prepare_env():
    if str(V2M_DIR) not in sys.path:
        sys.path.insert(0, str(V2M_DIR))
    if _BIN.is_dir():
        os.environ["PATH"] = str(_BIN) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(_BIN))
        except Exception:
            pass


_prepare_env()


def validate_model_dir(model_path: str) -> str:
    """校验模型目录是否包含可用模型文件，返回规范化路径。"""
    p = Path(model_path).expanduser()
    if p.is_file():
        p = p.parent
    has_gguf = any(p.glob("*.gguf"))
    has_onnx = (p / "embed_tokens.bin").is_file() or any(p.glob("encoder*.onnx"))
    if not (has_gguf or has_onnx):
        raise FileNotFoundError(
            f"模型目录中未找到模型文件：{p}\n"
            "需要 Qwen3-ASR-1.7B-dml（含 .gguf 或 encoder*.onnx + embed_tokens.bin + tokenizer.json）。"
        )
    return str(p.resolve())


class ASREngine:
    """封装 Qwen3-ASR 的加载与片段转录。所有耗时操作应在后台线程调用。"""

    def __init__(self, model_path: str, device: str = "dml"):
        self.model_path = validate_model_dir(model_path)
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        """加载模型（只加载一次，后续调用复用）。非线程安全，须在专用后台线程调用。"""
        with self._lock:
            if self._model is None:
                from inference.API.asr_api import load_qwen_model

                self._model = load_qwen_model(
                    self.model_path, device=self.device, use_cache=True
                )
            return self._model

    def transcribe_marks(
        self,
        wav_path: str,
        marks,
        languages: list[str] | None = None,
        on_progress=None,
        cancel_checker=None,
    ) -> list[str]:
        """对单个 wav 的标记片段逐段识别。

        marks: 已按 start 排序的 Mark 列表（或含 .start/.end 的对象）
        languages: 与 marks 等长的语言码列表（zh/ja/en）；缺省用每段自身的 lang
        返回: 与 marks 等长的文本列表（每段一个）
        """
        import numpy as np
        import soundfile as sf

        if not marks:
            return []

        from inference.API.asr_api import batch_transcribe_asr

        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        if data.ndim > 1:
            data = data.mean(axis=1)  # 混音到单声道
        total = float(len(data)) / sr

        model = self.load()

        chunks = []
        lang_list = []
        for i, m in enumerate(marks):
            s = max(0.0, float(m.start))
            e = min(total, float(m.end))
            if e - s <= 0.001:
                chunks.append({"offset": s, "waveform": np.zeros(0, dtype=np.float32)})
            else:
                chunk = data[int(s * sr): int(e * sr)]
                chunks.append({"offset": s, "waveform": chunk})
            if languages and i < len(languages) and languages[i]:
                lang_list.append(languages[i])
            else:
                lang_list.append(getattr(m, "lang", "zh") or "zh")

        texts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="lab_asr_") as tmp:
            # 逐语言批量识别：同一语言放一批（不同语言不能混批）
            for lang in dict.fromkeys(lang_list):
                idxs = [i for i, l in enumerate(lang_list) if l == lang]
                sub_chunks = [chunks[i] for i in idxs]
                if cancel_checker and cancel_checker():
                    raise InterruptedError("ASR 已取消")
                results, chunk_indices = batch_transcribe_asr(
                    chunks=sub_chunks,
                    sr=sr,
                    asr_model=model,
                    temp_dir_path=Path(tmp),
                    asr_batch_size=1,
                    language=lang,
                    cancel_checker=cancel_checker,
                    device=self.device,
                    force_subprocess=False,
                )
                # 结果与 chunk_indices 对齐，回填到对应 marks
                for res, ci in zip(results, chunk_indices):
                    text = ""
                    if res is not None:
                        text = str(getattr(res, "text", "") or "").strip()
                    texts.append((idxs[ci], text))
                if on_progress:
                    on_progress(len([t for t in texts if isinstance(t, tuple)]), len(marks))

        texts.sort(key=lambda x: x[0])
        return [t for _, t in texts]
