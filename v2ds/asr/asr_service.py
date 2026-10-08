# coding=utf-8
"""ASR 常驻服务的主进程侧客户端。

管理一个常驻的 asr_server.py 子进程：
- 模型加载一次后保持，多次转录复用（省去重复加载）
- load / transcribe / unload / ping 走 stdin/stdout 的 JSON 行协议
- 所有方法会阻塞等待子进程响应，应在后台线程调用
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

_SERVER = Path(__file__).resolve().parent / "asr_server.py"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ASRService:
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._log_path = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, model_path: str = ""):
        """启动常驻子进程（未启动才启动）。"""
        if self.is_running:
            return
        if model_path:
            self.model_path = model_path
        # 日志写到临时文件，避免把子进程的 DLL 噪音刷到 GUI 控制台
        import tempfile
        self._log_path = Path(tempfile.gettempdir()) / "lab_asr_server.log"
        cmd = [sys.executable, "-u", str(_SERVER), self.model_path]
        # 强制子进程的 stdin/stdout/stderr 走 UTF-8。
        # 否则在中文 Windows 下，子进程默认用 GBK(cp936) 写 stdout，
        # 转录返回中文时主进程按 UTF-8 严格解码就会抛
        # "utf-8 codec can't decode byte 0xd0..."。
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(self._log_path, "w", encoding="utf-8", errors="replace"),
            text=True,
            encoding="utf-8",
            errors="replace",  # 读取侧兜底：个别脏字节只替换、不抛异常
            env=env,
            creationflags=_CREATE_NO_WINDOW,
        )

    def _request(self, obj) -> dict:
        """发送一行 JSON，阻塞等待一行响应。须持有 _lock 调用。"""
        if not self.is_running:
            raise RuntimeError("ASR 服务未启动")
        try:
            self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            # 跳过子进程在 stdout 上可能打印的非 JSON 噪音行（如模型加载时的 print），
            # 直至读到一条合法的 JSON dict 响应，避免把日志当协议解析。
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("ASR 服务已退出（见 " + str(self._log_path) + "）")
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(resp, dict):
                    continue
                return resp
        except Exception as e:
            raise RuntimeError(f"ASR 服务通信失败：{e}")

    def load(self, model_path: str = ""):
        """加载模型并保持。阻塞直至加载完成。"""
        with self._lock:
            if not self.is_running:
                self.start(model_path)
            resp = self._request({"cmd": "load", "model": model_path or self.model_path})
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "加载失败"))
            return True

    def transcribe(self, wav_path: str, marks) -> list[str]:
        """转录。模型未加载则报错（调用方应先 load）。"""
        data = [{"start": m.start, "end": m.end,
                 "text": getattr(m, "text", ""), "lang": getattr(m, "lang", "zh") or "zh"}
                for m in marks]
        with self._lock:
            resp = self._request({"cmd": "transcribe", "wav": wav_path, "marks": data})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "转录失败"))
        return resp.get("result", [])

    def is_loaded(self) -> bool:
        with self._lock:
            resp = self._request({"cmd": "ping"})
            return bool(resp.get("ok") and resp.get("loaded"))

    def unload(self):
        """卸载并退出常驻进程，释放内存。"""
        with self._lock:
            if self.is_running:
                try:
                    self._request({"cmd": "unload"})
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=10)
                except Exception:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
            self._proc = None

    def stop(self):
        self.unload()
