# coding=utf-8
"""V2DS Studio (Vocal to Dataset Studio) 入口。用整合包 Python（python\\python.exe）运行。"""
import faulthandler
import os
import sys
import traceback
from pathlib import Path

# 原生致命错误（段错误/abort 等 excepthook 抓不到的）打印到 stderr，便于从 .bat 窗口看日志
faulthandler.enable()

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# inference 包就在本目录（APP_DIR 已加入 sys.path）；加载 llama.cpp 动态库
_BIN = APP_DIR / "inference" / "qwen3asr_dml" / "bin"
if _BIN.is_dir():
    os.environ["PATH"] = str(_BIN) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(_BIN))
    except Exception:
        pass

# 必须先于 PyQt5 导入 onnxruntime：PyQt5 加载的旧版 VC 运行时 DLL 与
# onnxruntime_pybind11_state 冲突，后导入会 DLL 初始化例程失败。
# 呼吸检测（tifa_onnx_runner）在按钮回调里才 import，那时 PyQt5 早已加载。
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass  # 未装 onnxruntime 的环境不阻塞启动（呼吸检测按钮再报错）


def _apply_dark_palette(app) -> None:
    """应用全局深色主题（Fusion + 深色 QPalette + 控件样式表）。"""
    app.setStyle("Fusion")
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPalette, QColor

    bg = QColor(30, 32, 38)
    panel = QColor(38, 40, 46)
    text = QColor(210, 214, 220)
    sel = QColor(58, 110, 200)

    pal = QPalette()
    pal.setColor(QPalette.Window, bg)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, QColor(24, 26, 32))
    pal.setColor(QPalette.AlternateBase, panel)
    pal.setColor(QPalette.ToolTipBase, panel)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, panel)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Link, QColor(90, 160, 255))
    pal.setColor(QPalette.Highlight, sel)
    pal.setColor(QPalette.HighlightedText, Qt.white)
    pal.setColor(QPalette.PlaceholderText, QColor(120, 125, 135))
    app.setPalette(pal)
    app.setStyleSheet("""
        QMenuBar { background: #20222a; color: #d2d6dc; }
        QMenuBar::item:selected { background: #2c2f3a; }
        QMenu { background: #262933; color: #d2d6dc; }
        QMenu::item:selected { background: #2c2f3a; }
        QStatusBar { background: #20222a; color: #b8bdc6; }
        QHeaderView::section { background: #262933; color: #cfd3da; border: 1px solid #33363f; }
        QTableWidget { background: #181a20; color: #d2d6dc; gridline-color: #2c2f38; }
        QListWidget { background: #181a20; color: #d2d6dc; }
        QTextEdit { background: #181a20; color: #d2d6dc; border: 1px solid #33363f; }
        QPushButton { background: #333641; color: #e2e5ea; border: 1px solid #454a57; padding: 3px 10px; border-radius: 3px; }
        QPushButton:hover { background: #3c404d; }
        QPushButton:pressed { background: #2a2d36; }
        QPushButton:disabled { color: #7a7f8a; }
        QSlider::groove:horizontal { height: 5px; background: #333641; border-radius: 2px; }
        QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #6aa0ff; }
        QComboBox { background: #262933; color: #d2d6dc; border: 1px solid #454a57; padding: 2px 6px; }
        QComboBox QAbstractItemView { background: #262933; color: #d2d6dc; selection-background-color: #2c2f3a; }
        QToolTip { background: #262933; color: #e2e5ea; border: 1px solid #454a57; }
    """)


def main() -> int:
    # 整合包 ASR 运行时内部用 multiprocessing 加载 DML 后端，
    # 必须在此引导期调用 freeze_support()，否则报
    # "An attempt has been made to start a new process before the
    #  current process has finished its bootstrapping phase."
    import multiprocessing
    multiprocessing.freeze_support()

    from PyQt5.QtWidgets import QApplication, QMessageBox
    from v2ds.core.config import Config
    from v2ds.ui.main_window import MainWindow

    class _SafeQApplication(QApplication):
        """槽函数里抛出的异常在这里兜底：先备份工程再提示，避免一次操作
        错误直接带走整个进程（旧版 PyQt 下可能 abort，未保存数据全丢）。"""

        def notify(self, receiver, event):
            try:
                return super().notify(receiver, event)
            except Exception:
                tb = traceback.format_exc()
                traceback.print_exc()
                paths = {}
                win = getattr(self, "_main_win", None)
                if win is not None:
                    try:
                        paths = win.crash_backup(tb)
                    except Exception:
                        pass
                try:
                    msg = "发生了一个错误（已自动备份，软件没有关闭）：\n\n" + tb[-1200:]
                    if paths.get("project"):
                        msg += f"\n\n工程备份：{paths['project']}"
                    if paths.get("wav"):
                        msg += f"\n音频备份：{paths['wav']}"
                    QMessageBox.critical(None, "V2DS Studio 出错", msg)
                except Exception:
                    pass
                return False

    app = _SafeQApplication(sys.argv)
    app.setApplicationName("V2DS Studio")
    _apply_dark_palette(app)
    config = Config()
    win = MainWindow(config)
    app._main_win = win
    win.show()
    return app.exec_()


def _excepthook(etype, value, tb):
    """事件循环之外的未捕获异常（如启动阶段）：同样尝试备份。"""
    text = "".join(traceback.format_exception(etype, value, tb))
    traceback.print_exception(etype, value, tb)
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        win = getattr(app, "_main_win", None) if app is not None else None
        if win is not None:
            win.crash_backup(text)
    except Exception:
        pass
    sys.__excepthook__(etype, value, tb)


if __name__ == "__main__":
    sys.excepthook = _excepthook
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
