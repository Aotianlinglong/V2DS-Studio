# coding=utf-8
"""V2DS Studio (Vocal to Dataset Studio) 入口。用整合包 Python（python\\python.exe）运行。"""
import os
import sys
import traceback
from pathlib import Path

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

    from PyQt5.QtWidgets import QApplication
    from config import Config
    from main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("V2DS Studio")
    _apply_dark_palette(app)
    config = Config()
    win = MainWindow(config)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
