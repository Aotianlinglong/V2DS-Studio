# coding=utf-8
"""VST3 独立弹窗：独立可移动窗口，含插件名 + 处理按钮。"""
from __future__ import annotations
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
)

_DARK = """
QWidget { background-color: #1e1f24; color: #e8e8e8; }
QLabel { color: #e8e8e8; font-size: 13px; }
QPushButton {
    background-color: #3a3d45; color: #fff; border: 1px solid #4c4f58;
    border-radius: 5px; padding: 7px 16px; font-size: 13px;
}
QPushButton:hover { background-color: #4a4d57; }
QPushButton:pressed { background-color: #2c2e35; }
"""


class VstWindow(QWidget):
    def __init__(self, main, plugin_path: str):
        super().__init__()
        self._main = main
        self._plugin_path = plugin_path
        name = os.path.splitext(os.path.basename(plugin_path))[0]
        self.setWindowTitle(f"VST3 · {name}")
        self.setStyleSheet(_DARK)
        self.setMinimumSize(340, 130)
        self.setWindowFlag(Qt.Window)  # 独立顶层窗口，带标题栏/关闭按钮/可移动

        lay = QVBoxLayout(self)
        tip = QLabel(f"插件：{name}\n参数在插件自带界面中调整；此窗口触发处理。")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        lay.addStretch(1)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        run_btn = QPushButton("处理选区/整段")
        run_btn.clicked.connect(self._run)
        btn_row.addWidget(run_btn)
        lay.addLayout(btn_row)

    def _run(self):
        if self._main is not None:
            self._main._vst_run_path(self._plugin_path)

    def closeEvent(self, e):
        if self._main is not None:
            self._main._close_vst_window(self)
        super().closeEvent(e)
