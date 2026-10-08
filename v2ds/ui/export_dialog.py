# coding=utf-8
"""导出设置弹窗：工程语言、按语言分文件夹、导出路径。"""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QLineEdit, QPushButton, QFileDialog,
)
from v2ds.core.config import LANG_MAP, LANG_CODES


class ExportDialog(QDialog):
    def __init__(self, default_lang: str = "zh", default_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出设置")
        self.setMinimumWidth(480)

        lay = QVBoxLayout(self)

        # 工程导出语言
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("工程导出语言："))
        self.lang_combo = QComboBox()
        for code in LANG_CODES:
            self.lang_combo.addItem(LANG_MAP[code], code)
        idx = self.lang_combo.findData(default_lang)
        self.lang_combo.setCurrentIndex(max(0, idx))
        row1.addWidget(self.lang_combo)
        row1.addStretch(1)
        lay.addLayout(row1)

        # 分开导出
        self.chk_split = QCheckBox("按标记语言分文件夹导出（zh/ja/en/yue 各一个子目录）")
        self.chk_split.setChecked(False)
        lay.addWidget(self.chk_split)

        lay.addWidget(QLabel("· DiffSinger 数据集要求同一语言；勾选后不同语言标记分到不同子数据集"))

        # 导出路径
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("导出路径："))
        self.path_edit = QLineEdit(default_path)
        self.path_edit.setPlaceholderText("默认：当前工程/data_out")
        path_row.addWidget(self.path_edit, 1)
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        lay.addLayout(path_row)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("确认")
        ok.clicked.connect(self._apply)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

        self.export_path = ""
        self.split_by_lang = False
        self.dataset_lang = "zh"

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", self.path_edit.text())
        if d:
            self.path_edit.setText(d)

    def _apply(self):
        self.dataset_lang = str(self.lang_combo.currentData())
        self.split_by_lang = bool(self.chk_split.isChecked())
        self.export_path = self.path_edit.text().strip()
        self.accept()
