# coding=utf-8
"""软件设置对话框：工程默认语言 + 音素/VST 管理。确认后写入 config.json。"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QPushButton,
    QComboBox, QHBoxLayout, QLabel, QMessageBox, QListWidget,
    QListWidgetItem, QCheckBox, QTableWidget, QTableWidgetItem,
)

from config import Config, LANG_MAP, LANG_CODES, normalize_quick_phonemes


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("软件设置")
        self.setMinimumWidth(560)

        lay = QVBoxLayout(self)
        form = QFormLayout()

        # 默认语言
        self.lang_combo = QComboBox()
        for code in LANG_CODES:
            self.lang_combo.addItem(LANG_MAP[code], code)
        cur = config.get("default_lang", "zh")
        idx = self.lang_combo.findData(cur)
        self.lang_combo.setCurrentIndex(max(0, idx))
        form.addRow("工程默认语言", self.lang_combo)

        form.addRow("", QLabel("· 默认语言作用于新标记；单个标记可在列表中单独改"))

        lay.addLayout(form)

        # 呼吸检测：EP 判别开关（关闭时句尾呼吸改插后置 AP）
        self.chk_breath_ep = QCheckBox("开启EP判别（呼吸检测在句尾插入 EP；关闭则改为后置 AP）")
        self.chk_breath_ep.setChecked(bool(self.config.get("breath_insert_ep", True)))
        lay.addWidget(self.chk_breath_ep)

        # ---- 内置特殊音素可见性（固定音素） ----
        from config import BUILTIN_PHONEMES, normalize_builtin_visibility
        lay.addWidget(QLabel("内置特殊音素（写死定义；勾选=在右键菜单显示）"))
        vis = normalize_builtin_visibility(self.config.get("builtin_phoneme_visibility"))
        self.builtin_checks = {}
        builtin_row = QHBoxLayout()
        for sym in BUILTIN_PHONEMES:
            side_txt = "后置" if BUILTIN_PHONEMES[sym]["side"] == "after" else "前置"
            label = sym if "（" in sym else f"{sym}（{side_txt}）"
            chk = QCheckBox(label)
            chk.setChecked(vis.get(sym, False))
            self.builtin_checks[sym] = chk
            builtin_row.addWidget(chk)
        builtin_row.addStretch(1)
        lay.addLayout(builtin_row)

        # ---- 自定义快捷插入音素面板 ----
        lay.addWidget(QLabel("自定义插入音素（带语言属性；前置=插在被点格之前，后置=紧随其后；SP/AP/EP/GS/AP（后置）/ja/cl 为内置，不可在此添加）"))
        self.quick_table = QTableWidget(0, 3)
        self.quick_table.setHorizontalHeaderLabels(["音素", "插入方式", ""])
        self.quick_table.horizontalHeader().setStretchLastSection(False)
        self.quick_table.setColumnWidth(0, 140)
        self.quick_table.setColumnWidth(1, 110)
        self.quick_table.setColumnWidth(2, 60)
        self.quick_table.setMaximumHeight(180)
        lay.addWidget(self.quick_table)
        add_row_btn = QPushButton("添加音素")
        add_row_btn.clicked.connect(lambda: self._add_quick_row("", "before"))
        lay.addWidget(add_row_btn)
        for it in normalize_quick_phonemes(self.config.get("quick_phonemes")):
            self._add_quick_row(it["symbol"], it["side"])

        # ---- VST3 管理：扫描 + 勾选启用 ----
        lay.addWidget(QLabel("VST3 插件（扫描后勾选要启用的，确认后右侧才显示）"))
        scan_row = QHBoxLayout()
        scan_btn = QPushButton("扫描 VST3 插件")
        scan_btn.clicked.connect(self._scan_vsts)
        scan_row.addWidget(scan_btn)
        scan_row.addStretch(1)
        lay.addLayout(scan_row)
        self.vst_list = QListWidget()
        self.vst_list.setMaximumHeight(200)
        self.vst_list.setStyleSheet(
            "QListWidget::item { border: 1px solid #666; border-radius: 3px; padding: 3px; margin: 1px; }"
            "QListWidget::item:selected { background: #3a3d45; }"
        )
        lay.addWidget(self.vst_list)
        self._scan_vsts()  # 打开即扫描

        # 悬停按钮显示说明
        self.chk_tooltips = QCheckBox("悬停时显示按钮说明")
        self.chk_tooltips.setChecked(bool(self.config.get("show_tooltips", True)))
        lay.addWidget(self.chk_tooltips)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("确认")
        ok.clicked.connect(self._apply)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

    def _add_quick_row(self, symbol: str, side: str):
        r = self.quick_table.rowCount()
        self.quick_table.insertRow(r)
        self.quick_table.setItem(r, 0, QTableWidgetItem(symbol))
        side_combo = QComboBox()
        side_combo.addItem("前置", "before")
        side_combo.addItem("后置", "after")
        side_combo.setCurrentIndex(1 if side == "after" else 0)
        self.quick_table.setCellWidget(r, 1, side_combo)
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(lambda _=False, row=r: self._del_quick_row(row))
        self.quick_table.setCellWidget(r, 2, del_btn)

    def _del_quick_row(self, row: int):
        # 删除按钮绑定的是创建时的行号，行号会变，按按钮定位当前行
        btn = self.sender()
        if btn is not None:
            idx = self.quick_table.indexAt(btn.pos())
            if idx.isValid():
                row = idx.row()
        if 0 <= row < self.quick_table.rowCount():
            self.quick_table.removeRow(row)

    def _collect_quick_phonemes(self) -> list[dict]:
        out = []
        for r in range(self.quick_table.rowCount()):
            it = self.quick_table.item(r, 0)
            sym = (it.text() if it else "").strip()
            if not sym:
                continue
            combo = self.quick_table.cellWidget(r, 1)
            side = str(combo.currentData()) if combo else "before"
            out.append({"symbol": sym, "side": side})
        return out

    def _scan_vsts(self):
        from vst_scan import scan
        enabled = set(self.config.get("enabled_vsts", []))
        self.vst_list.clear()
        try:
            items = scan()
        except Exception as e:
            QMessageBox.warning(self, "扫描失败", str(e))
            return
        for name, path in items:
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, path)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if path in enabled else Qt.Unchecked)
            it.setToolTip(path)
            self.vst_list.addItem(it)

    def _apply(self):
        self.config["default_lang"] = str(self.lang_combo.currentData())
        # 收集勾选启用的 VST
        enabled = []
        for i in range(self.vst_list.count()):
            it = self.vst_list.item(i)
            if it.checkState() == Qt.Checked:
                enabled.append(it.data(Qt.UserRole))
        self.config["enabled_vsts"] = enabled
        # 内置音素可见性
        self.config["builtin_phoneme_visibility"] = {
            sym: bool(chk.isChecked()) for sym, chk in self.builtin_checks.items()}
        self.config["breath_insert_ep"] = bool(self.chk_breath_ep.isChecked())
        # 自定义快捷音素（剔除与内置重名的行）
        from config import BUILTIN_PHONEMES as _BP
        quick = [it for it in self._collect_quick_phonemes()
                 if it["symbol"] not in _BP]
        self.config["quick_phonemes"] = quick or [
            {"symbol": "cl", "side": "before"},
            {"symbol": "n", "side": "before"},
        ]
        # 按钮悬停说明
        self.config["show_tooltips"] = bool(self.chk_tooltips.isChecked())
        try:
            self.config.save()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"配置保存失败：{e}")
            return
        self.accept()
