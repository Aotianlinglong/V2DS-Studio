# coding=utf-8
"""软件设置对话框：左侧分类导航 + 右侧详情页。确认后写入 config.json。

分类：通用设置 / 音素管理 / VST3 插件 / 快捷键说明 / PFML 操作说明。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
    QComboBox, QLabel, QMessageBox, QListWidget,
    QListWidgetItem, QCheckBox, QTableWidget, QTableWidgetItem,
    QStackedWidget, QWidget, QTextBrowser, QGroupBox,
)

from v2ds.core.config import Config, LANG_MAP, LANG_CODES, normalize_quick_phonemes


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("软件设置")
        self.resize(800, 580)

        root = QVBoxLayout(self)
        body = QHBoxLayout()

        # 左：分类导航
        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        self.nav.setStyleSheet(
            "QListWidget::item { padding: 8px 6px; }"
            "QListWidget::item:selected { background: #2d5f8a; }"
        )
        self.stack = QStackedWidget()
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        # 右：各分类详情页（顺序与左侧导航一一对应）
        self.nav.addItem("通用设置")
        self.stack.addWidget(self._page_general())
        self.nav.addItem("音素管理")
        self.stack.addWidget(self._page_phonemes())
        self.nav.addItem("VST3 插件")
        self.stack.addWidget(self._page_vst())
        self.nav.addItem("快捷键说明")
        self.stack.addWidget(self._page_shortcuts())
        self.nav.addItem("PFML 操作说明")
        self.stack.addWidget(self._page_pfml_guide())
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        # 底部按钮
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("确认")
        ok.clicked.connect(self._apply)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        root.addLayout(btns)

    # ---------------- 页面构建 ----------------
    def _page_general(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.lang_combo = QComboBox()
        for code in LANG_CODES:
            self.lang_combo.addItem(LANG_MAP[code], code)
        cur = self.config.get("default_lang", "zh")
        idx = self.lang_combo.findData(cur)
        self.lang_combo.setCurrentIndex(max(0, idx))
        form.addRow("新工程初始语言", self.lang_combo)
        form.addRow("", QLabel("· 仅作为新工程/旧工程未设置时的初始主语言；打开工程后，以文件列表上方的「工程主语言」为准"))
        lay.addLayout(form)

        self.chk_tooltips = QCheckBox("悬停时显示按钮说明")
        self.chk_tooltips.setChecked(bool(self.config.get("show_tooltips", True)))
        lay.addWidget(self.chk_tooltips)
        lay.addStretch(1)
        return w

    def _page_phonemes(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # 呼吸检测：EP 判别开关（关闭时句尾呼吸改插后置 AP）
        self.chk_breath_ep = QCheckBox("开启EP判别（呼吸检测在句尾插入 EP；关闭则改为后置 AP）")
        self.chk_breath_ep.setChecked(bool(self.config.get("breath_insert_ep", True)))
        lay.addWidget(self.chk_breath_ep)

        # ---- 内置特殊音素可见性（固定音素） ----
        from v2ds.core.config import BUILTIN_PHONEMES, normalize_builtin_visibility
        box = QGroupBox("内置特殊音素（写死定义；勾选=在右键菜单显示）")
        box_lay = QVBoxLayout(box)
        vis = normalize_builtin_visibility(self.config.get("builtin_phoneme_visibility"))
        self.builtin_checks = {}
        # 网格排布，避免音素多时一行放不下
        row_lay = None
        for i, sym in enumerate(BUILTIN_PHONEMES):
            if i % 5 == 0:
                row_lay = QHBoxLayout()
                box_lay.addLayout(row_lay)
            side_txt = "后置" if BUILTIN_PHONEMES[sym]["side"] == "after" else "前置"
            label = sym if "（" in sym else f"{sym}（{side_txt}）"
            chk = QCheckBox(label)
            chk.setChecked(vis.get(sym, False))
            self.builtin_checks[sym] = chk
            row_lay.addWidget(chk)
        if row_lay is not None:
            row_lay.addStretch(1)
        lay.addWidget(box)

        # ---- 自定义快捷插入音素面板 ----
        lay.addWidget(QLabel(
            "自定义插入音素（带语言属性；前置=插在被点格之前，后置=紧随其后；"
            "SP/AP/EP/GS/AP（后置）/ja/cl 为内置，不可在此添加）"))
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
        lay.addStretch(1)
        return w

    def _page_vst(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("VST3 插件（扫描后勾选要启用的，确认后右侧才显示）"))
        scan_row = QHBoxLayout()
        scan_btn = QPushButton("扫描 VST3 插件")
        scan_btn.clicked.connect(self._scan_vsts)
        scan_row.addWidget(scan_btn)
        scan_row.addStretch(1)
        lay.addLayout(scan_row)
        self.vst_list = QListWidget()
        self.vst_list.setStyleSheet(
            "QListWidget::item { border: 1px solid #666; border-radius: 3px; padding: 3px; margin: 1px; }"
            "QListWidget::item:selected { background: #3a3d45; }"
        )
        lay.addWidget(self.vst_list)
        self._scan_vsts()  # 打开即扫描
        return w

    def _page_shortcuts(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("快捷键说明（固定，不可修改）"))
        self.shortcut_table = QTableWidget(0, 2)
        self.shortcut_table.setHorizontalHeaderLabels(["按键", "功能"])
        self.shortcut_table.verticalHeader().setVisible(False)
        self.shortcut_table.horizontalHeader().setStretchLastSection(True)
        self.shortcut_table.setColumnWidth(0, 130)
        self.shortcut_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.shortcut_table.setSelectionMode(QTableWidget.NoSelection)
        self.shortcut_table.setFocusPolicy(Qt.NoFocus)
        shortcuts = [
            ("空格", "播放 / 暂停选中片段"),
            ("Shift + Q", "播放加速 0.1 倍（0.5~2.0，音调不变）"),
            ("Shift + W", "播放减速 0.1 倍（0.5~2.0，音调不变）"),
            ("M", "按当前框选区域新建标记"),
            ("W / S", "选中上一条 / 下一条标记"),
            ("↑ / ↓", "标记列表上移 / 下移"),
            ("Delete", "删除选中标记"),
            ("Ctrl + G", "合并选中标记"),
            ("X", "删除选定的音频区域"),
            ("Z", "选区静音"),
            ("C", "在播放头位置插入静音"),
            ("Ctrl + S", "保存工程"),
            ("Ctrl + Z", "撤销"),
            ("Ctrl + Shift + Z", "重做"),
        ]
        for key, desc in shortcuts:
            r = self.shortcut_table.rowCount()
            self.shortcut_table.insertRow(r)
            self.shortcut_table.setItem(r, 0, QTableWidgetItem(key))
            self.shortcut_table.setItem(r, 1, QTableWidgetItem(desc))
        lay.addWidget(self.shortcut_table)
        return w

    def _page_pfml_guide(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        brow = QTextBrowser()
        brow.setOpenExternalLinks(False)
        brow.setStyleSheet("QTextBrowser { background:#1e1e1e; color:#ddd; }")
        brow.setHtml(_PFML_GUIDE_HTML)
        lay.addWidget(brow)
        return w

    # ---------------- 逻辑 ----------------
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
        from fx.vst_scan import scan
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
        from v2ds.core.config import BUILTIN_PHONEMES as _BP
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


_PFML_GUIDE_HTML = """
<style>
  h3 { color:#7fd0c0; margin:14px 0 6px 0; }
  b  { color:#e8c07a; }
  li { margin:3px 0; }
  .k { color:#7fb0e0; }
</style>
<p>PFML 字符条是歌词标注视图：每个字/词是一个方块，方块下方显示读音。</p>

<h3>工具条按钮（歌词下方「PFML:」栏）</h3>
<ul>
  <li><b>发音</b>：拖选字符后点此，选词格候选读音。</li>
  <li><b>显示多音</b>：开关多音字/多音词的橙色横杠标记。</li>
  <li><b>插音素</b>：在光标处按前置/后置规则插入音素。</li>
  <li><b>手动分词</b>：把拖选区间合并成一个词格。</li>
  <li><b>设置语言</b>：为选中区间设置语言，方块底色随之变化。</li>
  <li><b>语言图例</b>：各色块对应的语言底色。</li>
  <li><b>清除标注</b>：清除选中区间的词/发音标注。</li>
  <li><b>自动分词</b>：按语言自动切格（会清空旧的词/发音标注）。</li>
</ul>

<h3>鼠标操作</h3>
<ul>
  <li><span class="k">左键</span>：单击选中方块；拖拽框选一段；长按方块后拖到目标读音松手即选定。</li>
  <li><span class="k">中键</span>：点「+xx」插入格取消该音素；点读音行取消发音；点方块本体取消分词。</li>
  <li><span class="k">右键方块</span>：菜单含<b>插入音素</b>、<b>设置语言</b>（含「取消语言」）、<b>手动分词</b>三组。</li>
  <li><span class="k">右键「+xx」插入格</span>：设置该插入音素的语言。</li>
</ul>

<h3>方块的视觉含义</h3>
<ul>
  <li>方块<b>底色</b> = 所属语言（对照语言图例）。</li>
  <li>顶部<b>彩色细带</b> = 2 字以上的词格（同格同色、异格异色）。</li>
  <li>底部<b>橙色横杠</b> = 多音字/多音词（勾选「显示多音」后出现）。</li>
  <li>「<b>+xx</b>」小格 = 插入音素（前置/后置按插入位置排布）。</li>
</ul>
"""