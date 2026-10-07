# coding=utf-8
"""V2DS Studio (Vocal to Dataset Studio) 主窗口：切片 + Lab 一体化标注软件。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QKeySequence, QColor, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView,
    QAbstractItemView, QTextEdit, QLabel, QSlider, QPushButton, QFileDialog,
    QMessageBox, QProgressDialog, QMenuBar, QMenu, QAction, QInputDialog,
    QToolBar, QShortcut, QLineEdit, QPlainTextEdit, QScrollArea,
    QApplication, QCheckBox,
)

from config import Config, LANG_MAP, LANG_CODES
from project import Project, Mark
from pfml import remap_pfml
from pfml_strip import PfmlCharStrip, LANG_BG, PronunciationPopup
from waveform_view import AudioDisplayState, WaveformView
from audio_loader import AudioLoader
from asr_engine import ASREngine
from asr_worker import ASRWorker
from asr_service import ASRService
from dialogs import SettingsDialog
from player import AudioPlayer

AUTOSAVE_MS = 5 * 60 * 1000   # 5 分钟自动保存


class MainWindow(QMainWindow):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.project = Project()
        self.state = AudioDisplayState()
        self.current_wav: str = ""

        self.asr_engine: ASREngine | None = None
        self._asr_worker: ASRWorker | None = None
        self._asr_service: ASRService | None = None
        self._asr_load_thread: ASRWorker | None = None
        self._audio_loader: AudioLoader | None = None

        self._loading_text_blocked = False
        self._edit_mark_idx = -1   # 当前编辑的标记索引（即使取消选中也保留）
        self._text_undo_pushed = False  # 文本编辑是否已入栈（合并连续编辑为一步）
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave)
        self._vst_windows = []

        self._init_ui()
        self._init_shortcuts()
        self._init_player()
        self._apply_window_geometry()

    # ---------------- UI ----------------
    def _init_ui(self):
        self.setWindowTitle("V2DS Studio — 切片与 Lab 标注")
        self._build_menu()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        # 主体：左文件列表 | 右（顶部工具行 + 波形频谱 + 下标记/文本）
        main_split = QSplitter(Qt.Horizontal)
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(160)
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)
        main_split.addWidget(self.file_list)

        right = QSplitter(Qt.Vertical)

        # 顶部工具行：时间 + 全局进度条 + 播放（横跨右侧编辑区，参考截图）
        tool = QWidget()
        tl = QVBoxLayout(tool)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        row = QHBoxLayout()
        self.pos_label = QLabel("0:00.000 / 0:00.000")
        self.pos_label.setMinimumWidth(175)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(260)
        self.device_combo.setToolTip("选择音频输出设备（无声时切换此项）")
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.btn_play = QPushButton("播放")
        self.btn_play.setToolTip("空格：播放/暂停选中片段")
        self.btn_play.clicked.connect(self._toggle_play_selected)
        row.addWidget(self.pos_label)
        row.addWidget(self.device_combo, 1)
        row.addWidget(self.btn_play)
        # 导出目录（界面显示，可修改；留空则用当前工程/data_out）
        row.addWidget(QLabel("导出目录:"))
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("默认：当前工程/data_out")
        self.export_dir_edit.setMinimumWidth(200)
        self.export_dir_edit.setToolTip("切片导出目标文件夹；留空则导出到当前工程下的 data_out")
        self.export_dir_edit.editingFinished.connect(self._save_export_dir)
        self.export_dir_edit.setText(self.config.get("export_dir", ""))
        row.addWidget(self.export_dir_edit, 1)
        btn_dir = QPushButton("…")
        btn_dir.setFixedWidth(30)
        btn_dir.setToolTip("选择导出目录")
        btn_dir.clicked.connect(self._pick_export_dir)
        row.addWidget(btn_dir)
        tl.addLayout(row)
        right.addWidget(tool)

        self.wave_view = WaveformView(self.state)
        right.addWidget(self.wave_view)

        bottom = QSplitter(Qt.Horizontal)
        # ---- 标记面板 ----
        mark_panel = QWidget()
        mp_lay = QVBoxLayout(mark_panel)
        mp_lay.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("标记列表"))
        head.addStretch(1)
        self.btn_autoslice = QPushButton("自动切分")
        self.btn_autoslice.setToolTip("按静音自动切分整段音频为标记")
        self.btn_autoslice.clicked.connect(self.auto_slice)
        head.addWidget(self.btn_autoslice)
        self.btn_asr = QPushButton("运行 ASR")
        self.btn_asr.setToolTip("按已标记片段逐段识别，写入文本（只识别选中的片段，未选中则识别全部）")
        self.btn_asr.clicked.connect(self._run_asr)
        head.addWidget(self.btn_asr)
        self.btn_tifa = QPushButton("TIFA选音")
        self.btn_tifa.setToolTip("用 tifa.cpp 对齐器按音频判断多音字/多音词读音并自动标音"
                                 "（只处理选中的片段，未选中则处理全部）")
        self.btn_tifa.clicked.connect(self._run_tifa)
        head.addWidget(self.btn_tifa)
        self.btn_breath = QPushButton("呼吸检测")
        self.btn_breath.setToolTip("对齐取词边界 + FBL 呼吸检测，把 AP/EP 插入对应字符位置"
                                   "（只处理选中的片段，未选中则处理全部；重跑自动清除上次结果）")
        self.btn_breath.clicked.connect(self._run_breath)
        head.addWidget(self.btn_breath)
        self.btn_asr_unload = QPushButton("卸载ASR模型")
        self.btn_asr_unload.setToolTip("释放 ASR 模型内存")
        self.btn_asr_unload.clicked.connect(self._asr_unload_model)
        self.btn_asr_unload.setEnabled(False)
        head.addWidget(self.btn_asr_unload)
        mp_lay.addLayout(head)
        self.mark_table = QTableWidget(0, 5)
        self.mark_table.setHorizontalHeaderLabels(["#", "起始(s)", "结束(s)", "时长(s)", "语言"])
        self.mark_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mark_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mark_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mark_table.verticalHeader().setVisible(False)
        self.mark_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mark_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.mark_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self.mark_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.mark_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.mark_table.itemSelectionChanged.connect(self._on_mark_selected)
        self.mark_table.cellClicked.connect(self._on_mark_cell_clicked)
        mp_lay.addWidget(self.mark_table)
        bottom.addWidget(mark_panel)

        # ---- 文本编辑区 ----
        text_panel = QWidget()
        tp_lay = QVBoxLayout(text_panel)
        tp_lay.setContentsMargins(4, 0, 0, 0)
        tp_lay.addWidget(QLabel("歌词文本（选中标记后编辑）"))
        # ---- PFML 工具条：在歌词里拖选字符后，点这些按钮做标注 ----
        pfml_bar = QHBoxLayout()
        pfml_bar.addWidget(QLabel("PFML:"))
        self.btn_pfml_pron = QPushButton("发音")
        self.btn_pfml_pron.setToolTip("给选中区间指定发音（弹出 G2P 候选）")
        self.btn_pfml_pron.clicked.connect(self._pfml_set_pron)
        pfml_bar.addWidget(self.btn_pfml_pron)
        self.chk_poly = QCheckBox("显示多音")
        self.chk_poly.setToolTip("勾选后在 PFML 方块下方用橙色横杠标出多音：中/粤按字、日/英按词；全局常态生效并记住")
        self.chk_poly.setChecked(bool(self.config.get("show_polyphonic", False)))
        self.chk_poly.toggled.connect(self._on_poly_toggled)
        pfml_bar.addWidget(self.chk_poly)
        self.btn_pfml_ins = QPushButton("插音素")
        self.btn_pfml_ins.setToolTip("在光标处插入一个音素（鼻音/气声）")
        self.btn_pfml_ins.clicked.connect(self._pfml_insert_phoneme)
        pfml_bar.addWidget(self.btn_pfml_ins)
        self.btn_pfml_word = QPushButton("手动分词")
        self.btn_pfml_word.setToolTip("把拖选区间标记为一个词/组（青条）；中键点击方格本体可取消分词")
        self.btn_pfml_word.clicked.connect(self._pfml_mark_word)
        pfml_bar.addWidget(self.btn_pfml_word)
        pfml_bar.addWidget(QLabel("语言:"))
        self.btn_pfml_setlang = QPushButton("设置语言")
        self.btn_pfml_setlang.setToolTip("选中方块后点此，弹出菜单选择语言（嵌套 scope）")
        self.btn_pfml_setlang.clicked.connect(self._pfml_set_lang)
        pfml_bar.addWidget(self.btn_pfml_setlang)
        # 语言颜色图例（色块 + 名称）
        for code in LANG_CODES:
            c = LANG_BG.get(code)
            if c is None:
                continue
            sw = QLabel()
            sw.setFixedSize(12, 12)
            sw.setStyleSheet(f"background:{c.name()}; border:1px solid #555;")
            sw.setToolTip(LANG_MAP[code])
            pfml_bar.addWidget(sw)
            lb = QLabel(LANG_MAP[code])
            lb.setToolTip(LANG_MAP[code])
            pfml_bar.addWidget(lb)
        self.btn_pfml_clear = QPushButton("清除标注")
        self.btn_pfml_clear.setToolTip("清除选中区间的词/发音标注")
        self.btn_pfml_clear.clicked.connect(self._pfml_clear)
        pfml_bar.addWidget(self.btn_pfml_clear)
        pfml_bar.addSpacing(8)
        # 分词工具：自动分词（按语言切格）+ 手动分词（把选中区间合并成词格；中键取消）
        self.btn_ja_tok = QPushButton("自动分词")
        self.btn_ja_tok.setToolTip("按语言自动切分格子：日语用分词器整段重分，中文/英文重置为逐字/逐词（清空旧标注）")
        self.btn_ja_tok.clicked.connect(self._on_auto_tokenize)
        pfml_bar.addWidget(self.btn_ja_tok)
        pfml_bar.addStretch(1)
        tp_lay.addLayout(pfml_bar)
        # ---- PFML 字符条：自动换行，随窗口宽度伸缩 ----
        self.pfml_strip = PfmlCharStrip()
        self.pfml_strip.selectionChanged.connect(self._on_strip_sel)
        self.pfml_strip.longPressRequested.connect(self._on_long_press)
        self.pfml_strip.middleClicked.connect(self._on_strip_middle)
        self.pfml_strip.readingMiddleClicked.connect(self._on_strip_reading_middle)
        self.pfml_strip.insertAtRequested.connect(self._on_strip_rightclick)
        self.pfml_strip.insRightClicked.connect(self._on_strip_ins_rightclick)
        self.pfml_strip.insertDoubleClicked.connect(self._on_strip_ins_dbl)
        self.pfml_strip.insSelected.connect(lambda pos: setattr(self, "_sel_ins_pos", pos))
        tp_lay.addWidget(self.pfml_strip)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("选中左侧标记后，在此编辑/核对歌词；拖选字符后用上方 PFML 按钮标注。")
        self.text_edit.textChanged.connect(self._on_text_changed)
        self.text_edit.installEventFilter(self)
        tp_lay.addWidget(self.text_edit)
        bottom.addWidget(text_panel)

        # ---- 歌词 Match 面板（纵向一列，在 VST 左边）----
        self.match_panel = QWidget()
        match_panel = self.match_panel
        mp_lay = QVBoxLayout(match_panel)
        mp_lay.setContentsMargins(4, 0, 0, 0)
        mp_lay.addWidget(QLabel("歌词 Match"))
        self.match_edit = QPlainTextEdit()
        self.match_edit.setPlaceholderText("粘贴这首 wav 的完整歌词，点 Match 自动按时长分配到各切片")
        mp_lay.addWidget(self.match_edit, 1)
        self.btn_match = QPushButton("Match 歌词")
        self.btn_match.clicked.connect(self._do_match_lyrics)
        mp_lay.addWidget(self.btn_match)
        bottom.addWidget(match_panel)

        # ---- VST3 面板（文本框右侧，只显示设置里勾选启用的）----
        self.vst_panel = QWidget()
        vst_panel = self.vst_panel
        vp_lay = QVBoxLayout(vst_panel)
        vp_lay.setContentsMargins(4, 0, 0, 0)
        vhead = QHBoxLayout()
        vhead.addWidget(QLabel("VST3（在设置里勾选启用）"))
        vhead.addStretch(1)
        vp_lay.addLayout(vhead)
        self.vst_list = QListWidget()
        self.vst_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.vst_list.itemClicked.connect(self._on_vst_clicked)
        vp_lay.addWidget(self.vst_list, 1)
        vbtn = QHBoxLayout()
        self.btn_vst_run = QPushButton("处理选区/整段")
        self.btn_vst_run.clicked.connect(self._vst_run)
        vbtn.addWidget(self.btn_vst_run)
        vp_lay.addLayout(vbtn)
        bottom.addWidget(vst_panel)
        bottom.setStretchFactor(0, 0)   # 标记列表固定
        bottom.setStretchFactor(1, 3)    # 歌词文本
        bottom.setStretchFactor(2, 2)    # Match
        bottom.setStretchFactor(3, 1)    # VST
        bottom.setChildrenCollapsible(False)
        self.bottom_split = bottom

        right.addWidget(bottom)
        right.setSizes([42, 400, 260])

        main_split.addWidget(right)
        main_split.setSizes([170, 830])
        root.addWidget(main_split, 1)

        self.reload_vst_panel()
        self._apply_tooltips()
        self.statusBar().showMessage("就绪")
        self._first_show = True

    def _apply_tooltips(self):
        """按设置开关显示/隐藏按钮的悬停说明。

        首次调用时把按钮原始 tooltip 记到动态属性，之后切换不丢原文。
        只作用于按钮类（QAbstractButton），不影响输入框/下拉框/菜单项。"""
        from PyQt5.QtWidgets import QAbstractButton
        enabled = bool(self.config.get("show_tooltips", True))
        for w in self.findChildren(QAbstractButton):
            orig = w.property("_lab_orig_tip")
            if orig is None:
                orig = w.toolTip()
                w.setProperty("_lab_orig_tip", orig)
            w.setToolTip(orig if enabled else "")

    def showEvent(self, e):
        super().showEvent(e)
        if self._first_show:
            self._first_show = False
            from PyQt5.QtCore import QTimer
            def _fix():
                saved = self.config.get("panel_sizes")
                if saved and len(saved) == 4:
                    self.bottom_split.setSizes([int(x) for x in saved])
                else:
                    self.bottom_split.setSizes([220, 500, 260, 180])
            QTimer.singleShot(100, _fix)

        # 信号
        self.wave_view.selChanged.connect(self._on_sel_changed)
        self.wave_view.playheadMoved.connect(self._on_playhead_moved)
        self.wave_view.markUpdated.connect(self._refresh_mark_table)
        self.wave_view.markEndpointDragging.connect(lambda _idx: self._push_undo())
        self.wave_view.playToggle.connect(self._toggle_play_selected)
        self.wave_view.canvasClicked.connect(self._clear_text_focus)
        self.wave_view.markClicked.connect(self._select_mark_by_index)
        # 点击输入框以外 → 取消输入焦点（应用级过滤器，让快捷键生效）
        QApplication.instance().installEventFilter(self)

    def _build_menu(self):
        bar = self.menuBar()
        m_file = bar.addMenu("文件")
        a = QAction("打开工作文件夹", self); a.triggered.connect(self.open_work_folder); m_file.addAction(a)
        a = QAction("打开工程文件…", self); a.triggered.connect(self.open_project); m_file.addAction(a)
        self.recent_menu = m_file.addMenu("最近工程")
        self._refresh_recent_menu()
        m_file.addSeparator()
        a = QAction("保存工程", self); a.setShortcut("Ctrl+S"); a.triggered.connect(self.save_project); m_file.addAction(a)
        a = QAction("另存工程为…", self); a.triggered.connect(self.save_project_as); m_file.addAction(a)
        m_file.addSeparator()
        a = QAction("导出切片…", self); a.triggered.connect(self.export_slices); m_file.addAction(a)
        a = QAction("退出", self); a.triggered.connect(self.close); m_file.addAction(a)
        m_edit = bar.addMenu("编辑")
        a = QAction("撤销", self); a.setShortcut("Ctrl+Z"); a.triggered.connect(self.undo); m_edit.addAction(a)
        a = QAction("重做", self); a.setShortcut("Ctrl+Shift+Z"); a.triggered.connect(self.redo); m_edit.addAction(a)
        m_edit.addSeparator()
        a = QAction("删除选定区域", self); a.setShortcut("X"); a.triggered.connect(self.delete_selection); m_edit.addAction(a)
        a = QAction("选区静音", self); a.setShortcut("Z"); a.triggered.connect(self.silence_selection); m_edit.addAction(a)
        a = QAction("在播放头插入静音…", self); a.setShortcut("C"); a.triggered.connect(self.insert_silence); m_edit.addAction(a)
        m_tool = bar.addMenu("工具")
        a = QAction("自动切分（按静音分段）…", self); a.triggered.connect(self.auto_slice)
        m_tool.addAction(a)
        m_tool.addSeparator()
        a = QAction("软件设置…", self); a.triggered.connect(self.open_settings); m_tool.addAction(a)

        # ---- 显示菜单：控制各面板开关 ----
        m_view = bar.addMenu("显示")
        self._view_actions = []
        for name, attr in [("歌词 Match 面板", "match_panel"),
                           ("VST3 面板", "vst_panel"),
                           ("标记列表", "left_panel")]:
            act = QAction(name, self, checkable=True)
            act.setChecked(True)
            act.triggered.connect(lambda checked, a=attr: self._toggle_panel(a, checked))
            m_view.addAction(act)
            self._view_actions.append((attr, act))

    def _toggle_panel(self, attr, visible):
        w = getattr(self, attr, None)
        if w is not None:
            w.setVisible(visible)

    def _init_shortcuts(self):
        QShortcut(QKeySequence("M"), self, activated=self._make_mark_from_sel)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected_marks)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_selected)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self._merge_selected_marks)

    def _init_player(self):
        self.player = AudioPlayer()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._tick_playback)
        self._play_timer.start()
        self._last_playhead_x = None
        self._undo_stack = []   # 撤销栈：[(wave副本, marks副本), ...]
        self._redo_stack = []
        self._populate_audio_devices()

    def _populate_audio_devices(self):
        """列出可用的音频输出设备（DirectSound 主声音 + WASAPI），默认选 Windows 主声音输出。"""
        try:
            import sounddevice as sd
            self.device_combo.blockSignals(True)
            self.device_combo.clear()
            self._device_index = []
            devs = sd.query_devices()
            preferred = None
            for i, d in enumerate(devs):
                if d["max_output_channels"] <= 0:
                    continue
                host = sd.query_hostapis(d["hostapi"])["name"]
                if host not in ("Windows WASAPI", "Windows DirectSound"):
                    continue
                label = f"{d['name']}  [{host}]"
                self._device_index.append(i)
                self.device_combo.addItem(label)
                if ("主声音" in d["name"]) or ("primary" in d["name"].lower()):
                    preferred = len(self._device_index) - 1
            self.device_combo.blockSignals(False)
            if preferred is not None:
                self.device_combo.setCurrentIndex(preferred)
                self._apply_device(preferred)
        except Exception as e:
            self.statusBar().showMessage(f"音频设备加载失败：{e}")

    def _on_device_changed(self, idx):
        if idx >= 0:
            self._apply_device(idx)

    def _apply_device(self, idx):
        if hasattr(self, "_device_index") and 0 <= idx < len(self._device_index):
            self.player.set_device(self._device_index[idx])
            self.statusBar().showMessage(f"音频输出：{self.device_combo.currentText()}")

    def _apply_window_geometry(self):
        self.resize(int(self.config.get("window_width", 1280)),
                    int(self.config.get("window_height", 820)))
        if self.config.get("window_maximized", False):
            self.showMaximized()

    # ---------------- 文件列表 / 工程 ----------------
    def open_work_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择工作文件夹")
        if not d:
            return
        self.project = Project()
        self.project.folder_path = d
        self._populate_file_list()
        self.statusBar().showMessage(f"已打开文件夹：{d}")

    def _populate_file_list(self):
        self.file_list.clear()
        for name in self.project.wav_names:
            self.file_list.addItem(QListWidgetItem(name))
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)

    def open_project(self):
        f, _ = QFileDialog.getOpenFileName(self, "打开工程文件", "", "V2DS 工程 (*.labproj)")
        if not f:
            return
        # 重置当前 wav，确保切换新工程时歌词面板随之刷新
        self.current_wav = ""
        self.match_edit.setPlainText("")
        try:
            self.project = Project.load(f)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"工程打开失败：{e}")
            return
        # 校验原始 wav
        missing = []
        for name in self.project.wav_names:
            if self.project.folder_path:
                p = Path(self.project.folder_path) / name
                if not p.is_file():
                    missing.append(name)
        self._populate_file_list()
        if missing:
            QMessageBox.warning(self, "提示",
                                "以下原始音频文件缺失（可能被移动/删除）：\n" + "\n".join(missing))
        self.statusBar().showMessage(f"已打开工程：{f}")
        self._add_recent(f)

    def save_project(self):
        # 若当前音频有未保存的编辑，先写回 wav（16bit，保留原采样率）
        if not self._flush_audio_edit():
            return
        # 保存当前 wav 的歌词原文到工程
        if self.current_wav:
            self.project.set_lyric(self.current_wav, self.match_edit.toPlainText())
        if self.project.project_file is None:
            return self.save_project_as()
        try:
            self.project.save(self.project.project_file)
            self.statusBar().showMessage(f"工程已保存：{self.project.project_file}")
            self._add_recent(self.project.project_file)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")

    def save_project_as(self):
        # 若当前音频有未保存的编辑，先写回 wav
        if not self._flush_audio_edit():
            return
        if not self.project.folder_path:
            default = str(Path.home() / "工程.labproj")
        else:
            default = str(Path(self.project.folder_path) / "工程.labproj")
        f, _ = QFileDialog.getSaveFileName(self, "保存工程", default, "V2DS 工程 (*.labproj)")
        if not f:
            return
        try:
            self.project.save(f)
            self.statusBar().showMessage(f"工程已保存：{f}")
            self._add_recent(f)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")

    def _add_recent(self, path):
        """把工程路径记入最近列表（最多 8 条），写入 config。"""
        rec = self.config.get("recent_projects", [])
        p = str(Path(path).resolve())
        if p in rec:
            rec.remove(p)
        rec.insert(0, p)
        self.config.data["recent_projects"] = rec[:8]
        try:
            self.config.save()
        except Exception:
            pass
        self._refresh_recent_menu()

    def _remove_recent(self, path):
        rec = self.config.get("recent_projects", [])
        if path in rec:
            rec.remove(path)
            self.config.data["recent_projects"] = rec
            try:
                self.config.save()
            except Exception:
                pass
            self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        self.recent_menu.clear()
        rec = self.config.get("recent_projects", [])
        for p in rec:
            a = QAction(Path(p).name, self)
            a.setToolTip(p)
            a.triggered.connect(lambda _=False, path=p: self._open_recent(path))
            self.recent_menu.addAction(a)
        if not rec:
            a = QAction("（无最近工程）", self)
            a.setEnabled(False)
            self.recent_menu.addAction(a)

    def _open_recent(self, path):
        if not Path(path).is_file():
            QMessageBox.warning(self, "提示", f"工程文件不存在：{path}")
            self._remove_recent(path)
            return
        try:
            self.project = Project.load(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"工程打开失败：{e}")
            return
        self._populate_file_list()
        self._add_recent(path)
        self.statusBar().showMessage(f"已打开工程：{path}")

    def _autosave(self):
        if self.project.dirty and self.project.project_file is not None:
            self.project.make_backup(self.project.project_file)
            try:
                self.project.save(self.project.project_file)
                self.statusBar().showMessage("已自动保存工程（5 分钟）")
            except Exception as e:
                QMessageBox.warning(self, "提示", f"自动保存失败：{e}")
        self._autosave_timer.start(AUTOSAVE_MS)

    def _mark_dirty(self):
        self.project.dirty = True
        self._autosave_timer.start(AUTOSAVE_MS)

    # ---------------- 音频加载 ----------------
    def _revert_file_selection(self):
        """文件切换前置校验失败时，把列表选中项回退到 current_wav，避免列表与内容脱节。"""
        from PyQt5.QtCore import QSignalBlocker
        with QSignalBlocker(self.file_list):
            self.file_list.clearSelection()
            for i in range(self.file_list.count()):
                if self.file_list.item(i).text() == self.current_wav:
                    self.file_list.setCurrentRow(i)
                    return

    def _on_file_selected(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        name = items[0].text()
        if not self.project.folder_path:
            self._revert_file_selection()
            return
        path = Path(self.project.folder_path) / name
        if not path.is_file():
            QMessageBox.warning(self, "提示", f"音频文件不存在：{path}")
            self._revert_file_selection()
            return
        # 同名重选：内容已对应该文件则跳过；否则不自愈性重刷，防止半切换状态下永久卡死
        if name == self.current_wav:
            if self.mark_table.rowCount() == len(self.project.marks_of(name)):
                return
        # 先保存当前 match 歌词到正在编辑的 wav
        if self.current_wav:
            self.project.set_lyric(self.current_wav, self.match_edit.toPlainText())
        # 播放停止等风险操作隔离：任何异常都不得阻断下面板数据切换
        try:
            self._stop_playback()
        except Exception:
            pass
        # ---- 数据切换集中完成（原子）：先更新状态，再刷新所有面板 ----
        self.current_wav = name
        self.state.marks = self.project.marks_of(name)
        self.state.selected_mark = -1
        self._edit_mark_idx = -1
        # 立即切换 marks（不等后台音频加载完），避免表格显示上一个文件的标记
        self._refresh_mark_table()
        # 切换歌词面板到新 wav 的歌词
        self.match_edit.setPlainText(self.project.lyric_of(name))
        # 后台加载音频（即使启动失败，面板内容也已经是新文件，不会脱节）
        self.statusBar().showMessage(f"加载音频：{name} …")
        try:
            self._audio_loader = AudioLoader(str(path), name, self)
            self._loading_name = name
            self._audio_loader.loaded.connect(self._on_audio_loaded)
            self._audio_loader.failed.connect(self._on_audio_failed)
            self._audio_loader.start()
        except Exception as ex:
            self._on_audio_failed(str(ex))

    def _on_audio_loaded(self, name, payload):
        # 只接受当前正在加载的 wav 的结果，旧 loader 完成直接丢弃
        if name != self.current_wav:
            return
        wave, sr, mel, mel_fps = payload
        self.state.set_audio(wave, sr, mel, mel_fps)
        self.player.load(wave, sr)
        self.state.marks = self.project.marks_of(self.current_wav)
        self._refresh_mark_table()
        self.wave_view.refresh()
        self.statusBar().showMessage(f"已加载：{self.current_wav}  {sr}Hz {self.state.duration:.1f}s")

    def _on_audio_failed(self, msg):
        self.state.wave = None
        self.wave_view.refresh()
        QMessageBox.warning(self, "提示", f"音频读取失败：{msg}")

    # ---------------- 标记 ----------------
    def _on_sel_changed(self, start, end):
        pass  # 选区状态在画布内维护，M 键读取

    def _make_mark_from_sel(self):
        st = self.state
        if not self.current_wav or not st.has_audio():
            return
        if st.sel_start < 0 or st.sel_end < 0:
            return
        if st.sel_end - st.sel_start < 0.05:
            return
        lang = self.config.get("default_lang", "zh")
        mark = Mark(st.sel_start, st.sel_end, "", lang)
        self._push_undo()
        self.project.add_mark(self.current_wav, mark)
        self.state.marks = self.project.marks_of(self.current_wav)
        self._refresh_mark_table()
        self._mark_dirty()
        st.sel_start = st.sel_end = -1.0
        # 建标后取消选中状态，播放头跳片段起点
        self.state.selected_mark = -1
        self.mark_table.clearSelection()
        st.play_pos = mark.start
        self._update_pos_label()
        self.wave_view.refresh()
        self.statusBar().showMessage(f"已新建标记 {mark.start:.3f}~{mark.end:.3f}，按空格播放")

    def _delete_selected_marks(self):
        rows = sorted({r.row() for r in self.mark_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._push_undo()
        self.project.remove_marks(self.current_wav, rows)
        self.state.marks = self.project.marks_of(self.current_wav)
        self._refresh_mark_table()
        self._mark_dirty()

    def _merge_selected_marks(self):
        """Ctrl+J：把选中的多个标记合并为一个（文本与 PFML 一起拼接）。"""
        marks = self.project.marks_of(self.current_wav)
        rows = sorted({r.row() for r in self.mark_table.selectedIndexes()})
        if len(rows) < 2:
            self.statusBar().showMessage("合并需要选中至少 2 个标记")
            return
        sel = [marks[i] for i in rows if 0 <= i < len(marks)]
        if len(sel) < 2:
            return
        sel.sort(key=lambda m: m.start)
        self._push_undo()
        # 文本拼接（非空段之间用空格连接），PFML 偏移与文本精确对齐：
        # 只累加非空段的长度，段间空格也只在两非空段之间才插入（占 1 偏移）
        merged_pfml = {"words": [], "overrides": [], "spans": []}
        new_text = ""
        offset = 0
        prev_nonempty = False
        for m in sel:
            t = m.text or ""
            if not t.strip():
                continue  # 空文本段不参与拼接，也不产生任何偏移
            if prev_nonempty:
                new_text += " "
                offset += 1  # 段间空格占一个位置
            base = offset
            new_text += t
            tlen = len(t)
            # 用位置版 pfml 做偏移累加（uid 版无法按位置加偏移）
            pf = m.pos_pfml()
            for kind in ("words", "overrides", "spans"):
                for it in pf.get(kind, []):
                    item = dict(it)
                    try:
                        item["begin"] = int(it["begin"]) + base
                        item["end"] = int(it["end"]) + base
                    except Exception:
                        continue
                    merged_pfml[kind].append(item)
            offset = base + tlen
            prev_nonempty = True
        # 新标记：start=最早, end=最晚, lang 取第一段；位置版 pfml 由 __init__ 转成 uid 版
        first = sel[0]
        new_mark = Mark(first.start, sel[-1].end, new_text, first.lang, merged_pfml)
        # 从原列表移除被合并的，插入新标记
        idxs = [marks.index(m) for m in sel]
        keep = [m for i, m in enumerate(marks) if i not in idxs]
        keep.append(new_mark)
        keep.sort(key=lambda m: m.start)
        self.project.set_marks(self.current_wav, keep)
        self.state.marks = keep
        self._refresh_mark_table()
        self._mark_dirty()
        self.statusBar().showMessage(
            f"已合并 {len(sel)} 个标记为 {new_mark.start:.3f}~{new_mark.end:.3f}")

    def _refresh_mark_table(self):
        self._loading_text_blocked = True
        self.mark_table.setRowCount(0)
        marks = self.state.marks if self.current_wav else []
        for i, m in enumerate(marks):
            self.mark_table.insertRow(i)
            self.mark_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.mark_table.setItem(i, 1, QTableWidgetItem(f"{m.start:.3f}"))
            self.mark_table.setItem(i, 2, QTableWidgetItem(f"{m.end:.3f}"))
            self.mark_table.setItem(i, 3, QTableWidgetItem(f"{m.duration:.3f}"))
            # Match 改动高亮：橙色（区别于选中蓝）
            if getattr(m, "_match_changed", False):
                bg = QColor("#7a5a1a")
                for col in range(4):
                    self.mark_table.item(i, col).setBackground(bg)
            combo = QComboBox()
            for code in LANG_CODES:
                combo.addItem(LANG_MAP[code], code)
            idx = combo.findData(m.lang)
            combo.setCurrentIndex(max(0, idx))
            combo.currentIndexChanged.connect(
                lambda _i, row=i: self._on_mark_lang_changed(row, _i)
            )
            self.mark_table.setCellWidget(i, 4, combo)
        self._loading_text_blocked = False
        # 刷新文本编辑区
        self._sync_text_edit()

    def _on_mark_lang_changed(self, row, combo_index):
        marks = self.project.marks_of(self.current_wav)
        if 0 <= row < len(marks):
            marks[row].lang = str(self.mark_table.cellWidget(row, 4).currentData())
            self._mark_dirty()

    def _on_mark_cell_clicked(self, row, col):
        marks = self.state.marks
        if not (0 <= row < len(marks)):
            return
        self.state.selected_mark = row
        m = marks[row]
        self._jump_to_mark(m)
        self._sync_text_edit()
        # 选择片段后播放头跳转片段起点（仅不在播放时）
        if not self.player.playing:
            self.state.play_pos = m.start
            self._update_pos_label()
        self.wave_view.refresh()

    def _on_mark_selected(self):
        rows = [r.row() for r in self.mark_table.selectedIndexes()]
        if rows:
            self.state.selected_mark = rows[0]
            self._edit_mark_idx = rows[0]   # 记住正在编辑的标记
            # 点进这个标记就消除 match 高亮（仅清背景色，不重建表格——避免冲掉选中状态）
            marks = self.project.marks_of(self.current_wav)
            if marks and 0 <= rows[0] < len(marks):
                mk = marks[rows[0]]
                if getattr(mk, "_match_changed", False):
                    mk._match_changed = False
                    for col in range(4):
                        it = self.mark_table.item(rows[0], col)
                        if it:
                            it.setBackground(QColor())
                # 片段中点居中画布（方向键/W/S 与鼠标点击行行为一致）
                self._jump_to_mark(mk)
                # 选择片段后播放头跳转片段起点（仅不在播放时）
                if not self.player.playing:
                    self.state.play_pos = mk.start
                    self._update_pos_label()
            self._sync_text_edit()
            self.wave_view.refresh()
        else:
            # 列表取消选中 → 同步取消画布选中（编辑标记保留，PFML 仍可编辑）
            if self.state.selected_mark != -1:
                self.state.selected_mark = -1
                self.wave_view.refresh()

    def _jump_to_mark(self, m: Mark):
        st = self.state
        if not st.has_audio():
            return
        vis = st.visible_seconds(self.wave_view.wave.width())
        # 片段中点对准视图中央，视图宽度保持面板当前宽度
        mid = (m.start + m.end) / 2.0
        st.view_start = mid - vis / 2.0
        st.clamp_view(self.wave_view.wave.width())
        self.wave_view.refresh()

    # ---------------- 文本编辑 ----------------
    def _sync_text_edit(self):
        marks = self.project.marks_of(self.current_wav)
        idx = self._edit_mark_idx
        if 0 <= idx < len(marks):
            text = marks[idx].text
            if self.text_edit.toPlainText() != text:
                self._loading_text_blocked = True
                self.text_edit.setPlainText(text)
                self._loading_text_blocked = False
            # 刷新字符条（传位置版，pfml_strip 只读渲染）
            self.pfml_strip.set_data(text, marks[idx].pos_pfml(),
                                     marks[idx].lang or "zh",
                                     self._strip_groups(marks[idx]),
                                     marks[idx].slots.uids,
                                     poly=self._poly_positions(marks[idx]))
        else:
            self._loading_text_blocked = True
            self.text_edit.clear()
            self._loading_text_blocked = False
            self.pfml_strip.set_data("", {}, "zh")
            self._pfml_anchor = (None, None)

    def _on_strip_sel(self, b, e):
        """字符条选中区间变化。"""
        self._pfml_sel = (b, e) if b >= 0 else None

    @staticmethod
    def _is_japanese_char(c: str) -> bool:
        """判断是否是日文字符（平假名/片假名/汉字/长音ー）。"""
        o = ord(c)
        return (0x3040 <= o <= 0x30FF       # 平假名+片假名
                or 0x4E00 <= o <= 0x9FFF     # 汉字
                or c == "ー" or c == "～")

    def _on_long_press(self, b, e, gx, gy):
        """长按方块：在方块下方弹出发音候选下拉，按住拖动、松手选定。"""
        self._show_pron_popup(b, e, gx, gy + self.pfml_strip.CELL + 4, grab=True)

    def _show_pron_popup(self, b, e, gx, gy, grab=False):
        """发音候选下拉。发音单位 = 该字所属的词格（分词/合并结果）。"""
        from g2p import list_candidates
        m = self._edit_mark()
        if m is None:
            return
        lang = self._lang_for_range(b, e, m)
        # 发音单位 = 该字所属的词格（分词器/手动合并的结果），统一各语言；单字格=单字
        unit = m.slots.unit_of_pos(b)
        if unit:
            b, e = unit
        self._pfml_sel = (b, e)
        sel_text = m.text[b:e]
        cands = list_candidates(sel_text, lang)
        rows = [(f"{c['script']}  →  {' '.join(c['phonemes'])}", ("pron", c["phonemes"]))
                for c in cands]
        rows.append(("手输音素…", ("manual", None)))
        popup = PronunciationPopup(self)
        popup.set_rows(rows)
        popup.selected.connect(
            lambda r, b_=b, e_=e, rows_=rows, popup_=popup:
                self._on_pron_pick(b_, e_, r, rows_, popup_))
        popup.open_at(gx, gy, grab=grab)
        self._pron_popup = popup

    def _on_pron_pick(self, b, e, row_idx, rows, popup):
        self.pfml_strip.cancel_long_press()
        self._pron_popup = None
        if row_idx < 0 or row_idx >= len(rows):
            popup.deleteLater()
            return  # 移出菜单松开 → 取消
        kind, payload = rows[row_idx][1]
        if kind == "manual":
            self._pfml_set_pron_manual(b, e)
        else:
            self._apply_pron(b, e, payload)
        popup.deleteLater()

    def _apply_pron(self, b, e, phonemes):
        """直接给 [b,e) 区间指定发音（候选选中后）。"""
        d = self._pfml_data()
        m = self._edit_mark()
        if not d or m is None:
            return
        self._push_undo()
        d["overrides"] = [o for o in d["overrides"]
                          if not (o["begin"] == b and o["end"] == e and not o.get("ins"))]
        d["overrides"].append({"begin": b, "end": e, "script": "", "phonemes": phonemes,
                               "key": m.text[b:e]})
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"发音：{' '.join(phonemes)}")
        self._pfml_refresh_strip()

    def _on_strip_ins_dbl(self, pos, sym="", key=None):
        """双击/中键插入音素格：只删除被点的那一格。

        key=(pos,k)：pos 位置上第 k 个插入音素（与 pfml_strip 布局顺序一致，
        同位置可挂多个同符号/前后置混合的插入，必须按条目身份精确删除，
        不能按 (pos, sym) 匹配——否则会一口气删掉同位置所有同符号插入）。
        """
        d = self._pfml_data()
        if not d:
            return
        here = [o for o in d["overrides"] if o["begin"] == pos and o["end"] == pos]
        target = None
        if key is not None:
            try:
                _p, k = key
                k = int(k)
            except (TypeError, ValueError):
                k = -1
            if 0 <= k < len(here):
                target = here[k]
        if target is None and sym:
            # 兜底（旧信号/异常索引）：按符号删一个
            target = next((o for o in here
                           if " ".join(o.get("phonemes", [])) == sym), None)
        if target is None:
            return
        self._push_undo()
        d["overrides"] = [o for o in d["overrides"] if o is not target]
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"删除了位置 {pos} 的插入音素 {sym}")
        self._pfml_refresh_strip()

    def _on_strip_ins_rightclick(self, key, global_pos):
        """右键插入音素格：独立小菜单，仅设置语言（4 语言 + 取消语言）。

        内置特殊音素（SP/AP/EP/GS）是无语言裸符号，菜单对其禁用并提示。
        key = (pos, k)：pos 位置第 k 个插入音素（与 pfml_strip 布局顺序一致）。
        """
        d = self._pfml_data()
        m = self._edit_mark()
        if not d or m is None or not key:
            return
        pos, k = key
        ins_list = [o for o in d["overrides"] if o["begin"] == pos and o["end"] == pos]
        if not (0 <= k < len(ins_list)):
            return
        item = ins_list[k]
        sym = " ".join(item.get("phonemes", []))
        from config import BUILTIN_PHONEMES
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtGui import QFont
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu {background:#2a2a2a; color:#eee;}"
            "QMenu::item:selected {background:#2d5f8a;}"
            "QMenu::item:disabled {color:#9b9b9b;}")
        title_font = QFont()
        title_font.setBold(True)
        t = menu.addAction(f"音素 {sym}：设置语言")
        t.setEnabled(False)
        t.setFont(title_font)
        is_builtin = sym in BUILTIN_PHONEMES
        if is_builtin:
            if BUILTIN_PHONEMES[sym].get("lang"):
                hint = menu.addAction("（内置音素，固定为 ja/cl，不可改语言）")
            else:
                hint = menu.addAction("（内置特殊音素，无语言属性）")
            hint.setEnabled(False)
        cur_lang = item.get("lang")
        for code in LANG_CODES:
            act = menu.addAction(LANG_MAP[code])
            act.setEnabled(not is_builtin)
            act.setCheckable(True)
            act.setChecked(cur_lang == code)
            act.triggered.connect(
                lambda _c=False, lang=code: self._set_ins_lang(key, lang))
        act = menu.addAction("取消语言")
        act.setEnabled(not is_builtin and cur_lang is not None)
        act.triggered.connect(lambda: self._set_ins_lang(key, None))
        menu.exec_(global_pos)

    def _set_ins_lang(self, key, lang):
        """设置/取消某个插入音素的语言属性。"""
        d = self._pfml_data()
        if not d or not key:
            return
        pos, k = key
        ins_list = [o for o in d["overrides"] if o["begin"] == pos and o["end"] == pos]
        if not (0 <= k < len(ins_list)):
            return
        self._push_undo()
        item = ins_list[k]
        if lang is None:
            item.pop("lang", None)
        else:
            item["lang"] = lang
        self._pfml_commit(d)
        self._mark_dirty()
        sym = " ".join(item.get("phonemes", []))
        self.statusBar().showMessage(
            f"音素 {sym} 语言 → {LANG_MAP.get(lang, '默认（标记主语言）') if lang else '默认（标记主语言）'}")
        self._pfml_refresh_strip()

    def _on_strip_rightclick(self, pos, global_pos):
        """字符条右键：纵向单列菜单。两个不可点击的分组标题（插入音素/设置语言），
        组内选项一次性全部展示；末尾一个可点击的「手动分词」单项。"""
        m = self._edit_mark()
        if m is None:
            return
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtGui import QFont
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu {background:#2a2a2a; color:#eee;}"
            "QMenu::item:selected {background:#2d5f8a;}"
            "QMenu::item:disabled {color:#9b9b9b;}")
        title_font = QFont()
        title_font.setBold(True)

        def add_title(text):
            act = menu.addAction(text)
            act.setEnabled(False)
            act.setFont(title_font)

        # ---- 分组：插入音素（在右键位置插入）----
        add_title("插入音素")
        from config import (normalize_quick_phonemes, normalize_builtin_visibility,
                            BUILTIN_PHONEMES)
        # 内置特殊音素（写死 side，可见性由设置控制）
        vis = normalize_builtin_visibility(self.config.get("builtin_phoneme_visibility"))
        for sym in BUILTIN_PHONEMES:
            if not vis.get(sym):
                continue
            side = BUILTIN_PHONEMES[sym]["side"]
            # 存储/导出的实际符号（如「AP（后置）」存 AP）；键名已含（后置）时不重复标注
            store = BUILTIN_PHONEMES[sym].get("symbol", sym)
            label = f"插入 {sym}"
            if side == "after" and "后置" not in sym:
                label += "（后置）"
            act = menu.addAction(label)
            act.triggered.connect(
                lambda _c=False, p=store, s=side: self._do_insert_phoneme(pos, p, s))
        # 自定义音素
        for it in normalize_quick_phonemes(self.config.get("quick_phonemes")):
            ph, side = it["symbol"], it["side"]
            act = menu.addAction(f"插入 {ph}" + ("（后置）" if side == "after" else ""))
            act.triggered.connect(
                lambda _c=False, p=ph, s=side: self._do_insert_phoneme(pos, p, s))
        menu.addSeparator()

        # ---- 分组：设置语言（作用于当前选中区间；无选中=右键处格子）----
        add_title("设置语言")
        r = self._pfml_selection_range()
        if not r:
            unit = m.slots.unit_of_pos(pos)
            if unit:
                r = unit
        for code in LANG_CODES:
            act = menu.addAction(LANG_MAP[code])
            act.setEnabled(r is not None)
            act.triggered.connect(lambda _c=False, lang=code: self._apply_span_lang(r, lang))
        act = menu.addAction("取消语言")
        act.setEnabled(r is not None)
        act.triggered.connect(lambda: self._apply_span_lang(r, None))
        menu.addSeparator()

        # ---- 单项：手动分词（作用于拖选区间）----
        act = menu.addAction("手动分词")
        act.triggered.connect(self._pfml_mark_word)

        menu.exec_(global_pos)

    def _do_insert_phoneme(self, pos, sym, side="before"):
        d = self._pfml_data()
        m = self._edit_mark()
        if not d or m is None:
            return
        self._push_undo()
        if sym is None:
            s, ok = QInputDialog.getText(self, "插入音素", "音素符号：")
            if not ok or not s.strip():
                self._undo_stack.pop()  # 取消则不占撤销位
                return
            sym = s.strip()
        # 后置：插在被点格的后面（位置=该格结尾，side 提示随位置版带入 uid 锚定）
        if side == "after":
            unit = m.slots.unit_of_pos(pos)
            if unit:
                pos = unit[1]
        # 插入音素的锚：取插入点附近 3 字符（前1+后1）做 key，消歧义；
        # 若该段文字被删，插入音素随之消失
        s0, s1, s2 = m.text[max(0, pos-1):pos], m.text[pos:pos+1], m.text[pos+1:pos+2]
        anchor = s0 + s1 + s2
        item = {"begin": pos, "end": pos,
                "script": "", "phonemes": [sym],
                "key": anchor, "ins": True}
        if side == "after":
            item["side"] = "after"
        d["overrides"].append(item)
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"在 {pos} 插入音素 {sym}")
        self._pfml_refresh_strip()

    def _on_text_changed(self):
        if self._loading_text_blocked:
            return
        marks = self.project.marks_of(self.current_wav)
        idx = self._edit_mark_idx
        if 0 <= idx < len(marks):
            # 文本编辑合并为一步撤销：从进入编辑起只入栈一次，直到失去焦点
            if not self._text_undo_pushed:
                self._push_undo()
                self._text_undo_pushed = True
            new_text = self.text_edit.toPlainText()
            m = marks[idx]
            # 更新文本并同步 uid 表；标注（uid 版）自动跟随，不现算编号
            m.set_text(new_text)
            # 防错：内容变化的方格，取消其发音/插入音素（避免留错标注）
            m.cancel_overrides_for(m.slots.last_changed)
            self._mark_dirty()
            # 实时同步字符条（uid 版标注映射回当前位置渲染）
            self.pfml_strip.set_data(
                m.text, m.pos_pfml(), m.lang or "zh", self._strip_groups(m),
                m.slots.uids, poly=self._poly_positions(m))

    # ---------------- 歌词 Match ----------------
    def _do_match_lyrics(self):
        """把用户贴的正确歌词，和各标记已有的 ASR text 对齐，填回正确歌词。"""
        import difflib
        marks = sorted(self.project.marks_of(self.current_wav), key=lambda m: m.start)
        if not marks:
            QMessageBox.warning(self, "提示", "当前 wav 没有标记")
            return
        # 对齐用去空白版本，但保留原文空格位置信息
        raw = self.match_edit.toPlainText().strip()
        if not raw:
            QMessageBox.warning(self, "提示", "先在上方粘贴整首正确歌词")
            return
        # user_nospace: 去空白后的对齐串；space_after[k] = user_nospace[k] 后面在 raw 里是否跟空格
        user_nospace = ""
        space_after = []   # space_after[k] = True 表示 user_nospace[k] 后应该有空格
        for idx, ch in enumerate(raw):
            if ch.isspace():
                continue
            user_nospace += ch
            # 看 raw 里 idx 后面是不是空格
            nxt = raw[idx + 1] if idx + 1 < len(raw) else ""
            space_after.append(bool(nxt and nxt.isspace()))
        user = user_nospace
        # 各标记现有 ASR text 拼接，建立 字符位置→标记下标 映射
        asr_full = ""
        char_to_mark = []   # asr_full[k] 属于 char_to_mark[k] 号标记
        for i, m in enumerate(marks):
            t = (m.text or "").strip()
            asr_full += t
            char_to_mark.extend([i] * len(t))
        if not asr_full:
            QMessageBox.warning(self, "提示", "各标记还没有 ASR 文本，先跑 ASR")
            return
        # 对齐 user（正确歌词）到 asr_full（ASR 结果）
        sm = difflib.SequenceMatcher(None, asr_full, user, autojunk=False)
        # result[i] = 标记 i 应填入的字符列表（从 user_nospace 取，带空格标记）
        result = {i: [] for i in range(len(marks))}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(j1, j2):
                    mk = char_to_mark[min(i1 + (k - j1), len(char_to_mark) - 1)]
                    ch = user[k]
                    if space_after[k]:
                        ch += " "
                    result[mk].append(ch)
            elif tag == "replace":
                n_asr = i2 - i1
                n_user = j2 - j1
                if n_asr == 0 or n_user == 0:
                    continue
                covered = set(char_to_mark[i1:i2])
                per = max(1, n_user // max(1, len(covered)))
                jj = j1
                for mk in sorted(covered):
                    chars = ""
                    for k in range(jj, min(jj + per, j2)):
                        c = user[k]
                        if space_after[k]:
                            c += " "
                        chars += c
                    result[mk].append(chars)
                    jj += per
            elif tag == "insert":
                if i1 < len(char_to_mark):
                    chars = ""
                    for k in range(j1, j2):
                        c = user[k]
                        if space_after[k]:
                            c += " "
                        chars += c
                    result[char_to_mark[min(i1, len(char_to_mark) - 1)]].append(chars)
        # 填回
        self._push_undo()
        changed = 0
        segs = ["".join(result[i]) for i in range(len(marks))]
        for i, m in enumerate(marks):
            new_text = segs[i]
            if new_text and new_text != m.text:
                # 整段替换：重切分发新 uid；日语自动分词；旧标注作废
                m.apply_matched_text(new_text)
                m._match_changed = True
                changed += 1
            else:
                m._match_changed = False
        self.project.dirty = True
        self._refresh_mark_table()
        self._sync_text_edit()
        self.statusBar().showMessage(f"Match 完成：{changed} 个标记歌词已对齐更新（橙色高亮）")


    def _pfml_selection_range(self):
        """返回当前选中的字符区间 (b,e)。优先字符条，其次 QTextEdit。"""
        # 用编辑中的标记（即使列表取消选中也保留），与 _pfml_data / 文本编辑一致
        marks = self.project.marks_of(self.current_wav)
        idx = self._edit_mark_idx
        if not marks or not (0 <= idx < len(marks)):
            return None
        if getattr(self, "_pfml_sel", None):
            return self._pfml_sel
        cur = self.text_edit.textCursor()
        if cur.selectionStart() != cur.selectionEnd():
            return (cur.selectionStart(), cur.selectionEnd())
        return None

    def _lang_for_range(self, b: int, e: int, m) -> str:
        """取区间 [b,e) 的 G2P 语言：
        有 span 标签 → 用 span 语言；无标签 → 跟随标记主语言。"""
        try:
            lang = self.pfml_strip._lang_at(b, e)
        except Exception:
            lang = None
        return lang or (m.lang or "zh")

    def _strip_groups(self, m):
        """返回跨 2+ 个字符的词格跨度 [(b,e),...]（任何语言：自动分词/手动合并的格子）。
        只有 >=2 字的词格才需要上色显示，单字格不上色。"""
        out = []
        for (b, e) in m.slots.slot_spans():
            if e - b >= 2:
                out.append((b, e))
        return out

    def _pfml_refresh_strip(self):
        marks = self.project.marks_of(self.current_wav)
        idx = self._edit_mark_idx
        if marks and 0 <= idx < len(marks):
            m = marks[idx]
            self.pfml_strip.set_data(m.text, m.pos_pfml(), m.lang or "zh",
                                     self._strip_groups(m), m.slots.uids,
                                     poly=self._poly_positions(m))

    def _on_auto_tokenize(self):
        """自动分词：按语言切分格子（日语=分词器；中/英=逐字/逐词），重置为自动分组。"""
        m = self._edit_mark()
        if m is None:
            self.statusBar().showMessage("自动分词：请先选中一条标记")
            return
        self._push_undo()
        m.slots.reset(m.text, m.lang)
        m.pfml = {}
        m._match_changed = True
        self._mark_dirty()
        self.statusBar().showMessage(f"已按 {m.lang or 'zh'} 自动分词（重置为自动分组，旧标注清空）")
        self._pfml_refresh_strip()

    def _on_strip_reading_middle(self, b, e):
        """中键点击方格下方读音行：只取消该处发音（不影响分词/语言标签）。"""
        d = self._pfml_data()
        if not d:
            return
        for o in list(d["overrides"]):
            if o["begin"] != o["end"] and o["begin"] <= b and e <= o["end"]:
                self._push_undo()
                d["overrides"].remove(o)
                self._pfml_commit(d)
                self._mark_dirty()
                self.statusBar().showMessage("已取消发音")
                self._pfml_refresh_strip()
                return
        self.statusBar().showMessage("该处没有可取消的读音")

    def _on_strip_middle(self, b, e):
        """中键点击方格本体：只取消该处分词（把词格拆回逐字）。

        发音、语言标签均不在此处理（语言标签由 Mark.ungroup_slot 保留）。"""
        m = self._edit_mark()
        if m is None:
            return
        unit = m.slots.unit_of_pos(b)
        if unit and (unit[1] - unit[0]) >= 2:
            self._push_undo()
            m.ungroup_slot(b)
            self._mark_dirty()
            self.statusBar().showMessage("已拆开该词格（取消分词）")
            self._pfml_refresh_strip()
            return
        self.statusBar().showMessage("该处不是词格，无需取消分词")

    def _pfml_data(self):
        """拿到当前编辑标记的位置版 pfml（begin/end），供交互函数读写。

        交互层基于位置（用户选中/点击的都是位置）。写入后须调 _pfml_commit()
        把位置版转回 uid 版存进 mark.pfml（mark 是唯一真相）。
        """
        marks = self.project.marks_of(self.current_wav)
        idx = self._edit_mark_idx
        if not marks or not (0 <= idx < len(marks)):
            return None
        m = marks[idx]
        pos = m.pos_pfml()
        pos.setdefault("words", [])
        pos.setdefault("overrides", [])
        pos.setdefault("spans", [])
        return pos

    def _pfml_commit(self, pos_pfml: dict):
        """把交互层操作后的位置版 pfml 转回 uid 版存进 mark.pfml。"""
        m = self._edit_mark()
        if m is None:
            return
        m.pfml = m.slots.to_uid_pfml(pos_pfml)

    def _pfml_mark_word(self):
        """手动分词：把拖选区间合并为一个词格（与自动分词同一格子系统）。
        中键点击该词格可取消（拆回逐字）。"""
        r = self._pfml_selection_range()
        m = self._edit_mark()
        if not r or m is None:
            self.statusBar().showMessage("先在歌词文本里拖选要合并的词")
            return
        b, e = r
        self._push_undo()
        if m.merge_slots(b, e):
            self._mark_dirty()
            self.statusBar().showMessage(f"已合并为一个词格 [{b},{e})")
        else:
            self.statusBar().showMessage("合并失败：范围不合法")
        self._pfml_refresh_strip()

    def _pfml_set_pron(self):
        from g2p import list_candidates
        r = self._pfml_selection_range()
        d = self._pfml_data()
        m = self._edit_mark()
        if not r or not d or m is None:
            self.statusBar().showMessage("先在歌词文本里拖选要指定发音的字")
            return
        b, e = r
        sel_text = m.text[b:e]
        self._push_undo()
        lang = self._lang_for_range(b, e, m)
        cands = list_candidates(sel_text, lang)
        items = [f"{c['script']}  →  {' '.join(c['phonemes'])}" for c in cands]
        items.append("（手输音素，空格分隔）")
        choice, ok = QInputDialog.getItem(self, f"指定发音：{sel_text}",
                                          "选候选：", items, 0, False)
        if not ok:
            return
        if choice.startswith("（手输"):
            manual, ok2 = QInputDialog.getText(self, "手写音素",
                                               "音素（空格分隔，如 ch ong）：")
            if not ok2:
                return
            phonemes = manual.split()
        else:
            phonemes = cands[items.index(choice)]["phonemes"]
        # 覆盖已有同区间 override
        d["overrides"] = [o for o in d["overrides"]
                          if not (o["begin"] == b and o["end"] == e and not o.get("ins"))]
        d["overrides"].append({"begin": b, "end": e,
                               "script": "", "phonemes": phonemes,
                               "key": m.text[b:e]})
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"发音已指定：{' '.join(phonemes)}")
        self._pfml_refresh_strip()

    def _on_poly_toggled(self, checked):
        # 全局常态设置：勾选状态保存进 config（重启保持），并重刷字符条更新横杠
        try:
            self.config.data["show_polyphonic"] = bool(checked)
            self.config.save()
        except Exception:
            pass
        self._pfml_refresh_strip()

    def _poly_positions(self, m) -> dict:
        """勾选"显示多音"时，返回 {起始偏移: 结束偏移} 的多音字/多音词跨度集合。

        中/粤按单字多音（每个字独立跨度）；日语/英语按"候选>=2"的整词判多音。
        """
        if not getattr(self, "chk_poly", None) or not self.chk_poly.isChecked():
            return {}
        lang = m.lang or "zh"
        try:
            if lang in ("zh", "yue"):
                from g2p import auto_polyphonic
                return {b: b + 1 for b in auto_polyphonic(m.text, lang)}
            # 日语/英语：多候选(>=2)的整词显示多音
            from g2p import list_candidates
            out = {}
            for b, e in m.slots.slot_spans():
                lab = m.text[b:e]
                if not lab or lab.isspace():
                    continue
                if len(list_candidates(lab, lang)) >= 2:
                    out[b] = e
            return out
        except Exception:
            return {}

    def _pfml_set_pron_manual(self, b, e):
        """对指定区间 [b,e) 直接弹手动音素输入框。"""
        from PyQt5.QtWidgets import QInputDialog
        d = self._pfml_data()
        m = self._edit_mark()
        if not d or m is None:
            return
        sel_text = m.text[b:e]
        manual, ok = QInputDialog.getText(
            self, f"手写音素：{sel_text}", "音素（空格分隔，如 ch ong）：")
        if not ok or not manual.strip():
            return
        phonemes = manual.split()
        if not phonemes:
            return
        self._push_undo()
        d["overrides"] = [o for o in d["overrides"]
                          if not (o["begin"] == b and o["end"] == e and not o.get("ins"))]
        d["overrides"].append({"begin": b, "end": e, "script": "", "phonemes": phonemes,
                               "key": m.text[b:e]})
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"发音已指定：{' '.join(phonemes)}")
        self._pfml_refresh_strip()

    def _pfml_insert_phoneme(self):
        d = self._pfml_data()
        m = self._edit_mark()
        if not d or m is None:
            self.statusBar().showMessage("先选中一个标记")
            return
        cur = self.text_edit.textCursor()
        pos = cur.position()
        sym, ok = QInputDialog.getText(self, "插入音素",
                                       "音素符号（如 n / AP / SP）：")
        if not ok or not sym.strip():
            return
        self._push_undo()
        d["overrides"] = [o for o in d["overrides"]
                          if not (o["begin"] == pos and o["end"] == pos)]
        s0, s1, s2 = m.text[max(0, pos-1):pos], m.text[pos:pos+1], m.text[pos+1:pos+2]
        anchor = s0 + s1 + s2
        d["overrides"].append({"begin": pos, "end": pos,
                               "script": "", "phonemes": [sym.strip()],
                               "key": anchor, "ins": True})
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"在 {pos} 插入音素 {sym.strip()}")
        self._pfml_refresh_strip()

    def _pfml_set_lang(self):
        r = self._pfml_selection_range()
        d = self._pfml_data()
        if not r or not d:
            self.statusBar().showMessage("先选中要指定语言的文字")
            return
        # 弹语言菜单
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet("QMenu {background:#2a2a2a; color:#eee;}"
                           "QMenu::item:selected {background:#2d5f8a;}")
        for code in LANG_CODES:
            act = menu.addAction(LANG_MAP[code])
            act.triggered.connect(lambda _c=False, lang=code: self._apply_span_lang(r, lang))
        act_clear = menu.addAction("取消语言")
        act_clear.triggered.connect(lambda: self._apply_span_lang(r, None))
        menu.exec_(self.btn_pfml_setlang.mapToGlobal(
            self.btn_pfml_setlang.rect().bottomLeft()))

    def _apply_span_lang(self, r, lang):
        d = self._pfml_data()
        m = self._edit_mark()
        if not d or not r or m is None:
            return
        self._push_undo()
        b, e = r
        if lang is None:
            # 取消：删掉正好覆盖 [b,e) 的区间
            d["spans"] = [s for s in d["spans"]
                          if not (s["begin"] == b and s["end"] == e)]
            self._pfml_commit(d)
            self._mark_dirty()
            self.statusBar().showMessage("取消语言区间")
            self._pfml_refresh_strip()
            return
        d["spans"].append({"begin": b, "end": e, "language": lang,
                           "key": m.text[b:e]})
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage(f"区间 [{b},{e}) 语言 = {LANG_MAP.get(lang, lang)}")
        self._pfml_refresh_strip()

    def _pfml_clear(self):
        d = self._pfml_data()
        if not d:
            return
        self._push_undo()
        d["words"] = []
        d["overrides"] = []
        d["spans"] = []
        self._pfml_commit(d)
        self._mark_dirty()
        self.statusBar().showMessage("已清除该标记全部 PFML 标注")
        self._pfml_refresh_strip()

    # ---------------- 播放 ----------------
    def _selected_mark(self):
        marks = self.project.marks_of(self.current_wav)
        idx = self.state.selected_mark
        if 0 <= idx < len(marks):
            return marks[idx]
        return None

    def _edit_mark(self):
        """当前正在编辑的标记（用 _edit_mark_idx，取消选中后仍保留）。"""
        marks = self.project.marks_of(self.current_wav)
        idx = self._edit_mark_idx
        if 0 <= idx < len(marks):
            return marks[idx]
        return None

    def _toggle_play_selected(self):
        st = self.state
        if not st.has_audio():
            return
        if self.player.playing:
            self.player.pause()
            self.state.auto_follow = False
            self.btn_play.setText("播放")
            self._update_pos_label()
            return
        m = self._selected_mark()
        if m is not None:
            # 选中标记：播放头局限在标记区域内播放
            start = min(max(st.play_pos, m.start), m.end - 0.05)
            end = m.end
        else:
            # 平时：从当前播放头位置播到音频结尾
            start = min(st.play_pos, st.duration)
            end = st.duration
        if start >= end:
            return
        if abs(self.player.start - start) < 1e-6 and abs(self.player.end - end) < 1e-6:
            self.player.toggle()
        else:
            self.player.play_slice(start, end)
        self.state.auto_follow = True
        self.btn_play.setText("暂停")
        self._update_pos_label()

    def _tick_playback(self):
        if not self.player.playing:
            return
        pos = self.player.current_pos()
        self.state.play_pos = float(np.clip(pos, 0.0, self.state.duration))
        self._follow_playhead()
        if pos >= self.player.end:
            # 到达播放范围结尾：停止（选中标记时即停在区域内）
            self.player.stop()
            self.state.play_pos = min(self.player.end, self.state.duration)
            self.state.auto_follow = False
            self.btn_play.setText("播放")
            self.wave_view.refresh()
            self._update_pos_label()
            return
        self._update_pos_label()
        self._repaint_playhead()

    def _repaint_playhead(self):
        """只重绘播放头附近区域：覆盖上一帧位置 + 当前位置，清除旧线避免滞留。"""
        st = self.state
        x = self.wave_view.wave._time_to_disp_x(st.play_pos)
        last = self._last_playhead_x
        if last is None:
            lo = hi = x
        else:
            lo = min(x, last)
            hi = max(x, last)
        wspan = max(26.0, (hi - lo) + 26)
        x0 = max(0, int(lo - 12))
        for c in (self.wave_view.wave, self.wave_view.spec):
            c.update(x0, 0, max(1, int(wspan)), max(1, c.height()))
        self._last_playhead_x = x

    def _follow_playhead(self):
        st = self.state
        w = self.wave_view.wave.width()
        vis = st.visible_seconds(w)
        right_edge = st.view_start + vis
        if st.play_pos > right_edge - 0.02:
            st.view_start = st.play_pos - vis * 0.2
            st.clamp_view(w)
            self.wave_view.refresh()
        elif st.play_pos < st.view_start:
            st.view_start = max(0.0, st.play_pos - vis * 0.02)
            self.wave_view.refresh()

    def _update_pos_label(self):
        st = self.state
        self.pos_label.setText(f"{self._fmt(st.play_pos)} / {self._fmt(st.duration)}")

    @staticmethod
    def _fmt(t: float) -> str:
        t = max(0.0, float(t))
        mm = int(t // 60)
        ss = t - mm * 60
        return f"{mm:02d}:{ss:06.3f}"

    def _on_playhead_moved(self, t):
        st = self.state
        m = self._selected_mark()
        if m is not None:
            t = float(np.clip(t, m.start, m.end))  # 选中标记时播放头限制在区域内
        st.play_pos = float(np.clip(t, 0.0, st.duration))
        # 点击画布移动播放头时取消选中状态
        self._clear_mark_selection(repaint=False)
        self.wave_view.refresh()
        self._update_pos_label()

    def _clear_mark_selection(self, repaint=True):
        if self.state.selected_mark != -1:
            self.state.selected_mark = -1
            self.mark_table.clearSelection()
            if repaint:
                self.wave_view.refresh()
                self._sync_text_edit()

    def _stop_playback(self):
        self.player.stop()
        self.btn_play.setText("播放")
        self.state.auto_follow = False

    # ---------------- 音频编辑 ----------------
    def _flush_audio_edit(self) -> bool:
        """把当前音频的未保存编辑写回 wav；成功返回 True。"""
        if not self.state.edited or not self.current_wav:
            return True
        path = Path(self.project.folder_path) / self.current_wav
        if not path.is_file():
            QMessageBox.warning(self, "提示", f"找不到源音频，无法写回修改：{self.current_wav}")
            return False
        try:
            import soundfile as sf
            sf.write(str(path), self.state.wave, self.state.sr, subtype="PCM_16")
            self.state.edited = False
            self.statusBar().showMessage(f"已写回音频修改：{self.current_wav}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "写回失败", str(e))
            return False

    def delete_selection(self):
        st = self.state
        if not st.has_audio() or not self.current_wav:
            QMessageBox.information(self, "提示", "没有加载音频"); return
        if st.sel_start < 0 or st.sel_end <= st.sel_start:
            QMessageBox.information(self, "提示", "请先用鼠标框选要删除的区域"); return
        sr = st.sr
        s = max(0, int(st.sel_start * sr)); e = min(len(st.wave), int(st.sel_end * sr))
        if s >= e: return
        s_sec, e_sec = s / sr, e / sr
        self._push_undo()
        st.wave = np.concatenate([st.wave[:s], st.wave[e:]]) if e < len(st.wave) else st.wave[:s].copy()
        self._shift_marks_after_delete(s_sec, e_sec)
        st.sel_start = st.sel_end = -1.0
        self._after_edit()
        self._refresh_mark_table()
        self.statusBar().showMessage(f"已删除选区 {s_sec:.3f}~{e_sec:.3f}s，标记已同步调整")

    def silence_selection(self):
        st = self.state
        if not st.has_audio() or not self.current_wav:
            QMessageBox.information(self, "提示", "没有加载音频"); return
        if st.sel_start < 0 or st.sel_end <= st.sel_start:
            QMessageBox.information(self, "提示", "请先用鼠标框选要静音的区域"); return
        sr = st.sr
        s = max(0, int(st.sel_start * sr)); e = min(len(st.wave), int(st.sel_end * sr))
        if s >= e: return
        self._push_undo()
        st.wave = st.wave.copy()
        st.wave[s:e] = 0.0
        # 同步频谱：静音区对应帧置 0（黑），频谱即时反映静音
        if st.mel is not None and st.mel_fps > 0:
            fs = max(0, int(s / sr * st.mel_fps))
            fe = min(st.mel.shape[1], int(e / sr * st.mel_fps))
            if fe > fs:
                st.mel = np.ascontiguousarray(st.mel)
                st.mel[:, fs:fe] = 0
        st.sel_start = st.sel_end = -1.0
        self._after_edit(recompute_mel=False)
        self.statusBar().showMessage("已将选区静音（幅值归零 = dB 负无穷）")

    def insert_silence(self):
        st = self.state
        if not st.has_audio() or not self.current_wav:
            QMessageBox.information(self, "提示", "没有加载音频"); return
        dur, ok = QInputDialog.getDouble(self, "插入静音", "静音时长（秒）：", 1.0, 0.001, 600.0, 3)
        if not ok: return
        sr = st.sr
        idx = max(0, min(len(st.wave), int(st.play_pos * sr)))
        n = max(1, int(dur * sr))
        self._push_undo()
        st.wave = np.concatenate([st.wave[:idx], np.zeros(n, dtype=np.float32), st.wave[idx:]])
        self._shift_marks_after_insert(st.play_pos, dur)
        self._after_edit()
        self._refresh_mark_table()
        self.statusBar().showMessage(f"已在播放头位置插入 {dur:.3f}s 静音，标记已同步后移")

    # ---- 自动切分（基于 RMS 静音检测） ----
    def auto_slice(self):
        st = self.state
        if not st.has_audio() or not self.current_wav:
            QMessageBox.information(self, "提示", "没有加载音频"); return
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle("自动切分（按静音分段）")
        form = QFormLayout(dlg)
        def _spin(lo, hi, val, step=1.0, dec=1):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
            s.setSingleStep(step); s.setValue(val); return s
        sp_th = _spin(-60.0, -10.0, -40.0, 1.0, 0)
        sp_minlen = _spin(0.1, 30.0, 5.0, 0.5, 2)
        sp_minint = _spin(0.05, 5.0, 0.3, 0.05, 2)
        sp_hop = _spin(1.0, 100.0, 10.0, 1.0, 0)
        sp_sil = _spin(0.0, 10.0, 1.0, 0.5, 2)
        form.addRow("静音阈值 (dB，越低切得越松)", sp_th)
        form.addRow("最片段短长度 (秒)", sp_minlen)
        form.addRow("最短静音间隔 (秒)", sp_minint)
        form.addRow("帧长 (ms，越小越精细)", sp_hop)
        form.addRow("两端保留静音 (秒)", sp_sil)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return
        from audio_slicer import slice_audio
        segs = slice_audio(st.wave, st.sr,
                           threshold_db=float(sp_th.value()),
                           min_length_ms=float(sp_minlen.value())*1000.0,
                           min_interval_ms=float(sp_minint.value())*1000.0,
                           hop_ms=float(sp_hop.value()),
                           max_sil_kept_ms=float(sp_sil.value())*1000.0)
        if not segs:
            QMessageBox.information(self, "结果", "未检测到任何有声片段，请调低阈值试试"); return
        # 若当前已有标记，询问是否覆盖
        marks = self.project.marks_of(self.current_wav)
        if marks:
            r = QMessageBox.question(self, "已有标记",
                f"当前已有 {len(marks)} 个标记。\n替换为自动切分的 {len(segs)} 段？",
                QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._push_undo()
        new_marks = [Mark(s, e, "", self._edit_mark().lang if self._edit_mark() else "zh")
                     for (s, e) in segs]
        self.project.set_marks(self.current_wav, new_marks)
        self.state.marks = new_marks
        self.state.selected_mark = 0
        self._edit_mark_idx = 0
        self._refresh_mark_table()
        self._sync_text_edit()
        self.wave_view.refresh()
        self.statusBar().showMessage(f"自动切分完成：共 {len(segs)} 段")

    # ---- 音频处理插件（fx/）：无选区整段，有选区仅处理选区 ----
    def _fx_target_range(self):
        st = self.state; sr = st.sr
        if st.sel_start >= 0 and st.sel_end > st.sel_start:
            s = max(0, int(st.sel_start * sr)); e = min(len(st.wave), int(st.sel_end * sr))
            return s, e, "选区"
        return 0, len(st.wave), "整段"

    def _apply_fx(self, display_name, proc):
        st = self.state
        if not st.has_audio() or not self.current_wav:
            QMessageBox.information(self, "提示", "没有加载音频"); return
        sr = st.sr
        s, e, scope = self._fx_target_range()
        if e <= s:
            return
        self._push_undo()
        seg = np.ascontiguousarray(st.wave[s:e])
        try:
            out = proc(seg, sr)
        except Exception as ex:
            if self._undo_stack:
                self._undo_stack.pop()
            QMessageBox.critical(self, "处理失败", f"{display_name}：{ex}"); return
        if len(out) != len(seg):
            out = out[:len(seg)] if len(out) >= len(seg) else np.pad(out, (0, len(seg) - len(out)))
        st.wave = st.wave.copy()
        st.wave[s:e] = out
        st.sel_start = st.sel_end = -1.0
        self._after_edit(recompute_mel=True)
        self.statusBar().showMessage(f"已对{scope}执行{display_name}（可撤销 Ctrl+Z）")

    def run_denoise(self):
        try:
            from fx import denoise
        except Exception as ex:
            QMessageBox.warning(self, "降噪未就绪",
                                f"无法加载 RX10 Voice De-noise：\n{ex}"); return
        strength, ok = QInputDialog.getDouble(self, "AI 降噪 (RX10 Voice De-noise)", "强度（0=原音，100=插件全开）：", 100, 0, 100, 0)
        if not ok:
            return
        self._apply_fx("降噪 (RX10)", lambda seg, sr: denoise.process(seg, sr, strength=strength))

    def run_declick(self):
        try:
            from fx import declick
        except Exception as ex:
            QMessageBox.warning(self, "去口水音未就绪",
                                f"无法加载 RX10 Mouth De-click：\n{ex}"); return
        strength, ok = QInputDialog.getDouble(self, "去口水音 (RX10 Mouth De-click)", "强度（0=原音，100=插件全开）：", 100, 0, 100, 0)
        if not ok:
            return
        self._apply_fx("去口水音 (RX10)", lambda seg, sr: declick.process(seg, sr, strength=strength))

    def run_deplosive(self):
        try:
            from fx import deplosive
        except Exception as ex:
            QMessageBox.warning(self, "去喷麦未就绪",
                                f"无法加载 RX10 De-plosive：\n{ex}"); return
        strength, ok = QInputDialog.getDouble(self, "去喷麦 (RX10 De-plosive)", "强度（0=原音，100=插件全开）：", 100, 0, 100, 0)
        if not ok:
            return
        self._apply_fx("去喷麦 (RX10)", lambda seg, sr: deplosive.process(seg, sr, strength=strength))

    # ---- VST3 通用加载器（启用列表来自设置/config）----
    def reload_vst_panel(self):
        """从 config['enabled_vsts'] 填充右侧 VST 面板。"""
        self.vst_list.clear()
        for path in self.config.get("enabled_vsts", []):
            it = QListWidgetItem(os.path.splitext(os.path.basename(path))[0])
            it.setData(Qt.UserRole, path)
            it.setToolTip(path)
            self.vst_list.addItem(it)

    def _on_vst_clicked(self, item):
        """单击插件名 → 弹出独立可移动窗口（含插件原生界面 + 处理按钮）。"""
        path = item.data(Qt.UserRole)
        if not path:
            return
        self._open_vst_window(path)

    def _open_vst_window(self, path):
        import subprocess
        if not self.current_wav or not self.project.folder_path:
            QMessageBox.information(self, "提示", "请先加载一个 wav 文件"); return
        wav_path = str(Path(self.project.folder_path) / self.current_wav)
        if not os.path.isfile(wav_path):
            QMessageBox.warning(self, "提示", f"找不到音频：{wav_path}"); return
        sub = Path(__file__).resolve().parent / "vst_subprocess.py"
        subprocess.Popen([sys.executable, str(sub), "editor", path, wav_path],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.statusBar().showMessage(f"已打开 {os.path.basename(path)}（调参数后选区处理）")

    def _close_vst_window(self, win):
        if win in self._vst_windows:
            self._vst_windows.remove(win)

    def _current_vst_path(self):
        it = self.vst_list.currentItem()
        if not it:
            QMessageBox.information(self, "提示", "请先在右侧 VST3 列表选中一个插件"); return None
        return it.data(Qt.UserRole)

    def _vst_run(self):
        path = self._current_vst_path()
        if not path: return
        self._vst_run_path(path)

    def _vst_run_path(self, path):
        st = self.state
        if not st.has_audio() or not self.current_wav:
            QMessageBox.information(self, "提示", "没有加载音频"); return
        import subprocess, tempfile, shutil
        import soundfile as sf
        sr = st.sr
        s, e, scope = self._fx_target_range()
        if e <= s:
            return
        self._push_undo()
        seg = np.ascontiguousarray(st.wave[s:e])
        tmpdir = tempfile.mkdtemp(prefix="lab_vst_")
        inw = os.path.join(tmpdir, "in.wav"); outw = os.path.join(tmpdir, "out.wav")
        try:
            sf.write(inw, seg, sr)
            sub = Path(__file__).resolve().parent / "vst_subprocess.py"
            r = subprocess.run([sys.executable, "-u", str(sub), "process", path, inw, outw],
                               capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode != 0:
                if self._undo_stack:
                    self._undo_stack.pop()
                QMessageBox.critical(self, "处理失败",
                                     r.stderr.decode("utf-8", "replace")[-1500:]); return
            out, _ = sf.read(outw, dtype="float32")
            out = np.asarray(out, dtype="float32").ravel()
            m = min(len(seg), len(out))
            st.wave = st.wave.copy()
            st.wave[s:s + m] = out[:m]
            st.sel_start = st.sel_end = -1.0
            self._after_edit(recompute_mel=True)
            self.statusBar().showMessage(f"已用 {os.path.basename(path)} 处理{scope}（可撤销）")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _after_edit(self, recompute_mel=True):
        st = self.state
        st.duration = float(len(st.wave)) / st.sr if len(st.wave) else 0.0
        st.edited = True
        self.project.dirty = True   # 音频被修改 → 工程标脏，关闭时提示保存并写回 wav
        # 同步波形绘制用的标记（shift 后 project 已更新）
        if self.current_wav and self.current_wav in self.project.wavs:
            st.marks = self.project.wavs[self.current_wav]
        if recompute_mel and st.mel is not None and len(st.wave):
            try:
                mel, fps = AudioLoader._compute_mel(st.wave, st.sr)
                st.mel, st.mel_fps = mel, fps
            except Exception as e:
                QMessageBox.warning(self, "提示", f"频谱重算失败：{e}")
        self.player.load(st.wave, st.sr)
        st.clamp_view(self.wave_view.wave.width())
        self.wave_view.refresh()
        self._update_pos_label()

    def _shift_marks_after_delete(self, s_sec, e_sec):
        """删除 [s,e] 后同步标记：后续标记前移、起点落在删除区内的移除、跨界标记缩短。"""
        d = e_sec - s_sec
        keep = []
        for m in self.project.marks_of(self.current_wav):
            ms, me = m.start, m.end
            if ms >= e_sec:
                ms -= d; me -= d
            elif ms >= s_sec:
                continue  # 起点落在删除区内 → 整段被删，移除
            elif me > s_sec:
                me -= d  # 跨删除区，终点前移
            ns = round(max(0.0, ms), 3); ne = round(max(0.0, me), 3)
            if ne > ns:
                cm = m.copy()
                cm.start, cm.end = ns, ne
                keep.append(cm)
        self.project.wavs[self.current_wav] = keep

    def _shift_marks_after_insert(self, pos_sec, dur_sec):
        """在 pos 处插入 dur 秒静音后，把插入点及之后的标记时间后移。"""
        for m in self.project.marks_of(self.current_wav):
            if m.start >= pos_sec:
                m.start = round(m.start + dur_sec, 3)
            if m.end >= pos_sec:
                m.end = round(m.end + dur_sec, 3)

    def _push_undo(self):
        import copy
        st = self.state
        wave = np.copy(st.wave) if st.wave is not None else None
        marks = [m.copy() for m in st.marks]
        self._undo_stack.append((wave, marks))
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _apply_edit_state(self, wave, marks):
        st = self.state
        st.wave = wave
        st.marks = marks
        if self.current_wav:
            self.project.wavs[self.current_wav] = marks
        self.project.dirty = True
        st.duration = float(len(wave)) / st.sr if len(wave) else 0.0
        st.edited = True
        if st.mel is not None and len(wave):
            try:
                mel, fps = AudioLoader._compute_mel(wave, st.sr)
                st.mel, st.mel_fps = mel, fps
            except Exception:
                pass
        self.player.load(wave, st.sr)
        st.clamp_view(self.wave_view.wave.width())
        # 恢复合理的选中/编辑索引（不越界）
        n = len(marks)
        if self._edit_mark_idx >= n:
            self._edit_mark_idx = max(0, n - 1)
        if self.state.selected_mark >= n:
            self.state.selected_mark = max(-1, n - 1)
        self.wave_view.refresh()
        self._refresh_mark_table()
        self._sync_text_edit()
        self._update_pos_label()
        self._text_undo_pushed = False

    def undo(self):
        import copy
        if not self._undo_stack:
            self.statusBar().showMessage("没有可撤销的操作"); return
        st = self.state
        cur = (np.copy(st.wave) if st.wave is not None else None,
               [copy.deepcopy(m) for m in st.marks])
        wave, marks = self._undo_stack.pop()
        self._redo_stack.append(cur)
        self._apply_edit_state(wave, marks)
        self.statusBar().showMessage("已撤销")

    def redo(self):
        import copy
        if not self._redo_stack:
            self.statusBar().showMessage("没有可重做的操作"); return
        st = self.state
        cur = (np.copy(st.wave) if st.wave is not None else None,
               [copy.deepcopy(m) for m in st.marks])
        wave, marks = self._redo_stack.pop()
        self._undo_stack.append(cur)
        self._apply_edit_state(wave, marks)
        self.statusBar().showMessage("已重做")

    def _is_text_input(self, w):
        """w 或其任一父级是文本输入控件（含 QTextEdit 内部 viewport）。"""
        from PyQt5.QtWidgets import (
            QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox,
        )
        while w is not None:
            if isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
                return True
            pw = getattr(w, 'parentWidget', None)
            w = pw() if callable(pw) else None
        return False

    def eventFilter(self, obj, ev):
        t = ev.type()
        # 标记列表有选中标记时：W(上)/S(下) 等同方向键切换标记
        if t == QEvent.KeyPress and self._mark_nav_key(ev):
            return True
        # 点击输入框以外 → 取消输入状态
        if t == QEvent.MouseButtonPress and not self._is_text_input(obj):
            self._clear_text_focus()
        # 文本框失去焦点 → 结束当前合并的文本编辑撤销单元
        if obj is self.text_edit and t == QEvent.FocusOut:
            self._text_undo_pushed = False
        return super().eventFilter(obj, ev)

    def _mark_nav_key(self, ev) -> bool:
        """W/S 在标记列表存在选中行时切换到上/下一条标记（等同方向键）。"""
        mods = ev.modifiers()
        if mods & (Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        k = ev.key()
        if k not in (Qt.Key_W, Qt.Key_S):
            return False
        fw = QApplication.focusWidget()
        if self._is_text_input(fw):
            return False
        # 模态对话框 / 右键菜单打开时不拦截（菜单自身要用按键）
        if QApplication.activeModalWidget() is not None:
            return False
        from PyQt5.QtWidgets import QMenu
        if fw is not None and isinstance(fw.window(), QMenu):
            return False
        rows = sorted({r.row() for r in self.mark_table.selectedIndexes()})
        if not rows:
            return False
        n = self.mark_table.rowCount()
        idx = rows[0] + (-1 if k == Qt.Key_W else 1)
        if not (0 <= idx < n):
            return False  # 到边界：与方向键一致，不处理也不拦截其他控件
        self._select_mark_by_index(idx)
        return True

    def _clear_text_focus(self):
        """点击输入框以外 → 取消所有输入框的编辑焦点（让快捷键生效）。"""
        self.text_edit.clearFocus()
        self.match_edit.clearFocus()
        self.export_dir_edit.clearFocus()
        self._text_undo_pushed = False

    def _select_mark_by_index(self, idx):
        """点击画布标记序号 → 选中该标记（同步表格高亮 + 文本编辑）。"""
        marks = self.project.marks_of(self.current_wav)
        if not marks or not (0 <= idx < len(marks)):
            return
        self.state.selected_mark = idx
        self._edit_mark_idx = idx
        # 消除 match 高亮
        if getattr(marks[idx], "_match_changed", False):
            marks[idx]._match_changed = False
            self._refresh_mark_table()
        # 同步左侧表格选中
        self.mark_table.selectRow(idx)
        self.mark_table.setCurrentCell(idx, 0)
        self._sync_text_edit()
        self.wave_view.refresh()

    # ---------------- ASR ----------------
    def _asr_service_get(self) -> ASRService:
        if self._asr_service is None:
            self._asr_service = ASRService()
        return self._asr_service

    def _asr_unload_model(self):
        svc = self._asr_service_get()
        try:
            svc.unload()
        except Exception:
            pass
        self.btn_asr_unload.setEnabled(False)
        self.statusBar().showMessage("ASR 模型已卸载，内存已释放。")

    def _run_asr(self):
        if not self.current_wav or not self.project.folder_path:
            QMessageBox.information(self, "提示", "请先在左侧选择音频文件")
            return
        marks = self.project.marks_of(self.current_wav)
        if not marks:
            QMessageBox.information(self, "提示", "当前音频没有标记片段，请先框选并按 M 新建标记")
            return
        model_path = self.config.get("onnx_asr_model_path", "")
        if not model_path:
            QMessageBox.warning(self, "提示", "未找到 Qwen3-ASR 模型（整合包 experiments/models 目录）")
            return
        # 只识别选中的标记；未选中则识别全部
        selected_rows = sorted({r.row() for r in self.mark_table.selectedIndexes()})
        if selected_rows:
            sel = [marks[i] for i in selected_rows if 0 <= i < len(marks)]
            if sel:
                marks = sel
        self._asr_target = marks  # 记录实际被识别的标记（回填用，避免错段）
        path = Path(self.project.folder_path) / self.current_wav
        self.btn_asr.setEnabled(False)
        # 通过常驻服务后台线程转录
        svc = self._asr_service_get()
        self._asr_progress = QProgressDialog(
            "正在运行 ASR（若未加载模型会先加载，请稍候）…", "取消", 0, len(marks), self)
        self._asr_progress.setWindowModality(Qt.WindowModal)
        self._asr_progress.setWindowTitle("ASR 识别")
        self._asr_progress.setMinimumDuration(0)
        self._asr_progress.setValue(0)
        self._asr_progress.canceled.connect(self._asr_cancel)

        from PyQt5.QtCore import QThread

        class _Transcribe(QThread):
            done = pyqtSignal(object)
            err = pyqtSignal(str)
            def run(self):
                try:
                    svc.load(model_path)  # 若已加载会复用，不重复加载
                    texts = svc.transcribe(str(path), list(marks))
                    self.done.emit((texts, len(marks)))
                except Exception as e:
                    self.err.emit(str(e))
        self._asr_transcribe_thread = _Transcribe()
        self._asr_transcribe_thread.done.connect(self._on_asr_done)
        self._asr_transcribe_thread.err.connect(self._on_asr_failed)
        self._asr_transcribe_thread.start()
        # 加载态下同步按钮状态
        self.btn_asr_unload.setEnabled(True)

    def _asr_cancel(self):
        t = getattr(self, "_asr_transcribe_thread", None)
        if t is not None and t.isRunning():
            # 常驻服务不支持逐条取消，直接忽略；仅在 UI 上关闭对话框
            pass
        if self._asr_progress:
            self._asr_progress.close()
        self._asr_progress = None
        self.btn_asr.setEnabled(True)

    def _on_asr_done(self, payload):
        texts, total = payload
        marks = self.project.marks_of(self.current_wav)
        # 只回填被识别的那部分（选中子集 → 对应位置的目标标记）
        target = getattr(self, "_asr_target", None)
        if not target:
            target = marks
        changed = 0
        for mark, t in zip(target, texts):
            if t and t != mark.text:
                # 整段替换：同步 slots（否则 text 与格子表脱节，发音挂不上）；日语自动分词
                mark.apply_matched_text(t)
                mark._match_changed = False  # ASR 覆盖不高亮（避免与歌词 match 冲突）
                changed += 1
            else:
                mark._match_changed = False
        if self._asr_progress:
            self._asr_progress.close()
        self._asr_progress = None
        self.btn_asr.setEnabled(True)
        self.btn_asr_unload.setEnabled(True)
        self._sync_text_edit()
        self._refresh_mark_table()
        self._mark_dirty()
        self.statusBar().showMessage(f"ASR 完成：{len(texts)} 段，更新 {changed} 段")
        QMessageBox.information(self, "完成", f"ASR 识别完成，共 {len(texts)} 段。")

    def _on_asr_failed(self, msg):
        if self._asr_progress:
            self._asr_progress.close()
        self._asr_progress = None
        self.btn_asr.setEnabled(True)
        QMessageBox.critical(self, "ASR 失败", msg)

    # ---------------- TIFA 自动选音 ----------------
    def _run_tifa(self):
        """按音频判断多音字/多音词读音并自动标音（TIFA ONNX 对齐）。

        与 ASR 同逻辑：只处理选中的标记，未选中则处理全部。
        后台线程只计算、主线程写回 override（避免跨线程改 mark）。
        """
        if not self.current_wav or not self.project.folder_path:
            QMessageBox.information(self, "提示", "请先在左侧选择音频文件")
            return
        import tifa_align
        if not tifa_align.align_available():
            QMessageBox.warning(self, "提示",
                "未找到 TIFA 对齐引擎（models/tifa/*.onnx）。")
            return
        marks = self.project.marks_of(self.current_wav)
        if not marks:
            QMessageBox.information(self, "提示", "当前音频没有标记片段，请先框选并按 M 新建标记")
            return
        # 只处理选中的标记；未选中则处理全部
        selected_rows = sorted({r.row() for r in self.mark_table.selectedIndexes()})
        if selected_rows:
            sel = [marks[i] for i in selected_rows if 0 <= i < len(marks)]
            if sel:
                marks = sel
        # 运行前检测：含疑似其他语言文本却没打语言副标签的片段。
        # 这些片段会被按主语言 G2P（如英文 what 在中文管道里只有 1 个无效候选），
        # TIFA 即使对齐成功也不会写入 → 先提示用户补标签。
        from tifa_align import detect_unlabeled_languages
        hits = []
        for mk in marks:
            for b, e, lang in detect_unlabeled_languages(mk):
                hits.append((mk, b, e, lang))
        if hits:
            lines = []
            for mk, b, e, lang in hits[:6]:
                ctx = mk.text[max(0, b - 3):min(len(mk.text), e + 3)]
                lines.append(f"• 『{ctx}』中的 {mk.text[b:e]} 疑似「{LANG_MAP.get(lang, lang)}」")
            more = "" if len(hits) <= 6 else f"\n…等共 {len(hits)} 处"
            ret = QMessageBox.question(
                self, "发现未标注语言的文本",
                "检测到以下文本含有非主语言内容，但没有设置语言标签：\n\n"
                + "\n".join(lines) + more + "\n\n仍要继续运行吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        path = Path(self.project.folder_path) / self.current_wav
        self._tifa_target = marks
        self.btn_tifa.setEnabled(False)
        self._tifa_progress = QProgressDialog(
            "正在运行 TIFA 自动选音（按音频判断多音字读音）…", "取消", 0, len(marks), self)
        self._tifa_progress.setWindowModality(Qt.WindowModal)
        self._tifa_progress.setWindowTitle("TIFA 自动选音")
        self._tifa_progress.setMinimumDuration(0)
        self._tifa_progress.setValue(0)

        from PyQt5.QtCore import QThread

        class _TifaRun(QThread):
            prog = pyqtSignal(int)
            done = pyqtSignal(object)     # (results, total)
            err = pyqtSignal(str)
            def run(self):
                import tempfile, shutil
                import tifa_align
                results = []
                tmp = None
                try:
                    tmp = tempfile.mkdtemp(prefix="tifa_run_")
                    for i, m in enumerate(self.marks):
                        mc = m.copy()      # 副本计算，避免后台线程改主线程对象
                        ok, chosen = tifa_align.compute_for_mark(mc, str(self.path), tmp)
                        if ok == 0:
                            results.append((i, chosen))
                        self.prog.emit(i + 1)
                    self.done.emit((results, len(self.marks)))
                except Exception as e:
                    self.err.emit(str(e))
                finally:
                    if tmp:
                        shutil.rmtree(tmp, ignore_errors=True)

        th = _TifaRun()
        th.marks = marks
        th.path = path
        th.prog.connect(self._tifa_progress.setValue)
        th.done.connect(self._on_tifa_done)
        th.err.connect(self._on_tifa_failed)
        self._tifa_thread = th
        th.start()

    def _write_tifa_overrides(self, m, chosen) -> int:
        """把一个标记的「音频判读音」写进 m.pfml（跳过已有手写 override）。"""
        pos = m.pos_pfml()
        pos.setdefault("overrides", [])
        existing = {(o["begin"], o["end"]) for o in pos["overrides"] if not o.get("ins")}
        written = 0
        for ch in chosen:
            b, e = ch["b"], ch["e"]
            if (b, e) in existing:
                continue  # 已有手动指定，不覆盖
            pos["overrides"] = [o for o in pos["overrides"]
                                if not (o["begin"] == b and o["end"] == e and not o.get("ins"))]
            pos["overrides"].append({"begin": b, "end": e, "script": ch["script"],
                                     "phonemes": ch["phonemes"], "key": m.text[b:e]})
            written += 1
        if written:
            m.pfml = m.slots.to_uid_pfml(pos)
        return written

    def _on_tifa_done(self, payload):
        results, total = payload
        marks = self.project.marks_of(self.current_wav)
        target = getattr(self, "_tifa_target", None)
        if not target:
            target = marks
        self._push_undo()   # 整个批量一步可撤销
        written = 0
        for i, chosen in results:
            if not (0 <= i < len(target)):
                continue
            written += self._write_tifa_overrides(target[i], chosen)
        if self._tifa_progress:
            self._tifa_progress.close()
        self._tifa_progress = None
        self.btn_tifa.setEnabled(True)
        self._pfml_refresh_strip()
        self._mark_dirty()
        self._refresh_mark_table()
        self._sync_text_edit()
        self.statusBar().showMessage(f"TIFA 自动选音完成：{total} 段，写入 {written} 处读音")
        QMessageBox.information(self, "完成", f"TIFA 自动选音完成，共 {total} 段，写入 {written} 处读音。")

    def _on_tifa_failed(self, msg):
        if self._tifa_progress:
            self._tifa_progress.close()
        self._tifa_progress = None
        self.btn_tifa.setEnabled(True)
        QMessageBox.critical(self, "TIFA 自动选音失败", msg)

    # ---------------- 呼吸检测（FBL ONNX + align 词边界） ----------------
    def _run_breath(self):
        """检测标记片段中的呼吸，把 AP/EP 插入对应字符位置。

        流程：切片 → align(词边界) → FBL ONNX 直接推理 → 只保留中心落在词间
        空隙的段（等价旧 FBL 流程"SP 中找 AP"）→ 映射回文本位置。
        重跑自动清除上次自动插入的呼吸音素（auto="breath" 标记）。
        """
        if not self.current_wav or not self.project.folder_path:
            QMessageBox.information(self, "提示", "请先在左侧选择音频文件")
            return
        import tifa_align
        import tifa_onnx_runner
        if not tifa_onnx_runner.fbl_available():
            QMessageBox.warning(self, "提示",
                "未找到呼吸检测模型（models/fbl/model.onnx）。")
            return
        marks = self.project.marks_of(self.current_wav)
        if not marks:
            QMessageBox.information(self, "提示", "当前音频没有标记片段")
            return
        selected_rows = sorted({r.row() for r in self.mark_table.selectedIndexes()})
        if selected_rows:
            sel = [marks[i] for i in selected_rows if 0 <= i < len(marks)]
            if sel:
                marks = sel
        path = Path(self.project.folder_path) / self.current_wav
        self._breath_target = marks
        self.btn_breath.setEnabled(False)
        self._breath_progress = QProgressDialog(
            "正在检测呼吸（对齐 + FBL 检测）…", "取消", 0, len(marks), self)
        self._breath_progress.setWindowModality(Qt.WindowModal)
        self._breath_progress.setWindowTitle("呼吸检测")
        self._breath_progress.setMinimumDuration(0)
        self._breath_progress.setValue(0)

        from PyQt5.QtCore import QThread

        class _BreathRun(QThread):
            prog = pyqtSignal(int)
            done = pyqtSignal(object)     # (results, total)
            err = pyqtSignal(str)
            def run(self):
                import tempfile, shutil
                import tifa_align as ta
                results = []
                tmp = None
                try:
                    tmp = tempfile.mkdtemp(prefix="breath_run_")
                    for i, m in enumerate(self.marks):
                        mc = m.copy()
                        ok, inserts = ta.breath_for_mark(mc, str(self.path), tmp)
                        if ok == 0:
                            results.append((i, inserts))
                        self.prog.emit(i + 1)
                    if tmp:
                        shutil.rmtree(tmp, ignore_errors=True)
                    self.done.emit((results, len(self.marks)))
                except Exception as e:
                    if tmp:
                        shutil.rmtree(tmp, ignore_errors=True)
                    self.err.emit(str(e))

        th = _BreathRun()
        th.marks = marks
        th.path = path
        th.prog.connect(self._breath_progress.setValue)
        th.done.connect(self._on_breath_done)
        th.err.connect(self._on_breath_failed)
        self._breath_thread = th
        th.start()

    def _write_breath_inserts(self, m, inserts) -> int:
        """把呼吸检测的 AP/EP 写进 m.pfml（位置版→uid 版）。

        先清除上次自动插入的呼吸音素（auto="breath"），再写本次结果。
        关闭 EP 判定时：末尾 EP 改为 AP（后置）。
        """
        pos = m.pos_pfml()
        pos.setdefault("overrides", [])
        pos["overrides"] = [o for o in pos["overrides"] if o.get("auto") != "breath"]
        if not self.config.get("breath_insert_ep", True):
            for it in inserts:
                if it.get("sym") == "EP":
                    it["sym"] = "AP"
                    it["side"] = "after"
        written = 0
        for it in inserts:
            b = it["b"]
            item = {"begin": b, "end": b, "script": "",
                    "phonemes": [it["sym"]],
                    "key": m.text[max(0, b - 1):b + 2],
                    "ins": True, "auto": "breath"}
            if it.get("side") == "after":
                item["side"] = "after"
            pos["overrides"].append(item)
            written += 1
        # 即使本次无检出，也要提交以清除旧自动插入
        m.pfml = m.slots.to_uid_pfml(pos)
        return written

    def _on_breath_done(self, payload):
        results, total = payload
        marks = self.project.marks_of(self.current_wav)
        target = getattr(self, "_breath_target", None) or marks
        self._push_undo()
        written = 0
        res_map = {i: ins for i, ins in results}
        for i in range(len(target)):
            # 所有目标标记都要清旧自动插入；有检出的写新
            written += self._write_breath_inserts(target[i], res_map.get(i, []))
        if self._breath_progress:
            self._breath_progress.close()
        self._breath_progress = None
        self.btn_breath.setEnabled(True)
        self._pfml_refresh_strip()
        self._mark_dirty()
        self._refresh_mark_table()
        self._sync_text_edit()
        self.statusBar().showMessage(f"呼吸检测完成：{total} 段，插入 {written} 处 AP/EP")
        QMessageBox.information(self, "完成",
                                f"呼吸检测完成，共 {total} 段，插入 {written} 处 AP/EP。")

    def _on_breath_failed(self, msg):
        if self._breath_progress:
            self._breath_progress.close()
        self._breath_progress = None
        self.btn_breath.setEnabled(True)
        QMessageBox.critical(self, "呼吸检测失败", msg)

    # ---------------- 导出 ----------------
    def _save_export_dir(self):
        txt = self.export_dir_edit.text().strip()
        self.config.data["export_dir"] = txt
        try:
            self.config.save()
        except Exception:
            pass

    def _pick_export_dir(self):
        start = self.export_dir_edit.text().strip() or str(self.project.folder_path or Path.home())
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", start)
        if d:
            self.export_dir_edit.setText(d)
            self._save_export_dir()

    def _resolve_export_dir(self) -> str:
        txt = self.export_dir_edit.text().strip()
        base = Path(txt) if txt else Path(self.project.folder_path) / "data_out"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            base = Path(self.project.folder_path)
        return str(base)

    def export_slices(self):
        if not self.project.folder_path or not self.project.wavs:
            QMessageBox.information(self, "提示", "没有可导出的标记")
            return
        # 校验超 20s
        over = []
        total_count = 0
        for name, marks in self.project.wavs.items():
            for m in marks:
                total_count += 1
                if m.duration > 20.0:
                    over.append(f"{name}  {m.start:.3f}~{m.end:.3f} ({m.duration:.1f}s)")
        if over:
            QMessageBox.warning(self, "片段超 20 秒",
                                "以下片段超过 20 秒，无法导出，请先拆分：\n" + "\n".join(over))
            return
        if total_count == 0:
            QMessageBox.information(self, "提示", "工程中没有标记")
            return
        import tifa_align
        if not tifa_align.align_available():
            QMessageBox.warning(self, "提示",
                "未找到 TIFA 对齐引擎（models/tifa/*.onnx），无法生成 TextGrid。")
            return

        # 导出设置弹窗：语言 + 分文件夹 + 路径
        from export_dialog import ExportDialog
        dlg = ExportDialog(self.config.get("default_lang", "zh"),
                           self.export_dir_edit.text().strip(), self)
        if not dlg.exec_():
            return
        out_dir = dlg.export_path or str(Path(self.project.folder_path) / "data_out")
        self.export_dir_edit.setText(dlg.export_path)
        self._save_export_dir()

        self._export_total = total_count
        self._export_progress = QProgressDialog(
            "正在导出切片并生成 TextGrid（TIFA 对齐）…", "取消", 0, total_count + 1, self)
        self._export_progress.setWindowModality(Qt.WindowModal)
        self._export_progress.setMinimumDuration(0)
        self._export_progress.setValue(0)

        from PyQt5.QtCore import QThread, pyqtSignal

        class _ExportRun(QThread):
            prog = pyqtSignal(int)
            lbl = pyqtSignal(str)
            done = pyqtSignal(object)   # (exported, skipped, tg_failed)
            err = pyqtSignal(str)
            def run(self):
                try:
                    r = self.mw._do_export(self.out_dir, self.lang, self.split,
                                           self.prog, self.mw._export_progress,
                                           self.lbl, self.mw._export_total)
                    self.done.emit(r)
                except Exception as e:
                    self.err.emit(str(e))

        th = _ExportRun()
        th.mw = self
        th.out_dir = out_dir
        th.lang = dlg.dataset_lang
        th.split = dlg.split_by_lang
        th.prog.connect(self._export_progress.setValue)
        th.lbl.connect(self._export_progress.setLabelText)
        th.done.connect(self._on_export_done)
        th.err.connect(self._on_export_err)
        self._export_thread = th
        th.start()

    def _on_export_done(self, result):
        exported, tg_failed, out_dir = result
        self._export_progress.close()
        msg = f"导出完成：{exported} 个切片（wav + pfml + TextGrid + statistics）到：\n{out_dir}"
        if tg_failed:
            msg += f"\n{tg_failed} 个 TextGrid 对齐失败（wav/pfml 已导出）"
        QMessageBox.information(self, "完成", msg)

    def _on_export_err(self, msg):
        self._export_progress.close()
        QMessageBox.critical(self, "导出失败", msg)

    def _do_export(self, out_dir: str, dataset_lang: str, split: bool,
                   prog, progdlg, lbl=None, total_count=0) -> tuple[int, int, int, str]:
        """导出：wav + pfml → TIFA 对齐生成 TextGrid + statistics。

        目录结构：
          不分开：<out>/wav/{name}.wav,.pfml  +  <out>/TextGrid/{name}.TextGrid
          分开：  <out>/<lang>/wav/...       +  <out>/<lang>/TextGrid/...
        不分开时：
          - 主语言 == dataset_lang 的标记正常导出（无语言前缀）
          - 非主语言标记也以带语言标签形式导出（<scope language=…> 保留）
            对齐时 extra_langs=工程内所有副语言（TIFA -L），确保音素解析正确
        """
        import soundfile as sf
        from pfml import to_pfml
        import tifa_onnx_runner as onnxr
        # 收集工程内所有副语言（不含 dataset_lang），供 -L 传参
        all_langs = set()
        for _name, marks in self.project.wavs.items():
            for m in marks:
                all_langs.add(m.lang or "zh")
        other_langs = sorted(all_langs - {dataset_lang})

        exported = tg_failed = done = 0
        seq_map = {}   # (stem, lang_folder) -> seq
        # 每个语言文件夹聚合 metrics 和 score_words
        folder_metrics: dict[str, list] = {}
        folder_scores: dict[str, list] = {}
        # 切片命名：以 wav 文件名为前缀，每个 wav 内按序 _01/_02/…，不同 wav 序号不衔接
        for name, marks in self.project.wavs.items():
            path = Path(self.project.folder_path) / name
            if not path.is_file():
                continue
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if data.ndim > 1:
                data = data.mean(axis=1)
            stem = Path(name).stem
            for m in sorted(marks, key=lambda m: m.start):
                if progdlg.wasCanceled():
                    return exported, tg_failed, out_dir
                lang = m.lang or "zh"
                s, e = int(m.start * sr), min(len(data), int(m.end * sr))
                if s >= e:
                    done += 1
                    prog.emit(done)
                    continue
                folder_lang = lang if split else ""
                key = (stem, folder_lang)
                seq_map[key] = seq_map.get(key, 0) + 1
                base = f"{stem}_{seq_map[key]:02d}"
                root = Path(out_dir) / folder_lang if split else Path(out_dir)
                wav_dir = root / "wav"
                tg_dir = root / "TextGrid"
                wav_dir.mkdir(parents=True, exist_ok=True)
                tg_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / f"{base}.wav"
                sf.write(str(wav_path), data[s:e], sr, subtype="PCM_16")
                # PFML：主语言=m.lang，内嵌词边界/发音/音素/嵌套语言（uid 版映射回位置）
                pfml_text = to_pfml(m.text or "", lang, m.pos_pfml())
                (wav_dir / f"{base}.pfml").write_text(pfml_text, encoding="utf-8")
                # TextGrid：用最终 wav+pfml 现跑 TIFA 对齐；副语言标记激活 -L
                # 主语言前缀剥除以数据集语言为准（TIFA -l），副语言音素保留 lang/ 前缀
                ok, _tg, _err, extra = onnxr.run_align_onnx(
                    wav_path, pfml_text, lang, tg_dir,
                    extra_langs=other_langs if lang != dataset_lang else None,
                    main_lang=dataset_lang,
                )
                if not ok:
                    tg_failed += 1
                exported += 1
                # 聚合 metrics + scores 到语言文件夹
                fkey = folder_lang or "_root"
                metrics = extra.get("metrics") or {}
                if metrics:
                    folder_metrics.setdefault(fkey, []).append(
                        {"identifier": base, **metrics})
                sw = extra.get("score_words") or []
                if sw:
                    folder_scores.setdefault(fkey, []).append(
                        {"identifier": base, "words": sw})
                done += 1
                prog.emit(done)
        # 写 statistics（分语言时每个子文件夹一份，含图表）
        if lbl is not None:
            lbl.emit("正在生成 statistics 统计文件…")
        for fkey, metrics_list in folder_metrics.items():
            stat_dir = (Path(out_dir) / (fkey if fkey != "_root" else "")
                        / "TextGrid" / "statistics")
            onnxr.write_statistics(stat_dir, metrics_list,
                                   folder_scores.get(fkey, []))
        if lbl is not None:
            prog.emit(total_count + 1)
        return exported, tg_failed, out_dir

    # ---------------- 设置 ----------------
    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec_():
            # 若 ASR 模型路径改变，重置引擎
            self.asr_engine = None
            self.reload_vst_panel()
            self._apply_tooltips()
            self.statusBar().showMessage("设置已更新")

    # ---------------- 关闭 ----------------
    def closeEvent(self, e):
        if self.project.dirty:
            r = QMessageBox.question(
                self, "未保存修改",
                "工程有未保存的修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if r == QMessageBox.Save:
                self.save_project()
            elif r == QMessageBox.Cancel:
                e.ignore()
                return
        # 保存窗口尺寸和最大化状态
        self.config["window_width"] = self.width()
        self.config["window_height"] = self.height()
        self.config["window_maximized"] = self.isMaximized()
        # 保存底部面板分隔条尺寸
        self.config["panel_sizes"] = self.bottom_split.sizes()
        self.config.save()
        # 卸载常驻 ASR 服务，释放内存
        if self._asr_service is not None:
            try:
                self._asr_service.unload()
            except Exception:
                pass
        e.accept()
