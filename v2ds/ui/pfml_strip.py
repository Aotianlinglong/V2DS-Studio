# coding=utf-8
"""PFML 字符条：把歌词铺成一个个方块，自动换行，像 MaxLabel 那样直观显示标注。

- 一个字符一块（拉丁文按词）
- 单击选中 / 拖拽选一段 / 双击打开发音弹窗
- 方块下方显示指定读音
- 词边界：底部青色横杠
- 插入音素：+n 方块（占一格宽）
"""
from __future__ import annotations
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QTimer, QPoint
from PyQt5.QtGui import QPainter, QFont, QColor, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy


def _is_latin(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "'’-")


# 语言 → 方块底色（主语言/无区间用中性灰；不同语言用低饱和色，不加图例）
LANG_BG = {
    "zh": QColor(70, 90, 70),      # 暗绿
    "ja": QColor(90, 80, 60),      # 暗琥珀
    "en": QColor(60, 80, 100),     # 暗蓝
    "yue": QColor(90, 60, 70),     # 暗紫红
}
_DEFAULT_BG = QColor(42, 42, 42)

# 多音字底部横杠：低饱和橙（警示但不刺眼，适配黑色背景）
_POLY_BAR = QColor(201, 149, 86)


class PfmlCharStrip(QWidget):
    """字符条。text + pfml dict → 可视化方块，自动换行。"""

    selectionChanged = pyqtSignal(int, int)   # begin, end（字符区间）
    insertDoubleClicked = pyqtSignal(int, str, object)  # 插入音素位置 + 音素符号 + 唯一 key (b,k)
    insertAtRequested = pyqtSignal(int, object)  # 右键请求插入音素：位置 + 全局pos
    insRightClicked = pyqtSignal(object, object)  # 右键插入音素格：key(b,k) + 全局pos
    insSelected = pyqtSignal(int)               # 选中了哪个插入音素 pos（-1=取消）
    middleClicked = pyqtSignal(int, int)       # 中键点击方格本体：取消该处分词 (b,e)
    readingMiddleClicked = pyqtSignal(int, int)  # 中键点击方格下方读音：取消发音 (b,e)
    longPressRequested = pyqtSignal(int, int, int, int)  # 长按字符 (b,e,全局x,全局y)

    LONG_PRESS_MS = 260        # 长按判定时长
    LONG_PRESS_MOVE = 6        # 长按期间允许的移动抖动（px）

    CELL = 26          # 方块边长
    PAD = 4
    GAP = 12           # 空格断句处的额外间距
    READING_H = 18     # 读音行高度（要给 g/q/y/p 等下伸字母留空间）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.pfml = {}
        self.lang = "zh"
        self.sel_b = -1
        self.sel_e = -1
        self._sel_ins = -1
        self._press_pos = None
        self._press_xy = None
        self._lp_cell = None
        self._lp_timer = QTimer(self)
        self._lp_timer.setSingleShot(True)
        self._lp_timer.setInterval(self.LONG_PRESS_MS)
        self._lp_timer.timeout.connect(self._on_lp_timeout)
        self._word_reading = {}
        self._groups = []  # 词格跨度 [(b,e),...]（无数据时为空）
        self._slot_uids = None  # 每位置格子 uid（无数据时 None → 只按空格断句）
        self._poly = {}  # 多音字/多音词跨度 {起始偏移: 结束偏移}（显示多音时非空）
        # 预计算每个方块：[(type, b, e, x, y, w, txt)]
        self._cells = []
        self._row_h = self.CELL + self.READING_H + self.PAD
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setMinimumHeight(self._row_h + 4)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#1e1e1e;")

    # ---------- 数据 ----------
    def set_data(self, text: str, pfml: dict, lang: str, groups=None, uids=None,
                 poly=None):
        self.text = text or ""
        self.lang = lang or "zh"
        # 日语/多字词"大方格"跨度 [(b,e),...]，供顶部落色（仅 2+ 字词格）
        self._groups = list(groups or [])
        # 多音字/多音词跨度（底部橙色横杠）
        self._poly = dict(poly or {})
        # 每位置的格子 uid（>=0 属某格 → 画方块；<0 空格/独立标点 → 只留间距）
        self._slot_uids = list(uids) if uids is not None else None
        # 编号由 main_window 在文本编辑时用 diff 实时迁移并写回存储，
        # 这里直接使用传入的 pfml（不再做字符匹配）。
        self.pfml = {k: list(v) for k, v in (pfml or {}).items()}
        self.sel_b = self.sel_e = -1
        self._sel_ins = -1
        self._layout()
        self.update()
    def _overrides_at(self, b: int, e: int):
        out = []
        for o in self.pfml.get("overrides", []):
            if o["begin"] == o["end"]:
                continue
            if o["begin"] <= b and e <= o["end"]:
                out.append(o)
        return out

    def _is_word(self, b: int, e: int) -> bool:
        for w in self.pfml.get("words", []):
            if w["begin"] == b and w["end"] == e:
                return True
        return False

    def _group_color(self, b: int, e: int):
        """词格颜色：基于跨度散列得到稳定的低饱和不同色（同格同色、异格异色）。"""
        from PyQt5.QtGui import QColor
        h = (b * 137.508 + e * 93.7) % 360.0
        return QColor.fromHslF(h / 360.0, 0.5, 0.83)

    def _lang_at(self, b: int, e: int):
        """返回覆盖 [b,e) 的 span 语言；无则 None。"""
        for s in self.pfml.get("spans", []):
            if s["begin"] <= b and e <= s["end"]:
                return s.get("language")
        return None

    def _layout(self):
        """按当前宽度自动换行布局。"""
        from PyQt5.QtGui import QFontMetrics
        ins_map = {}
        for o in self.pfml.get("overrides", []):
            if o["begin"] != o["end"]:
                continue
            ins_map.setdefault(o["begin"], []).append(" ".join(o.get("phonemes", [])))

        # 读音（含单字和词）：预计算文本宽度，布局时把后续方块推开
        reading_font = QFont("Microsoft YaHei", 8)
        rfm = QFontMetrics(reading_font)
        self._word_reading = {}
        for o in self.pfml.get("overrides", []):
            if o["begin"] == o["end"]:
                continue  # 插入音素单独处理
            if not o.get("phonemes"):
                continue
            ph = " ".join(o["phonemes"])
            tw = rfm.horizontalAdvance(ph)
            span = (o["end"] - o["begin"]) * (self.CELL + self.PAD) - self.PAD
            self._word_reading[o["begin"]] = {
                "end": o["end"], "ph": ph,
                "w": tw,           # 读音文本宽度
                "span": span,      # 覆盖字符的格子总宽
                "need_push": tw > span}

        raw = []
        i = 0
        n = len(self.text)
        while i < n:
            ch = self.text[i]
            if _is_latin(ch):
                j = i
                while j < n and _is_latin(self.text[j]):
                    j += 1
                raw.append((i, j, self.text[i:j]))
                i = j
            else:
                raw.append((i, i + 1, ch))
                i += 1

        avail = max(self.width() - self.PAD * 2, self.CELL * 2)
        self._cells = []
        x = self.PAD
        row = 0
        cur_word = None  # {"end","start_x","w","row"}
        pending_gap = False  # 遇到空格：下一个真实字符前加一段断句间距
        for b, e, txt in raw:
            # 空格不占格子，只作为断句分隔；独立标点同理（不属于任何格子）
            # 但分隔位置上可能挂有插入音素（如后置 EP 落在尾随标点位置）——先渲染，否则永远不显示
            if txt.isspace() or (self._slot_uids is not None and
                                 not any(self._slot_uids[p] >= 0 for p in range(b, e))):
                for k, sym in enumerate(ins_map.get(b, [])):
                    w = self.CELL
                    if x + w > avail:
                        x = self.PAD
                        row += 1
                        if cur_word is not None:
                            cur_word["start_row"] = row
                            cur_word["start_x"] = x
                    self._cells.append({"type": "ins", "b": b, "e": b,
                                        "x": x, "row": row, "w": w, "txt": "+" + sym,
                                        "sym": sym, "key": (b, k)})
                    x += w + self.PAD
                pending_gap = True
                continue
            # 上一个字符后有空格 → 放置前多空一段（行首不加）
            if pending_gap and x > self.PAD:
                x += self.GAP
            pending_gap = False
            for k, sym in enumerate(ins_map.get(b, [])):
                w = self.CELL
                if x + w > avail:
                    x = self.PAD
                    row += 1
                    if cur_word is not None:
                        cur_word["start_row"] = row
                        cur_word["start_x"] = x
                self._cells.append({"type": "ins", "b": b, "e": b,
                                    "x": x, "row": row, "w": w, "txt": "+" + sym,
                                    "sym": sym, "key": (b, k)})
                x += w + self.PAD
            if len(txt) > 1 and txt.isascii():
                w = max(self.CELL, len(txt) * 8)
            else:
                w = self.CELL
            if x + w > avail:
                x = self.PAD
                row += 1
                if cur_word is not None:
                    cur_word["start_row"] = row
                    cur_word["start_x"] = x
            self._cells.append({"type": "char", "b": b, "e": e,
                                "x": x, "row": row, "w": w, "txt": txt})
            x += w + self.PAD
            # 进入新词
            if b in self._word_reading:
                wr = self._word_reading[b]
                cur_word = {"wb": b, "end": wr["end"], "start_x": x - w - self.PAD,
                            "w": wr["w"], "row": row,
                            "need_push": wr["need_push"],
                            "cell_start": len(self._cells) - 1}
            # 当前词结束：若读音比格区宽，词格整体右移 half 让出左侧，后续右移 extra 让出右侧
            if cur_word is not None and e >= cur_word["end"] and row == cur_word["row"]:
                word_w = x - cur_word["start_x"] - self.PAD
                # 用实际格子宽度判断（拉丁合并块后，预估值不准）
                if cur_word.get("wb") in self._word_reading:
                    self._word_reading[cur_word["wb"]]["span"] = word_w
                need = cur_word["w"] > word_w
                if need:
                    extra = cur_word["w"] - word_w
                    if extra > 0:
                        half = extra / 2.0
                        # 词格右移 half，使左侧留出等距空间
                        cs = cur_word.get("cell_start", 0)
                        for cc in self._cells[cs:]:
                            if cc["row"] == row:
                                cc["x"] = cc["x"] + half
                        # 后续方块右移 extra，使右侧与左侧对称
                        x += extra
                cur_word = None
        for k, sym in enumerate(ins_map.get(n, [])):
            w = self.CELL
            if x + w > avail:
                x = self.PAD
                row += 1
            self._cells.append({"type": "ins", "b": n, "e": n,
                                "x": x, "row": row, "w": w, "txt": "+" + sym,
                                "sym": sym, "key": (n, k)})
            x += w + self.PAD
        n_rows = row + 1
        # 按格子真实几何重算每个词读音的 span（拉丁合并块后，原按字符数算的 span 偏大）
        first_by_b = {}
        for c in self._cells:
            if c["type"] == "char" and c["b"] not in first_by_b:
                first_by_b[c["b"]] = c
        for wb, wr in self._word_reading.items():
            fc = first_by_b.get(wb)
            if not fc:
                continue
            fe = wr["end"]
            # 同区间内、同行的最右格子
            right = fc
            for c in self._cells:
                if c["type"] != "char":
                    continue
                if c["row"] != fc["row"]:
                    continue
                if c["b"] >= wb and c["e"] <= fe and c["x"] + c["w"] > right["x"] + right["w"]:
                    right = c
            wr["span"] = (right["x"] + right["w"]) - fc["x"]
            wr["need_push"] = wr["w"] > wr["span"]
        self.setMinimumHeight(n_rows * self._row_h + 6)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.text:
            self._layout()
            self.update()

    # ---------- 绘制 ----------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        ch_h = self.CELL
        # 建立字符区间 -> 方块 的索引，用于定位词起点的横坐标
        cell_by_b = {}
        for c in self._cells:
            if c["type"] == "char":
                cell_by_b.setdefault(c["b"], c)

        font = QFont("Microsoft YaHei", 10)
        p.setFont(font)
        for c in self._cells:
            x = int(c["x"])
            y = 2 + c["row"] * self._row_h
            w = int(c["w"])
            b, e = c["b"], c["e"]
            selected = (self.sel_b >= 0 and b >= self.sel_b and e <= self.sel_e)
            if c["type"] == "ins":
                rect = QRect(x, y + ch_h // 4, ch_h, ch_h // 2)
                if c.get("key") == self._sel_ins:
                    p.setPen(QPen(QColor("#e06060"), 2))
                    p.setBrush(QBrush(QColor("#5a2a2a")))
                else:
                    p.setPen(QPen(QColor("#888")))
                    p.setBrush(QBrush(QColor("#2a2a2a")))
                p.drawRect(rect)
                p.setPen(QPen(QColor("#bbb")))
                p.drawText(rect, Qt.AlignCenter, c["txt"])
                continue
            rect = QRect(x, y, w, ch_h)
            if selected:
                p.setBrush(QBrush(QColor("#2d5f8a")))
            else:
                lang = self._lang_at(b, e)
                p.setBrush(QBrush(LANG_BG.get(lang, _DEFAULT_BG)))
            p.setPen(QPen(QColor("#555")))
            p.drawRect(rect)
            p.setPen(QPen(QColor("#eee")))
            p.drawText(rect, Qt.AlignCenter, c["txt"])
            if self._is_word(b, e):
                p.setPen(QPen(QColor("#4fd1c5"), 3))
                p.drawLine(x, y + ch_h - 2, x + w, y + ch_h - 2)

        # 多音字/多音词：每个所属方格各自画一条低饱和橙横杠（不跨越词格）
        if self._poly:
            poly_spans = list(self._poly.items())
            for c in self._cells:
                if c["type"] != "char":
                    continue
                b, e = c["b"], c["e"]
                for pb, pe in poly_spans:
                    if pb <= b and e <= pe:
                        y0 = 2 + c["row"] * self._row_h + ch_h - 4
                        p.fillRect(int(c["x"]), y0, int(c["w"]), 4, _POLY_BAR)
                        break

        # 词格（2+ 字）：在每个小方块最上方画一条低饱和色带，象征一个词格
        # 每个词格用稳定的伪随机低饱和色（基于跨度散列），同格同色、异格异色
        if self._groups:
            for gb, ge in self._groups:
                col = self._group_color(gb, ge)
                p.setPen(QPen(QColor(col), 1))
                for c in self._cells:
                    if c["type"] != "char":
                        continue
                    cb, ce = c["b"], c["e"]
                    if gb <= cb and ce <= ge:
                        x0 = int(c["x"]); w0 = int(c["w"])
                        y0 = 2 + c["row"] * self._row_h
                        p.fillRect(x0, y0, w0, 4, col)

        # 读音（单字与词）：以词的跨度中心为基准，严格居中；左右溢出相等
        for b, wr in self._word_reading.items():
            bc = cell_by_b.get(b)
            if not bc:
                continue
            y = 2 + bc["row"] * self._row_h + ch_h
            span = wr.get("span", int(bc["w"]))
            cx = int(bc["x"]) + span / 2.0        # 词跨度中心
            f = QFont("Microsoft YaHei", 8)
            p.setFont(f)
            p.setPen(QPen(QColor("#5fd3c0")))
            tw = p.fontMetrics().horizontalAdvance(wr["ph"])
            x0 = cx - tw / 2.0
            # 若左溢出超出左边缘，强制从 PAD 开始，避免被裁
            if x0 < self.PAD:
                x0 = float(self.PAD)
            p.drawText(QRect(int(x0), y, tw + self.PAD, self.READING_H),
                       Qt.AlignLeft | Qt.AlignVCenter, wr["ph"])
        p.end()

    # ---------- 鼠标 ----------
    def _cell_at(self, mx: float, my: float):
        for c in self._cells:
            cy = 2 + c["row"] * self._row_h
            if c["x"] <= mx <= c["x"] + c["w"] and cy <= my <= cy + self._row_h:
                return c
        return None

    def mousePressEvent(self, e):
        c = self._cell_at(e.x(), e.y())
        if e.button() == Qt.RightButton:
            # 右键插入音素格 → 独立小菜单（设置语言）；右键其他 → 主菜单
            if c is not None and c["type"] == "ins":
                self.insRightClicked.emit(c.get("key"), e.globalPos())
                return
            pos = c["b"] if c else 0
            self.insertAtRequested.emit(pos, e.globalPos())
            return
        if e.button() == Qt.MiddleButton:
            # 中键 = 按点击部位精确取消：
            # 插入音素格 → 取消该音素；方格下方读音区 → 取消发音；方格本体 → 取消分词
            if c is None:
                return
            if c["type"] == "ins":
                self.insertDoubleClicked.emit(c["b"], c.get("sym", ""), c.get("key"))
                return
            cy = 2 + c["row"] * self._row_h
            if e.y() >= cy + self.CELL:
                # 方格下方读音行
                self.readingMiddleClicked.emit(c["b"], c["e"])
            else:
                self.middleClicked.emit(c["b"], c["e"])
            return
        if c is None:
            self.sel_b = self.sel_e = -1
            self.selectionChanged.emit(-1, -1)
            self.update()
            return
        self._press_pos = c
        if e.button() == Qt.LeftButton:
            if c["type"] == "ins":
                self._sel_ins = c.get("key")
                self.sel_b = self.sel_e = -1
                self.selectionChanged.emit(-1, -1)
                self.insSelected.emit(c["b"])
            else:
                self._sel_ins = -1
                self.insSelected.emit(-1)
                self.sel_b = c["b"]
                self.sel_e = c["e"]
                self.selectionChanged.emit(self.sel_b, self.sel_e)
                # 长按选音：记录按下位置，静止超时后弹出发音候选下拉
                self._press_xy = (e.x(), e.y())
                self._lp_cell = c
                self._lp_timer.start()
            self.update()

    def mouseMoveEvent(self, e):
        if self._press_pos is None:
            return
        # 按下后若移动超阈值（不是长按而是拖拽选段）→ 取消长按计时
        if self._lp_timer.isActive() and self._press_xy:
            if abs(e.x() - self._press_xy[0]) > self.LONG_PRESS_MOVE or \
                    abs(e.y() - self._press_xy[1]) > self.LONG_PRESS_MOVE:
                self._lp_timer.stop()
        c = self._cell_at(e.x(), e.y())
        if c is None or c["type"] != "char":
            return
        b = min(self._press_pos["b"], c["b"])
        en = max(self._press_pos["e"], c["e"])
        self.sel_b, self.sel_e = b, en
        self.selectionChanged.emit(b, en)
        self.update()

    def mouseReleaseEvent(self, e):
        self._lp_timer.stop()
        self._press_pos = None
        self._lp_cell = None

    def _on_lp_timeout(self):
        """静止长按达到阈值：发出发音候选下拉请求（全局位置=方块左上）。"""
        c = self._lp_cell
        self._lp_cell = None
        self._press_pos = None
        if c is None or c["type"] != "char":
            return
        pt = self.mapToGlobal(QPoint(int(c["x"]), 2 + c["row"] * self._row_h))
        self.longPressRequested.emit(c["b"], c["e"], pt.x(), pt.y())

    def cancel_long_press(self):
        """弹层收起/选定后由上层调用，复位长按状态。"""
        self._lp_timer.stop()
        self._press_pos = None
        self._lp_cell = None

    def mouseDoubleClickEvent(self, e):
        c = self._cell_at(e.x(), e.y())
        if c is None:
            return
        if c["type"] == "ins":
            self.insertDoubleClicked.emit(c["b"], c.get("sym", ""), c.get("key"))


class PronunciationPopup(QWidget):
    """长按弹出的发音候选下拉：拖动高亮、松手选定、移出取消。

    - ``grab=True``（长按路径）：弹出时 grabMouse，承接还在按住的鼠标，
      用户不必抬手，直接拖到目标项松开即选定。
    - ``grab=False``（双击/兼容路径）：普通点击选项选定。
    ``selected`` 发选项行号（-1 = 取消）。
    """
    selected = pyqtSignal(int)
    ROW_H = 28
    MAX_W = 260

    def __init__(self, parent=None):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self._rows = []      # [(label, payload)]
        self._hover = -1
        self._grabbed = False

    def set_rows(self, rows):
        self._rows = list(rows)
        self.setFixedSize(self.MAX_W, self.ROW_H * len(self._rows))

    def open_at(self, x, y, grab=True):
        self._hover = -1
        self._grabbed = grab
        self.move(int(x), int(y))
        self.show()
        self.raise_()
        if grab:
            self.grabMouse()

    def _row_at(self, pos):
        n = len(self._rows)
        if 0 <= pos.y() < self.ROW_H * n:
            return pos.y() // self.ROW_H
        return -1

    def _highlight(self, row):
        if row != self._hover:
            self._hover = row
            self.update()

    def mousePressEvent(self, e):
        self._highlight(self._row_at(e.pos()))

    def mouseMoveEvent(self, e):
        self._highlight(self._row_at(e.pos()))

    def mouseReleaseEvent(self, e):
        row = self._row_at(e.pos())
        if self._grabbed:
            self.releaseMouse()
            self._grabbed = False
        self.hide()
        self.selected.emit(row)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#2a2a2a"))
        p.setFont(QFont("Microsoft YaHei", 9))
        for i, (label, _payload) in enumerate(self._rows):
            y = i * self.ROW_H
            if i == self._hover:
                p.fillRect(0, y, self.width(), self.ROW_H, QColor("#2d5f8a"))
            p.setPen(QPen(QColor("#eee")))
            p.drawText(QRect(8, y, self.width() - 16, self.ROW_H),
                       Qt.AlignVCenter | Qt.AlignLeft, label)
        p.setPen(QPen(QColor("#555")))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.end()
