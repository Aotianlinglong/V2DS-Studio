# coding=utf-8
"""波形 + 梅尔频谱显示组件。

结构：WaveformView 是一个垂直 QSplitter：
  - 上：WaveformCanvas（波形）
  - 下：SpectrogramCanvas（梅尔频谱）
两者共享 AudioDisplayState（缩放/平移/选区/播放头/标记）。

交互（按用户已确认的方案）：
  - 滚轮：以鼠标位置为锚缩放画布
  - 左键拖拽：框选选区（松开后可在主窗口按 M 建标记）
  - 左键单击：移动播放头
  - 标记端点：hover 到端点把手，拖拽调整 start/end（松开才刷新列表）
  - 播放中：画布自动跟随播放头；停止后手动操作
  - 框选/端点不超出音频 [0, duration]
"""
from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QSizePolicy,
)

DRAG_THRESHOLD = 5          # 判定“点击 vs 拖拽”的像素阈值
ENDPOINT_HIT_PX = 6         # 端点把手命中半径（像素）


# ---- 梅尔频谱 colormap（近似 viridis）----
def _make_cmap() -> np.ndarray:
    # 简易 256 色渐变，从黑紫到黄绿到亮黄
    stops = np.array([
        [12, 5, 51], [50, 20, 90], [70, 70, 130], [50, 130, 120],
        [90, 190, 80], [190, 220, 60], [250, 250, 120],
    ], dtype=np.float32)
    n = 256
    pos = np.linspace(0, 1, len(stops))
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for c in range(3):
        cmap[:, c] = np.interp(np.linspace(0, 1, n), pos, stops[:, c]).astype(np.uint8)
    return cmap


_CMAP = _make_cmap()


def _pick_ruler_step(sec_per_px: float) -> float:
    target_px = 90.0
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if step / sec_per_px >= target_px:
            return step
    return 600.0


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    if t >= 3600:
        return f"{int(t // 3600)}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}"
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:05.2f}"


class AudioDisplayState:
    """波形/频谱共享的音频数据与视图状态。"""

    def __init__(self):
        self.wave: np.ndarray | None = None       # float32 mono
        self.sr: int = 44100
        self.duration: float = 0.0
        self.mel: np.ndarray | None = None        # uint8 [n_mels, n_frames]
        self.mel_fps: float = 0.0
        self.view_start: float = 0.0              # 可视区左端时间(秒)
        self.sec_per_px: float = 0.01
        self.sel_start: float = -1.0              # -1 表示无选区
        self.sel_end: float = -1.0
        self.play_pos: float = 0.0
        self.marks: list = []                     # list[Mark]
        self.selected_mark: int = -1
        self.auto_follow: bool = True
        self.edited: bool = False              # 音频被编辑（删除/静音/插入），保存时写回 wav
        self.peak: float = 1.0                 # 当前文件最大振幅（波形自动匹配用）

    def has_audio(self) -> bool:
        return self.wave is not None and len(self.wave) > 0

    def set_audio(self, wave: np.ndarray, sr: int, mel=None, mel_fps=0.0):
        self.wave = wave
        self.sr = sr
        self.duration = float(len(wave)) / sr if len(wave) else 0.0
        self.mel = mel
        self.mel_fps = mel_fps
        self.view_start = 0.0
        # 默认整段可见：预留左右 5% 边距
        if self.duration > 0:
            self.sec_per_px = (self.duration * 1.08) / 1000.0
        else:
            self.sec_per_px = 0.01
        self.sel_start = self.sel_end = -1.0
        self.play_pos = 0.0
        self.auto_follow = True
        self.edited = False
        self.peak = float(max(1e-6, np.max(np.abs(wave)))) if len(wave) else 1.0

    def visible_seconds(self, width_px: int) -> float:
        return max(0.0, (width_px - 40)) * self.sec_per_px

    def clamp_view(self, width_px: int):
        vis = self.visible_seconds(width_px)
        max_start = max(0.0, self.duration - vis)
        self.view_start = min(max(0.0, self.view_start), max_start)


class _BaseCanvas(QWidget):
    """波形/频谱画布基类：统一缩放、选区、播放头、标记端点交互与覆盖绘制。"""

    selChanged = pyqtSignal(float, float)          # (start, end)，无选区为 -1
    playheadMoved = pyqtSignal(float)
    markEndpointDragging = pyqtSignal(int)
    markUpdated = pyqtSignal(int)                  # 端点拖拽完成，通知刷新列表
    viewChanged = pyqtSignal()                     # 缩放/平移变化，通知另一画布同步
    syncDrag = pyqtSignal()                        # 选区/标记端点拖拽，通知另一画布同步
    playToggle = pyqtSignal()                      # 右键点击 → 播放/暂停
    canvasClicked = pyqtSignal()                   # 画布被点击 → 取消文本框焦点
    markClicked = pyqtSignal(int)                  # 点击标记序号 → 选中该标记

    def __init__(self, state: AudioDisplayState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMouseTracking(True)
        # 画布可通过点击获得焦点：使速度/文本等输入框在点击画布时由系统级
        # 焦点切换确定失焦（仅在 NoFocus 控件的点击槽里 clearFocus，在
        # Windows 上会被随后的鼠标事件后处理还原焦点）。
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._press_pos = None
        self._drag_mode = None        # None / 'select' / 'endpoint' / 'sel_start' / 'sel_end' / 'playhead'
        self._drag_mark = -1
        self._drag_is_start = True
        self._pan_start = 0.0
        self._hover_endpoint = (-1, False)  # (mark_idx, is_start) or (-1, False)
        self._last_overlay_xs = []    # 上一帧覆盖层端点像素，拖拽时一并重绘防滞留

    # ---- 坐标换算 ----
    def _edge(self) -> float:
        return 20.0

    def _plot_w(self) -> float:
        return max(1.0, self.width() - self._edge() * 2)

    def time_to_x(self, t: float) -> float:
        return self._edge() + (t - self.state.view_start) / self.state.sec_per_px

    def _time_to_disp_x(self, t: float) -> float:
        """音频在画布中的显示 x：音频短于可视时拉伸填满绘制区（右侧不留白），
        否则按真实时间比例。波形/频谱/标记/刻度共用，保证三者不错位。"""
        st = self.state
        if st.duration > 0 and st.visible_seconds(self.width()) > st.duration:
            return self._edge() + (t / st.duration) * self._plot_w()
        return self.time_to_x(t)

    def x_to_time(self, x: float) -> float:
        st = self.state
        # 与 _time_to_disp_x 互为逆映射：拉伸模式也要按拉伸比例反算
        if st.duration > 0 and st.visible_seconds(self.width()) > st.duration:
            t = (x - self._edge()) / self._plot_w() * st.duration
        else:
            t = st.view_start + (x - self._edge()) * st.sec_per_px
        return float(np.clip(t, 0.0, st.duration))

    # ---- 覆盖层绘制：选区 / 播放头 / 标记 ----
    def draw_overlay(self, p: QPainter, height: float):
        st = self.state
        edge0 = self._edge()
        edge1 = edge0 + self._plot_w()

        def _clip(t):
            x = self._time_to_disp_x(t)
            return float(np.clip(x, edge0, edge1))

        # 选区（部分在可视区时显示可视部分，与波形一样受画布显示区限制）
        if st.sel_start >= 0 and st.sel_end >= st.sel_start:
            x0 = _clip(st.sel_start)
            x1 = _clip(st.sel_end)
            if x1 >= edge0 and x0 <= edge1:
                p.fillRect(QRectF(x0, 0, max(0.0, x1 - x0), height), QColor(90, 150, 255, 46))
                pen = QPen(QColor(120, 170, 255, 200)); pen.setWidth(1)
                p.setPen(pen)
                p.drawLine(QPointF(x0, 0), QPointF(x0, height))
                p.drawLine(QPointF(x1, 0), QPointF(x1, height))
        # 标记（虚线始末 + 端点把手；与画布显示区求交，部分可见时画可见部分，不再整条消失）
        for idx, m in enumerate(st.marks):
            raw0 = self._time_to_disp_x(m.start)
            raw1 = self._time_to_disp_x(m.end)
            if raw1 < edge0 or raw0 > edge1:
                continue  # 整个标记在可视区外，不画
            x0 = float(np.clip(raw0, edge0, edge1))
            x1 = float(np.clip(raw1, edge0, edge1))
            vis0 = edge0 - 1e-9 <= raw0 <= edge1 + 1e-9
            vis1 = edge0 - 1e-9 <= raw1 <= edge1 + 1e-9
            sel = (idx == st.selected_mark)
            if sel:
                # 无色相灰白高亮（仅选中时填充；未选中不填充颜色，只有白虚线）
                p.fillRect(QRectF(x0, 0, max(0.0, x1 - x0), height), QColor(215, 215, 225, 26))
            dash = QPen(QColor(255, 255, 255, 235) if sel else QColor(205, 210, 220, 210))
            dash.setWidth(1)
            if sel:
                dash.setStyle(Qt.SolidLine)
            else:
                dash.setStyle(Qt.DashLine)
            p.setPen(dash)
            # 只在端点真在可视区内时才画边界竖线，避免出界端点被 clip 后"卡在边缘"
            if vis0:
                p.drawLine(QPointF(x0, 0), QPointF(x0, height))
            if vis1:
                p.drawLine(QPointF(x1, 0), QPointF(x1, height))
            # 端点把手（白色，与红色播放头区分；仅当该端点真在可视区内时画）
            p.setBrush(QColor(240, 240, 245, 230))
            p.setPen(Qt.NoPen)
            if vis0:
                p.drawRect(QRectF(raw0 - 3, height * 0.5 - 6, 6, 12))
            if vis1:
                p.drawRect(QRectF(raw1 - 3, height * 0.5 - 6, 6, 12))
            # 序号标签（标记片段中间，可点击选中；只要序号位置在可视区内就画）
            cx = (x0 + x1) / 2
            if edge0 <= cx <= edge1:
                f = p.font()
                f.setPointSize(8)
                f.setBold(True)
                p.setFont(f)
                p.setPen(QPen(QColor(230, 235, 245, 235)))
                num_rect = QRectF(cx - 20, 2, 40, 14)
                p.drawText(num_rect, Qt.AlignCenter, str(idx + 1))
        # 播放头（单点线，仅当在可视区内时画）
        if 0.0 <= st.play_pos <= st.duration:
            xp = self._time_to_disp_x(st.play_pos)
            if edge0 - 1e-9 <= xp <= edge1 + 1e-9:
                pen = QPen(QColor(255, 60, 60, 230)); pen.setWidth(1)
                p.setPen(pen)
                p.drawLine(QPointF(xp, 0), QPointF(xp, height))

    def _repaint_overlay(self):
        """局部重绘覆盖层（选区/标记边界/播放头）所在区域，两画布同步刷新。
        同时纳入上一帧端点位置，拖拽时清除旧线，避免滞留。"""
        st = self.state
        xs = []
        if st.sel_start >= 0:
            xs.append(self._time_to_disp_x(st.sel_start))
            xs.append(self._time_to_disp_x(st.sel_end))
        for m in st.marks:
            xs.append(self._time_to_disp_x(m.start))
            xs.append(self._time_to_disp_x(m.end))
        if 0.0 <= st.play_pos <= st.duration:
            xs.append(self._time_to_disp_x(st.play_pos))
        cur = xs[:]
        if self._last_overlay_xs:
            xs = xs + self._last_overlay_xs
        self._last_overlay_xs = cur
        if xs:
            x0 = max(0, int(min(xs)) - 4)
            x1 = int(max(xs)) + 4
            self.update(x0, 0, max(1, x1 - x0), max(1, self.height()))
        else:
            self.update()

    # ---- 缩放 ----
    def wheelEvent(self, e):
        st = self.state
        if not st.has_audio():
            return
        factor = 0.85 if e.angleDelta().y() > 0 else 1.18
        anchor_t = self.x_to_time(e.pos().x())
        new_spp = st.sec_per_px * factor
        new_spp = float(np.clip(new_spp, st.duration / 100000.0 if st.duration > 0 else 1e-6, st.duration * 2.0 if st.duration > 0 else 1e6))
        # 保持锚点时间在鼠标处
        x = e.pos().x()
        st.sec_per_px = new_spp
        st.view_start = anchor_t - (x - self._edge()) * new_spp
        st.clamp_view(self.width())
        self.update()
        self.viewChanged.emit()
        e.accept()

    # ---- 鼠标交互 ----
    def _hit_endpoint(self, pos) -> tuple[int, bool] | None:
        st = self.state
        for idx, m in enumerate(st.marks):
            x0 = self._time_to_disp_x(m.start)
            x1 = self._time_to_disp_x(m.end)
            if abs(pos.x() - x0) <= ENDPOINT_HIT_PX:
                return (idx, True)
            if abs(pos.x() - x1) <= ENDPOINT_HIT_PX:
                return (idx, False)
        return None

    def _hit_sel_boundary(self, pos) -> str | None:
        """检测鼠标是否在选区左/右边界上，返回 'sel_start' / 'sel_end' 或 None。"""
        st = self.state
        if st.sel_start < 0 or st.sel_end < st.sel_start:
            return None
        x0 = self._time_to_disp_x(st.sel_start)
        x1 = self._time_to_disp_x(st.sel_end)
        if abs(pos.x() - x0) <= ENDPOINT_HIT_PX:
            return "sel_start"
        if abs(pos.x() - x1) <= ENDPOINT_HIT_PX:
            return "sel_end"
        return None

    def _hit_mark_number(self, pos) -> int | None:
        """点击标记中间序号 → 返回标记索引。"""
        st = self.state
        edge0 = self._edge()
        edge1 = edge0 + self._plot_w()
        for idx, m in enumerate(st.marks):
            raw0 = self._time_to_disp_x(m.start)
            raw1 = self._time_to_disp_x(m.end)
            if raw1 < edge0 - 1 or raw0 > edge1 + 1:
                continue
            x0 = float(np.clip(raw0, edge0, edge1))
            x1 = float(np.clip(raw1, edge0, edge1))
            cx = (x0 + x1) / 2
            if not (edge0 <= cx <= edge1):
                continue
            # 命中热区比绘制文字稍大，方便点中
            num_rect = QRectF(cx - 26, 0, 52, 20)
            if num_rect.contains(pos):
                return idx
        return None

    def mousePressEvent(self, e):
        self.setFocus(Qt.MouseFocusReason)
        self.canvasClicked.emit()
        if not self.state.has_audio():
            return
        if e.button() == Qt.MiddleButton:
            # 中键（滚轮键）拖拽：平移视图
            self._drag_mode = "pan"
            self._press_pos = e.pos()
            self._pan_start = self.state.view_start
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.RightButton:
            # 右键单击 → 播放/暂停
            self.playToggle.emit()
            return
        if e.button() != Qt.LeftButton:
            return
        # 点击标记序号 → 选中该标记（最高优先）
        num_idx = self._hit_mark_number(e.pos())
        if num_idx is not None:
            self.markClicked.emit(num_idx)
            return
        # 选区边界拖拽（优先于标记端点）
        selb = self._hit_sel_boundary(e.pos())
        if selb is not None:
            self._drag_mode = selb  # 'sel_start' / 'sel_end'
            self._press_pos = e.pos()
            return
        hit = self._hit_endpoint(e.pos())
        if hit is not None:
            self._drag_mode = "endpoint"
            self._drag_mark, self._drag_is_start = hit
            self._press_pos = e.pos()
            self.markEndpointDragging.emit(self._drag_mark)
            return
        self._press_pos = e.pos()
        self._drag_mode = "select"
        self.state.sel_start = self.state.sel_end = -1.0

    def mouseMoveEvent(self, e):
        st = self.state
        # 中键拖拽：平移视图（波形/频谱同步）
        if self._drag_mode == "pan" and self._press_pos is not None:
            dx = e.pos().x() - self._press_pos.x()
            st.view_start = self._pan_start - dx * st.sec_per_px
            st.clamp_view(self.width())
            self.update()
            self.viewChanged.emit()
            return
        # hover 提示：选区边界 / 标记端点
        if self._drag_mode is None:
            if st.sel_start >= 0 and self._hit_sel_boundary(e.pos()):
                self.setCursor(Qt.SizeHorCursor)
            else:
                hit = self._hit_endpoint(e.pos())
                self._hover_endpoint = hit if hit else (-1, False)
                self.setCursor(Qt.SizeHorCursor if hit else Qt.ArrowCursor)
            self.update()
            return
        if self._drag_mode == "select" and self._press_pos is not None:
            t = self.x_to_time(e.pos().x())
            anchor = self.x_to_time(self._press_pos.x())
            st.sel_start = min(anchor, t)
            st.sel_end = max(anchor, t)
            st.sel_start = max(0.0, st.sel_start)
            st.sel_end = min(st.duration, st.sel_end)
            self.selChanged.emit(st.sel_start, st.sel_end)
            self._repaint_overlay()
            self.syncDrag.emit()
        elif self._drag_mode in ("sel_start", "sel_end") and self._press_pos is not None:
            t = self.x_to_time(e.pos().x())
            t = float(np.clip(t, 0.0, st.duration))
            if self._drag_mode == "sel_start":
                if t <= st.sel_end - 0.05:
                    st.sel_start = round(t, 3)
            else:
                if t >= st.sel_start + 0.05:
                    st.sel_end = round(t, 3)
            self.selChanged.emit(st.sel_start, st.sel_end)
            self._repaint_overlay()
            self.syncDrag.emit()
        elif self._drag_mode == "endpoint" and self._drag_mark >= 0:
            m = st.marks[self._drag_mark]
            t = self.x_to_time(e.pos().x())
            t = float(np.clip(t, 0.0, st.duration))
            if self._drag_is_start:
                if t <= m.end - 0.05:
                    m.start = round(t, 3)
            else:
                if t >= m.start + 0.05:
                    m.end = round(t, 3)
            self._repaint_overlay()
            self.syncDrag.emit()

    def mouseReleaseEvent(self, e):
        st = self.state
        if e.button() == Qt.MiddleButton and self._drag_mode == "pan":
            self._drag_mode = None
            self._press_pos = None
            self.setCursor(Qt.ArrowCursor)
            return
        if e.button() != Qt.LeftButton:
            return
        if self._drag_mode in ("sel_start", "sel_end"):
            self._drag_mode = None
            self._press_pos = None
            return
        if self._drag_mode == "select":
            # 若是单击（几乎没移动）→ 移动播放头
            if self._press_pos is not None:
                dx = e.pos().x() - self._press_pos.x()
                if abs(dx) <= DRAG_THRESHOLD:
                    st.play_pos = self.x_to_time(e.pos().x())
                    self.playheadMoved.emit(st.play_pos)
                    st.sel_start = st.sel_end = -1.0
                    self.selChanged.emit(-1.0, -1.0)
                else:
                    if st.sel_end - st.sel_start < 0.01:
                        st.sel_start = st.sel_end = -1.0
                        self.selChanged.emit(-1.0, -1.0)
            self.update()
        elif self._drag_mode == "endpoint" and self._drag_mark >= 0:
            self.markUpdated.emit(self._drag_mark)  # 松开后通知刷新列表
        self._drag_mode = None
        self._press_pos = None
        self._drag_mark = -1


class WaveformCanvas(_BaseCanvas):
    """上：波形画布。"""

    def paintEvent(self, e):
        st = self.state
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(22, 24, 30))
        if not st.has_audio():
            p.setPen(QColor(150, 155, 165))
            p.drawText(self.rect(), Qt.AlignCenter, "未加载音频")
            return
        edge = self._edge()
        plot_w = self._plot_w()
        h = self.height()
        ruler_h = 18.0
        top = ruler_h
        plot_h = max(1.0, h - ruler_h)
        mid = top + plot_h / 2.0
        # 波形按当前文件最大振幅自动匹配（用户确认：按文件最大波形匹配，不用实时）
        amp = plot_h * 0.45 / st.peak
        self._draw_ruler(p, edge, plot_w, top, ruler_h)
        p.setPen(QPen(QColor(60, 64, 74)))
        p.drawLine(QPointF(edge, mid), QPointF(edge + plot_w, mid))
        x0_px = int(edge)
        x1_px = int(edge + plot_w)
        # 音频始终拉伸填满绘制区（缩小到全段后右侧不留白）；标记用 _time_to_disp_x 保持同一比例
        clip = p.clipBoundingRect()
        c0 = max(x0_px, int(np.floor(clip.left())))
        c1 = min(x1_px, int(np.ceil(clip.right())))
        if c1 <= c0:
            c0, c1 = x0_px, x1_px
        n_cols = max(1, c1 - c0)
        t_a = self.x_to_time(c0)
        t_b = self.x_to_time(c1)
        i0 = max(0, int(t_a * st.sr))
        i1 = min(len(st.wave), max(i0 + 1, int(t_b * st.sr)))
        wave = st.wave[i0:i1]
        if len(wave) == 0:
            self.draw_overlay(p, h)
            p.end()
            return
        pen = QPen(QColor(120, 190, 255)); pen.setWidth(1)
        p.setPen(pen)
        block = max(1, len(wave) // n_cols)
        n = len(wave) // block
        if n >= 1:
            view = wave[: n * block].reshape(n, block)
            mn = view.min(axis=1)
            mx = view.max(axis=1)
            for c in range(n):
                x = c0 + c
                if x > x1_px:
                    break
                y0 = mid - mx[c] * amp
                y1 = mid - mn[c] * amp
                p.drawLine(QPointF(x, y0), QPointF(x, y1))
        self.draw_overlay(p, h)
        p.end()

    def _draw_ruler(self, p, edge, plot_w, top, ruler_h):
        st = self.state
        stretch = st.duration > 0 and st.visible_seconds(self.width()) > st.duration
        if stretch:
            # 音频短于可视（拉伸填满画布）：刻度固定在音频总长上，不随缩小继续变化，与画布/标记停住一致
            step = _pick_ruler_step(st.duration / plot_w)
            t0 = 0.0
            t1 = st.duration
        else:
            step = _pick_ruler_step(st.sec_per_px)
            t0 = st.view_start
            t1 = t0 + st.visible_seconds(self.width())
        cur = math.floor(t0 / step) * step
        p.setPen(QPen(QColor(130, 138, 150)))
        p.drawLine(QPointF(edge, top + ruler_h - 1), QPointF(edge + plot_w, top + ruler_h - 1))
        while cur <= t1:
            x = self._time_to_disp_x(cur)
            if edge <= x <= edge + plot_w:
                p.drawLine(QPointF(x, top + ruler_h - 5), QPointF(x, top + ruler_h - 1))
                p.drawText(QRectF(x - 45, top, 90, ruler_h - 5),
                           Qt.AlignCenter | Qt.AlignBottom, _fmt_time(cur))
            cur += step


class SpectrogramCanvas(_BaseCanvas):
    """下：梅尔频谱画布。"""

    def paintEvent(self, e):
        st = self.state
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 18, 24))
        if not st.has_audio():
            p.setPen(QColor(150, 155, 165))
            p.drawText(self.rect(), Qt.AlignCenter, "未加载频谱")
            return
        edge = self._edge()
        plot_w = self._plot_w()
        h = self.height()
        # 画频谱图
        if st.mel is not None and st.mel_fps > 0:
            # 音频始终拉伸填满绘制区（缩小到全段后右侧不留白）
            clip = p.clipBoundingRect()
            c0 = max(int(edge), int(np.floor(clip.left())))
            c1 = min(int(edge + plot_w), int(np.ceil(clip.right())))
            if c1 <= c0:
                c0, c1 = int(edge), int(edge + plot_w)
            t_a = self.x_to_time(c0)
            t_b = self.x_to_time(c1)
            f0 = max(0, int(t_a * st.mel_fps))
            f1 = max(f0 + 1, int(t_b * st.mel_fps))
            f1 = min(st.mel.shape[1], f1)
            block = st.mel[:, f0:f1]
            if block.shape[1] > 1:
                target_w = max(1, c1 - c0)
                if block.shape[1] != target_w:
                    # 线性插值到目标宽度
                    xs = np.linspace(0, block.shape[1] - 1, target_w)
                    from numpy import interp
                    out = np.empty((block.shape[0], target_w), dtype=np.float32)
                    for r in range(block.shape[0]):
                        out[r] = interp(xs, np.arange(block.shape[1]), block[r])
                    block = out
                block = np.flipud(block)  # 垂直翻转：低频在下、高频在上
                rgb = _CMAP[block.astype(np.uint8)]  # (mel, w, 3)
                img = QImage(rgb.data, block.shape[1], block.shape[0],
                             block.shape[1] * 3, QImage.Format_RGB888).copy()
                p.drawImage(QRectF(c0, 0, target_w, h), img)
        self.draw_overlay(p, h)
        p.end()


class WaveformView(QWidget):
    """上下分栏容器：波形（上）+ 频谱（下），分隔条可拖动调整高度。"""

    selChanged = pyqtSignal(float, float)
    playheadMoved = pyqtSignal(float)
    markUpdated = pyqtSignal(int)
    markEndpointDragging = pyqtSignal(int)
    playToggle = pyqtSignal()
    canvasClicked = pyqtSignal()
    markClicked = pyqtSignal(int)

    def __init__(self, state: AudioDisplayState, parent=None):
        super().__init__(parent)
        self.state = state
        self.splitter = QSplitter(Qt.Vertical, self)
        self.wave = WaveformCanvas(state, self)
        self.spec = SpectrogramCanvas(state, self)
        self.splitter.addWidget(self.wave)
        self.splitter.addWidget(self.spec)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.splitter)
        # 转发信号
        self.wave.selChanged.connect(self.selChanged)
        self.spec.selChanged.connect(self.selChanged)
        self.wave.playheadMoved.connect(self.playheadMoved)
        self.spec.playheadMoved.connect(self.playheadMoved)
        self.wave.markUpdated.connect(self.markUpdated)
        self.spec.markUpdated.connect(self.markUpdated)
        self.wave.markEndpointDragging.connect(self.markEndpointDragging)
        self.spec.markEndpointDragging.connect(self.markEndpointDragging)
        self.wave.viewChanged.connect(self._sync_canvases)
        self.spec.viewChanged.connect(self._sync_canvases)
        self.wave.syncDrag.connect(self._on_sync_overlay)
        self.spec.syncDrag.connect(self._on_sync_overlay)
        self.wave.playToggle.connect(self.playToggle)
        self.spec.playToggle.connect(self.playToggle)
        self.wave.canvasClicked.connect(self.canvasClicked)
        self.spec.canvasClicked.connect(self.canvasClicked)
        self.wave.markClicked.connect(self.markClicked)
        self.spec.markClicked.connect(self.markClicked)

    def set_divider_height(self, wave_height: int):
        total = self.splitter.height()
        if total > 0:
            self.splitter.setSizes([wave_height, max(60, total - wave_height)])

    def _sync_canvases(self):
        self.wave.update()
        self.spec.update()

    def _on_sync_overlay(self):
        self.wave._repaint_overlay()
        self.spec._repaint_overlay()

    def refresh(self):
        self.wave.update()
        self.spec.update()
