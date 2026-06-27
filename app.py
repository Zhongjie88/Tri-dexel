"""
app.py — Tri-dexel NC Machining Simulation GUI

Layout
------
┌──────────────────────────┬──────────────────────────────────────┐
│  Left panel (controls)   │  Right panel (embedded 3-D viewport) │
│  ────────────────────    │  ─────────────────────────────────── │
│  FEEDSTOCK               │                                       │
│  TOOLPATH                │       PyVista QtInteractor            │
│  TOOL                    │       (rotatable while running)       │
│  SIMULATION              │                                       │
│  ▶ Start / ■ Stop / ↺    │                                       │
└──────────────────────────┴──────────────────────────────────────┘

Dependencies
------------
  pip install PyQt5 pyvistaqt pyvista scikit-image numpy
"""

import sys
import math
import os
import numpy as np

os.environ.setdefault("QT_API", "pyqt6")   # tell pyvistaqt to use PyQt6

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel,
    QDoubleSpinBox, QProgressBar, QSlider,
    QFrame, QSizePolicy, QScrollArea,
    QRadioButton, QButtonGroup, QCheckBox,
    QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont

import pyvista as pv
from pyvistaqt import QtInteractor

from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import BallEndMill, FlatEndMill
from src.simulation.engine import SimulationEngine
from src.gcode.parser import GCodeParser


# ══════════════════════════════════════════════════════════════════════════
# Colour palette
# ══════════════════════════════════════════════════════════════════════════

BG      = "#111214"   # window background
PANEL   = "#181a1f"   # left panel
CARD    = "#24272e"   # widget / group background
FIELD   = "#1d2026"   # input background
ACCENT  = "#d9a441"   # brass highlight
SUCCESS = "#38a169"   # green
WARNING = "#d69e2e"   # amber
DANGER  = "#d64545"   # red
T1      = "#f3f4f1"   # primary text
T2      = "#a7adb5"   # secondary text
T3      = "#4b515b"   # muted / borders
SURF    = "#c8b89a"   # stock material colour
TOOL_C  = "#52b6d9"   # tool colour
BOX_C   = "#6a7c90"   # wireframe colour
CUT_CMAP = "turbo"     # high-contrast cut-depth map
G0_C    = "#45aaf2"   # G0 rapid moves — sky blue
G1_C    = "#fd9644"   # G1 feed  moves — amber

# ══════════════════════════════════════════════════════════════════════════
# Toolpath generators
# ══════════════════════════════════════════════════════════════════════════

def make_serpentine(x0=10, x1=90, y0=10, y1=90,
                    z_cut=20.0, step=6.0, retract=55.0):
    segs, y, d = [], y0, 1
    while y <= y1 + 1e-6:
        xs, xe = (x0, x1) if d == 1 else (x1, x0)
        segs.append(((xs, y, retract), (xs, y, z_cut)))
        segs.append(((xs, y, z_cut),   (xe, y, z_cut)))
        segs.append(((xe, y, z_cut),   (xe, y, retract)))
        y += step
        d *= -1
    return segs


def make_spiral(cx=50, cy=50, r0=38, r1=5,
                z_cut=20.0, step=5.0, n=72):
    segs, r = [], r0
    while r >= r1 - 1e-6:
        pts = [
            (cx + r * math.cos(2 * math.pi * k / n),
             cy + r * math.sin(2 * math.pi * k / n),
             z_cut)
            for k in range(n + 1)
        ]
        segs.append(((pts[0][0], pts[0][1], z_cut + 15), pts[0]))
        for k in range(len(pts) - 1):
            segs.append((pts[k], pts[k + 1]))
        r -= step
    return segs


STRATEGIES = {
    "Serpentine Pocket": make_serpentine,
    "Spiral Contour":    make_spiral,
}


# ══════════════════════════════════════════════════════════════════════════
# Worker threads
# ══════════════════════════════════════════════════════════════════════════

class SimWorker(QThread):
    """Background simulation thread — emits height-map frames for the GUI."""

    progress    = pyqtSignal(int, int)     # (current, total)
    frame_ready = pyqtSignal(object, object)  # (np.ndarray hmap, list tool_pos)
    finished    = pyqtSignal()

    def __init__(self, engine: SimulationEngine, toolpath: list,
                 emit_every: int = 1, delay_ms: int = 100):
        super().__init__()
        self.engine     = engine
        self.toolpath   = toolpath
        self.emit_every = emit_every
        self.delay_ms   = delay_ms    # writable during run for live speed control
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        total = len(self.toolpath)
        for idx, (seg_s, seg_e) in enumerate(self.toolpath):
            if self._stop:
                break
            self.engine.simulate_move(seg_s, seg_e)
            self.progress.emit(idx + 1, total)
            emit_every = max(1, int(self.emit_every))
            if idx % emit_every == 0 or idx == total - 1:
                hmap = self.engine.stock.z_grid.height_map().copy()
                self.frame_ready.emit(hmap, list(seg_e))
            d = self.delay_ms
            if d > 0:
                self.msleep(d)
        self.finished.emit()


class MeshWorker(QThread):
    """Builds a smooth marching-cubes mesh after simulation in the background."""

    mesh_ready = pyqtSignal(object)    # pv.PolyData

    def __init__(self, stock):
        super().__init__()
        self.stock = stock

    def run(self):
        from src.reconstruction.mesh import voxel_to_mesh
        voxels = self.stock.to_voxel_grid()
        mesh = voxel_to_mesh(voxels, self.stock.bounds, self.stock.resolution)
        self.mesh_ready.emit(mesh)


# ══════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tri-dexel NC Machining Simulator")
        self.resize(1520, 900)
        self.setMinimumSize(1280, 760)

        # simulation state
        self._stock: TriDexelStock | None       = None
        self._engine: SimulationEngine | None   = None
        self._worker: SimWorker | None          = None
        self._mesh_worker: MeshWorker | None    = None
        self._surface: pv.StructuredGrid | None = None
        self._surface_actor                     = None
        self._tool_actor                        = None
        self._gcode_actors: list                = []
        self._xx = self._yy                     = None
        self._display_i_idx = self._display_j_idx = None
        self._display_i_edges = self._display_j_edges = None

        # file state
        self._stock_file: str | None = None
        self._gcode_file: str | None = None
        self._gcode_moves: list | None = None   # parsed on file browse; used for overlay
        self._last_scene_bounds: tuple | None = None

        # render throttle — simulation thread writes here; QTimer reads at 30 fps
        self._pending_hmap: np.ndarray | None = None
        self._pending_tool_pos: list | None    = None
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)          # ≈ 30 fps
        self._render_timer.timeout.connect(self._render_pending_frame)

        self._build_ui()
        self._apply_stylesheet()
        self._idle_scene()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        row.addWidget(self._build_left_panel(), stretch=0)

        vp_frame = QFrame()
        vp_frame.setObjectName("vp_frame")
        vp_layout = QVBoxLayout(vp_frame)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        vp_layout.setSpacing(0)
        view_header = QFrame()
        view_header.setObjectName("view_header")
        vh = QHBoxLayout(view_header)
        vh.setContentsMargins(24, 16, 24, 16)
        vh.setSpacing(16)
        view_title = QLabel("Simulation View")
        view_title.setObjectName("view_title")
        view_hint = QLabel("Left drag rotate | Right drag zoom | Middle drag pan")
        view_hint.setObjectName("view_hint")
        view_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        vh.addWidget(view_title)
        vh.addStretch()
        vh.addWidget(view_hint)
        vp_layout.addWidget(view_header)
        self.plotter = QtInteractor(vp_frame)
        vp_layout.addWidget(self.plotter)
        row.addWidget(vp_frame, stretch=1)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("left_panel")
        panel.setFixedWidth(360)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        header = QFrame()
        header.setObjectName("panel_header")
        hv = QVBoxLayout(header)
        hv.setContentsMargins(22, 22, 22, 18)
        hv.setSpacing(5)
        logo = QLabel("TRI-DEXEL")
        logo.setObjectName("logo")
        subtitle = QLabel("NC machining simulation")
        subtitle.setObjectName("subtitle")
        hv.addWidget(logo)
        hv.addWidget(subtitle)
        col.addWidget(header)

        # content area with padding
        content = QWidget()
        content.setObjectName("content")
        cv = QVBoxLayout(content)
        cv.setContentsMargins(22, 20, 22, 20)
        cv.setSpacing(14)

        # ── FEEDSTOCK ─────────────────────────────────────────────────
        cv.addWidget(self._section_label("FEEDSTOCK"))

        self._stock_grp = QButtonGroup(self)
        self.rb_box   = QRadioButton("Box Stock")
        self.rb_file  = QRadioButton("Import File  (.STL / .STEP)")
        self.rb_box.setChecked(True)
        self._stock_grp.addButton(self.rb_box,  0)
        self._stock_grp.addButton(self.rb_file, 1)
        cv.addWidget(self.rb_box)

        # Box dims
        self.w_box_dims = QWidget()
        bd = QGridLayout(self.w_box_dims)
        bd.setContentsMargins(20, 2, 0, 2)
        bd.setHorizontalSpacing(12)
        bd.setVerticalSpacing(8)
        for row_idx, (lbl, attr, val) in enumerate([
                ("Width", "sp_bx", 100.0),
                ("Depth", "sp_by", 100.0),
                ("Height", "sp_bz",  50.0),
        ]):
            label = QLabel(lbl)
            label.setObjectName("field_label")
            label.setFixedWidth(68)
            bd.addWidget(label, row_idx, 0)
            sp = QDoubleSpinBox()
            sp.setRange(1.0, 9999.0)
            sp.setValue(val)
            sp.setSuffix(" mm")
            sp.setDecimals(0)
            sp.setMinimumWidth(0)
            setattr(self, attr, sp)
            bd.addWidget(sp, row_idx, 1)
        bd.setColumnStretch(1, 1)
        cv.addWidget(self.w_box_dims)

        cv.addWidget(self.rb_file)

        # STL file row
        self.w_stl = QWidget()
        self.w_stl.setVisible(False)
        sf = QVBoxLayout(self.w_stl)
        sf.setContentsMargins(20, 2, 0, 2)
        sf.setSpacing(8)
        self.lbl_stock_path = QLabel("No file selected")
        self.lbl_stock_path.setObjectName("filepath")
        self.lbl_stock_path.setWordWrap(True)
        self.lbl_stock_path.setFixedHeight(44)
        self.lbl_stock_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        btn_stl = QPushButton("Browse")
        btn_stl.setObjectName("browse_btn")
        btn_stl.clicked.connect(self._browse_stock_file)
        sf.addWidget(self.lbl_stock_path)
        sf.addWidget(btn_stl)
        cv.addWidget(self.w_stl)

        self.rb_file.toggled.connect(lambda on: (
            self.w_stl.setVisible(on),
            self.w_box_dims.setVisible(not on),
        ))

        # ── TOOLPATH ──────────────────────────────────────────────────
        cv.addWidget(self._divider())
        cv.addWidget(self._section_label("TOOLPATH"))

        self._path_grp = QButtonGroup(self)
        self.rb_builtin = QRadioButton("Built-in Strategy")
        self.rb_gcode   = QRadioButton("G-code File  (.nc / .gcode)")
        self.rb_builtin.setChecked(True)
        self._path_grp.addButton(self.rb_builtin, 0)
        self._path_grp.addButton(self.rb_gcode,   1)
        cv.addWidget(self.rb_builtin)

        # Strategy combo
        self.w_strategy = QWidget()
        sw = QHBoxLayout(self.w_strategy)
        sw.setContentsMargins(20, 2, 0, 2)
        sw.setSpacing(8)
        self.cb_strategy = QComboBox()
        self.cb_strategy.addItems(list(STRATEGIES.keys()))
        sw.addWidget(self.cb_strategy)
        cv.addWidget(self.w_strategy)

        cv.addWidget(self.rb_gcode)

        # G-code file row
        self.w_gcode = QWidget()
        self.w_gcode.setVisible(False)
        gf = QVBoxLayout(self.w_gcode)
        gf.setContentsMargins(20, 2, 0, 2)
        gf.setSpacing(8)
        self.lbl_gcode_path = QLabel("No file selected")
        self.lbl_gcode_path.setObjectName("filepath")
        self.lbl_gcode_path.setWordWrap(True)
        self.lbl_gcode_path.setFixedHeight(44)
        self.lbl_gcode_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        btn_gc = QPushButton("Browse")
        btn_gc.setObjectName("browse_btn")
        btn_gc.clicked.connect(self._browse_gcode_file)
        gf.addWidget(self.lbl_gcode_path)
        gf.addWidget(btn_gc)
        self.chk_gcode_overlay = QCheckBox("Show G-code path")
        self.chk_gcode_overlay.setObjectName("switch")
        self.chk_gcode_overlay.setChecked(True)
        self.chk_gcode_overlay.toggled.connect(self._on_gcode_overlay_toggled)
        gf.addWidget(self.chk_gcode_overlay)
        cv.addWidget(self.w_gcode)

        self.rb_gcode.toggled.connect(lambda on: (
            self.w_gcode.setVisible(on),
            self.w_strategy.setVisible(not on),
        ))

        # ── TOOL ──────────────────────────────────────────────────────
        cv.addWidget(self._divider())
        cv.addWidget(self._section_label("TOOL"))

        tg = QGridLayout()
        tg.setHorizontalSpacing(12)
        tg.setVerticalSpacing(10)
        tg.addWidget(QLabel("Type"), 0, 0)
        self.cb_tool = QComboBox()
        self.cb_tool.addItems(["Ball-end Mill", "Flat-end Mill"])
        tg.addWidget(self.cb_tool, 0, 1)
        tg.addWidget(QLabel("Radius"), 1, 0)
        self.sp_radius = QDoubleSpinBox()
        self.sp_radius.setRange(0.5, 50.0)
        self.sp_radius.setValue(5.0)
        self.sp_radius.setSingleStep(0.5)
        self.sp_radius.setSuffix(" mm")
        tg.addWidget(self.sp_radius, 1, 1)
        cv.addLayout(tg)

        # ── SIMULATION ────────────────────────────────────────────────
        cv.addWidget(self._divider())
        cv.addWidget(self._section_label("SIMULATION"))

        sg = QGridLayout()
        sg.setHorizontalSpacing(12)
        sg.setVerticalSpacing(10)
        sg.addWidget(QLabel("Resolution"), 0, 0)
        self.sp_res = QDoubleSpinBox()
        self.sp_res.setRange(0.1, 5.0)
        self.sp_res.setValue(1.0)
        self.sp_res.setSingleStep(0.25)
        self.sp_res.setSuffix(" mm")
        sg.addWidget(self.sp_res, 0, 1)
        cv.addLayout(sg)

        # Speed slider
        cv.addWidget(QLabel("Simulation Speed"))
        sp_row = QWidget()
        sr = QHBoxLayout(sp_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(10)
        lbl_s = QLabel("Slow")
        lbl_s.setObjectName("muted")
        self.sl_speed = QSlider(Qt.Orientation.Horizontal)
        self.sl_speed.setRange(0, 10)
        self.sl_speed.setValue(4)
        self.sl_speed.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sl_speed.setTickInterval(2)
        lbl_f = QLabel("Fast")
        lbl_f.setObjectName("muted")
        sr.addWidget(lbl_s)
        sr.addWidget(self.sl_speed, stretch=1)
        sr.addWidget(lbl_f)
        cv.addWidget(sp_row)

        delay_ms = self._speed_delay_ms(self.sl_speed.value())
        self.lbl_speed_val = QLabel(f"{delay_ms} ms delay")
        self.lbl_speed_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_speed_val.setObjectName("muted")
        cv.addWidget(self.lbl_speed_val)
        self.sl_speed.valueChanged.connect(self._on_speed_changed)

        cv.addStretch()

        # ── BUTTONS ───────────────────────────────────────────────────
        cv.addWidget(self._divider())

        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(54)
        self.btn_start.setAutoDefault(False)
        self.btn_start.setDefault(False)
        self.btn_start.clicked.connect(self._start)
        cv.addWidget(self.btn_start)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(44)
        self.btn_stop.setAutoDefault(False)
        self.btn_stop.setDefault(False)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setFixedHeight(44)
        self.btn_reset.setAutoDefault(False)
        self.btn_reset.setDefault(False)
        self.btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_reset)
        cv.addLayout(btn_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(9)
        self.progress_bar.setTextVisible(False)
        cv.addWidget(self.progress_bar)

        self.lbl_seg = QLabel("Idle")
        self.lbl_seg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_seg.setObjectName("muted")
        self.lbl_seg.setFixedHeight(24)
        self.lbl_seg.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        cv.addWidget(self.lbl_seg)

        # Status
        cv.addWidget(self._divider())
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setObjectName("status")
        self.lbl_status.setFixedHeight(48)
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        cv.addWidget(self.lbl_status)

        scroll = QScrollArea()
        scroll.setObjectName("control_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)
        col.addWidget(scroll, stretch=1)
        return panel

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_lbl")
        return lbl

    @staticmethod
    def _divider() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setObjectName("divider")
        f.setFixedHeight(1)
        return f

    # ── Stylesheet ─────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {BG};
                color: {T1};
                font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
                font-size: 15px;
            }}

            QFrame#left_panel {{
                background: {PANEL};
                border-right: 1px solid {T3};
            }}
            QWidget#content  {{ background: {PANEL}; }}
            QFrame#vp_frame {{ background: {BG}; }}
            QScrollArea#control_scroll {{
                background: {PANEL};
                border: none;
            }}
            QScrollArea#control_scroll > QWidget > QWidget {{
                background: {PANEL};
            }}
            QScrollBar:vertical {{
                background: {PANEL};
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {T3};
                border-radius: 5px;
                min-height: 32px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0;
                background: transparent;
            }}

            QFrame#panel_header {{
                background: #202228;
                border-bottom: 1px solid {T3};
            }}
            QLabel#logo {{
                color: {ACCENT};
                background: transparent;
                font-size: 26px;
                font-weight: bold;
            }}
            QLabel#subtitle {{
                color: {T2};
                background: transparent;
                font-size: 14px;
            }}

            QFrame#view_header {{
                background: #181a1f;
                border-bottom: 1px solid {T3};
            }}
            QLabel#view_title {{
                background: transparent;
                color: {T1};
                font-size: 17px;
                font-weight: bold;
            }}
            QLabel#view_hint {{
                background: transparent;
                color: {T2};
                font-size: 13px;
            }}

            QLabel#section_lbl {{
                color: {ACCENT};
                font-size: 13px;
                font-weight: bold;
                padding: 6px 0 2px 0;
            }}

            QLabel#muted {{ color: {T2}; font-size: 13px; }}
            QLabel#field_label {{
                color: {T2};
                font-size: 14px;
                background: transparent;
            }}

            QLabel#filepath {{
                color: {T2};
                font-size: 13px;
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 3px;
                padding: 8px 10px;
                min-height: 30px;
            }}

            QLabel#status {{
                color: {ACCENT};
                font-size: 14px;
                font-weight: bold;
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 3px;
                padding: 10px 12px;
            }}

            QFrame#divider {{ background: {T3}; }}

            QComboBox, QDoubleSpinBox {{
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 3px;
                padding: 7px 10px;
                color: {T1};
                min-height: 36px;
            }}
            QComboBox:hover, QDoubleSpinBox:hover {{
                border-color: {ACCENT};
            }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            QComboBox QAbstractItemView {{
                background: {FIELD};
                border: 1px solid {T3};
                selection-background-color: {ACCENT};
                selection-color: #17140b;
                color: {T1};
            }}

            QRadioButton {{
                color: {T2};
                spacing: 10px;
                font-size: 15px;
            }}
            QRadioButton:checked {{ color: {T1}; }}
            QRadioButton::indicator {{
                width: 17px; height: 17px;
                border-radius: 9px;
                border: 2px solid {T3};
                background: transparent;
            }}
            QRadioButton::indicator:checked {{
                border: 2px solid {ACCENT};
                background: {ACCENT};
            }}

            QCheckBox#switch {{
                color: {T2};
                spacing: 10px;
                font-size: 14px;
                padding: 4px 0;
            }}
            QCheckBox#switch::indicator {{
                width: 42px;
                height: 22px;
                border-radius: 11px;
                border: 1px solid {T3};
                background: #2b2f38;
            }}
            QCheckBox#switch::indicator:checked {{
                border: 1px solid {ACCENT};
                background: {ACCENT};
            }}

            QPushButton#browse_btn {{
                background: {FIELD};
                color: {T2};
                border: 1px solid {T3};
                border-radius: 3px;
                font-size: 14px;
                padding: 8px 12px;
                min-height: 36px;
            }}
            QPushButton#browse_btn:hover {{
                border-color: {ACCENT};
                color: {T1};
            }}

            QPushButton {{
                background: {FIELD};
                color: {T2};
                border: 1px solid {T3};
                border-radius: 3px;
                font-size: 15px;
                padding: 9px 10px;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background: #2b2f38;
                border-color: {ACCENT};
                color: {T1};
            }}
            QPushButton:disabled {{
                background: {BG};
                color: {T3};
                border-color: #24272e;
            }}
            QPushButton#btn_start {{
                background: {SUCCESS};
                color: #07140c;
                border: 1px solid transparent;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton#btn_start:hover  {{ background: #46b979; color: #07140c; }}
            QPushButton#btn_start:disabled {{
                background: #193222;
                color: {T3};
                border: 1px solid transparent;
            }}
            QPushButton#btn_stop {{
                background: {DANGER};
                color: #180606;
                border: 1px solid transparent;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton#btn_stop:hover  {{ background: #e25555; color: #180606; }}
            QPushButton#btn_stop:disabled {{
                background: #1c0f0f;
                color: {T3};
                border: 1px solid transparent;
            }}
            QPushButton#btn_reset {{
                color: {T1};
            }}

            QProgressBar {{
                background: {FIELD};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 3px;
            }}

            QSlider::groove:horizontal {{
                background: {FIELD};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 20px; height: 20px;
                border-radius: 10px;
                margin: -7px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 2px;
            }}
        """)

    # ── 3-D scene ──────────────────────────────────────────────────────

    def _gcode_lines_to_meshes(self, moves):
        """Build two PolyData line meshes from parsed G-code moves.

        Returns (rapid_mesh | None, feed_mesh | None).
        G0 moves → rapid_mesh; G1 moves → feed_mesh.
        """
        rapid_pts: list = []
        feed_pts:  list = []
        pos = [0.0, 0.0, 0.0]
        for m in moves:
            cur = [m.x, m.y, m.z]
            target = rapid_pts if m.rapid else feed_pts
            target.extend([pos[:], cur[:]])
            pos = cur

        def _build(pts_list):
            if not pts_list:
                return None
            pts = np.array(pts_list, dtype=float)     # (2*N, 3)
            n = len(pts) // 2
            cells = np.empty(n * 3, dtype=np.intp)
            cells[0::3] = 2
            cells[1::3] = np.arange(n) * 2
            cells[2::3] = np.arange(n) * 2 + 1
            mesh = pv.PolyData()
            mesh.points = pts
            mesh.lines = cells
            return mesh

        return _build(rapid_pts), _build(feed_pts)

    def _add_gcode_overlay(self):
        """Add G0/G1 toolpath lines to the current plotter scene."""
        self._gcode_actors = []
        if not self._gcode_moves:
            return
        visible = self.chk_gcode_overlay.isChecked()
        rapid_mesh, feed_mesh = self._gcode_lines_to_meshes(self._gcode_moves)
        if rapid_mesh is not None:
            actor = self.plotter.add_mesh(
                rapid_mesh, color=G0_C,
                line_width=3.0, opacity=0.72,
                render_lines_as_tubes=True,
                lighting=False,
            )
            actor.SetVisibility(visible)
            self._gcode_actors.append(actor)
        if feed_mesh is not None:
            actor = self.plotter.add_mesh(
                feed_mesh, color=G1_C,
                line_width=4.0, opacity=0.96,
                render_lines_as_tubes=True,
                lighting=False,
            )
            actor.SetVisibility(visible)
            self._gcode_actors.append(actor)
        # Inline text legend (more reliable across PyVista versions than add_legend)
        y = 0.04
        if feed_mesh is not None:
            actor = self.plotter.add_text(
                "-- G1 Feed", position=(0.01, y),
                font_size=9, color=G1_C, shadow=True,
                viewport=True,
            )
            actor.SetVisibility(visible)
            self._gcode_actors.append(actor)
            y += 0.05
        if rapid_mesh is not None:
            actor = self.plotter.add_text(
                "-- G0 Rapid", position=(0.01, y),
                font_size=9, color=G0_C, shadow=True,
                viewport=True,
            )
            actor.SetVisibility(visible)
            self._gcode_actors.append(actor)

    def _set_gcode_overlay_visible(self, visible: bool):
        for actor in self._gcode_actors:
            if actor is not None:
                actor.SetVisibility(visible)
        self.plotter.render()

    def _idle_scene(self):
        """Default view: solid box representing the configured stock."""
        self.plotter.clear()
        self._gcode_actors = []
        self.plotter.set_background(BG, top="#12192e")

        if self.rb_file.isChecked() and self._stock_file:
            return  # already showing STL preview

        w = self.sp_bx.value()
        d = self.sp_by.value()
        h = self.sp_bz.value()
        bounds = (0.0, w, 0.0, d, 0.0, h)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        box = pv.Box(bounds=bounds)
        show_gcode = self._gcode_moves and self.chk_gcode_overlay.isChecked()
        stock_opacity = 0.35 if show_gcode else 1.0
        self.plotter.add_mesh(
            box, color=SURF, opacity=stock_opacity,
            smooth_shading=True, ambient=0.15, diffuse=0.8,
        )
        self.plotter.add_mesh(box, style="wireframe", color=BOX_C,
                               opacity=0.25, line_width=1.0)
        self.plotter.add_text(
            "Start a simulation to begin",
            position="lower_edge", font_size=10, color=T3,
        )
        self._add_gcode_overlay()
        self.plotter.show_axes()
        self._set_default_camera(x_min, x_max, y_min, y_max, z_min, z_max)
        self.plotter.render()

    def _preview_stl(self, path: str):
        """Show the loaded mesh/CAD file in the viewport as a preview."""
        try:
            from src.stock.stl_importer import load_mesh

            mesh = load_mesh(path)
            self.plotter.clear()
            self._gcode_actors = []
            self.plotter.set_background(BG, top="#12192e")
            show_gcode = self._gcode_moves and self.chk_gcode_overlay.isChecked()
            stock_opacity = 0.38 if show_gcode else 1.0
            self.plotter.add_mesh(
                mesh, color=SURF, opacity=stock_opacity,
                smooth_shading=True, ambient=0.15, diffuse=0.8,
            )
            b = mesh.bounds
            x_min, x_max, y_min, y_max, z_min, z_max = (
                b[0], b[1], b[2], b[3], b[4], b[5])
            box = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
            self.plotter.add_mesh(box, style="wireframe", color=BOX_C,
                                   opacity=0.2, line_width=1.0)
            self.plotter.add_text(
                f"{os.path.basename(path)}\n"
                f"{x_max-x_min:.1f} x {y_max-y_min:.1f} x {z_max-z_min:.1f} mm",
                position="lower_edge", font_size=9, color=T2,
            )
            self._add_gcode_overlay()
            self.plotter.show_axes()
            self._set_default_camera(x_min, x_max, y_min, y_max, z_min, z_max)
            self.plotter.render()
            self._set_status("Stock loaded", SUCCESS)
        except Exception as exc:
            self._set_status("Preview failed", DANGER)
            QMessageBox.warning(
                self,
                "Preview failed",
                self._format_import_error(path, exc),
            )

    @staticmethod
    def _format_import_error(path: str, exc: Exception) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".step", ".stp"):
            return (
                "Unable to load this STEP file.\n\n"
                "STEP is a CAD/B-Rep format, so it must be converted to a "
                "surface mesh before simulation. This project tries gmsh first.\n\n"
                f"File:\n{path}\n\n"
                f"Reason:\n{exc}"
            )
        return f"Unable to load this mesh file:\n{path}\n\nReason:\n{exc}"

    def _set_default_camera(self, x_min, x_max, y_min, y_max, z_min, z_max):
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        cz = (z_min + z_max) / 2
        d  = max(x_max - x_min, y_max - y_min) * 1.5
        self.plotter.camera_position = [
            (cx + d * 0.85, cy - d * 0.95, cz + d * 0.80),
            (cx, cy, cz),
            (0, 0, 1),
        ]

    def _build_scene(self, bounds):
        """(Re)build the 3-D scene for a new simulation run."""
        self.plotter.clear()
        self._gcode_actors = []
        self.plotter.set_background(BG, top="#12192e")
        self._last_scene_bounds = bounds

        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        s = self._stock

        max_live_axis = 180
        n_disp_i = min(s.nx, max_live_axis)
        n_disp_j = min(s.ny, max_live_axis)
        self._display_i_edges = np.linspace(0, s.nx, n_disp_i + 1, dtype=int)
        self._display_j_edges = np.linspace(0, s.ny, n_disp_j + 1, dtype=int)
        self._display_i_idx = (
            (self._display_i_edges[:-1] + self._display_i_edges[1:] - 1) // 2
        )
        self._display_j_idx = (
            (self._display_j_edges[:-1] + self._display_j_edges[1:] - 1) // 2
        )
        live_stride = max(
            1,
            math.ceil(s.nx / len(self._display_i_idx)),
            math.ceil(s.ny / len(self._display_j_idx)),
        )

        x_full = np.array([s.z_grid.row_center(i) for i in range(s.nx)])
        y_full = np.array([s.z_grid.col_center(j) for j in range(s.ny)])
        x = x_full[self._display_i_idx]
        y = y_full[self._display_j_idx]
        self._xx, self._yy = np.meshgrid(x, y, indexing="ij")

        # Static base
        base_pts  = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min],
                               [x_max, y_max, z_min], [x_min, y_max, z_min]], dtype=float)
        base_face = np.array([4, 0, 1, 2, 3])
        self.plotter.add_mesh(pv.PolyData(base_pts, base_face),
                               color=SURF, ambient=0.15, diffuse=0.80,
                               smooth_shading=False, opacity=0.35)

        # Dynamic height-map surface. Color is cut depth, not world height,
        # so machined regions stay visually distinct from untouched stock.
        zz = np.full((len(self._display_i_idx), len(self._display_j_idx)), z_max)
        self._surface = pv.StructuredGrid(self._xx, self._yy, zz)
        self._surface["cut_depth"] = np.zeros(self._surface.n_points)
        self._surface_actor = self.plotter.add_mesh(
            self._surface,
            scalars="cut_depth",
            clim=(0.0, max(1.0, z_max - z_min)),
            cmap=CUT_CMAP,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Cut depth",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.18,
                "width": 0.08,
                "height": 0.55,
                "color": T1,
                "label_font_size": 9,
                "title_font_size": 10,
            },
            smooth_shading=False,
            show_edges=(live_stride == 1),
            edge_color="#171a1f",
            line_width=0.25,
            ambient=0.25, diffuse=0.85, specular=0.15,
        )
        if s.nx != len(self._display_i_idx) or s.ny != len(self._display_j_idx):
            self.plotter.add_text(
                f"Live preview decimated {live_stride}x; final mesh uses full resolution",
                position="lower_right", font_size=8, color=T2, shadow=True,
            )

        top_ref = pv.Plane(
            center=((x_min + x_max) / 2, (y_min + y_max) / 2, z_max + 0.02),
            direction=(0, 0, 1),
            i_size=(x_max - x_min),
            j_size=(y_max - y_min),
            i_resolution=8,
            j_resolution=8,
        )
        self.plotter.add_mesh(
            top_ref, style="wireframe", color=ACCENT,
            line_width=1.0, opacity=0.28,
        )

        # Wireframe box
        box = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
        self.plotter.add_mesh(box, style="wireframe", color=BOX_C,
                               line_width=1.4, opacity=0.42)

        # Tool sphere
        r = self._engine.tool.radius
        sphere = pv.Sphere(radius=r, theta_resolution=28, phi_resolution=28)
        self._tool_actor = self.plotter.add_mesh(
            sphere, color=TOOL_C, opacity=0.95,
            smooth_shading=True, specular=0.75,
        )
        self._tool_actor.position = [
            x_min + (x_max - x_min) * 0.1,
            y_min + (y_max - y_min) * 0.1,
            z_max,
        ]

        self._add_gcode_overlay()
        self.plotter.show_axes()
        self._set_default_camera(x_min, x_max, y_min, y_max, z_min, z_max)
        self.plotter.render()

    # ── Frame / progress slots ─────────────────────────────────────────

    def _apply_hmap_to_surface(self, hmap: np.ndarray) -> None:
        if self._display_i_edges is not None and self._display_j_edges is not None:
            hmap = self._pool_hmap_for_display(hmap)
        zz  = np.nan_to_num(hmap, nan=self._stock.z_min)
        # Update only the Z column of the existing point array.
        # VTK/PyVista StructuredGrid stores points in Fortran order (i varies fastest),
        # which matches zz.ravel(order='F') for our indexing='ij' meshgrid.
        pts = self._surface.points.copy()
        pts[:, 2] = zz.ravel(order='F')
        self._surface.points = pts                       # property setter → Modified()
        depth = np.maximum(0.0, self._stock.z_max - pts[:, 2])
        self._surface["cut_depth"] = depth
        points = self._surface.GetPoints()
        if points is not None:
            points.Modified()
        scalars = self._surface.GetPointData().GetScalars()
        if scalars is not None:
            scalars.Modified()
        self._surface.Modified()
        if self._surface_actor is not None:
            self._surface_actor.mapper.Modified()

    def _pool_hmap_for_display(self, hmap: np.ndarray) -> np.ndarray:
        """Conservative live-preview downsample.

        Each displayed point uses the minimum Z in its source block. A normal
        point sample can miss a narrow tool mark; min-pooling cannot.
        """
        ei = self._display_i_edges
        ej = self._display_j_edges
        pooled = np.empty((len(ei) - 1, len(ej) - 1), dtype=float)
        for oi, (i0, i1) in enumerate(zip(ei[:-1], ei[1:])):
            row = hmap[i0:i1, :]
            for oj, (j0, j1) in enumerate(zip(ej[:-1], ej[1:])):
                block = row[:, j0:j1]
                pooled[oi, oj] = np.nanmin(block)
        return pooled

    @pyqtSlot(object, object)
    def _on_frame(self, hmap, tool_pos):
        # Just cache the latest data; the QTimer fires the actual render at 30 fps.
        # This prevents signal-queue pile-up when simulation runs faster than the GPU.
        self._pending_hmap     = hmap
        self._pending_tool_pos = tool_pos

    def _render_pending_frame(self):
        """Called by self._render_timer every 33 ms — one render per tick at most."""
        hmap = self._pending_hmap
        if hmap is None or self._surface is None or self._xx is None:
            return
        self._pending_hmap = None           # consume
        pos = self._pending_tool_pos
        try:
            self._apply_hmap_to_surface(hmap)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Render error: {exc}", DANGER)
            return
        if self._tool_actor is not None and pos is not None:
            self._tool_actor.position = pos
        self.plotter.render()

    @pyqtSlot(int, int)
    def _on_progress(self, cur: int, total: int):
        pct = cur * 100 // total
        self.progress_bar.setValue(pct)
        self.lbl_seg.setText(f"{cur} / {total} segments")

    @pyqtSlot()
    def _on_finished(self):
        self._render_timer.stop()
        self._pending_hmap = None

        if self._stock is not None and self._surface is not None:
            self._apply_hmap_to_surface(self._stock.z_grid.height_map())
            self.plotter.render()

        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_seg.setText("Rebuilding smooth mesh...")
        self._set_status("Meshing", WARNING)

        self._mesh_worker = MeshWorker(self._stock)
        self._mesh_worker.mesh_ready.connect(self._on_mesh_ready)
        self._mesh_worker.start()

    @pyqtSlot(object)
    def _on_mesh_ready(self, mesh: "pv.PolyData"):
        if self._surface_actor is not None:
            self.plotter.remove_actor(self._surface_actor)
            self._surface_actor = None
        if self._tool_actor is not None:
            self.plotter.remove_actor(self._tool_actor)
            self._tool_actor = None

        z_max = self._stock.z_max
        z_min = self._stock.z_min
        cut_depth = np.maximum(0.0, z_max - mesh.points[:, 2])
        mesh["cut_depth"] = cut_depth

        self._surface_actor = self.plotter.add_mesh(
            mesh,
            scalars="cut_depth",
            clim=(0.0, max(1.0, z_max - z_min)),
            cmap=CUT_CMAP,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Cut depth",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.18,
                "width": 0.08,
                "height": 0.55,
                "color": T1,
                "label_font_size": 9,
                "title_font_size": 10,
            },
            smooth_shading=False,
            show_edges=True,
            edge_color="#1a1d22",
            line_width=0.2,
            ambient=0.28, diffuse=0.82, specular=0.18,
        )

        bounds = self._stock.bounds
        box = pv.Box(bounds=bounds)
        self.plotter.add_mesh(
            box, style="wireframe", color=BOX_C,
            line_width=1.4, opacity=0.42,
        )
        x_min, x_max, y_min, y_max, _, _ = bounds
        top_ref = pv.Plane(
            center=((x_min + x_max) / 2, (y_min + y_max) / 2, z_max + 0.02),
            direction=(0, 0, 1),
            i_size=(x_max - x_min),
            j_size=(y_max - y_min),
            i_resolution=8,
            j_resolution=8,
        )
        self.plotter.add_mesh(
            top_ref, style="wireframe", color=ACCENT,
            line_width=1.0, opacity=0.24,
        )
        self._add_gcode_overlay()
        self.plotter.render()

        self.lbl_seg.setText("Idle")
        self.btn_start.setEnabled(True)
        self._set_status("Complete", SUCCESS)

    # ── Speed ──────────────────────────────────────────────────────────

    @staticmethod
    def _speed_delay_ms(val: int) -> int:
        return max(0, (10 - val) * 15)

    @pyqtSlot(int)
    def _on_speed_changed(self, val: int):
        delay_ms = self._speed_delay_ms(val)
        self.lbl_speed_val.setText(f"{delay_ms} ms delay")
        if self._worker is not None and self._worker.isRunning():
            self._worker.delay_ms = delay_ms

    @pyqtSlot(bool)
    def _on_gcode_overlay_toggled(self, checked: bool):
        if self._gcode_actors:
            self._set_gcode_overlay_visible(checked)
            return
        if self._stock is not None and self._last_scene_bounds is not None:
            self._build_scene(self._last_scene_bounds)
        elif self.rb_file.isChecked() and self._stock_file:
            self._preview_stl(self._stock_file)
        else:
            self._idle_scene()

    # ── File browse ────────────────────────────────────────────────────

    @pyqtSlot()
    def _browse_stock_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Feedstock File", "",
            "Mesh Files (*.stl *.step *.stp *.obj *.ply);;All Files (*)"
        )
        if not path:
            return
        self._stock_file = path
        name = os.path.basename(path)
        self.lbl_stock_path.setText(name)
        self.lbl_stock_path.setStyleSheet(
            f"color:{T1};font-size:12px;background:{FIELD};"
            f"border:1px solid {ACCENT};border-radius:3px;padding:7px 9px;"
        )
        self._preview_stl(path)

    @pyqtSlot()
    def _browse_gcode_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select G-code File", "",
            "CNC Programs (*.nc *.gcode *.tap *.cnc *.ngc);;All Files (*)"
        )
        if not path:
            return
        self._gcode_file = path
        name = os.path.basename(path)
        self.lbl_gcode_path.setText(name)
        self.lbl_gcode_path.setStyleSheet(
            f"color:{T1};font-size:12px;background:{FIELD};"
            f"border:1px solid {ACCENT};border-radius:3px;padding:7px 9px;"
        )
        try:
            self._gcode_moves = GCodeParser().parse_file(path)
            n_f = sum(1 for m in self._gcode_moves if not m.rapid)
            n_r = len(self._gcode_moves) - n_f
            # Rebuild the idle/preview scene so the toolpath overlay appears immediately
            if self.rb_file.isChecked() and self._stock_file:
                self._preview_stl(self._stock_file)
            else:
                self._idle_scene()
            self._set_status(f"G-code: {n_f} feed, {n_r} rapid", SUCCESS)
        except Exception as exc:
            self._gcode_moves = None
            self._set_status(f"Parse error: {exc}", DANGER)

    # ── Toolpath builder ───────────────────────────────────────────────

    def _build_builtin_toolpath(self, strategy_idx: int, bounds: tuple) -> list:
        """Generate a built-in toolpath scaled to the given stock bounds."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        mx = (x_max - x_min) * 0.10
        my = (y_max - y_min) * 0.10
        z_cut    = z_min + (z_max - z_min) * 0.50
        retract  = z_max + max(5.0, (z_max - z_min) * 0.12)
        step_x   = (x_max - x_min) * 0.06
        step_y   = (y_max - y_min) * 0.06

        if strategy_idx == 0:
            return make_serpentine(
                x0=x_min + mx, x1=x_max - mx,
                y0=y_min + my, y1=y_max - my,
                z_cut=z_cut, step=max(step_x, step_y), retract=retract,
            )
        else:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            r0 = min(x_max - x_min, y_max - y_min) / 2 * 0.85
            r1 = max(3.0, r0 * 0.08)
            return make_spiral(
                cx=cx, cy=cy, r0=r0, r1=r1, z_cut=z_cut,
            )

    # ── Button handlers ────────────────────────────────────────────────

    def _start(self):
        self.btn_start.setEnabled(False)   # prevent double-click during sync loading
        launched = False
        try:
            res = self.sp_res.value()

            # ── Build stock ───────────────────────────────────────────
            if self.rb_file.isChecked():
                if not self._stock_file:
                    self._set_status("Please select a stock file", WARNING)
                    return
                self._set_status("Loading stock", WARNING)
                QApplication.processEvents()

                from src.stock.stl_importer import load_mesh, initialize_stock_from_mesh
                try:
                    mesh = load_mesh(self._stock_file)
                except RuntimeError as exc:
                    QMessageBox.critical(self, "Load Error", str(exc))
                    self._set_status("Load failed", DANGER)
                    return

                b = mesh.bounds
                bounds = (float(b[0]), float(b[1]),
                          float(b[2]), float(b[3]),
                          float(b[4]), float(b[5]))
                self._stock = TriDexelStock(bounds, res)

                self._set_status("Initialising dexels", WARNING)
                QApplication.processEvents()
                initialize_stock_from_mesh(self._stock, mesh)

            else:
                w = self.sp_bx.value()
                d = self.sp_by.value()
                h = self.sp_bz.value()
                bounds = (0.0, w, 0.0, d, 0.0, h)
                self._stock = TriDexelStock(bounds, res)
                self._stock.initialize_box_stock()

            # ── Build tool ────────────────────────────────────────────
            r    = self.sp_radius.value()
            tool = BallEndMill(r) if self.cb_tool.currentIndex() == 0 \
                   else FlatEndMill(r)
            self._engine = SimulationEngine(self._stock, tool)

            # ── Build toolpath ────────────────────────────────────────
            if self.rb_gcode.isChecked():
                if not self._gcode_file:
                    self._set_status("Please select a G-code file", WARNING)
                    return
                parser = GCodeParser()
                moves  = parser.parse_file(self._gcode_file)
                self._gcode_moves = moves   # keep for _build_scene overlay
                toolpath = []
                prev = None
                for m in moves:
                    cur = (m.x, m.y, m.z)
                    if prev is not None and not m.rapid:
                        toolpath.append((prev, cur))
                    prev = cur
                if not toolpath:
                    self._set_status("No cutting moves found in G-code", WARNING)
                    return
            else:
                toolpath = self._build_builtin_toolpath(
                    self.cb_strategy.currentIndex(), bounds
                )

            # ── Launch ────────────────────────────────────────────────
            self._build_scene(bounds)

            delay_ms = self._speed_delay_ms(self.sl_speed.value())
            self._worker = SimWorker(
                self._engine, toolpath, emit_every=1, delay_ms=delay_ms
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.finished.connect(self._on_finished)
            self._pending_hmap = None
            self._pending_tool_pos = None
            self._render_timer.start()
            self._worker.start()
            launched = True   # simulation is now running; _on_mesh_ready re-enables Start

            self.btn_stop.setEnabled(True)
            self.progress_bar.setValue(0)
            n = len(toolpath)
            self._set_status(f"Simulating ({n} moves)", SUCCESS)

        except Exception as exc:
            QMessageBox.critical(self, "Simulation Error", str(exc))
            self._set_status(f"Error: {exc}", DANGER)
            self.btn_stop.setEnabled(False)

        finally:
            # Re-enable Start if the worker never launched (validation failures,
            # file errors, exceptions).  If launched=True, the button stays
            # disabled until _on_mesh_ready fires after the simulation finishes.
            if not launched:
                self.btn_start.setEnabled(True)

    def _stop(self):
        self._render_timer.stop()
        self._pending_hmap = None
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)
        self._set_status("Stopped", DANGER)

    def _reset(self):
        self._render_timer.stop()
        self._pending_hmap = None
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        if self._mesh_worker and self._mesh_worker.isRunning():
            self._mesh_worker.wait()
        self._surface       = None
        self._surface_actor = None
        self._tool_actor    = None
        self.progress_bar.setValue(0)
        self.lbl_seg.setText("Idle")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("Ready", ACCENT)
        if self.rb_file.isChecked() and self._stock_file:
            self._preview_stl(self._stock_file)
        else:
            self._idle_scene()

    def _set_status(self, text: str, color: str = ACCENT):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f"color:{color};font-size:13px;font-weight:bold;"
            f"background:{FIELD};border:1px solid {T3};"
            f"border-radius:3px;padding:8px 10px;"
        )

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        if self._mesh_worker and self._mesh_worker.isRunning():
            self._mesh_worker.wait()
        self.plotter.close()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
