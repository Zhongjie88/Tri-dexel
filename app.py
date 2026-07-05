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
import time
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
from src.simulation.sv_engine import SweptVolumeSimulationEngine
from src.simulation.collision import collision_summary
from src.gcode import GCodeParser, SiemensSinumerikParser
from src.motion.gcode import (
    gcode_moves_from_canonical_moves,
    tool_pose_segments_from_gcode_moves,
    all_pose_segments_from_gcode_moves,
)
from src.motion.pose import ToolPose


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

DEFAULT_STOCK_PATH = (
    r"C:\Users\a0903\Downloads\數位雙生\4-29-S-CUT\4-29-S-CUT"
    r"\S_CUT_feedstock_310x210x80_same_origin.stl"
)
DEFAULT_GCODE_PATH = (
    r"C:\Users\a0903\Downloads\數位雙生\4-29-S-CUT\4-29-S-CUT"
    r"\EM12_ROUGH_B01\EM12_ROUGH_B01.mpf"
)
DEFAULT_TOOL_RADIUS = 10.0


# ══════════════════════════════════════════════════════════════════════════
# Worker threads
# ══════════════════════════════════════════════════════════════════════════

class SimWorker(QThread):
    """Background simulation thread — emits height-map frames for the GUI."""

    progress    = pyqtSignal(int, int)     # (current, total)
    frame_ready = pyqtSignal(object, object)  # (np.ndarray hmap, list tool_pos)
    finished    = pyqtSignal()

    def __init__(
        self,
        engine: SimulationEngine,
        toolpath: list,
        emit_every: int = 1,
        delay_ms: int = 100,
        step_multiplier: float = 1.0,
        frame_interval_ms: int = 0,
        segment_batch: int = 1,
        preview_i_idx=None,
        preview_j_idx=None,
        live_surface: bool = True,
    ):
        super().__init__()
        self.engine     = engine
        self.toolpath   = toolpath
        self.emit_every = emit_every
        self.delay_ms   = delay_ms    # writable during run for live speed control
        self.step_multiplier = step_multiplier
        self.frame_interval_ms = frame_interval_ms
        self.segment_batch = segment_batch
        self.preview_i_idx = preview_i_idx
        self.preview_j_idx = preview_j_idx
        self.live_surface = live_surface
        self._stop      = False

    def stop(self):
        self._stop = True

    def _preview_height_map(self) -> np.ndarray:
        full = self.engine.stock.z_grid.height_map()   # O(1) cached copy, shape (nx, ny)
        if self.preview_i_idx is None or self.preview_j_idx is None:
            return full
        rows = np.asarray(self.preview_i_idx, dtype=int)
        cols = np.asarray(self.preview_j_idx, dtype=int)
        return full[np.ix_(rows, cols)]                 # vectorised numpy indexing, O(n)

    def run(self):
        if type(self.engine) is SimulationEngine:
            self._run_legacy_continuous()
            self.finished.emit()
            return

        total = len(self.toolpath)
        last_frame_t = 0.0
        last_progress_t = 0.0
        idx = 0
        while idx < total and not self._stop:
            batch = max(1, int(self.segment_batch))
            # Geometry accuracy must not depend on animation speed.
            step = self.engine.stock.resolution
            last_pos = None
            for _ in range(batch):
                if idx >= total or self._stop:
                    break
                seg_s, seg_e = self.toolpath[idx]
                if isinstance(seg_s, ToolPose) and isinstance(seg_e, ToolPose):
                    is_rapid = seg_e.motion_type == "G0"
                    if not is_rapid:
                        self.engine.simulate_pose_move(seg_s, seg_e, step=step)
                    last_pos = {
                        "pos": tuple(float(v) for v in seg_e.position),
                        "axis": tuple(float(v) for v in seg_e.axis),
                    }
                    if is_rapid:
                        # Emit immediately so rapid repositioning is visible;
                        # do not wait for the batch-end rate-limited emit.
                        hmap = self._preview_height_map() if self.live_surface else None
                        self.frame_ready.emit(hmap, last_pos)
                        last_frame_t = time.perf_counter()
                        last_pos = None  # consumed; prevent double-emit below
                else:
                    self.engine.simulate_move(seg_s, seg_e, step=step)
                    last_pos = {"pos": seg_e, "axis": (0.0, 0.0, 1.0)}
                idx += 1

            now = time.perf_counter()
            if now - last_progress_t >= 0.05 or idx == total - 1:
                self.progress.emit(idx, total)
                last_progress_t = now

            emit_every = max(1, int(self.emit_every))
            interval_s = max(0, int(self.frame_interval_ms)) / 1000.0
            frame_due = interval_s <= 0.0 or now - last_frame_t >= interval_s
            if ((idx % emit_every == 0 and frame_due) or idx >= total) and last_pos is not None:
                hmap = self._preview_height_map() if self.live_surface else None
                self.frame_ready.emit(hmap, last_pos)
                last_frame_t = now
            d = self.delay_ms
            if d > 0:
                self.msleep(d)
        self.finished.emit()

    def _run_legacy_continuous(self):
        total = len(self.toolpath)
        last_frame_t = 0.0
        last_progress_t = 0.0
        # Geometry accuracy must not depend on animation speed.
        step = self.engine.stock.resolution
        interval_s = max(0, int(self.frame_interval_ms)) / 1000.0

        def _progress(cur: int, n_total: int, pos):
            nonlocal last_frame_t, last_progress_t
            if self._stop:
                return
            now = time.perf_counter()
            if now - last_progress_t >= 0.05 or cur >= n_total:
                self.progress.emit(cur, n_total)
                last_progress_t = now
            frame_due = interval_s <= 0.0 or now - last_frame_t >= interval_s
            if frame_due or cur >= n_total:
                hmap = self._preview_height_map() if self.live_surface else None
                self.frame_ready.emit(hmap, {"pos": list(pos), "axis": [0.0, 0.0, 1.0]})
                last_frame_t = now
            d = self.delay_ms
            if d > 0:
                self.msleep(d)

        self.engine.simulate_toolpath(
            self.toolpath,
            step=step,
            progress_callback=_progress,
            stop_callback=lambda: self._stop,
        )


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
        self._engine: SimulationEngine | SweptVolumeSimulationEngine | None = None
        self._worker: SimWorker | None          = None
        self._mesh_worker: MeshWorker | None    = None
        self._surface: pv.StructuredGrid | None = None
        self._surface_actor                     = None
        self._tool_actor                        = None
        self._result_mesh: pv.PolyData | None   = None
        self._gcode_actors: list                = []
        self._coverage_actors: list             = []
        self._xx = self._yy                     = None
        self._display_i_idx = self._display_j_idx = None
        self._display_i_edges = self._display_j_edges = None

        # file state
        self._stock_file: str | None = None
        self._gcode_file: str | None = None
        self._gcode_moves: list | None = None   # parsed on file browse; used for overlay
        self._canonical_program = None
        self._last_scene_bounds: tuple | None = None

        # render throttle — simulation thread writes here; QTimer reads at 30 fps
        self._pending_hmap: np.ndarray | None = None
        self._pending_tool_pos: list | None    = None
        self._pending_frame_dirty = False
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)          # ≈ 30 fps
        self._render_timer.timeout.connect(self._render_pending_frame)

        self._build_ui()
        self._apply_stylesheet()
        self._idle_scene()
        QTimer.singleShot(0, self._load_default_inputs)

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
        self.chk_coverage_overlay = QCheckBox("Show coverage gaps")
        self.chk_coverage_overlay.setObjectName("switch")
        self.chk_coverage_overlay.setChecked(False)
        self.chk_coverage_overlay.toggled.connect(self._on_coverage_overlay_toggled)
        gf.addWidget(self.chk_coverage_overlay)
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
        self.cb_tool.setCurrentIndex(1)
        tg.addWidget(self.cb_tool, 0, 1)
        tg.addWidget(QLabel("Radius"), 1, 0)
        self.sp_radius = QDoubleSpinBox()
        self.sp_radius.setRange(0.5, 50.0)
        self.sp_radius.setValue(DEFAULT_TOOL_RADIUS)
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
        sg.addWidget(QLabel("Method"), 1, 0)
        self.cb_sim_method = QComboBox()
        self.cb_sim_method.addItems(["Legacy Sampling", "Swept Volume"])
        sg.addWidget(self.cb_sim_method, 1, 1)
        cv.addLayout(sg)

        self.chk_collision = QCheckBox("Check shank / holder collision")
        self.chk_collision.setObjectName("switch")
        self.chk_collision.setChecked(False)
        self.chk_collision.setEnabled(False)
        cv.addWidget(self.chk_collision)
        self.cb_sim_method.currentIndexChanged.connect(self._on_sim_method_changed)
        self.chk_collision.toggled.connect(lambda _checked: self._update_collision_status())

        self.chk_final_only = QCheckBox("Fast final-only preview")
        self.chk_final_only.setObjectName("switch")
        self.chk_final_only.setChecked(False)
        self.chk_final_only.toggled.connect(
            lambda _checked: self._on_speed_changed(self.sl_speed.value())
        )
        cv.addWidget(self.chk_final_only)

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

        delay_ms, step_multiplier, frame_interval_ms, segment_batch = self._speed_settings(
            self.sl_speed.value()
        )
        self.lbl_speed_val = QLabel(
            self._speed_label(delay_ms, step_multiplier, frame_interval_ms, segment_batch)
        )
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

        self.btn_export = QPushButton("Export Result")
        self.btn_export.setObjectName("browse_btn")
        self.btn_export.setFixedHeight(42)
        self.btn_export.setAutoDefault(False)
        self.btn_export.setDefault(False)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_result_mesh)
        cv.addWidget(self.btn_export)

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

        self.lbl_collision = QLabel("Collision check: Swept only")
        self.lbl_collision.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_collision.setObjectName("collision_status")
        self.lbl_collision.setFixedHeight(30)
        self.lbl_collision.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        cv.addWidget(self.lbl_collision)

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

            QLabel#collision_status {{
                color: {T2};
                font-size: 13px;
                font-weight: bold;
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 3px;
                padding: 6px 8px;
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
        for m in moves:
            start = [m.start_x, m.start_y, m.start_z]
            cur = [m.x, m.y, m.z]
            target = rapid_pts if m.rapid else feed_pts
            target.extend([start, cur])

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

    def _clear_coverage_overlay(self):
        for actor in self._coverage_actors:
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self._coverage_actors = []

    def _coverage_gap_mesh(self, bounds: tuple[float, ...]) -> pv.PolyData | None:
        if not self._gcode_moves:
            return None
        x_min, x_max, y_min, y_max, _, z_max = bounds
        radius = float(self.sp_radius.value())
        cell = max(float(self.sp_res.value()), 2.0)
        xs = np.arange(x_min, x_max + 1e-9, cell)
        ys = np.arange(y_min, y_max + 1e-9, cell)
        if len(xs) < 2 or len(ys) < 2:
            return None

        covered = np.zeros((len(xs), len(ys)), dtype=bool)
        feed_moves = [m for m in self._gcode_moves if not m.rapid]
        sample_step = max(cell, radius * 0.5)
        r2 = radius * radius
        for m in feed_moves:
            sx, sy, sz = m.start_x, m.start_y, m.start_z
            ex, ey, ez = m.x, m.y, m.z
            length = math.dist((sx, sy, sz), (ex, ey, ez))
            n = max(1, math.ceil(length / sample_step))
            for k in range(n + 1):
                t = k / n
                x = sx + (ex - sx) * t
                y = sy + (ey - sy) * t
                i0 = max(0, int(math.floor((x - radius - x_min) / cell)))
                i1 = min(len(xs) - 1, int(math.ceil((x + radius - x_min) / cell)))
                j0 = max(0, int(math.floor((y - radius - y_min) / cell)))
                j1 = min(len(ys) - 1, int(math.ceil((y + radius - y_min) / cell)))
                if i0 > i1 or j0 > j1:
                    continue
                xx = xs[i0 : i1 + 1, np.newaxis]
                yy = ys[np.newaxis, j0 : j1 + 1]
                covered[i0 : i1 + 1, j0 : j1 + 1] |= (xx - x) ** 2 + (yy - y) ** 2 <= r2

        gap = ~covered
        if not np.any(gap):
            return None

        z = z_max + max(0.05, self.sp_res.value() * 0.1)
        points: list[tuple[float, float, float]] = []
        faces: list[int] = []
        half = cell * 0.5
        for i, j in np.argwhere(gap):
            x = float(xs[i])
            y = float(ys[j])
            base = len(points)
            points.extend(
                [
                    (x - half, y - half, z),
                    (x + half, y - half, z),
                    (x + half, y + half, z),
                    (x - half, y + half, z),
                ]
            )
            faces.extend([4, base, base + 1, base + 2, base + 3])
        mesh = pv.PolyData(np.asarray(points), np.asarray(faces, dtype=np.int64))
        mesh["coverage_gap"] = np.ones(mesh.n_points)
        return mesh

    def _add_coverage_overlay(self):
        self._clear_coverage_overlay()
        if not hasattr(self, "chk_coverage_overlay") or not self.chk_coverage_overlay.isChecked():
            return
        if self._last_scene_bounds is not None:
            bounds = self._last_scene_bounds
        elif self.rb_file.isChecked() and self._stock_file:
            from src.stock.stl_importer import load_mesh

            bounds = tuple(float(v) for v in load_mesh(self._stock_file).bounds)
        else:
            bounds = (
                0.0,
                self.sp_bx.value(),
                0.0,
                self.sp_by.value(),
                0.0,
                self.sp_bz.value(),
            )
        mesh = self._coverage_gap_mesh(bounds)
        if mesh is None:
            actor = self.plotter.add_text(
                "Coverage gaps: none",
                position=(0.01, 0.14),
                font_size=9,
                color=SUCCESS,
                shadow=True,
                viewport=True,
            )
            self._coverage_actors.append(actor)
            return
        actor = self.plotter.add_mesh(
            mesh,
            color=DANGER,
            opacity=0.38,
            lighting=False,
            show_edges=False,
        )
        self._coverage_actors.append(actor)
        actor = self.plotter.add_text(
            "Red = no cutter footprint coverage",
            position=(0.01, 0.14),
            font_size=9,
            color=DANGER,
            shadow=True,
            viewport=True,
        )
        self._coverage_actors.append(actor)

    def _idle_scene(self):
        """Default view: solid box representing the configured stock."""
        self.plotter.clear()
        self._gcode_actors = []
        self._coverage_actors = []
        self.plotter.set_background(BG, top="#12192e")

        if self.rb_file.isChecked() and self._stock_file:
            return  # already showing STL preview

        w = self.sp_bx.value()
        d = self.sp_by.value()
        h = self.sp_bz.value()
        bounds = (0.0, w, 0.0, d, 0.0, h)
        self._last_scene_bounds = bounds
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
        self._add_coverage_overlay()
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
            self._coverage_actors = []
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
            self._last_scene_bounds = (
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
                float(z_min),
                float(z_max),
            )
            box = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
            self.plotter.add_mesh(box, style="wireframe", color=BOX_C,
                                   opacity=0.2, line_width=1.0)
            self.plotter.add_text(
                f"{os.path.basename(path)}\n"
                f"{x_max-x_min:.1f} x {y_max-y_min:.1f} x {z_max-z_min:.1f} mm",
                position="lower_edge", font_size=9, color=T2,
            )
            self._add_gcode_overlay()
            self._add_coverage_overlay()
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
        self._coverage_actors = []
        self.plotter.set_background(BG, top="#12192e")
        self._last_scene_bounds = bounds

        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        s = self._stock

        max_live_axis = self._live_preview_axis()
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

        # Tool visual mesh: cutter tip at local origin, shank extends along +Z.
        tool_mesh = self._build_tool_visual_mesh()
        self._tool_actor = self.plotter.add_mesh(
            tool_mesh, color=TOOL_C, opacity=0.95,
            smooth_shading=True, specular=0.75,
        )
        self._tool_actor.position = [
            x_min + (x_max - x_min) * 0.1,
            y_min + (y_max - y_min) * 0.1,
            z_max,
        ]

        self._add_gcode_overlay()
        self._add_coverage_overlay()
        self.plotter.show_axes()
        self._set_default_camera(x_min, x_max, y_min, y_max, z_min, z_max)
        self.plotter.render()

    # ── Frame / progress slots ─────────────────────────────────────────

    def _apply_hmap_to_surface(self, hmap: np.ndarray) -> None:
        expected_shape = (
            len(self._display_i_idx) if self._display_i_idx is not None else 0,
            len(self._display_j_idx) if self._display_j_idx is not None else 0,
        )
        if (
            hmap.shape != expected_shape
            and self._display_i_edges is not None
            and self._display_j_edges is not None
        ):
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

    def _surface_of_revolution_mesh(self, rings: list[list[tuple[float, float, float]]]) -> pv.PolyData:
        """Build a wrapped quad mesh from rings sampled around local +Z."""
        if len(rings) < 2 or len(rings[0]) < 3:
            raise ValueError("tool mesh requires at least two rings and three sectors")
        n_v = len(rings)
        n_u = len(rings[0])
        points = np.asarray([pt for ring in rings for pt in ring], dtype=float)
        faces: list[int] = []
        for j in range(n_v - 1):
            for i in range(n_u):
                a = j * n_u + i
                b = j * n_u + ((i + 1) % n_u)
                c = (j + 1) * n_u + ((i + 1) % n_u)
                d = (j + 1) * n_u + i
                faces.extend([4, a, b, c, d])
        return pv.PolyData(points, np.asarray(faces, dtype=np.int64)).clean()

    def _disk_mesh(self, radius: float, z: float, sectors: int, normal_down: bool) -> pv.PolyData:
        points = [(0.0, 0.0, z)]
        for i in range(sectors):
            a = 2.0 * math.pi * i / sectors
            points.append((radius * math.cos(a), radius * math.sin(a), z))
        faces: list[int] = []
        for i in range(sectors):
            a = i + 1
            b = ((i + 1) % sectors) + 1
            if normal_down:
                faces.extend([3, 0, b, a])
            else:
                faces.extend([3, 0, a, b])
        return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))

    def _build_tool_visual_mesh(self) -> pv.PolyData:
        """Create a cutter-shaped visual mesh in local coordinates with tip at origin."""
        tool = self._engine.tool
        r = float(tool.radius)
        sectors = 48
        height = max(4.0 * r, r + 8.0)

        if isinstance(tool, BallEndMill):
            rings = []
            for j in range(18):
                theta = 0.5 * math.pi * j / 17
                rr = r * math.sin(theta)
                z = r - r * math.cos(theta)
                ring = []
                for i in range(sectors):
                    a = 2.0 * math.pi * i / sectors
                    ring.append((rr * math.cos(a), rr * math.sin(a), z))
                rings.append(ring)
            for z in np.linspace(r, r + height, 18)[1:]:
                ring = []
                for i in range(sectors):
                    a = 2.0 * math.pi * i / sectors
                    ring.append((r * math.cos(a), r * math.sin(a), float(z)))
                rings.append(ring)
            mesh = self._surface_of_revolution_mesh(rings)
            top = self._disk_mesh(r, r + height, sectors, normal_down=False)
            return mesh.merge(top).clean()

        rings = []
        for z in np.linspace(0.0, height, 24):
            ring = []
            for i in range(sectors):
                a = 2.0 * math.pi * i / sectors
                ring.append((r * math.cos(a), r * math.sin(a), float(z)))
            rings.append(ring)
        side = self._surface_of_revolution_mesh(rings)
        bottom = self._disk_mesh(r, 0.0, sectors, normal_down=True)
        top = self._disk_mesh(r, height, sectors, normal_down=False)
        return side.merge(bottom).merge(top).clean()

    @pyqtSlot(object, object)
    def _on_frame(self, hmap, tool_pos):
        # Just cache the latest data; the QTimer fires the actual render at 30 fps.
        # This prevents signal-queue pile-up when simulation runs faster than the GPU.
        self._pending_hmap     = hmap
        self._pending_tool_pos = tool_pos
        self._pending_frame_dirty = True

    def _render_pending_frame(self):
        """Called by self._render_timer every 33 ms — one render per tick at most."""
        hmap = self._pending_hmap
        pos = self._pending_tool_pos
        if not self._pending_frame_dirty:
            return
        if self._surface is None or self._xx is None:
            self._pending_frame_dirty = False
            return
        self._pending_hmap = None           # consume
        self._pending_frame_dirty = False
        if hmap is not None:
            try:
                self._apply_hmap_to_surface(hmap)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._set_status(f"Render error: {exc}", DANGER)
                return
        if self._tool_actor is not None and pos is not None:
            if isinstance(pos, dict):
                self._apply_tool_pose_to_actor(
                    self._tool_actor, pos["pos"], pos.get("axis", [0.0, 0.0, 1.0])
                )
            else:
                self._tool_actor.position = pos
        self.plotter.render()

    def _apply_tool_pose_to_actor(self, actor, pos, axis) -> None:
        """Move and orient the tool actor: tip at pos, spindle pointing along axis."""
        z = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(z))
        z = z / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        # Build an orthonormal frame with z as the tool axis.
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[2]) > 0.95 else np.array([0.0, 0.0, 1.0])
        x = np.cross(ref, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.eye(4, dtype=float)
        R[:3, 0] = x
        R[:3, 1] = y
        R[:3, 2] = z
        # user_matrix rotates mesh around its local origin (tip), then
        # actor.position translates in world space.
        actor.user_matrix = R
        actor.position = list(pos)

    @pyqtSlot(int, int)
    def _on_progress(self, cur: int, total: int):
        pct = cur * 100 // total
        self.progress_bar.setValue(pct)
        self.lbl_seg.setText(f"{cur} / {total} segments")

    @pyqtSlot()
    def _on_finished(self):
        self._render_timer.stop()
        self._pending_hmap = None
        self._pending_frame_dirty = False

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
        self._result_mesh = mesh.copy(deep=True)
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
        self._add_coverage_overlay()
        self.plotter.render()

        self.lbl_seg.setText("Idle")
        self.btn_start.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._update_collision_status()
        collision_count = len(getattr(self._engine, "collision_events", []))
        if collision_count:
            self._set_status(f"Complete with {collision_count} collision samples", DANGER)
        else:
            self._set_status("Complete", SUCCESS)

    # ── Speed ──────────────────────────────────────────────────────────

    def _export_result_mesh(self):
        if self._result_mesh is None:
            self._set_status("No finished result to export", WARNING)
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Machined Stock",
            "",
            "STL + STEP (*.stl *.step);;STL Mesh (*.stl);;STEP Faceted CAD (*.step *.stp)",
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if not ext:
            if "STL + STEP" in selected_filter:
                path += ".stl"
                ext = ".stl"
            elif "STEP" in selected_filter:
                path += ".step"
                ext = ".step"
            else:
                path += ".stl"
                ext = ".stl"

        try:
            mesh = self._result_mesh.triangulate().clean()
            if "STL + STEP" in selected_filter:
                base, chosen_ext = os.path.splitext(path)
                if chosen_ext.lower() in (".stl", ".step", ".stp"):
                    stl_path = base + ".stl"
                    step_path = base + ".step"
                else:
                    stl_path = path + ".stl"
                    step_path = path + ".step"
                mesh.save(stl_path, binary=True)
                self._save_faceted_step(mesh, step_path)
                path = f"{stl_path}, {step_path}"
            elif ext == ".stl":
                mesh.save(path, binary=True)
            elif ext in (".step", ".stp"):
                self._save_faceted_step(mesh, path)
            else:
                raise ValueError("Please choose .stl, .step, or .stp")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            self._set_status("Export failed", DANGER)
            return

        self._set_status(f"Exported: {os.path.basename(path)}", SUCCESS)

    # STEP defaults to a compact machined top-surface B-Spline. The full
    # faceted solid is too large for practical SolidWorks assembly workflows.
    _STEP_SURFACE_MAX_GRID = 80

    def _save_faceted_step(self, mesh: "pv.PolyData", path: str) -> None:
        """Export the machined top surface as a compact OpenCascade STEP file."""
        from src.reconstruction.occ_step_export import export_zmap_surface_as_occ_step

        resolution = getattr(self._stock, "resolution", 1.0)
        result = export_zmap_surface_as_occ_step(
            self._stock,
            path,
            max_grid=self._STEP_SURFACE_MAX_GRID,
            tolerance=max(float(resolution) * 0.2, 0.05),
        )
        self._set_status(
            "STEP surface fitted "
            f"{result.source_shape[0]}x{result.source_shape[1]} -> "
            f"{result.output_shape[0]}x{result.output_shape[1]} control samples",
            WARNING,
        )

    @staticmethod
    def _speed_delay_ms(val: int) -> int:
        return max(0, (10 - val) * 15)

    def _live_preview_axis(self) -> int:
        val = self.sl_speed.value() if hasattr(self, "sl_speed") else 4
        if val >= 10:
            return 60
        if val >= 9:
            return 72
        if val >= 8:
            return 90
        if val >= 6:
            return 120
        return 180

    @classmethod
    def _speed_settings(cls, val: int) -> tuple[int, float, int, int]:
        delay_ms = cls._speed_delay_ms(val)
        if val >= 10:
            return delay_ms, 4.0, 80, 120
        if val >= 9:
            return delay_ms, 3.0, 60, 60
        if val >= 8:
            return delay_ms, 2.0, 45, 25
        if val >= 6:
            return delay_ms, 1.5, 33, 8
        return delay_ms, 1.0, 0, 1

    def _speed_label(
        self,
        delay_ms: int,
        step_multiplier: float,
        frame_interval_ms: int,
        segment_batch: int,
    ) -> str:
        if step_multiplier <= 1.0 and frame_interval_ms <= 0 and segment_batch <= 1:
            base = f"{delay_ms} ms delay"
        else:
            base = (
            f"{delay_ms} ms delay, "
            f"batch {segment_batch}, live {self._live_preview_axis()}px"
            )
        if hasattr(self, "chk_final_only") and self.chk_final_only.isChecked():
            return f"{base}, final only"
        return base

    @pyqtSlot(int)
    def _on_speed_changed(self, val: int):
        delay_ms, step_multiplier, frame_interval_ms, segment_batch = self._speed_settings(val)
        self.lbl_speed_val.setText(
            self._speed_label(delay_ms, step_multiplier, frame_interval_ms, segment_batch)
        )
        if self._worker is not None and self._worker.isRunning():
            self._worker.delay_ms = delay_ms
            self._worker.step_multiplier = step_multiplier
            self._worker.frame_interval_ms = frame_interval_ms
            self._worker.segment_batch = segment_batch

    def _on_sim_method_changed(self, index: int):
        is_swept = index == 1
        self.chk_collision.setEnabled(is_swept)
        if not is_swept:
            self.chk_collision.setChecked(False)
        self._update_collision_status()

    def _update_collision_status(self):
        if not hasattr(self, "lbl_collision"):
            return
        events = getattr(self._engine, "collision_events", None)
        if events:
            summary = collision_summary(events)
            detail = ", ".join(
                f"{name} {count}" for name, count in sorted(summary.items())
            )
            self.lbl_collision.setText(f"Collision warning: {detail}")
            self.lbl_collision.setStyleSheet(
                f"color:{DANGER};font-size:13px;font-weight:bold;"
                f"background:{FIELD};border:1px solid {DANGER};"
                f"border-radius:3px;padding:6px 8px;"
            )
            return
        if self.chk_collision.isChecked() and self.chk_collision.isEnabled():
            text = "Collision check: No contact"
            color = SUCCESS
            border = SUCCESS
        elif self.chk_collision.isEnabled():
            text = "Collision check: Off"
            color = T2
            border = T3
        else:
            text = "Collision check: Swept only"
            color = T2
            border = T3
        self.lbl_collision.setText(text)
        self.lbl_collision.setStyleSheet(
            f"color:{color};font-size:13px;font-weight:bold;"
            f"background:{FIELD};border:1px solid {border};"
            f"border-radius:3px;padding:6px 8px;"
        )

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

    @pyqtSlot(bool)
    def _on_coverage_overlay_toggled(self, checked: bool):
        if checked:
            self._add_coverage_overlay()
        else:
            self._clear_coverage_overlay()
        self.plotter.render()

    # ── File browse ────────────────────────────────────────────────────

    def _style_selected_file_label(self, label: QLabel, path: str) -> None:
        label.setText(os.path.basename(path))
        label.setStyleSheet(
            f"color:{T1};font-size:12px;background:{FIELD};"
            f"border:1px solid {ACCENT};border-radius:3px;padding:7px 9px;"
        )

    def _load_default_inputs(self) -> None:
        self.sp_radius.setValue(DEFAULT_TOOL_RADIUS)
        loaded_parts = []

        if os.path.exists(DEFAULT_STOCK_PATH):
            self.rb_file.setChecked(True)
            self._stock_file = DEFAULT_STOCK_PATH
            self._style_selected_file_label(self.lbl_stock_path, DEFAULT_STOCK_PATH)
            loaded_parts.append("stock")

        if os.path.exists(DEFAULT_GCODE_PATH):
            self.rb_gcode.setChecked(True)
            self._gcode_file = DEFAULT_GCODE_PATH
            self._style_selected_file_label(self.lbl_gcode_path, DEFAULT_GCODE_PATH)
            try:
                self._canonical_program = self._load_canonical_gcode(DEFAULT_GCODE_PATH)
                self._gcode_moves = gcode_moves_from_canonical_moves(
                    self._canonical_program.moves
                )
                loaded_parts.append("G-code")
            except Exception as exc:
                self._gcode_moves = None
                self._canonical_program = None
                self._set_status(f"Default G-code parse error: {exc}", DANGER)
                return

        if self._stock_file:
            self._preview_stl(self._stock_file)
        elif self._gcode_moves:
            self._idle_scene()

        missing = []
        if not os.path.exists(DEFAULT_STOCK_PATH):
            missing.append("stock")
        if not os.path.exists(DEFAULT_GCODE_PATH):
            missing.append("G-code")
        if missing:
            self._set_status(f"Default missing: {', '.join(missing)}", WARNING)
        elif loaded_parts:
            n_f = sum(1 for m in self._gcode_moves or [] if not m.rapid)
            self._set_status(
                f"Default loaded: {', '.join(loaded_parts)}, radius {DEFAULT_TOOL_RADIUS:g} mm, {n_f} feed",
                SUCCESS,
            )

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
            self._canonical_program = self._load_canonical_gcode(path)
            self._gcode_moves = gcode_moves_from_canonical_moves(
                self._canonical_program.moves
            )
            n_f = sum(1 for m in self._gcode_moves if not m.rapid)
            n_r = len(self._gcode_moves) - n_f
            # Rebuild the idle/preview scene so the toolpath overlay appears immediately
            if self.rb_file.isChecked() and self._stock_file:
                self._preview_stl(self._stock_file)
            else:
                self._idle_scene()
            warn_count = len(self._canonical_program.warnings)
            controller = self._canonical_program.controller
            suffix = f", {warn_count} warnings" if warn_count else ""
            self._set_status(f"G-code [{controller}]: {n_f} feed, {n_r} rapid{suffix}", SUCCESS)
        except Exception as exc:
            self._gcode_moves = None
            self._canonical_program = None
            self._set_status(f"Parse error: {exc}", DANGER)

    # ── Toolpath builder ───────────────────────────────────────────────

    def _align_toolpath_to_stock(
        self,
        toolpath: list,
        bounds: tuple,
    ) -> list:
        """Translate a toolpath so cutting moves are centred over the stock.

        G-code files often use machine coordinates that are completely different
        from the stock's local coordinate system.  This method computes an XYZ
        offset that:
          • centres the XY footprint of cutting moves over the stock XY centre
          • aligns the highest cutting-move Z with the stock top surface (z_max)

        Supports both ToolPose segment lists and plain ndarray (start, end) pairs.
        If the toolpath is already well inside the stock bounds (offset < 1 mm)
        the original list is returned unchanged.
        """
        # Collect endpoint positions of cutting (non-rapid) moves.
        # Segments can be (ToolPose, ToolPose) for swept-volume mode
        # or (tuple|ndarray, tuple|ndarray) for standard canonical mode.
        cut_positions = []
        for seg_s, seg_e in toolpath:
            if isinstance(seg_e, ToolPose):
                if seg_e.motion_type != "G0":
                    cut_positions.append(seg_e.position)
            elif seg_e is not None:
                cut_positions.append(np.asarray(seg_e, dtype=float))

        if not cut_positions:
            return toolpath

        cut_pos = np.array(cut_positions)
        gx_min, gx_max = float(cut_pos[:, 0].min()), float(cut_pos[:, 0].max())
        gy_min, gy_max = float(cut_pos[:, 1].min()), float(cut_pos[:, 1].max())
        gz_min, gz_max = float(cut_pos[:, 2].min()), float(cut_pos[:, 2].max())

        sx_min, sx_max, sy_min, sy_max, sz_min, sz_max = bounds
        sz_range = max(sz_max - sz_min, 1.0)

        dx = (sx_min + sx_max) / 2.0 - (gx_min + gx_max) / 2.0
        dy = (sy_min + sy_max) / 2.0 - (gy_min + gy_max) / 2.0
        dz = sz_max - gz_max

        # Guard: if dz is implausibly large (G1 safe-height moves inflated gz_max),
        # try aligning gz_min to sz_min instead — this handles work-coordinate
        # G-code where cuts go from slightly above zero down to negative Z.
        if abs(dz) > sz_range * 10:
            dz = sz_min - gz_min

        if abs(dx) < 1.0 and abs(dy) < 1.0 and abs(dz) < 1.0:
            return toolpath

        print(
            f"[Auto-align] G-code → stock offset:"
            f" dx={dx:.2f}  dy={dy:.2f}  dz={dz:.2f}"
        )
        offset = np.array([dx, dy, dz], dtype=float)

        def _shift_pose(pose: ToolPose) -> ToolPose:
            return ToolPose(
                pose.position + offset,
                pose.rotation,
                feed=pose.feed,
                line_no=pose.line_no,
                tool_id=pose.tool_id,
                motion_type=pose.motion_type,
                source_line=pose.source_line,
                segment_index=pose.segment_index,
                segment_count=pose.segment_count,
                arc_center=pose.arc_center,
                arc_radius=pose.arc_radius,
                arc_direction=pose.arc_direction,
            )

        new_toolpath = []
        for seg_s, seg_e in toolpath:
            if isinstance(seg_s, ToolPose) and isinstance(seg_e, ToolPose):
                new_toolpath.append((_shift_pose(seg_s), _shift_pose(seg_e)))
            elif not isinstance(seg_s, ToolPose) and not isinstance(seg_e, ToolPose):
                # plain tuple or ndarray positions (standard canonical mode)
                s_arr = np.asarray(seg_s, dtype=float) + offset
                e_arr = np.asarray(seg_e, dtype=float) + offset
                new_toolpath.append((tuple(s_arr), tuple(e_arr)))
            else:
                new_toolpath.append((seg_s, seg_e))
        return new_toolpath

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

    def _make_gcode_parser(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mpf":
            return SiemensSinumerikParser()
        return GCodeParser()

    def _load_canonical_gcode(self, path: str):
        parser = self._make_gcode_parser(path)
        return parser.parse_file_canonical(path)

    @staticmethod
    def _toolpath_from_canonical(program) -> list:
        return [
            (move.start, move.end)
            for move in program.moves
            if not move.rapid
        ]

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
            use_swept = self.cb_sim_method.currentIndex() == 1
            if use_swept:
                self._engine = SweptVolumeSimulationEngine(
                    self._stock,
                    tool,
                    radial_segments=8,
                    axial_segments=3,
                    use_envelope=False,
                    subdivide_moves=False,
                    legacy_z_topdown=True,
                    detect_collision=self.chk_collision.isChecked(),
                    collision_pose_samples=2,
                    collision_n_u=8,
                    collision_n_v=3,
                )
            else:
                self._engine = SimulationEngine(self._stock, tool)
            self._update_collision_status()

            # ── Build toolpath ────────────────────────────────────────
            if self.rb_gcode.isChecked():
                if not self._gcode_file:
                    self._set_status("Please select a G-code file", WARNING)
                    return
                program = self._load_canonical_gcode(self._gcode_file)
                self._canonical_program = program
                self._gcode_moves = gcode_moves_from_canonical_moves(program.moves)
                if use_swept:
                    # Include G0 rapid moves so the tool visually follows the
                    # full path; SimWorker skips simulation for rapid segments.
                    toolpath = all_pose_segments_from_gcode_moves(self._gcode_moves)
                else:
                    toolpath = self._toolpath_from_canonical(program)
                if not toolpath:
                    self._set_status("No cutting moves found in G-code", WARNING)
                    return
                # Auto-align: G-code may use machine coordinates; shift so
                # cutting moves are centred on the stock and the highest cut
                # aligns with the stock top surface.
                toolpath = self._align_toolpath_to_stock(toolpath, bounds)
            else:
                toolpath = self._build_builtin_toolpath(
                    self.cb_strategy.currentIndex(), bounds
                )

            # ── Launch ────────────────────────────────────────────────
            self._result_mesh = None
            self.btn_export.setEnabled(False)
            run_note = ""

            self._build_scene(bounds)

            delay_ms, step_multiplier, frame_interval_ms, segment_batch = self._speed_settings(
                self.sl_speed.value()
            )
            self._worker = SimWorker(
                self._engine,
                toolpath,
                emit_every=1,
                delay_ms=delay_ms,
                step_multiplier=step_multiplier,
                frame_interval_ms=frame_interval_ms,
                segment_batch=segment_batch,
                preview_i_idx=self._display_i_idx,
                preview_j_idx=self._display_j_idx,
                live_surface=not self.chk_final_only.isChecked(),
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.finished.connect(self._on_finished)
            self._pending_hmap = None
            self._pending_tool_pos = None
            self._pending_frame_dirty = False
            self._render_timer.start()
            self._worker.start()
            launched = True   # simulation is now running; _on_mesh_ready re-enables Start

            self.btn_stop.setEnabled(True)
            self.progress_bar.setValue(0)
            n = len(toolpath)
            method = "Swept Volume" if use_swept else "Legacy Continuous"
            self._set_status(f"Simulating {method} ({n} moves){run_note}", SUCCESS)
            self._update_collision_status()

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
        self._pending_frame_dirty = False
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)
        self._set_status("Stopped", DANGER)

    def _reset(self):
        self._render_timer.stop()
        self._pending_hmap = None
        self._pending_frame_dirty = False
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        if self._mesh_worker and self._mesh_worker.isRunning():
            self._mesh_worker.wait()
        self._surface       = None
        self._surface_actor = None
        self._tool_actor    = None
        self._result_mesh   = None
        self.progress_bar.setValue(0)
        self.lbl_seg.setText("Idle")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._engine = None
        self._update_collision_status()
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
