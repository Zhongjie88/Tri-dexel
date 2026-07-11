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
  pip install PyQt6 pyvistaqt pyvista scikit-image numpy
"""

import sys
import math
import os
import time
from pathlib import Path
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
    QDockWidget, QFormLayout, QPlainTextEdit,
    QGroupBox, QToolButton, QTabWidget, QStyle,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot, QEvent, QSize, QPointF
from PyQt6.QtGui import (
    QFont, QPainter, QColor, QPen, QBrush, QPolygonF, QRadialGradient, QPixmap,
    QIcon,
)

import pyvista as pv
from pyvistaqt import QtInteractor

from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import BallEndMill, FlatEndMill
from src.simulation.engine import SimulationEngine
from src.simulation.sv_engine import SweptVolumeSimulationEngine
from src.simulation.collision import collision_summary
from src.reconstruction.mesh import height_map_to_surface
from src.gcode import GCodeParser, SiemensSinumerikParser
from src.motion.gcode import (
    gcode_moves_from_canonical_moves,
    tool_pose_from_gcode_move,
    all_pose_segments_from_gcode_moves,
)
from src.motion.pose import ToolPose


# ══════════════════════════════════════════════════════════════════════════
# Colour palette
# ══════════════════════════════════════════════════════════════════════════

BG      = "#e8edf5"   # window / viewport background
PANEL   = "#f3f6fb"   # dock and ribbon panels
CARD    = "#edf2f8"   # widget / group background
FIELD   = "#fbfcff"   # input background
ACCENT  = "#1d5fae"   # IMACT blue highlight
SUCCESS = "#35a852"   # green
WARNING = "#f59e0b"   # amber
DANGER  = "#dc3545"   # red
T1      = "#162033"   # primary text
T2      = "#4f5d72"   # secondary text
T3      = "#b8c4d3"   # muted / borders
SURF    = "#aab2bd"   # stock material colour
TOOL_C  = "#1d5fae"   # tool colour
BOX_C   = "#2b65b5"   # wireframe colour
CUT_CMAP = "turbo"     # high-contrast cut-depth map
G0_C    = "#2f80ed"   # G0 rapid moves
G1_C    = "#e0ad2f"   # G1 feed moves

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

_EXAMPLES_DIR = Path(__file__).parent / "examples"
DEFAULT_STOCK_PATH = ""                                     # no default STL bundled
DEFAULT_GCODE_PATH = ""                                     # no default G-code
DEFAULT_TOOL_RADIUS = 5.0
FIVE_AXIS_DEMO_STOCK_PATH = ""
FIVE_AXIS_DEMO_GCODE_PATH = ""
FIVE_AXIS_DEMO_RADIUS = 6.0


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
        self.frame_interval_ms = frame_interval_ms
        self.segment_batch = segment_batch
        self.preview_i_idx = preview_i_idx
        self.preview_j_idx = preview_j_idx
        self.live_surface = live_surface
        self._stop      = False
        self.profile = {
            "sim_s": 0.0,
            "cut_s": 0.0,
            "hmap_s": 0.0,
            "emit_s": 0.0,
            "cut_samples": 0,
            "hmap_count": 0,
            "frame_count": 0,
        }

    def stop(self):
        self._stop = True

    def _preview_height_map(self) -> np.ndarray:
        t0 = time.perf_counter()
        full = self.engine.stock.z_grid.height_map()   # O(1) cached copy, shape (nx, ny)
        try:
            if self.preview_i_idx is None or self.preview_j_idx is None:
                return full
            rows = np.asarray(self.preview_i_idx, dtype=int)
            cols = np.asarray(self.preview_j_idx, dtype=int)
            return full[np.ix_(rows, cols)]                 # vectorised numpy indexing, O(n)
        finally:
            self.profile["hmap_s"] += time.perf_counter() - t0
            self.profile["hmap_count"] += 1

    def run(self):
        t0 = time.perf_counter()
        try:
            if type(self.engine) is SimulationEngine:
                self._run_legacy_continuous()
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
                            cut_t = time.perf_counter()
                            self.engine.simulate_pose_move(seg_s, seg_e, step=step)
                            self.profile["cut_s"] += time.perf_counter() - cut_t
                        last_pos = {
                            "pos": tuple(float(v) for v in seg_e.position),
                            "axis": tuple(float(v) for v in seg_e.axis),
                        }
                        if is_rapid:
                            # Emit immediately so rapid repositioning is visible;
                            # do not wait for the batch-end rate-limited emit.
                            hmap = self._preview_height_map() if self.live_surface else None
                            emit_t = time.perf_counter()
                            self.frame_ready.emit(hmap, last_pos)
                            self.profile["emit_s"] += time.perf_counter() - emit_t
                            self.profile["frame_count"] += 1
                            last_frame_t = time.perf_counter()
                            last_pos = None  # consumed; prevent double-emit below
                    else:
                        cut_t = time.perf_counter()
                        self.engine.simulate_move(seg_s, seg_e, step=step)
                        self.profile["cut_s"] += time.perf_counter() - cut_t
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
                    emit_t = time.perf_counter()
                    self.frame_ready.emit(hmap, last_pos)
                    self.profile["emit_s"] += time.perf_counter() - emit_t
                    self.profile["frame_count"] += 1
                    last_frame_t = now
                d = self.delay_ms
                if d > 0:
                    self.msleep(d)
        finally:
            self.profile["sim_s"] = time.perf_counter() - t0
            self.finished.emit()

    def _run_legacy_continuous(self):
        total = len(self.toolpath)
        if (
            self.toolpath
            and isinstance(self.toolpath[0][0], ToolPose)
            and isinstance(self.toolpath[0][1], ToolPose)
        ):
            self._run_legacy_oriented()
            return

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

        cut_t = time.perf_counter()
        self.engine.simulate_toolpath(
            self.toolpath,
            step=step,
            progress_callback=_progress,
            stop_callback=lambda: self._stop,
        )
        self.profile["cut_s"] += time.perf_counter() - cut_t

    def _run_legacy_oriented(self):
        total = len(self.toolpath)
        step = self.engine.stock.resolution
        sample_u = 32
        sample_v = 12
        # hmap is expensive; only recompute at the original frame_interval_ms rate.
        hmap_interval_s = max(0.033, int(self.frame_interval_ms) / 1000.0)
        # Tool position is cheap — update at 30 fps so the cutter visually moves
        # smoothly even when simulation runs faster than frame_interval_ms.
        TOOL_INTERVAL_S = 1.0 / 30.0   # 33 ms

        last_hmap_t = 0.0
        last_tool_t = 0.0
        last_progress_t = 0.0

        def _emit(cur: int, n_total: int, pose, force: bool = False):
            nonlocal last_hmap_t, last_tool_t, last_progress_t
            if self._stop:
                return
            now = time.perf_counter()
            if force or now - last_progress_t >= 0.05 or cur >= n_total:
                self.progress.emit(cur, n_total)
                last_progress_t = now
            tool_due = force or now - last_tool_t >= TOOL_INTERVAL_S or cur >= n_total
            hmap_due = force or now - last_hmap_t >= hmap_interval_s or cur >= n_total
            if tool_due:
                hmap = self._preview_height_map() if (self.live_surface and hmap_due) else None
                self.frame_ready.emit(
                    hmap,
                    {
                        "pos": tuple(float(v) for v in pose.position),
                        "axis": tuple(float(v) for v in pose.axis),
                    },
                )
                last_tool_t = now
                if hmap_due:
                    last_hmap_t = now
            d = self.delay_ms
            if d > 0:
                self.msleep(d)

        # Arc-length carry-over: same optimisation as simulate_toolpath — only
        # sample every `step` mm of *accumulated* travel, not per segment.
        # Without this, arc-linearised G-code (many 0.1 mm segments) is ~10×
        # slower because each segment forces at least one apply_tool_at call.
        def _apply_pose(pose: ToolPose) -> None:
            cut_t = time.perf_counter()
            self.engine.apply_tool_pose_at(pose, n_u=sample_u, n_v=sample_v)
            self.profile["cut_s"] += time.perf_counter() - cut_t
            self.profile["cut_samples"] += 1

        carry = 0.0
        prev_dir: tuple[float, float, float] | None = None
        corner_cos = math.cos(math.radians(5.0))

        for idx, (start, end) in enumerate(self.toolpath, start=1):
            if self._stop:
                break
            is_rapid = end.motion_type == "G0"
            if is_rapid:
                carry = 0.0
                prev_dir = None
                _emit(idx, total, end, force=True)
                continue

            x1, y1, z1 = float(start.position[0]), float(start.position[1]), float(start.position[2])
            x2, y2, z2 = float(end.position[0]),   float(end.position[1]),   float(end.position[2])
            dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
            length = math.sqrt(dx * dx + dy * dy + dz * dz)

            if length <= 1e-12:
                _apply_pose(end)
                carry = 0.0
                prev_dir = None
            else:
                cur_dir: tuple[float, float, float] = (dx / length, dy / length, dz / length)
                if prev_dir is not None:
                    dot = prev_dir[0]*cur_dir[0] + prev_dir[1]*cur_dir[1] + prev_dir[2]*cur_dir[2]
                    if dot < corner_cos:
                        _apply_pose(start)
                        carry = 0.0
                t = carry / length
                while t <= 1.0:
                    _apply_pose(start.interpolate(end, t))
                    t += step / length
                carry = (t - 1.0) * length
                prev_dir = cur_dir

            _emit(idx, total, end)


class MeshWorker(QThread):
    """Builds a smooth marching-cubes mesh after simulation in the background."""

    mesh_ready = pyqtSignal(object)    # pv.PolyData
    error      = pyqtSignal(str)       # error message if meshing fails

    def __init__(self, stock):
        super().__init__()
        self.stock = stock
        self.elapsed_s = 0.0

    def run(self):
        t0 = time.perf_counter()
        try:
            from src.reconstruction.mesh import voxel_to_mesh
            voxels = self.stock.to_voxel_grid()
            mesh = voxel_to_mesh(voxels, self.stock.bounds, self.stock.resolution)
            self.elapsed_s = time.perf_counter() - t0
            self.mesh_ready.emit(mesh)
        except Exception as exc:
            self.elapsed_s = time.perf_counter() - t0
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════
# Gizmo overlay
# ══════════════════════════════════════════════════════════════════════════

class GizmoOverlay(QWidget):
    """Transparent widget painted over the plotter's bottom-right corner.

    Draws X / Y / Z axis arrows in a fixed isometric projection so the user
    always knows the world orientation of the 3-D viewport.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(82, 82)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = 24, 60  # isometric origin in widget coords

        # (dx, dy, hex_color, label) — fixed isometric projection
        axes = [
            ( 44,  -8, "#e05555", "X"),
            (-24, -13, "#55a855", "Y"),
            (  0, -42, "#4a9adb", "Z"),
        ]
        for dx, dy, color_hex, label in axes:
            color = QColor(color_hex)
            p.setPen(QPen(color, 2))
            ex, ey = cx + dx, cy + dy
            p.drawLine(cx, cy, ex, ey)
            # Arrow head
            ang = math.atan2(-dy, dx)
            tip, half = 7, 0.5
            p.drawLine(ex, ey,
                       int(ex - tip * math.cos(ang - half)),
                       int(ey + tip * math.sin(ang - half)))
            p.drawLine(ex, ey,
                       int(ex - tip * math.cos(ang + half)),
                       int(ey + tip * math.sin(ang + half)))
            # Label
            p.setPen(QPen(color.lighter(140), 1))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            lx = ex + (5 if dx >= 0 else -14)
            ly = ey + (5 if dy >= 0 else -3)
            p.drawText(lx, ly, label)


class BrandLogoWidget(QWidget):
    """Ribbon logo widget.

    Loads the user's logo from assets when available.  The fallback drawing is
    intentionally compact so the ribbon does not look like a placeholder.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("brand_logo_mark")
        self.setFixedSize(112, 40)
        self._pixmap = self._load_logo_pixmap()

    @staticmethod
    def _load_logo_pixmap() -> QPixmap | None:
        path = app_logo_path()
        if path is not None:
            pix = QPixmap(str(path))
            if not pix.isNull():
                return pix
        return None

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
            return

        # Fallback mark based on the supplied IMACT logo direction.
        cx, cy = self.width() / 2, self.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1f242c"))
        p.drawRoundedRect(1, 3, self.width() - 2, self.height() - 6, 5, 5)

        p.save()
        p.translate(cx, cy)
        p.rotate(-16)
        p.setPen(QPen(QColor("#3d75c7"), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0, 0), 31, 12)
        p.rotate(34)
        p.drawEllipse(QPointF(0, 0), 31, 12)
        p.restore()

        yellow = QColor("#f2c84b")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(yellow)
        p.drawRect(int(cx - 3), 7, 6, self.height() - 14)
        p.drawPolygon(QPolygonF([
            QPointF(cx - 3, cy - 2),
            QPointF(cx + 19, cy + 4),
            QPointF(cx - 3, cy + 10),
        ]))

        p.setPen(QPen(QColor("#101820"), 1))
        p.setFont(QFont("Segoe UI", 5, QFont.Weight.Bold))
        p.drawText(21, int(cy + 6), "I.M.A.C.T")


def app_logo_path() -> Path | None:
    """Return the first available application logo path."""
    base = Path(__file__).resolve().parent
    candidates = [
        base / "assets" / "logo.png",
        base / "assets" / "imact_logo.png",
        base / "assets" / "imact-logo.png",
        base / "logo.png",
        base / "imact_logo.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def app_icon() -> QIcon:
    """Return the app icon used by the native window and taskbar."""
    path = app_logo_path()
    return QIcon(str(path)) if path is not None else QIcon()


# ══════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("")
        self.setWindowIcon(app_icon())
        self.resize(1520, 900)
        self.setMinimumSize(1280, 760)

        # simulation state
        self._stock: TriDexelStock | None       = None
        self._engine: SimulationEngine | SweptVolumeSimulationEngine | None = None
        self._worker: SimWorker | None          = None
        self._mesh_worker: MeshWorker | None    = None
        self._surface: pv.StructuredGrid | None = None
        self._surface_actor                     = None
        self._feature_edge_actor                = None
        self._tool_actor                        = None
        self._result_mesh: pv.PolyData | None   = None
        self._display_mesh: pv.PolyData | None  = None
        self._last_sim_method_idx               = 0
        self._last_display_quality_idx          = 0
        self._mesh_worker_display_result        = True
        self._display_data_state                = "Not ready"
        self._export_data_state                 = "Not ready"
        self._gcode_actors: list                = []
        self._coverage_actors: list             = []
        self._xx = self._yy                     = None
        self._display_i_idx = self._display_j_idx = None
        self._display_i_edges = self._display_j_edges = None

        # file state
        self._stock_file: str | None = None
        self._gcode_file: str | None = None
        self._gcode_moves: list | None = None          # parsed on file browse; raw machine coords
        self._display_gcode_moves: list | None = None  # auto-aligned to stock coords for overlay
        self._canonical_program = None
        self._last_scene_bounds: tuple | None = None
        self._last_sim_profile: dict | None = None

        # UI dock/console refs (set by _build_ui; guard log() before init)
        self.console: QPlainTextEdit | None = None
        self._left_dock: QDockWidget | None = None
        self._right_dock: QDockWidget | None = None
        self._console_dock: QDockWidget | None = None

        # render throttle — simulation thread writes here; QTimer reads at 30 fps
        self._pending_hmap: np.ndarray | None = None
        self._pending_tool_pos: list | None    = None
        self._pending_frame_dirty = False
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)          # ≈ 30 fps
        self._render_timer.timeout.connect(self._render_pending_frame)
        self._fa_preview_timer = QTimer(self)
        self._fa_preview_timer.setInterval(33)
        self._fa_preview_timer.timeout.connect(self._five_axis_preview_tick)
        self._fa_preview_poses: list[ToolPose] = []
        self._fa_preview_moves: list = []
        self._fa_preview_index = 0
        self._fa_preview_tool_actor = None
        self._fa_preview_axis_actor = None
        self._fa_preview_feed_actor = None
        self._fa_preview_path_actors: list = []

        self._build_ui()
        self._apply_stylesheet()
        self._idle_scene()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        # ── Ribbon (replaces menu bar) ────────────────────────────────────
        self.setMenuWidget(self._build_ribbon())

        # ── Central widget: view header + 3-D plotter ─────────────────────
        central = QFrame()
        central.setObjectName("vp_frame")
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        view_header = QFrame()
        view_header.setObjectName("view_header")
        vh = QHBoxLayout(view_header)
        vh.setContentsMargins(16, 6, 16, 6)
        vh.setSpacing(14)
        view_title = QLabel("Simulation View")
        view_title.setObjectName("view_title")
        vh.addWidget(view_title)
        # Info chips — updated dynamically via _sync_view_chips()
        self.lbl_view_mode = QLabel("Mode: -")
        self.lbl_view_mode.setObjectName("view_chip_mode")
        self.lbl_view_tool = QLabel()
        self.lbl_view_tool.setObjectName("view_chip_tool")
        self.lbl_view_tool.setVisible(False)
        vh.addSpacing(6)
        vh.addWidget(self.lbl_view_mode)
        vh.addWidget(self.lbl_view_tool)
        vh.addStretch()
        view_hint = QLabel("LMB Rotate  |  RMB Pan  |  Wheel Zoom")
        view_hint.setObjectName("view_hint")
        view_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        vh.addWidget(view_hint)
        cl.addWidget(view_header)

        self.plotter = QtInteractor(central)
        cl.addWidget(self.plotter, stretch=1)

        # Keep the VTK ViewCube, but do not draw the old black XYZ overlay.
        self._gizmo = None
        # VTK orientation cube — initialise after the render window is ready
        QTimer.singleShot(0, self._init_view_cube)

        self.setCentralWidget(central)

        # ── Left properties dock ──────────────────────────────────────────
        self._left_dock = self._build_left_dock()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)
        self.resizeDocks([self._left_dock], [340], Qt.Orientation.Horizontal)

        self._right_dock = self._build_right_dock()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._right_dock)
        self.resizeDocks([self._right_dock], [330], Qt.Orientation.Horizontal)

        # ── Console dock (bottom, hidden by default) ──────────────────────
        self._console_dock = self._build_console_dock()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)

        # ── Status bar ────────────────────────────────────────────────────
        sb = self.statusBar()
        sb.setObjectName("main_statusbar")
        sb.setSizeGripEnabled(False)

        # Left: dot + overall status + slim progress bar
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setObjectName("status")
        sb.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.setTextVisible(False)
        sb.addWidget(self.progress_bar)

        # Right: info segments (permanent)
        def _sb_sep():
            s = QFrame()
            s.setFrameShape(QFrame.Shape.VLine)
            s.setObjectName("sb_sep")
            return s

        self.lbl_method   = QLabel("Legacy Surface Fast")
        self.lbl_method.setObjectName("sb_info")
        self.lbl_res_disp = QLabel("Resolution: 1.00 mm")
        self.lbl_res_disp.setObjectName("sb_info")
        self.lbl_rad_disp = QLabel("Tool Radius: 5.00 mm")
        self.lbl_rad_disp.setObjectName("sb_info")
        self.lbl_seg      = QLabel("Segments: -")
        self.lbl_seg.setObjectName("sb_info")
        self.lbl_collision = QLabel("Collision: -")
        self.lbl_collision.setObjectName("collision_status")

        for w in [_sb_sep(), self.lbl_method,
                  _sb_sep(), self.lbl_res_disp,
                  _sb_sep(), self.lbl_rad_disp,
                  _sb_sep(), self.lbl_seg,
                  _sb_sep(), self.lbl_collision]:
            sb.addPermanentWidget(w)

        # Wire live updates
        self.cb_sim_method.currentIndexChanged.connect(self._sync_status_bar)
        self.cb_display_quality.currentIndexChanged.connect(self._sync_status_bar)
        self.sp_res.valueChanged.connect(self._sync_status_bar)
        self.sp_radius.valueChanged.connect(self._sync_status_bar)
        self.sp_cutting_length.valueChanged.connect(self._sync_status_bar)
        self.cb_tool.currentIndexChanged.connect(self._sync_status_bar)
        self._sync_status_bar()
        self._set_data_state("Not ready", "Not ready")

    def _sync_status_bar(self, *_):
        method_map = {0: "Legacy Surface Fast", 1: "Legacy Full Tri-Dexel", 2: "Swept Volume"}
        m = method_map.get(self.cb_sim_method.currentIndex(), "—")
        self.lbl_method.setText(m)
        self.lbl_res_disp.setText(f"Resolution: {self.sp_res.value():.2f} mm")
        self.lbl_rad_disp.setText(f"Tool Radius: {self.sp_radius.value():.2f} mm")
        if hasattr(self, "lbl_operation_title"):
            self.lbl_operation_title.setText(f"Operation: {m}")
            self.lbl_op_tool.setText(self.cb_tool.currentText())
            self.lbl_op_radius.setText(f"{self.sp_radius.value():.2f} mm")
            self.lbl_op_cutting_length.setText(
                f"{self.sp_cutting_length.value():.2f} mm"
            )
            self.lbl_op_resolution.setText(f"{self.sp_res.value():.2f} mm")
            self.lbl_op_method.setText(m)
            self.lbl_op_speed.setText(self.lbl_speed_val.text() if hasattr(self, "lbl_speed_val") else "-")
            collision = "Enabled" if self.chk_collision.isChecked() and self.chk_collision.isEnabled() else "Off"
            self.lbl_op_collision.setText(collision)
            self.lbl_op_display_quality.setText(self._display_quality_label())

    def _display_quality_index(self) -> int:
        if hasattr(self, "cb_display_quality"):
            return int(self.cb_display_quality.currentIndex())
        return int(self._last_display_quality_idx)

    def _display_quality_label(self, index: int | None = None) -> str:
        labels = {
            0: "Fast Surface",
            1: "Balanced Mesh",
            2: "Full Mesh",
        }
        idx = self._display_quality_index() if index is None else int(index)
        return labels.get(idx, "Fast Surface")

    def _set_data_state(
        self,
        display: str | None = None,
        export: str | None = None,
    ) -> None:
        if display is not None:
            self._display_data_state = display
        if export is not None:
            self._export_data_state = export
        if hasattr(self, "lbl_op_display_data"):
            self.lbl_op_display_data.setText(self._display_data_state)
        if hasattr(self, "lbl_op_export_data"):
            self.lbl_op_export_data.setText(self._export_data_state)
        if hasattr(self, "lbl_operation_state"):
            if self._export_data_state.startswith("Ready"):
                self.lbl_operation_state.setText("Ready")
            elif "Building" in self._export_data_state:
                self.lbl_operation_state.setText("Building export mesh")
            else:
                self.lbl_operation_state.setText("Ready")

    def _sync_view_chips(self, mode: str = "Ready"):
        tool_label = self.cb_tool.currentText()
        self.lbl_view_mode.setText(f"Mode: {mode}")
        self.lbl_view_tool.setText(
            f"Tool: {tool_label}  /  Radius: {self.sp_radius.value():.2f} mm"
        )
        self.lbl_view_tool.setVisible(True)

    def _set_button_icon(
        self,
        button: QPushButton,
        icon_name: str,
        size: int = 16,
    ) -> None:
        """Attach a native Qt standard icon to a command button."""
        standard_pixmap = getattr(QStyle.StandardPixmap, icon_name, None)
        if standard_pixmap is None:
            return
        button.setIcon(self.style().standardIcon(standard_pixmap))
        button.setIconSize(QSize(size, size))

    def _build_left_dock(self) -> QDockWidget:
        dock = QDockWidget("Operation Tree", self)
        dock.setObjectName("left_dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        scroll = QScrollArea()
        scroll.setObjectName("control_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("content")
        cv = QVBoxLayout(content)
        cv.setContentsMargins(12, 14, 12, 14)
        cv.setSpacing(16)

        # ── FEEDSTOCK ─────────────────────────────────────────────────
        grp_stock = QGroupBox("Project / Stock")
        grp_stock.setObjectName("dock_group")
        sv = QVBoxLayout(grp_stock)
        sv.setContentsMargins(14, 14, 14, 14)
        sv.setSpacing(10)

        self._stock_grp = QButtonGroup(self)
        self.rb_box   = QRadioButton("Box Stock")
        self.rb_file  = QRadioButton("Import File  (.STL / .STEP)")
        self.rb_box.setChecked(True)
        self._stock_grp.addButton(self.rb_box,  0)
        self._stock_grp.addButton(self.rb_file, 1)
        sv.addWidget(self.rb_box)

        self.w_box_dims = QWidget()
        form_dims = QFormLayout(self.w_box_dims)
        form_dims.setContentsMargins(8, 4, 0, 4)
        form_dims.setSpacing(8)
        form_dims.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label_text, attr, val in [
            ("Width",  "sp_bx", 100.0),
            ("Depth",  "sp_by", 100.0),
            ("Height", "sp_bz",  50.0),
        ]:
            sp = QDoubleSpinBox()
            sp.setRange(1.0, 9999.0)
            sp.setValue(val)
            sp.setSuffix(" mm")
            sp.setDecimals(0)
            setattr(self, attr, sp)
            lbl = QLabel(label_text)
            lbl.setObjectName("field_label")
            form_dims.addRow(lbl, sp)
        sv.addWidget(self.w_box_dims)

        sv.addWidget(self.rb_file)

        self.w_stl = QWidget()
        self.w_stl.setVisible(False)
        sf = QVBoxLayout(self.w_stl)
        sf.setContentsMargins(8, 6, 0, 6)
        sf.setSpacing(8)
        self.lbl_stock_path = QLabel("No file selected")
        self.lbl_stock_path.setObjectName("filepath")
        self.lbl_stock_path.setWordWrap(False)
        self.lbl_stock_path.setMinimumWidth(0)
        self.lbl_stock_path.setFixedHeight(34)
        self.lbl_stock_path.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        btn_stl = QPushButton("Browse ...")
        btn_stl.setObjectName("browse_btn")
        self._set_button_icon(btn_stl, "SP_DirOpenIcon")
        btn_stl.clicked.connect(self._browse_stock_file)
        sf.addWidget(self.lbl_stock_path)
        sf.addWidget(btn_stl)
        sv.addWidget(self.w_stl)

        self.rb_file.toggled.connect(lambda on: (
            self.w_stl.setVisible(on),
            self.w_box_dims.setVisible(not on),
        ))
        cv.addWidget(grp_stock)

        # ── TOOLPATH ──────────────────────────────────────────────────
        grp_path = QGroupBox("Toolpath")
        grp_path.setObjectName("dock_group")
        tv = QVBoxLayout(grp_path)
        tv.setContentsMargins(14, 14, 14, 14)
        tv.setSpacing(10)

        self._path_grp = QButtonGroup(self)
        self.rb_builtin = QRadioButton("Built-in Strategy")
        self.rb_gcode   = QRadioButton("G-code File  (.nc / .gcode)")
        self.rb_gcode.setChecked(True)
        self.rb_builtin.setVisible(False)
        self._path_grp.addButton(self.rb_builtin, 0)
        self._path_grp.addButton(self.rb_gcode,   1)

        self.w_strategy = QWidget()
        self.w_strategy.setVisible(False)
        sw = QHBoxLayout(self.w_strategy)
        sw.setContentsMargins(8, 4, 0, 4)
        sw.setSpacing(8)
        self.cb_strategy = QComboBox()
        self.cb_strategy.addItems(list(STRATEGIES.keys()))
        sw.addWidget(self.cb_strategy)

        tv.addWidget(self.rb_gcode)

        self.w_gcode = QWidget()
        self.w_gcode.setVisible(True)
        gf = QVBoxLayout(self.w_gcode)
        gf.setContentsMargins(8, 6, 0, 6)
        gf.setSpacing(8)
        self.lbl_gcode_path = QLabel("No file selected")
        self.lbl_gcode_path.setObjectName("filepath")
        self.lbl_gcode_path.setWordWrap(False)
        self.lbl_gcode_path.setMinimumWidth(0)
        self.lbl_gcode_path.setFixedHeight(34)
        self.lbl_gcode_path.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        btn_gc = QPushButton("Browse ...")
        btn_gc.setObjectName("browse_btn")
        self._set_button_icon(btn_gc, "SP_DialogOpenButton")
        btn_gc.clicked.connect(self._browse_gcode_file)
        gf.addWidget(self.lbl_gcode_path)
        gf.addWidget(btn_gc)
        self.chk_gcode_overlay = QCheckBox("Show G-code Path")
        self.chk_gcode_overlay.setObjectName("switch")
        self.chk_gcode_overlay.setChecked(True)
        self.chk_gcode_overlay.toggled.connect(self._on_gcode_overlay_toggled)
        gf.addWidget(self.chk_gcode_overlay)
        self.chk_coverage_overlay = QCheckBox("Show Coverage Gaps")
        self.chk_coverage_overlay.setObjectName("switch")
        self.chk_coverage_overlay.setChecked(False)
        self.chk_coverage_overlay.toggled.connect(self._on_coverage_overlay_toggled)
        gf.addWidget(self.chk_coverage_overlay)
        tv.addWidget(self.w_gcode)

        self.rb_gcode.toggled.connect(lambda on: self.w_gcode.setVisible(on))
        cv.addWidget(grp_path)

        # ── TOOL ──────────────────────────────────────────────────────
        grp_tool = QGroupBox("Tool")
        grp_tool.setObjectName("dock_group")
        tov = QFormLayout(grp_tool)
        tov.setContentsMargins(14, 14, 14, 14)
        tov.setSpacing(10)
        tov.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.cb_tool = QComboBox()
        self.cb_tool.addItems(["Ball-end Mill", "Flat-end Mill"])
        self.cb_tool.setCurrentIndex(1)
        tov.addRow(QLabel("Type"), self.cb_tool)
        self.sp_radius = QDoubleSpinBox()
        self.sp_radius.setRange(0.5, 50.0)
        self.sp_radius.setValue(DEFAULT_TOOL_RADIUS)
        self.sp_radius.setSingleStep(0.5)
        self.sp_radius.setSuffix(" mm")
        tov.addRow(QLabel("Radius"), self.sp_radius)
        self.sp_cutting_length = QDoubleSpinBox()
        self.sp_cutting_length.setRange(0.5, 300.0)
        self.sp_cutting_length.setValue(20.0)
        self.sp_cutting_length.setSingleStep(1.0)
        self.sp_cutting_length.setSuffix(" mm")
        self.sp_cutting_length.setToolTip(
            "Effective flute/cutting length. Geometry above this is collision-only."
        )
        tov.addRow(QLabel("Cut Len"), self.sp_cutting_length)
        cv.addWidget(grp_tool)

        # ── SIMULATION ────────────────────────────────────────────────
        grp_sim = QGroupBox("Simulation")
        grp_sim.setObjectName("dock_group")
        simv = QFormLayout(grp_sim)
        simv.setContentsMargins(14, 14, 14, 14)
        simv.setSpacing(10)
        simv.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_res = QDoubleSpinBox()
        self.sp_res.setRange(0.1, 5.0)
        self.sp_res.setValue(1.0)
        self.sp_res.setSingleStep(0.25)
        self.sp_res.setSuffix(" mm")
        simv.addRow(QLabel("Resolution"), self.sp_res)
        self.cb_sim_method = QComboBox()
        self.cb_sim_method.addItems(
            ["Legacy Surface Fast", "Legacy Full Tri-Dexel", "Swept Volume"]
        )
        simv.addRow(QLabel("Method"), self.cb_sim_method)
        self.cb_display_quality = QComboBox()
        self.cb_display_quality.addItems(
            ["Fast Surface", "Balanced Mesh", "Full Mesh"]
        )
        self.cb_display_quality.setCurrentIndex(0)
        simv.addRow(QLabel("Display"), self.cb_display_quality)
        self.chk_collision = QCheckBox("Check Shank / Holder Collision")
        self.chk_collision.setObjectName("switch")
        self.chk_collision.setChecked(False)
        self.chk_collision.setEnabled(False)
        self.chk_final_only = QCheckBox("Fast Final-only Preview")
        self.chk_final_only.setObjectName("switch")
        self.chk_final_only.setChecked(False)
        # Speed slider
        spd_w = QWidget()
        spd_h = QHBoxLayout(spd_w)
        spd_h.setContentsMargins(0, 0, 0, 0)
        spd_h.setSpacing(6)
        lbl_s = QLabel("0.25x"); lbl_s.setObjectName("muted")
        self.sl_speed = QSlider(Qt.Orientation.Horizontal)
        self.sl_speed.setRange(0, 10)
        self.sl_speed.setValue(4)
        self.sl_speed.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sl_speed.setTickInterval(1)
        self.sl_speed.setToolTip("Preview playback speed. Geometry accuracy is unchanged.")
        lbl_f = QLabel("32x"); lbl_f.setObjectName("muted")
        spd_h.addWidget(lbl_s)
        spd_h.addWidget(self.sl_speed, stretch=1)
        spd_h.addWidget(lbl_f)
        simv.addRow(QLabel("Speed"), spd_w)
        self.lbl_speed_val = QLabel(self._speed_label(self.sl_speed.value()))
        self.lbl_speed_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_speed_val.setObjectName("muted")
        self.lbl_speed_val.setToolTip(
            "Display multiplier only; simulation accuracy still follows resolution."
        )
        simv.addRow(self.lbl_speed_val)
        self.cb_sim_method.currentIndexChanged.connect(self._on_sim_method_changed)
        self.cb_display_quality.currentIndexChanged.connect(
            self._on_display_quality_changed
        )
        self.chk_collision.toggled.connect(
            lambda _: (self._update_collision_status(), self._sync_status_bar())
        )
        self.chk_final_only.toggled.connect(
            lambda _: self._on_speed_changed(self.sl_speed.value())
        )
        self.sl_speed.valueChanged.connect(self._on_speed_changed)
        cv.addWidget(grp_sim)

        cv.addStretch()

        scroll.setWidget(content)
        dock.setWidget(scroll)
        return dock

    def _build_right_dock(self) -> QDockWidget:
        """Build a CAM-style operation inspector on the right side."""
        dock = QDockWidget("Operation Settings", self)
        dock.setObjectName("right_dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        root = QWidget()
        root.setObjectName("operation_panel")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("operation_header")
        hv = QVBoxLayout(header)
        hv.setContentsMargins(14, 10, 14, 10)
        hv.setSpacing(3)
        self.lbl_operation_title = QLabel("Operation: Legacy Simulation")
        self.lbl_operation_title.setObjectName("operation_title")
        self.lbl_operation_state = QLabel("Ready")
        self.lbl_operation_state.setObjectName("operation_state")
        hv.addWidget(self.lbl_operation_title)
        hv.addWidget(self.lbl_operation_state)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.setObjectName("operation_tabs")
        tabs.setDocumentMode(True)

        tool_tab = QWidget()
        tool_form = QFormLayout(tool_tab)
        tool_form.setContentsMargins(14, 14, 14, 14)
        tool_form.setSpacing(8)
        tool_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_op_tool = QLabel()
        self.lbl_op_radius = QLabel()
        self.lbl_op_cutting_length = QLabel()
        self.lbl_op_resolution = QLabel()
        tool_form.addRow(QLabel("Tool"), self.lbl_op_tool)
        tool_form.addRow(QLabel("Radius"), self.lbl_op_radius)
        tool_form.addRow(QLabel("Cut Len"), self.lbl_op_cutting_length)
        tool_form.addRow(QLabel("Resolution"), self.lbl_op_resolution)

        sim_tab = QWidget()
        sim_form = QFormLayout(sim_tab)
        sim_form.setContentsMargins(14, 14, 14, 14)
        sim_form.setSpacing(8)
        sim_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_op_method = QLabel()
        self.lbl_op_speed = QLabel()
        self.lbl_op_collision = QLabel()
        sim_form.addRow(QLabel("Mode"), self.lbl_op_method)
        sim_form.addRow(QLabel("Speed"), self.lbl_op_speed)
        sim_form.addRow(QLabel("Collision"), self.lbl_op_collision)

        display_tab = QWidget()
        display_form = QFormLayout(display_tab)
        display_form.setContentsMargins(14, 14, 14, 14)
        display_form.setSpacing(8)
        display_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_op_path = QLabel("G-code overlay follows the viewport toggle")
        self.lbl_op_stock = QLabel("Stock display uses current feedstock")
        self.lbl_op_view = QLabel("ViewCube: click face to snap view")
        self.lbl_op_display_quality = QLabel()
        self.lbl_op_display_data = QLabel(self._display_data_state)
        self.lbl_op_export_data = QLabel(self._export_data_state)
        for lbl in (
            self.lbl_op_path,
            self.lbl_op_stock,
            self.lbl_op_view,
            self.lbl_op_display_quality,
            self.lbl_op_display_data,
            self.lbl_op_export_data,
        ):
            lbl.setWordWrap(True)
        display_form.addRow(QLabel("Quality"), self.lbl_op_display_quality)
        display_form.addRow(QLabel("Display Data"), self.lbl_op_display_data)
        display_form.addRow(QLabel("Export Data"), self.lbl_op_export_data)
        display_form.addRow(QLabel("Toolpath"), self.lbl_op_path)
        display_form.addRow(QLabel("Stock"), self.lbl_op_stock)
        display_form.addRow(QLabel("View"), self.lbl_op_view)

        tabs.addTab(tool_tab, "Tool")
        tabs.addTab(sim_tab, "Simulation")
        tabs.addTab(display_tab, "Display")
        layout.addWidget(tabs, stretch=1)

        dock.setWidget(root)
        return dock

    # ── Ribbon ─────────────────────────────────────────────────────────

    def _build_ribbon(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("ribbon_bar")
        bar.setFixedHeight(130)
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QFrame()
        title.setObjectName("ribbon_titlebar")
        title.setFixedHeight(46)
        tl = QHBoxLayout(title)
        tl.setContentsMargins(12, 0, 10, 2)
        tl.setSpacing(10)

        logo = BrandLogoWidget()
        logo.setFixedSize(82, 30)
        tl.addWidget(logo)

        title_text = QWidget()
        title_text.setObjectName("ribbon_title_text")
        title_text.setFixedHeight(30)
        tc = QVBoxLayout(title_text)
        tc.setContentsMargins(0, 0, 0, 0)
        tc.setSpacing(0)
        app_name = QLabel("IMACT CAM")
        app_name.setObjectName("ribbon_app_name")
        app_name.setFixedHeight(30)
        app_name.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        tc.addWidget(app_name)
        tl.addWidget(title_text)
        tl.addSpacing(18)

        project = QLabel("Machining Workspace")
        project.setObjectName("ribbon_project")
        tl.addWidget(project)
        tl.addStretch()

        btn_help = QPushButton("?")
        btn_help.setObjectName("ribbon_right_btn")
        btn_help.setFixedSize(26, 24)
        self._set_button_icon(btn_help, "SP_DialogHelpButton", 14)
        btn_cfg = QPushButton("Settings")
        btn_cfg.setObjectName("ribbon_right_btn")
        btn_cfg.setFixedHeight(24)
        btn_cfg.setFixedWidth(86)
        self._set_button_icon(btn_cfg, "SP_FileDialogDetailedView", 14)
        tl.addWidget(btn_help)
        tl.addWidget(btn_cfg)
        outer.addWidget(title)

        tabs = QTabWidget()
        tabs.setObjectName("ribbon")
        tabs.setDocumentMode(True)
        tabs.setFixedHeight(84)
        tabs.addTab(self._build_ribbon_tab_settings(),  "Setup")
        tabs.addTab(self._build_ribbon_tab_toolpath(),  "Toolpath")
        tabs.addTab(self._build_ribbon_tab_simulate(),  "Simulate")
        tabs.addTab(self._build_ribbon_tab_export(),    "Export")
        outer.addWidget(tabs)

        return bar

    def _build_ribbon_tab_settings(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("ribbon_tab")
        h = QHBoxLayout(tab)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(10)
        btn_props = QPushButton("Operation Tree")
        btn_props.setFixedHeight(28)
        btn_props.setFixedWidth(130)
        self._set_button_icon(btn_props, "SP_FileDialogListView")
        btn_props.clicked.connect(
            lambda: self._left_dock.setVisible(not self._left_dock.isVisible())
            if self._left_dock else None
        )
        h.addWidget(btn_props)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.NoFrame)
        sep.setObjectName("ribbon_sep")
        sep.setFixedSize(1, 32)
        h.addWidget(sep)
        btn_console = QPushButton("Console")
        btn_console.setFixedHeight(28)
        btn_console.setFixedWidth(96)
        self._set_button_icon(btn_console, "SP_FileDialogInfoView")
        btn_console.clicked.connect(
            lambda: self._console_dock.setVisible(not self._console_dock.isVisible())
            if self._console_dock else None
        )
        h.addWidget(btn_console)
        h.addStretch()
        return tab

    def _build_ribbon_tab_toolpath(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("ribbon_tab")
        h = QHBoxLayout(tab)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(10)
        btn_load = QPushButton("Load G-code…")
        btn_load.setFixedHeight(28)
        btn_load.setFixedWidth(136)
        self._set_button_icon(btn_load, "SP_DialogOpenButton")
        btn_load.clicked.connect(lambda: (
            self.rb_gcode.setChecked(True),
            self._browse_gcode_file(),
        ))
        h.addWidget(btn_load)
        h.addStretch()
        return tab

    def _build_ribbon_tab_simulate(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("ribbon_tab")
        h = QHBoxLayout(tab)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(8)

        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(32)
        self.btn_start.setFixedWidth(158)
        self._set_button_icon(self.btn_start, "SP_MediaPlay")
        self.btn_start.setAutoDefault(False)
        self.btn_start.setDefault(False)
        self.btn_start.clicked.connect(self._start)
        h.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setFixedWidth(86)
        self._set_button_icon(self.btn_stop, "SP_MediaStop")
        self.btn_stop.setAutoDefault(False)
        self.btn_stop.setDefault(False)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        h.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setFixedHeight(32)
        self.btn_reset.setFixedWidth(90)
        self._set_button_icon(self.btn_reset, "SP_BrowserReload")
        self.btn_reset.setAutoDefault(False)
        self.btn_reset.setDefault(False)
        self.btn_reset.clicked.connect(self._reset)
        h.addWidget(self.btn_reset)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.NoFrame)
        sep2.setObjectName("ribbon_sep")
        sep2.setFixedSize(1, 32)
        h.addWidget(sep2)
        h.addStretch()
        return tab

    def _build_ribbon_tab_export(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("ribbon_tab")
        h = QHBoxLayout(tab)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(8)
        self.btn_export = QPushButton("Export Mesh…")
        self.btn_export.setObjectName("browse_btn")
        self.btn_export.setFixedHeight(28)
        self.btn_export.setFixedWidth(146)
        self._set_button_icon(self.btn_export, "SP_DialogSaveButton")
        self.btn_export.setAutoDefault(False)
        self.btn_export.setDefault(False)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_result_mesh)
        h.addWidget(self.btn_export)
        h.addStretch()
        return tab

    # ── Console dock & event routing ────────────────────────────────────

    def _build_console_dock(self) -> QDockWidget:
        dock = QDockWidget("Console Output", self)
        dock.setObjectName("console_dock")
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setObjectName("console")
        self.console.setMaximumBlockCount(2000)
        dock.setWidget(self.console)
        dock.setMaximumHeight(180)
        dock.hide()
        return dock

    def log(self, msg: str) -> None:
        if self.console is not None:
            self.console.appendPlainText(msg)
        self.statusBar().showMessage(msg, 5000)

    def eventFilter(self, obj, event) -> bool:
        if self._gizmo is not None and obj is self.plotter and event.type() == QEvent.Type.Resize:
            self._gizmo.move(obj.width() - 88, obj.height() - 88)
        return super().eventFilter(obj, event)

    def _init_view_cube(self) -> None:
        """Add a genuine VTK 3D orientation cube (rotates with camera)."""
        try:
            import vtk
        except ImportError:
            return

        _VP = (0.89, 0.02, 0.99, 0.125)

        cube = vtk.vtkAnnotatedCubeActor()
        cube.SetXPlusFaceText("R")
        cube.SetXMinusFaceText("L")
        cube.SetYPlusFaceText("F")
        cube.SetYMinusFaceText("B")
        cube.SetZPlusFaceText("T")
        cube.SetZMinusFaceText("Bt")
        cube.SetFaceTextScale(0.28)

        cp = cube.GetCubeProperty()
        cp.SetColor(0.20, 0.27, 0.52)
        cp.SetAmbient(0.35)
        cp.SetDiffuse(0.75)
        cp.SetSpecular(0.30)
        cp.SetSpecularPower(20.0)

        ep = cube.GetTextEdgesProperty()
        ep.SetColor(0.70, 0.80, 1.00)
        ep.SetLineWidth(1.0)

        # Per-face property map — used for hover highlighting
        self._face_props = {
            "T":  cube.GetZPlusFaceProperty(),
            "Bt": cube.GetZMinusFaceProperty(),
            "F":  cube.GetYPlusFaceProperty(),
            "B":  cube.GetYMinusFaceProperty(),
            "R":  cube.GetXPlusFaceProperty(),
            "L":  cube.GetXMinusFaceProperty(),
        }
        self._face_normal_color = (0.88, 0.92, 1.00)
        self._face_hover_color  = (0.30, 0.60, 1.00)
        self._hover_face: "str | None" = None
        for fp in self._face_props.values():
            fp.SetColor(*self._face_normal_color)
            fp.SetAmbient(0.45)
            fp.SetDiffuse(0.55)

        # Use raw VTK interactor — self.plotter.iren is a pyvistaqt wrapper
        vtk_iren = self.plotter.ren_win.GetInteractor()

        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(cube)
        widget.SetInteractor(vtk_iren)
        widget.SetViewport(*_VP)
        widget.SetEnabled(1)
        widget.InteractiveOff()

        self._orient_widget = widget
        self._orient_cube   = cube
        self._vcube_vp      = _VP

        self._cube_renderer = None
        for getter in (widget.GetRenderer, widget.GetCurrentRenderer):
            try:
                r = getter()
            except Exception:
                r = None
            if r is not None:
                self._cube_renderer = r
                break

        # Fallback for older VTK builds where the widget renderer is only
        # discoverable through the render-window renderer collection.
        it = self.plotter.ren_win.GetRenderers()
        it.InitTraversal()
        r = it.GetNextItem()
        while r is not None:
            vp = r.GetViewport()
            if abs(vp[0] - _VP[0]) < 0.02:
                self._cube_renderer = r
                break
            r = it.GetNextItem()

        vtk_iren.AddObserver("MouseMoveEvent",       self._on_vcube_hover,  2.0)
        vtk_iren.AddObserver("LeftButtonPressEvent", self._on_vcube_click,  2.0)

    # ── ViewCube helpers ────────────────────────────────────────────────

    def _vcube_in_vp(self, x: int, y: int) -> bool:
        """Return True when window-pixel (x,y) falls inside the cube viewport."""
        try:
            w, h = self.plotter.ren_win.GetSize()
        except Exception:
            w, h = self.plotter.window_size
        if w <= 0 or h <= 0:
            return False
        cube_ren = getattr(self, "_cube_renderer", None)
        vp = cube_ren.GetViewport() if cube_ren is not None else self._vcube_vp
        return vp[0] <= x / w <= vp[2] and vp[1] <= y / h <= vp[3]

    def _vcube_pick_face(self, x: int, y: int) -> "str | None":
        """Return face name ('T','F','R','B','L','Bt') or None."""
        import vtk
        cube_ren = self._cube_renderer
        if cube_ren is None:
            return None

        _FACE_MAP = {
            ("Z", True): "T",  ("Z", False): "Bt",
            ("Y", True): "F",  ("Y", False): "B",
            ("X", True): "R",  ("X", False): "L",
        }

        def _dominant(nx, ny, nz):
            _, axis, sv = max((abs(nx), "X", nx), (abs(ny), "Y", ny), (abs(nz), "Z", nz))
            return _FACE_MAP.get((axis, sv > 0))

        # ① Cell picker — gives exact face normal
        cell_picker = vtk.vtkCellPicker()
        cell_picker.SetTolerance(0.02)
        cell_picker.Pick(x, y, 0, cube_ren)
        if cell_picker.GetActor() is not None:
            return _dominant(*cell_picker.GetPickNormal())

        # ② Prop picker fallback — less precise but catches the assembly surface
        prop_picker = vtk.vtkPropPicker()
        if prop_picker.Pick(x, y, 0, cube_ren):
            px, py, pz = prop_picker.GetPickPosition()
            if any((px, py, pz)):          # non-zero hit position
                return _dominant(px, py, pz)

        return None

    def _vcube_apply_highlight(self, face: "str | None") -> None:
        """Recolour all cube faces; highlight the named one."""
        nm, hv = self._face_normal_color, self._face_hover_color
        for name, fp in self._face_props.items():
            fp.SetColor(*(hv if name == face else nm))

    def _on_vcube_hover(self, caller, _event) -> None:
        """Highlight the cube face under the cursor on mouse move."""
        if not hasattr(self, "_orient_widget"):
            return
        x, y = caller.GetEventPosition()
        if not self._vcube_in_vp(x, y):
            face = None
        else:
            face = self._vcube_pick_face(x, y)
        if face != self._hover_face:
            self._hover_face = face
            self._vcube_apply_highlight(face)
            self.plotter.render()

    def _on_vcube_click(self, caller, _event) -> None:
        """Highlight the clicked face then snap camera to that ortho view."""
        if not hasattr(self, "_orient_widget"):
            return
        x, y = caller.GetEventPosition()
        if not self._vcube_in_vp(x, y):
            return

        face = self._vcube_pick_face(x, y)
        if face is None:
            self._snap_nearest_ortho()
            return

        # Snap immediately; the face highlight stays while the cursor remains
        # over the cube, which gives direct CAD-style feedback.
        self._hover_face = face
        self._vcube_apply_highlight(face)
        self._execute_vcube_snap(face)

    def _execute_vcube_snap(self, face: str) -> None:
        """Execute camera snap and restore neutral face colours."""
        _SNAP = {
            "T":  self.plotter.view_xy,
            "Bt": lambda: self.plotter.view_xy(negative=True),
            "F":  self.plotter.view_xz,
            "B":  lambda: self.plotter.view_xz(negative=True),
            "R":  self.plotter.view_yz,
            "L":  lambda: self.plotter.view_yz(negative=True),
        }
        fn = _SNAP.get(face)
        if fn:
            fn()
        self._vcube_apply_highlight(self._hover_face)  # keep current hover state
        self.plotter.render()

    def _snap_nearest_ortho(self) -> None:
        """Snap to the dominant orthographic axis closest to current camera."""
        cam = self.plotter.camera
        pos = cam.position
        fp  = cam.focal_point
        dx, dy, dz = pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]
        _, axis, sign_val = max((abs(dx), "X", dx), (abs(dy), "Y", dy), (abs(dz), "Z", dz))
        {
            ("Z", True):  self.plotter.view_xy,
            ("Z", False): lambda: self.plotter.view_xy(negative=True),
            ("Y", True):  self.plotter.view_xz,
            ("Y", False): lambda: self.plotter.view_xz(negative=True),
            ("X", True):  self.plotter.view_yz,
            ("X", False): lambda: self.plotter.view_yz(negative=True),
        }.get((axis, sign_val > 0), self.plotter.view_isometric)()
        self.plotter.render()

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_card(title: str) -> tuple:
        """Return (card QFrame, content QVBoxLayout) for a labelled section."""
        card = QFrame()
        card.setObjectName("section_card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("section_header")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(12, 7, 12, 7)
        hh.setSpacing(8)
        accent = QFrame()
        accent.setObjectName("section_accent")
        accent.setFixedSize(3, 16)
        lbl = QLabel(title)
        lbl.setObjectName("section_lbl")
        hh.addWidget(accent)
        hh.addWidget(lbl)
        hh.addStretch()
        outer.addWidget(header)

        body = QWidget()
        body.setObjectName("section_body")
        vl = QVBoxLayout(body)
        vl.setContentsMargins(12, 10, 12, 12)
        vl.setSpacing(8)
        outer.addWidget(body)
        return card, vl

    def _build_action_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("action_footer")
        fv = QVBoxLayout(footer)
        fv.setContentsMargins(10, 10, 10, 10)
        fv.setSpacing(7)

        footer_lbl = QLabel("RUN CONTROL")
        footer_lbl.setObjectName("footer_lbl")
        fv.addWidget(footer_lbl)

        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(48)
        self._set_button_icon(self.btn_start, "SP_MediaPlay")
        self.btn_start.setAutoDefault(False)
        self.btn_start.setDefault(False)
        self.btn_start.clicked.connect(self._start)
        fv.addWidget(self.btn_start)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(38)
        self._set_button_icon(self.btn_stop, "SP_MediaStop")
        self.btn_stop.setAutoDefault(False)
        self.btn_stop.setDefault(False)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setFixedHeight(38)
        self._set_button_icon(self.btn_reset, "SP_BrowserReload")
        self.btn_reset.setAutoDefault(False)
        self.btn_reset.setDefault(False)
        self.btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_reset)
        fv.addLayout(btn_row)

        self.btn_export = QPushButton("Export Mesh")
        self.btn_export.setObjectName("browse_btn")
        self.btn_export.setFixedHeight(36)
        self._set_button_icon(self.btn_export, "SP_DialogSaveButton")
        self.btn_export.setAutoDefault(False)
        self.btn_export.setDefault(False)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_result_mesh)
        fv.addWidget(self.btn_export)

        prog_row = QWidget()
        pr = QHBoxLayout(prog_row)
        pr.setContentsMargins(0, 2, 0, 0)
        pr.setSpacing(8)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        pr.addWidget(self.progress_bar, stretch=1)
        self.lbl_seg = QLabel("Idle")
        self.lbl_seg.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_seg.setObjectName("muted")
        self.lbl_seg.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lbl_seg.setFixedHeight(16)
        pr.addWidget(self.lbl_seg)
        fv.addWidget(prog_row)

        self.lbl_collision = QLabel("Collision check: Swept only")
        self.lbl_collision.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_collision.setObjectName("collision_status")
        self.lbl_collision.setFixedHeight(22)
        self.lbl_collision.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        fv.addWidget(self.lbl_collision)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setObjectName("status")
        self.lbl_status.setFixedHeight(34)
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        fv.addWidget(self.lbl_status)

        return footer

    # ── Stylesheet ─────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        CARD_BORDER = "#2d3139"
        CARD_BORDER_ACTIVE = "#3b414b"
        MONO = "'Consolas', 'JetBrains Mono', 'Courier New', monospace"
        SANS = "'Segoe UI', 'SF Pro Text', Arial, sans-serif"
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {BG};
                color: {T1};
                font-family: {SANS};
                font-size: 15px;
            }}

            QFrame#left_panel {{
                background: {PANEL};
                border-right: 1px solid {T3};
            }}
            QWidget#content {{ background: {PANEL}; }}
            QFrame#vp_frame {{ background: {BG}; }}
            QScrollArea#control_scroll {{
                background: {PANEL};
                border: none;
            }}
            QScrollArea#control_scroll > QWidget > QWidget {{
                background: {PANEL};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {T3};
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0;
                background: transparent;
            }}

            /* Panel header */
            QFrame#panel_header {{
                background: #191d24;
                border-bottom: 1px solid {T3};
            }}
            QLabel#logo {{
                color: {ACCENT};
                background: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 21px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QLabel#subtitle {{
                color: {T2};
                background: transparent;
                font-size: 13px;
            }}
            QLabel#mode_badge {{
                color: {SUCCESS};
                background: #111a15;
                border: 1px solid #2d6f45;
                border-radius: 2px;
                font-family: {MONO};
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }}

            /* Viewport header */
            QFrame#view_header {{
                background: #171a20;
                border-bottom: 1px solid {T3};
            }}
            QLabel#view_title {{
                background: transparent;
                color: {T1};
                font-size: 16px;
                font-weight: bold;
            }}
            QFrame#command_strip {{
                background: #20242b;
                border: 1px solid {T3};
                border-radius: 2px;
            }}
            QLabel#command_chip,
            QLabel#command_chip_active {{
                background: transparent;
                color: {T2};
                font-family: {MONO};
                font-size: 11px;
                font-weight: bold;
                padding: 0 10px;
                border-radius: 2px;
            }}
            QLabel#command_chip_active {{
                background: #2b302f;
                color: {ACCENT};
                border: 1px solid #4a4f38;
            }}
            QLabel#view_hint {{
                background: transparent;
                color: {T2};
                font-size: 13px;
            }}

            /* Property manager sections */
            QFrame#section_card {{
                background: #20242b;
                border: 1px solid {CARD_BORDER};
                border-radius: 3px;
            }}
            QFrame#section_card:hover {{
                border: 1px solid {CARD_BORDER_ACTIVE};
            }}
            QFrame#section_header {{
                background: #262b34;
                border-bottom: 1px solid {CARD_BORDER};
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }}
            QFrame#section_accent {{
                background: {ACCENT};
                border-radius: 1px;
            }}
            QWidget#section_body {{
                background: transparent;
            }}
            QLabel#section_lbl {{
                color: {T1};
                font-size: 13px;
                font-weight: bold;
                letter-spacing: 0.5px;
                background: transparent;
            }}

            /* Action footer */
            QFrame#action_footer {{
                background: #181b21;
                border-top: 1px solid {T3};
            }}
            QLabel#footer_lbl {{
                color: {ACCENT};
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
                background: transparent;
            }}

            /* Labels */
            QLabel {{ background: transparent; }}
            QLabel#muted {{ color: {T2}; font-size: 13px; }}
            QLabel#field_label {{
                color: {T2};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#filepath {{
                color: {T2};
                font-family: {MONO};
                font-size: 13px;
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 2px;
                padding: 6px 9px;
            }}
            QLabel#status {{
                color: {ACCENT};
                font-size: 13px;
                font-weight: bold;
                background: transparent;
            }}
            QLabel#collision_status {{
                color: {T2};
                font-size: 12px;
                background: transparent;
            }}

            /* Inputs */
            QComboBox, QDoubleSpinBox {{
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 2px;
                padding: 4px 10px;
                color: {T1};
                font-family: {SANS};
                font-size: 14px;
                min-height: 30px;
            }}
            QComboBox:hover, QDoubleSpinBox:hover,
            QComboBox:focus, QDoubleSpinBox:focus {{
                border-color: {ACCENT};
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {FIELD};
                border: 1px solid {T3};
                selection-background-color: {ACCENT};
                selection-color: #17140b;
                color: {T1};
                font-size: 14px;
            }}

            /* Radio buttons */
            QRadioButton {{
                color: {T2};
                spacing: 9px;
                font-size: 14px;
                background: transparent;
                min-height: 24px;
            }}
            QRadioButton:checked {{ color: {T1}; }}
            QRadioButton::indicator {{
                width: 16px; height: 16px;
                border-radius: 8px;
                border: 2px solid {T3};
                background: transparent;
            }}
            QRadioButton::indicator:checked {{
                border: 2px solid {ACCENT};
                background: {ACCENT};
            }}

            /* Toggle checkboxes */
            QCheckBox#switch {{
                color: {T2};
                spacing: 10px;
                font-size: 14px;
                padding: 3px 0;
                background: transparent;
            }}
            QCheckBox#switch::indicator {{
                width: 36px; height: 18px;
                border-radius: 9px;
                border: 1px solid {T3};
                background: #2b2f38;
            }}
            QCheckBox#switch::indicator:checked {{
                border: 1px solid {ACCENT};
                background: {ACCENT};
            }}

            /* Buttons */
            QPushButton {{
                background: {FIELD};
                color: {T1};
                border: 1px solid {T3};
                border-radius: 3px;
                font-size: 14px;
                font-weight: 500;
                padding: 5px 14px;
                min-height: 32px;
            }}
            QPushButton:hover {{
                background: #2b2f38;
                border-color: {ACCENT};
                color: {T1};
            }}
            QPushButton:disabled {{
                background: {BG};
                color: {T3};
                border-color: {CARD};
            }}
            QPushButton#browse_btn {{
                background: {FIELD};
                color: {T1};
                border: 1px solid {T3};
                border-radius: 3px;
                font-size: 13px;
            }}
            QPushButton#browse_btn:hover {{
                border-color: {ACCENT};
                color: {T1};
            }}
            QPushButton#btn_start {{
                background: {SUCCESS};
                color: #07140c;
                border: 1px solid transparent;
                font-size: 16px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }}
            QPushButton#btn_start:hover {{ background: #46b979; color: #07140c; }}
            QPushButton#btn_start:disabled {{
                background: #193222;
                color: {T3};
                border: 1px solid transparent;
            }}
            QPushButton#btn_stop {{
                background: {DANGER};
                color: #fff;
                border: 1px solid transparent;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#btn_stop:hover {{ background: #e25555; color: #fff; }}
            QPushButton#btn_stop:disabled {{
                background: #1c0f0f;
                color: {T3};
                border: 1px solid transparent;
            }}
            QPushButton#btn_reset {{ color: {T1}; font-size: 14px; }}

            /* Progress bar */
            QProgressBar {{
                background: {FIELD};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 4px;
            }}

            /* Speed slider */
            QSlider::groove:horizontal {{
                background: {FIELD};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 16px; height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 2px;
            }}

            /* ── Ribbon bar wrapper ────────────────────── */
            QFrame#ribbon_bar {{
                background: {BG};
                border-bottom: 1px solid {T3};
            }}
            QFrame#ribbon_logo {{
                background: #0d0f14;
                border-right: 1px solid {T3};
            }}
            QLabel#ribbon_icon {{
                color: {ACCENT};
                font-size: 28px;
                background: transparent;
            }}
            QLabel#ribbon_app_name {{
                color: {T1};
                font-size: 15px;
                font-weight: bold;
                background: transparent;
                letter-spacing: 0.5px;
            }}
            QLabel#ribbon_app_sub {{
                color: {T2};
                font-size: 11px;
                background: transparent;
            }}
            QFrame#ribbon_right {{
                background: transparent;
            }}
            QPushButton#ribbon_right_btn {{
                background: transparent;
                border: none;
                color: {T2};
                font-size: 12px;
                padding: 2px 6px;
                min-height: 22px;
            }}
            QPushButton#ribbon_right_btn:hover {{
                background: {CARD};
                color: {T1};
                border-radius: 3px;
            }}

            /* ── View header chips ──────────────────────── */
            QLabel#view_chip_mode {{
                background: #1a3a4a;
                color: #5bbfdf;
                border: 1px solid #2a5a7a;
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#view_chip_tool {{
                background: {CARD};
                color: {T2};
                border: 1px solid {T3};
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 12px;
            }}

            /* ── Status bar info segments ───────────────── */
            QLabel#sb_info {{
                color: {T2};
                font-size: 12px;
                padding: 0 6px;
                background: transparent;
            }}
            QFrame#sb_sep {{
                background: {T3};
                max-width: 1px;
                margin-top: 4px;
                margin-bottom: 4px;
            }}

            /* ── Ribbon ────────────────────────────────── */
            QTabWidget#ribbon {{
                background: {PANEL};
            }}
            QTabWidget#ribbon::pane {{
                background: {PANEL};
                border: none;
                border-bottom: 1px solid {T3};
            }}
            QTabWidget#ribbon > QTabBar {{
                background: {BG};
            }}
            QTabWidget#ribbon > QTabBar::tab {{
                background: {BG};
                color: {T2};
                min-width: 72px;
                height: 26px;
                padding: 2px 14px;
                border: none;
                font-size: 13px;
                font-weight: 500;
            }}
            QTabWidget#ribbon > QTabBar::tab:selected {{
                background: {PANEL};
                color: {T1};
                border-bottom: 2px solid {ACCENT};
            }}
            QTabWidget#ribbon > QTabBar::tab:hover:!selected {{
                background: {CARD};
                color: {T1};
            }}
            QWidget#ribbon_tab {{
                background: {PANEL};
            }}
            QFrame#ribbon_sep {{
                background: {T3};
                max-width: 1px;
                margin-top: 6px;
                margin-bottom: 6px;
            }}

            /* ── Dock widget ───────────────────────────── */
            QDockWidget {{
                font-size: 13px;
                color: {T2};
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
            }}
            QDockWidget::title {{
                background: #141720;
                padding: 6px 10px;
                font-size: 13px;
                font-weight: 600;
                color: {T1};
                border-bottom: 1px solid {T3};
                text-align: left;
            }}
            QDockWidget > QWidget {{
                background: {PANEL};
            }}

            /* ── Group boxes in dock ───────────────────── */
            QGroupBox#dock_group {{
                background: {CARD};
                border: 1px solid {T3};
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 4px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            QGroupBox#dock_group::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                top: 4px;
                color: {T2};
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
                background: transparent;
            }}

            /* ── Console ───────────────────────────────── */
            QPlainTextEdit#console {{
                background: #0e1014;
                color: {T2};
                font-family: {MONO};
                font-size: 13px;
                border: none;
                selection-background-color: {ACCENT};
                selection-color: #0e1014;
            }}

            /* ── Status bar ────────────────────────────── */
            QStatusBar#main_statusbar {{
                background: #0e1014;
                border-top: 1px solid {T3};
                font-size: 13px;
                color: {T2};
            }}
            QStatusBar#main_statusbar QLabel {{
                background: transparent;
                color: {T2};
                font-size: 13px;
                padding: 0 8px;
            }}
            QStatusBar#main_statusbar QLabel#status {{
                color: {ACCENT};
                font-size: 13px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0 8px;
            }}
            QStatusBar#main_statusbar QLabel#collision_status {{
                color: {T2};
                font-size: 12px;
                background: transparent;
                border: none;
            }}
            QStatusBar#main_statusbar QProgressBar {{
                background: {FIELD};
                border: 1px solid {T3};
                border-radius: 3px;
            }}

            /* Light CAM theme overrides */
            QMainWindow, QWidget {{
                background: {BG};
                color: {T1};
                font-family: {SANS};
                font-size: 13px;
            }}
            QFrame#vp_frame {{
                background: #eef3f8;
            }}
            QWidget#content,
            QScrollArea#control_scroll,
            QScrollArea#control_scroll > QWidget > QWidget,
            QDockWidget > QWidget {{
                background: {PANEL};
            }}
            QScrollBar:vertical {{
                background: #f1f3f6;
                width: 9px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #b7c1cf;
                border-radius: 4px;
                min-height: 28px;
            }}

            QFrame#ribbon_bar {{
                background: #ffffff;
                border-bottom: 1px solid #cfd6df;
            }}
            QFrame#ribbon_titlebar {{
                background: #ffffff;
                border-bottom: 1px solid #d9dee7;
            }}
            QFrame#ribbon_logo {{
                background: #ffffff;
                border-right: 1px solid #d9dee7;
            }}
            QWidget#brand_logo_mark {{
                background: transparent;
            }}
            QLabel#ribbon_app_name {{
                color: #1f2933;
                font-size: 17px;
                font-weight: 800;
            }}
            QLabel#ribbon_app_sub {{
                color: #6b7280;
                font-size: 11px;
                font-weight: 500;
            }}
            QLabel#ribbon_project {{
                color: #4b5563;
                font-size: 12px;
                font-weight: 600;
                padding-left: 12px;
                border-left: 1px solid #d9dee7;
            }}
            QTabWidget#ribbon,
            QTabWidget#ribbon::pane,
            QWidget#ribbon_tab {{
                background: #ffffff;
            }}
            QTabWidget#ribbon::pane {{
                border: none;
                border-bottom: 1px solid #d7dde6;
            }}
            QTabWidget#ribbon > QTabBar {{
                background: #ffffff;
            }}
            QTabWidget#ribbon > QTabBar::tab {{
                background: #ffffff;
                color: #1f2933;
                min-width: 76px;
                height: 26px;
                padding: 2px 14px;
                border: none;
                font-size: 13px;
                font-weight: 600;
            }}
            QTabWidget#ribbon > QTabBar::tab:selected {{
                color: {ACCENT};
                border-bottom: 3px solid {ACCENT};
            }}
            QTabWidget#ribbon > QTabBar::tab:hover:!selected {{
                background: #edf4ff;
                color: {ACCENT};
            }}
            QFrame#ribbon_sep {{
                background: #d9dee7;
                max-width: 1px;
                margin-top: 5px;
                margin-bottom: 5px;
            }}
            QFrame#ribbon_right {{
                background: #ffffff;
            }}
            QPushButton#ribbon_right_btn {{
                background: transparent;
                border: none;
                color: #374151;
                font-size: 12px;
                padding: 2px 6px;
                min-height: 22px;
            }}
            QPushButton#ribbon_right_btn:hover {{
                background: #eef4fb;
                color: {ACCENT};
                border-radius: 3px;
            }}

            QFrame#view_header {{
                background: #ffffff;
                border-bottom: 1px solid #d7dde6;
            }}
            QLabel#view_title {{
                color: #1f2933;
                font-size: 14px;
                font-weight: 700;
            }}
            QLabel#view_hint {{
                color: #697586;
                font-size: 12px;
            }}
            QLabel#view_chip_mode {{
                background: #e8f2ff;
                color: {ACCENT};
                border: 1px solid #b7d4f5;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#view_chip_tool {{
                background: #f6f8fb;
                color: #4b5563;
                border: 1px solid #d7dde6;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 12px;
            }}

            QDockWidget {{
                color: #1f2933;
                font-size: 13px;
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
            }}
            QDockWidget::title {{
                background: #ffffff;
                color: #111827;
                border-bottom: 1px solid #d7dde6;
                padding: 7px 10px;
                font-size: 13px;
                font-weight: 700;
                text-align: left;
            }}
            QGroupBox#dock_group {{
                background: #ffffff;
                border: 1px solid #d7dde6;
                border-radius: 4px;
                margin-top: 20px;
                padding-top: 4px;
                font-size: 12px;
                font-weight: 700;
            }}
            QGroupBox#dock_group::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 4px;
                color: #111827;
                background: #ffffff;
                font-size: 12px;
                font-weight: 700;
                padding: 0 4px;
            }}
            QWidget#operation_panel {{
                background: #ffffff;
            }}
            QFrame#operation_header {{
                background: #ffffff;
                border-bottom: 1px solid #d7dde6;
            }}
            QLabel#operation_title {{
                color: #111827;
                font-size: 13px;
                font-weight: 700;
            }}
            QLabel#operation_state {{
                color: {SUCCESS};
                font-size: 12px;
                font-weight: 600;
            }}
            QTabWidget#operation_tabs {{
                background: #ffffff;
            }}
            QTabWidget#operation_tabs::pane {{
                background: #ffffff;
                border: none;
                border-top: 1px solid #d7dde6;
            }}
            QTabWidget#operation_tabs > QTabBar {{
                background: #ffffff;
            }}
            QTabWidget#operation_tabs > QTabBar::tab {{
                background: #ffffff;
                color: #374151;
                height: 28px;
                min-width: 72px;
                padding: 0 8px;
                border: none;
                font-size: 12px;
                font-weight: 600;
            }}
            QTabWidget#operation_tabs > QTabBar::tab:selected {{
                color: {ACCENT};
                border-bottom: 2px solid {ACCENT};
            }}
            QTabWidget#operation_tabs > QTabBar::tab:hover:!selected {{
                background: #edf4ff;
                color: {ACCENT};
            }}

            QLabel,
            QLabel#field_label,
            QLabel#muted {{
                background: transparent;
                color: #4b5563;
                font-size: 12px;
            }}
            QLabel#field_label {{
                color: #374151;
                font-weight: 600;
            }}
            QLabel#filepath {{
                background: #f9fafb;
                color: #4b5563;
                border: 1px solid #d1d7e0;
                border-radius: 3px;
                padding: 6px 8px;
                font-size: 12px;
            }}

            QComboBox, QDoubleSpinBox {{
                background: #ffffff;
                border: 1px solid #cbd3df;
                border-radius: 3px;
                color: #1f2933;
                padding: 4px 8px;
                min-height: 28px;
                font-size: 12px;
            }}
            QComboBox:hover, QDoubleSpinBox:hover,
            QComboBox:focus, QDoubleSpinBox:focus {{
                border-color: {ACCENT};
                background: #ffffff;
            }}
            QComboBox QAbstractItemView {{
                background: #ffffff;
                border: 1px solid #cbd3df;
                selection-background-color: #dbeafe;
                selection-color: #111827;
                color: #111827;
            }}

            QRadioButton, QCheckBox#switch {{
                color: #374151;
                background: transparent;
                font-size: 12px;
                spacing: 8px;
                min-height: 22px;
            }}
            QRadioButton::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid #9aa7b8;
                background: #ffffff;
            }}
            QRadioButton::indicator:checked {{
                border: 4px solid {ACCENT};
                background: #ffffff;
            }}
            QCheckBox#switch::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 2px;
                border: 1px solid #9aa7b8;
                background: #ffffff;
            }}
            QCheckBox#switch::indicator:checked {{
                border: 1px solid {ACCENT};
                background: {ACCENT};
            }}

            QPushButton {{
                background: #ffffff;
                color: #1f2933;
                border: 1px solid #cbd3df;
                border-radius: 3px;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 10px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background: #eef6ff;
                border-color: {ACCENT};
                color: {ACCENT};
            }}
            QPushButton:pressed {{
                background: #dbeafe;
            }}
            QPushButton:disabled {{
                background: #f3f5f8;
                color: #9aa4b2;
                border-color: #d7dde6;
            }}
            QPushButton#browse_btn {{
                background: #f9fafb;
                color: #1f2933;
                border: 1px solid #cbd3df;
                border-radius: 3px;
                font-size: 12px;
            }}
            QPushButton#btn_start {{
                background: {ACCENT};
                color: #ffffff;
                border: 1px solid #0f66bd;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#btn_start:hover {{
                background: #0f66bd;
                color: #ffffff;
            }}
            QPushButton#btn_start:disabled {{
                background: #b9d7f6;
                color: #ffffff;
                border: 1px solid #b9d7f6;
            }}
            QPushButton#btn_stop {{
                background: #ffffff;
                color: {DANGER};
                border: 1px solid #ef9ca5;
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton#btn_stop:hover {{
                background: #fff0f2;
                color: {DANGER};
            }}
            QPushButton#btn_stop:disabled {{
                background: #f3f5f8;
                color: #c7ced8;
                border: 1px solid #d7dde6;
            }}
            QPushButton#btn_reset {{
                color: #1f2933;
            }}

            QProgressBar {{
                background: #e5e9ef;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 3px;
            }}
            QSlider::groove:horizontal {{
                background: #d7dde6;
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 2px;
            }}

            QPlainTextEdit#console {{
                background: #ffffff;
                color: #374151;
                font-family: {MONO};
                font-size: 12px;
                border-top: 1px solid #d7dde6;
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }}
            QStatusBar#main_statusbar {{
                background: #ffffff;
                border-top: 1px solid #d7dde6;
                color: #4b5563;
                font-size: 12px;
            }}
            QStatusBar#main_statusbar QLabel {{
                background: transparent;
                color: #4b5563;
                font-size: 12px;
                padding: 0 8px;
            }}
            QStatusBar#main_statusbar QLabel#status {{
                color: #1f2933;
                font-weight: 700;
                border: none;
            }}
            QStatusBar#main_statusbar QLabel#collision_status {{
                color: #4b5563;
                border: none;
            }}
            QStatusBar#main_statusbar QProgressBar {{
                background: #e5e9ef;
                border: 1px solid #d7dde6;
                border-radius: 3px;
            }}

            /* IMACT brand tint overrides */
            QMainWindow, QWidget {{
                background: {BG};
                color: {T1};
            }}
            QFrame#vp_frame {{
                background: #dfe7f2;
            }}
            QWidget#content,
            QScrollArea#control_scroll,
            QScrollArea#control_scroll > QWidget > QWidget,
            QDockWidget > QWidget {{
                background: #edf2f8;
            }}
            QScrollBar:vertical {{
                background: #dce4ef;
                width: 9px;
            }}
            QScrollBar::handle:vertical {{
                background: #8fa0b6;
                border-radius: 4px;
            }}

            QFrame#ribbon_bar {{
                background: #eef3f9;
                border-bottom: 1px solid #aab7c8;
            }}
            QFrame#ribbon_titlebar {{
                background: #17243e;
                border-bottom: 2px solid #f2c84b;
            }}
            QWidget#ribbon_title_text {{
                background: #17243e;
            }}
            QFrame#ribbon_titlebar QLabel {{
                background: transparent;
                padding: 0;
            }}
            QLabel#ribbon_app_name {{
                color: #ffffff;
                font-size: 17px;
                font-weight: 800;
            }}
            QLabel#ribbon_app_sub {{
                color: #cbd6e5;
            }}
            QLabel#ribbon_project {{
                background: #17243e;
                color: #ffffff;
                border-left: 1px solid #3d4d68;
            }}
            QFrame#ribbon_logo,
            QFrame#ribbon_right,
            QTabWidget#ribbon,
            QTabWidget#ribbon::pane,
            QTabWidget#ribbon > QTabBar,
            QWidget#ribbon_tab {{
                background: #eef3f9;
            }}
            QFrame#ribbon_logo {{
                border-right: 1px solid #c5cfdd;
            }}
            QTabWidget#ribbon::pane {{
                border: none;
                border-bottom: 1px solid #c5cfdd;
            }}
            QTabWidget#ribbon > QTabBar::tab {{
                background: #eef3f9;
                color: #17243e;
            }}
            QTabWidget#ribbon > QTabBar::tab:selected {{
                color: {ACCENT};
                border-bottom: 3px solid #f2c84b;
            }}
            QTabWidget#ribbon > QTabBar::tab:hover:!selected {{
                background: #dfe9f6;
                color: {ACCENT};
            }}
            QFrame#ribbon_sep {{
                background: #aab7c8;
            }}
            QPushButton#ribbon_right_btn {{
                color: #d8e1ee;
            }}
            QPushButton#ribbon_right_btn:hover {{
                background: #223352;
                color: #f2c84b;
            }}

            QFrame#view_header {{
                background: #f1f5fa;
                border-bottom: 1px solid #c5cfdd;
            }}
            QLabel#view_title {{
                color: #17243e;
            }}
            QLabel#view_chip_mode {{
                background: #dfeafb;
                color: {ACCENT};
                border: 1px solid #9fb8dc;
            }}
            QLabel#view_chip_tool {{
                background: #e9eff7;
                border: 1px solid #c5cfdd;
            }}

            QDockWidget::title {{
                background: #dfe7f2;
                color: #17243e;
                border-bottom: 1px solid #c5cfdd;
            }}
            QGroupBox#dock_group {{
                background: #f5f8fc;
                border: 1px solid #c5cfdd;
            }}
            QGroupBox#dock_group::title {{
                background: #f5f8fc;
                color: #17243e;
            }}
            QWidget#operation_panel,
            QTabWidget#operation_tabs,
            QTabWidget#operation_tabs::pane,
            QTabWidget#operation_tabs > QTabBar {{
                background: #eef3f9;
            }}
            QFrame#operation_header {{
                background: #dfe7f2;
                border-bottom: 1px solid #c5cfdd;
            }}
            QTabWidget#operation_tabs > QTabBar::tab {{
                background: #eef3f9;
            }}
            QTabWidget#operation_tabs > QTabBar::tab:selected {{
                color: {ACCENT};
                border-bottom: 2px solid #f2c84b;
            }}

            QLabel#filepath,
            QComboBox,
            QDoubleSpinBox {{
                background: #fbfcff;
                border: 1px solid #bdc8d8;
            }}
            QPushButton,
            QPushButton#browse_btn {{
                background: #f6f9fd;
                border: 1px solid #bdc8d8;
                color: #17243e;
            }}
            QPushButton:hover,
            QPushButton#browse_btn:hover {{
                background: #e5effb;
                border-color: {ACCENT};
                color: {ACCENT};
            }}
            QPushButton#btn_start {{
                background: {ACCENT};
                color: #ffffff;
                border: 1px solid #164e91;
            }}
            QPushButton#btn_start:hover {{
                background: #164e91;
                color: #ffffff;
            }}
            QPushButton#btn_reset {{
                color: #17243e;
            }}
            QCheckBox#switch::indicator:checked {{
                background: {ACCENT};
                border: 1px solid {ACCENT};
            }}
            QRadioButton::indicator:checked {{
                border: 4px solid {ACCENT};
            }}
            QProgressBar::chunk,
            QSlider::handle:horizontal,
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
            }}

            QPlainTextEdit#console {{
                background: #eef3f9;
                color: #17243e;
                border-top: 1px solid #c5cfdd;
            }}
            QStatusBar#main_statusbar {{
                background: #17243e;
                border-top: 2px solid #f2c84b;
                color: #d8e1ee;
            }}
            QStatusBar#main_statusbar QLabel {{
                color: #d8e1ee;
            }}
            QStatusBar#main_statusbar QLabel#status {{
                color: #f2c84b;
            }}
            QStatusBar#main_statusbar QLabel#collision_status {{
                color: #d8e1ee;
            }}
            QStatusBar#main_statusbar QProgressBar {{
                background: #2a3853;
                border: 1px solid #435575;
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

    def _clear_gcode_overlay(self):
        for actor in self._gcode_actors:
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self._gcode_actors = []

    def _add_gcode_overlay(self):
        """Add G0/G1 toolpath lines to the current plotter scene."""
        self._clear_gcode_overlay()
        moves = self._display_gcode_moves or self._gcode_moves
        if not moves:
            return
        visible = self.chk_gcode_overlay.isChecked()
        rapid_mesh, feed_mesh = self._gcode_lines_to_meshes(moves)
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
        moves = self._display_gcode_moves or self._gcode_moves
        if not moves:
            return None
        x_min, x_max, y_min, y_max, _, z_max = bounds
        radius = float(self.sp_radius.value())
        cell = max(float(self.sp_res.value()), 2.0)
        xs = np.arange(x_min, x_max + 1e-9, cell)
        ys = np.arange(y_min, y_max + 1e-9, cell)
        if len(xs) < 2 or len(ys) < 2:
            return None

        covered = np.zeros((len(xs), len(ys)), dtype=bool)
        feed_moves = [m for m in moves if not m.rapid]
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
        self.plotter.set_background(BG, top="#f5f8fc")

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
            self.plotter.set_background(BG, top="#f5f8fc")
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
        self.plotter.set_background(BG, top="#f5f8fc")
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

        x_full = s.z_grid.row_centers
        y_full = s.z_grid.col_centers
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
        """Conservative live-preview min-pool downsample (fully vectorized).

        Each displayed point uses the minimum Z in its source block so that
        narrow tool marks are never hidden by point-sampling.
        """
        ei = self._display_i_edges
        ej = self._display_j_edges
        # Replace NaN with +inf so np.minimum.reduceat ignores empty cells.
        data = np.where(np.isnan(hmap), np.inf, hmap)
        # Min-reduce along the i (row) axis, then the j (col) axis.
        temp   = np.minimum.reduceat(data, ei[:-1], axis=0)   # (n_i, ny)
        pooled = np.minimum.reduceat(temp, ej[:-1], axis=1)   # (n_i, n_j)
        pooled[np.isinf(pooled)] = np.nan
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
        return self._build_preview_tool_mesh(
            radius=r,
            ball=isinstance(tool, BallEndMill),
        )

    def _build_preview_tool_mesh(self, radius: float, ball: bool = True) -> pv.PolyData:
        """Create a cutter-shaped visual mesh in local coordinates with tip at origin."""
        r = float(radius)
        sectors = 48
        height = max(4.0 * r, r + 8.0)

        if ball:
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

    def _five_axis_moves_from_file(self, path: str) -> list:
        program = self._load_canonical_gcode(path)
        moves = gcode_moves_from_canonical_moves(program.moves)
        return [
            move
            for move in moves
            if move.a is not None or move.b is not None or move.c is not None
        ]

    @staticmethod
    def _resolve_default_demo_path(path: str, filename: str) -> str:
        if os.path.exists(path):
            return path
        root = os.path.join(os.path.expanduser("~"), "Downloads")
        for dirpath, _dirnames, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return path

    def _five_axis_path_meshes(self, moves: list) -> tuple[pv.PolyData | None, pv.PolyData | None]:
        rapid_pts: list = []
        feed_pts: list = []
        prev = None
        for move in moves:
            cur = (float(move.x), float(move.y), float(move.z))
            if prev is not None:
                target = rapid_pts if move.rapid else feed_pts
                target.extend([prev, cur])
            prev = cur

        def build(points: list) -> pv.PolyData | None:
            if not points:
                return None
            pts = np.asarray(points, dtype=float)
            n = len(pts) // 2
            cells = np.empty(n * 3, dtype=np.intp)
            cells[0::3] = 2
            cells[1::3] = np.arange(n) * 2
            cells[2::3] = np.arange(n) * 2 + 1
            mesh = pv.PolyData()
            mesh.points = pts
            mesh.lines = cells
            return mesh

        return build(rapid_pts), build(feed_pts)

    def _load_five_axis_demo(self) -> None:
        self._pause_five_axis_preview()
        stock_path = self._resolve_default_demo_path(FIVE_AXIS_DEMO_STOCK_PATH, "粗完S.stl")
        gcode_path = self._resolve_default_demo_path(FIVE_AXIS_DEMO_GCODE_PATH, "01_TRAORI_TEST.MPF")
        if not os.path.exists(stock_path):
            QMessageBox.warning(self, "5-Axis Preview", f"Stock file not found:\n{stock_path}")
            return
        if not os.path.exists(gcode_path):
            QMessageBox.warning(self, "5-Axis Preview", f"G-code file not found:\n{gcode_path}")
            return

        from src.stock.stl_importer import load_mesh

        mesh = load_mesh(stock_path)
        moves = self._five_axis_moves_from_file(gcode_path)
        poses = [tool_pose_from_gcode_move(move) for move in moves]
        if len(poses) < 2:
            QMessageBox.warning(self, "5-Axis Preview", "No TRAORI rotary moves found.")
            return

        self._stock_file = stock_path
        self._gcode_file = gcode_path
        self.rb_file.setChecked(True)
        self.rb_gcode.setChecked(True)
        self.cb_tool.setCurrentIndex(0)
        self.sp_radius.setValue(FIVE_AXIS_DEMO_RADIUS)
        self._style_selected_file_label(self.lbl_stock_path, stock_path)
        self._style_selected_file_label(self.lbl_gcode_path, gcode_path)
        self._canonical_program = self._load_canonical_gcode(gcode_path)
        self._gcode_moves = gcode_moves_from_canonical_moves(self._canonical_program.moves)
        self._display_gcode_moves = self._gcode_moves

        self._fa_preview_moves = moves
        self._fa_preview_poses = poses
        self._fa_preview_index = 0
        self.plotter.clear()
        self._gcode_actors = []
        self._coverage_actors = []
        self._fa_preview_path_actors = []
        self._fa_preview_axis_actor = None
        self._fa_preview_feed_actor = None
        self.plotter.set_background(BG, top="#f5f8fc")

        self.plotter.add_mesh(
            mesh,
            color=SURF,
            opacity=0.42,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.78,
        )
        b = mesh.bounds
        bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]))
        self._last_scene_bounds = bounds
        self.plotter.add_mesh(
            pv.Box(bounds=bounds),
            style="wireframe",
            color=BOX_C,
            opacity=0.32,
            line_width=1.0,
        )

        rapid_mesh, feed_mesh = self._five_axis_path_meshes(moves)
        if rapid_mesh is not None:
            self._fa_preview_path_actors.append(
                self.plotter.add_mesh(
                    rapid_mesh,
                    color=G0_C,
                    line_width=2.2,
                    opacity=0.62,
                    render_lines_as_tubes=True,
                    lighting=False,
                )
            )
        if feed_mesh is not None:
            self._fa_preview_path_actors.append(
                self.plotter.add_mesh(
                    feed_mesh,
                    color=G1_C,
                    line_width=3.2,
                    opacity=0.95,
                    render_lines_as_tubes=True,
                    lighting=False,
                )
            )

        tool_mesh = self._build_preview_tool_mesh(FIVE_AXIS_DEMO_RADIUS, ball=True)
        self._fa_preview_tool_actor = self.plotter.add_mesh(
            tool_mesh,
            color=TOOL_C,
            opacity=0.95,
            smooth_shading=True,
            specular=0.7,
        )
        self.plotter.show_axes()
        self._set_default_camera(*bounds)
        self._render_five_axis_preview_frame(0)
        self._set_status(
            f"5-axis preview ready: {len(moves)} TRAORI moves. Press Play 5-Axis.",
            SUCCESS,
        )

    def _render_five_axis_preview_frame(self, index: int) -> None:
        if not self._fa_preview_poses or self._fa_preview_tool_actor is None:
            return
        index = max(0, min(int(index), len(self._fa_preview_poses) - 1))
        pose = self._fa_preview_poses[index]
        move = self._fa_preview_moves[index]
        pos = tuple(float(v) for v in pose.position)
        axis = tuple(float(v) for v in pose.axis)
        self._apply_tool_pose_to_actor(self._fa_preview_tool_actor, pos, axis)

        for actor in (self._fa_preview_axis_actor, self._fa_preview_feed_actor):
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        length = max(FIVE_AXIS_DEMO_RADIUS * 5.0, 28.0)
        self._fa_preview_axis_actor = self.plotter.add_mesh(
            pv.Arrow(start=pos, direction=axis, scale=length),
            color=G0_C,
            opacity=0.95,
        )
        if index > 0:
            prev = self._fa_preview_poses[index - 1].position
        else:
            prev = self._fa_preview_poses[min(1, len(self._fa_preview_poses) - 1)].position
        feed = pose.position - prev
        feed_norm = float(np.linalg.norm(feed))
        if feed_norm > 1e-9:
            self._fa_preview_feed_actor = self.plotter.add_mesh(
                pv.Arrow(start=pos, direction=tuple(feed / feed_norm), scale=length * 0.65),
                color=SUCCESS,
                opacity=0.9,
            )
        else:
            self._fa_preview_feed_actor = None

        self.lbl_seg.setText(
            f"5-axis {index + 1} / {len(self._fa_preview_poses)} | "
            f"L{move.line_no} {move.motion_type} | "
            f"A{(move.a or 0.0):.3f} C{(move.c or 0.0):.3f}"
        )
        self._set_status(
            f"Tool axis=({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})",
            WARNING,
        )
        self.plotter.render()

    def _play_five_axis_preview(self) -> None:
        if not self._fa_preview_poses:
            self._load_five_axis_demo()
        if not self._fa_preview_poses:
            return
        if self._fa_preview_index >= len(self._fa_preview_poses) - 1:
            self._fa_preview_index = 0
        self._fa_preview_timer.start()

    def _pause_five_axis_preview(self) -> None:
        if hasattr(self, "_fa_preview_timer"):
            self._fa_preview_timer.stop()

    def _stop_five_axis_preview(self) -> None:
        self._pause_five_axis_preview()
        self._fa_preview_index = 0
        if self._fa_preview_poses:
            self._render_five_axis_preview_frame(0)

    def _five_axis_preview_tick(self) -> None:
        if not self._fa_preview_poses:
            self._fa_preview_timer.stop()
            return
        self._render_five_axis_preview_frame(self._fa_preview_index)
        self._fa_preview_index += 1
        if self._fa_preview_index >= len(self._fa_preview_poses):
            self._fa_preview_timer.stop()
            self._fa_preview_index = len(self._fa_preview_poses) - 1

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
        M = np.eye(4, dtype=float)
        M[:3, 0] = x
        M[:3, 1] = y
        M[:3, 2] = z
        M[:3, 3] = np.asarray(pos, dtype=float)
        # Keep rotation and translation in one matrix. Mixing user_matrix with
        # actor.position can apply transforms in an unexpected order in VTK.
        actor.position = (0.0, 0.0, 0.0)
        actor.user_matrix = M

    @pyqtSlot(int, int)
    def _on_progress(self, cur: int, total: int):
        pct = cur * 100 // total
        self.progress_bar.setValue(pct)
        self.lbl_seg.setText(f"{cur} / {total} segments")

    def _mesh_for_display(
        self,
        mesh: "pv.PolyData",
        quality_idx: int | None = None,
    ) -> "pv.PolyData":
        """Return a lighter mesh for interactive display.

        The full-resolution mesh stays in `_result_mesh` for export.  The
        viewport actor can use a reduced mesh so orbit and pan remain responsive
        after large marching-cubes reconstructions.
        """
        quality = self._display_quality_index() if quality_idx is None else int(quality_idx)
        if quality == 2:
            return mesh

        n_cells = int(mesh.n_cells)
        target_cells = 250_000
        if n_cells <= target_cells:
            return mesh

        reduction = min(0.88, max(0.0, 1.0 - target_cells / max(1, n_cells)))
        try:
            display = mesh.decimate_pro(
                reduction,
                preserve_topology=True,
                feature_angle=35.0,
            )
            return display.clean()
        except Exception as exc:
            self.log(f"Display decimation skipped: {exc}")
            return mesh

    def _mesh_with_normals(
        self,
        mesh: "pv.PolyData",
        *,
        split_vertices: bool,
        point_normals: bool = True,
        cell_normals: bool = True,
    ) -> "pv.PolyData":
        """Return a cleaned triangle mesh with stable normals for shading/export."""
        prepared = mesh.triangulate().clean()
        try:
            prepared = prepared.compute_normals(
                point_normals=point_normals,
                cell_normals=cell_normals,
                split_vertices=split_vertices,
                consistent_normals=True,
                auto_orient_normals=True,
                feature_angle=38.0,
                inplace=False,
            )
        except Exception as exc:
            self.log(f"Normal rebuild skipped: {exc}")
        return prepared

    def _remove_feature_edges(self) -> None:
        if self._feature_edge_actor is not None:
            try:
                self.plotter.remove_actor(self._feature_edge_actor)
            except Exception:
                pass
            self._feature_edge_actor = None

    def _add_feature_edge_overlay(self, mesh: "pv.PolyData") -> None:
        """Overlay only visual feature edges, not every triangle edge."""
        self._remove_feature_edges()
        try:
            edges = mesh.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False,
                feature_angle=42.0,
            ).clean()
        except Exception as exc:
            self.log(f"Feature edge overlay skipped: {exc}")
            return

        if edges.n_cells <= 0:
            return
        max_edge_cells = 80_000
        if edges.n_cells > max_edge_cells:
            self.log(f"Feature edge overlay skipped: {edges.n_cells:,} line cells")
            return

        self._feature_edge_actor = self.plotter.add_mesh(
            edges,
            color="#172033",
            line_width=1.0,
            opacity=0.42,
            render_lines_as_tubes=False,
            pickable=False,
        )

    def _finish_mesh_ready_ui(self) -> None:
        self.lbl_seg.setText("Idle")
        self.btn_start.setEnabled(True)
        self.btn_export.setEnabled(self._result_mesh is not None)
        self._update_collision_status()
        collision_count = len(getattr(self._engine, "collision_events", []))
        profile = self._last_sim_profile or {}
        mesh_s = float(getattr(self._mesh_worker, "elapsed_s", 0.0) or 0.0)
        perf_note = ""
        if profile:
            perf_note = (
                f" | sim {profile.get('sim_s', 0.0):.2f}s"
                f", cut {profile.get('cut_s', 0.0):.2f}s"
                f", mesh {mesh_s:.2f}s"
                f", hmap {int(profile.get('hmap_count', 0))}x"
            )
        if collision_count:
            self._set_status(
                f"Complete with {collision_count} collision samples{perf_note}",
                DANGER,
            )
        else:
            self._set_status(f"Complete{perf_note}", SUCCESS)

    def _show_final_height_surface(self) -> None:
        if self._stock is None:
            return

        hmap = self._stock.z_grid.height_map()
        x_min, x_max, y_min, y_max, z_min, z_max = self._stock.bounds
        surface = height_map_to_surface(hmap, x_min, x_max, y_min, y_max)
        cut_depth = np.maximum(0.0, z_max - surface.points[:, 2])
        surface["cut_depth"] = cut_depth

        if self._surface_actor is not None:
            self.plotter.remove_actor(self._surface_actor)
            self._surface_actor = None
        self._remove_feature_edges()
        if self._tool_actor is not None:
            self.plotter.remove_actor(self._tool_actor)
            self._tool_actor = None

        self._surface = surface
        try:
            self._display_mesh = surface.extract_surface().triangulate().clean()
        except Exception:
            self._display_mesh = None

        self._surface_actor = self.plotter.add_mesh(
            surface,
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
            smooth_shading=True,
            show_edges=False,
            ambient=0.34,
            diffuse=0.80,
            specular=0.18,
        )
        self._add_gcode_overlay()
        self._add_coverage_overlay()
        n_cells = int(getattr(self._display_mesh, "n_cells", surface.n_cells))
        self._set_data_state(display=f"Ready: Fast Surface ({n_cells:,} cells)")

    def _show_result_mesh(self, mesh: "pv.PolyData") -> None:
        if self._stock is None:
            return

        quality = self._last_display_quality_idx
        display_mesh = self._mesh_for_display(mesh, quality)
        if self._surface_actor is not None:
            self.plotter.remove_actor(self._surface_actor)
            self._surface_actor = None
        self._remove_feature_edges()
        if self._tool_actor is not None:
            self.plotter.remove_actor(self._tool_actor)
            self._tool_actor = None

        z_max = self._stock.z_max
        z_min = self._stock.z_min
        display_mesh = self._mesh_with_normals(display_mesh, split_vertices=True)
        cut_depth = np.maximum(0.0, z_max - display_mesh.points[:, 2])
        display_mesh["cut_depth"] = cut_depth
        self._display_mesh = display_mesh

        self._surface_actor = self.plotter.add_mesh(
            display_mesh,
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
            smooth_shading=True,
            show_edges=False,
            ambient=0.24,
            diffuse=0.88,
            specular=0.34,
            specular_power=18,
        )
        self._add_feature_edge_overlay(display_mesh)
        quality_name = self._display_quality_label(quality)
        self._set_data_state(
            display=f"Ready: {quality_name} ({int(display_mesh.n_cells):,} cells)"
        )
        self._add_gcode_overlay()
        self._add_coverage_overlay()

    @pyqtSlot()
    def _on_finished(self):
        self._render_timer.stop()
        self._pending_hmap = None
        self._pending_frame_dirty = False
        self._last_sim_profile = dict(getattr(self._worker, "profile", {}) or {})
        if self._last_sim_profile:
            print(
                "[Profile] "
                f"sim={self._last_sim_profile.get('sim_s', 0.0):.3f}s "
                f"cut={self._last_sim_profile.get('cut_s', 0.0):.3f}s "
                f"hmap={self._last_sim_profile.get('hmap_s', 0.0):.3f}s/"
                f"{int(self._last_sim_profile.get('hmap_count', 0))} "
                f"frames={int(self._last_sim_profile.get('frame_count', 0))} "
                f"samples={int(self._last_sim_profile.get('cut_samples', 0))}"
            )

        self._last_display_quality_idx = self._display_quality_index()
        use_fast_surface_display = self._last_display_quality_idx == 0

        if use_fast_surface_display:
            self._show_final_height_surface()
            self.plotter.render()
        elif self._stock is not None and self._surface is not None:
            self._apply_hmap_to_surface(self._stock.z_grid.height_map())
            self.plotter.render()

        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        if use_fast_surface_display:
            self.lbl_seg.setText("Building export mesh...")
            self._set_data_state(export="Building full export mesh...")
            self._set_status("Display ready, building export mesh", WARNING)
        else:
            self.lbl_seg.setText("Rebuilding smooth mesh...")
            self._set_data_state(
                display=f"Building {self._display_quality_label()}...",
                export="Building full export mesh...",
            )
            self._set_status("Building display/export mesh", WARNING)

        self._mesh_worker_display_result = not use_fast_surface_display
        self._mesh_worker = MeshWorker(self._stock)
        self._mesh_worker.mesh_ready.connect(self._on_mesh_ready)
        self._mesh_worker.error.connect(self._on_mesh_error)
        self._mesh_worker.start()

    @pyqtSlot(object)
    def _on_mesh_ready(self, mesh: "pv.PolyData"):
        self._result_mesh = mesh.copy(deep=True)
        self._set_data_state(
            export=f"Ready: Full Mesh ({int(self._result_mesh.n_cells):,} cells)"
        )
        if not self._mesh_worker_display_result:
            self._finish_mesh_ready_ui()
            return

        self._show_result_mesh(mesh)
        self.plotter.render()

        self._finish_mesh_ready_ui()

    @pyqtSlot(str)
    def _on_mesh_error(self, message: str):
        """Called when MeshWorker fails; re-enables UI so the user isn't stuck."""
        self.lbl_seg.setText("Idle")
        self.btn_start.setEnabled(True)
        self._set_data_state(export=f"Mesh error: {message}")
        self._set_status(f"Mesh error: {message}", DANGER)

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
            mesh = self._mesh_with_normals(
                self._result_mesh,
                split_vertices=True,
                point_normals=True,
                cell_normals=True,
            )
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
    def _speed_settings(cls, val: int) -> tuple[int, int, int]:
        delay_ms = cls._speed_delay_ms(val)
        if val >= 10:
            return delay_ms, 80, 120
        if val >= 9:
            return delay_ms, 60, 60
        if val >= 8:
            return delay_ms, 45, 25
        if val >= 6:
            return delay_ms, 33, 8
        return delay_ms, 0, 1

    @staticmethod
    def _speed_multiplier(val: int) -> float:
        multipliers = {
            0: 0.25,
            1: 0.33,
            2: 0.5,
            3: 0.75,
            4: 1.0,
            5: 1.5,
            6: 2.0,
            7: 4.0,
            8: 8.0,
            9: 16.0,
            10: 32.0,
        }
        return multipliers.get(int(val), 1.0)

    @staticmethod
    def _format_multiplier(value: float) -> str:
        if value >= 10:
            return f"{value:.0f}x"
        if value >= 1:
            return f"{value:g}x"
        return f"{value:.2g}x"

    def _speed_label(self, val: int) -> str:
        base = f"{self._format_multiplier(self._speed_multiplier(val))} preview speed"
        if hasattr(self, "chk_final_only") and self.chk_final_only.isChecked():
            return f"{base}, final only"
        return base

    @pyqtSlot(int)
    def _on_speed_changed(self, val: int):
        delay_ms, frame_interval_ms, segment_batch = self._speed_settings(val)
        self.lbl_speed_val.setText(self._speed_label(val))
        self._sync_status_bar()
        if self._worker is not None and self._worker.isRunning():
            self._worker.delay_ms = delay_ms
            self._worker.frame_interval_ms = frame_interval_ms
            self._worker.segment_batch = segment_batch

    def _on_sim_method_changed(self, index: int):
        is_swept = index == 2
        self.chk_collision.setEnabled(is_swept)
        if not is_swept:
            self.chk_collision.setChecked(False)
        self._update_collision_status()

    def _on_display_quality_changed(self, index: int):
        self._last_display_quality_idx = int(index)
        self._sync_status_bar()

        worker_running = self._worker is not None and self._worker.isRunning()
        mesh_running = self._mesh_worker is not None and self._mesh_worker.isRunning()
        if worker_running:
            return

        if mesh_running:
            self._mesh_worker_display_result = self._last_display_quality_idx != 0
            if self._last_display_quality_idx == 0 and self._stock is not None:
                self._show_final_height_surface()
                self.plotter.render()
                self._set_data_state(export="Building full export mesh...")
                self._set_status("Display switched to fast surface; export mesh still building", WARNING)
            else:
                self._set_data_state(
                    display=f"Will show {self._display_quality_label()} when mesh is ready"
                )
                self._set_status("Display mesh will update when background mesh is ready", WARNING)
            return

        if self._stock is None:
            return

        if self._last_display_quality_idx == 0:
            self._show_final_height_surface()
            self.plotter.render()
            return

        if self._result_mesh is not None:
            self._show_result_mesh(self._result_mesh)
            self.plotter.render()
        else:
            self._set_data_state(display=f"{self._display_quality_label()} requires completed mesh")

    def _update_collision_status(self):
        if not hasattr(self, "lbl_collision"):
            return
        events = getattr(self._engine, "collision_events", None)
        if events:
            summary = collision_summary(events)
            detail = ", ".join(
                f"{name} {count}" for name, count in sorted(summary.items())
            )
            self.lbl_collision.setText(f"Collision: {detail}")
            self.lbl_collision.setStyleSheet(
                f"color:{DANGER};font-weight:bold;background:transparent;border:none;"
            )
            return
        if self.chk_collision.isChecked() and self.chk_collision.isEnabled():
            text = "Collision check: No contact"
            color = SUCCESS
        elif self.chk_collision.isEnabled():
            text = "Collision check: Off"
            color = T2
        else:
            text = "Collision check: Swept only"
            color = T2
        self.lbl_collision.setText(text)
        self.lbl_collision.setStyleSheet(
            f"color:{color};background:transparent;border:none;"
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
        label.setToolTip(path)
        label.setWordWrap(False)
        label.setMinimumWidth(0)
        label.setFixedHeight(34)
        label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        label.setStyleSheet(
            f"color:{T1};font-size:12px;background:{FIELD};"
            f"font-family:'Consolas','JetBrains Mono','Courier New',monospace;"
            f"border:1px solid {ACCENT};border-radius:5px;padding:7px 9px;"
        )

    def _load_default_inputs(self) -> None:
        """Default project loading is intentionally disabled."""
        return

    @pyqtSlot()
    def _browse_stock_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Feedstock File", "",
            "Mesh Files (*.stl *.step *.stp *.obj *.ply);;All Files (*)"
        )
        if not path:
            return
        self._stock_file = path
        self._style_selected_file_label(self.lbl_stock_path, path)
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
        self._style_selected_file_label(self.lbl_gcode_path, path)
        try:
            self._canonical_program = self._load_canonical_gcode(path)
            self._gcode_moves = gcode_moves_from_canonical_moves(
                self._canonical_program.moves
            )
            self._display_gcode_moves = None  # will be computed at simulation start
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
            self._display_gcode_moves = None
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

        zero = np.zeros(3, dtype=float)
        if abs(dx) < 1.0 and abs(dy) < 1.0 and abs(dz) < 1.0:
            return toolpath, zero

        offset = np.array([dx, dy, dz], dtype=float)
        print(
            f"[Auto-align] G-code → stock offset:"
            f" dx={dx:.2f}  dy={dy:.2f}  dz={dz:.2f}"
        )

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
        return new_toolpath, offset

    def _shift_gcode_moves(self, moves: list, offset: np.ndarray) -> list:
        """Return a copy of GCodeMove list with XYZ positions shifted by offset."""
        from src.gcode.parser import GCodeMove
        dx, dy, dz = float(offset[0]), float(offset[1]), float(offset[2])
        shifted = []
        for m in moves:
            arc_center = None
            if m.arc_center is not None:
                arc_center = (float(m.arc_center[0]) + dx, float(m.arc_center[1]) + dy)
            shifted.append(GCodeMove(
                x=float(m.x) + dx,
                y=float(m.y) + dy,
                z=float(m.z) + dz,
                feed=m.feed,
                rapid=m.rapid,
                line_no=m.line_no,
                motion_type=m.motion_type,
                source_line=m.source_line,
                segment_index=m.segment_index,
                segment_count=m.segment_count,
                arc_center=arc_center,
                arc_radius=m.arc_radius,
                arc_direction=m.arc_direction,
                plane=m.plane,
                controller=m.controller,
                warnings=m.warnings,
                start_x=float(m.start_x) + dx,
                start_y=float(m.start_y) + dy,
                start_z=float(m.start_z) + dz,
                a=m.a,
                b=m.b,
                c=m.c,
                start_a=m.start_a,
                start_b=m.start_b,
                start_c=m.start_c,
            ))
        return shifted

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
        self._pause_five_axis_preview()
        self.btn_start.setEnabled(False)   # prevent double-click during sync loading
        launched = False
        try:
            res = self.sp_res.value()
            r   = self.sp_radius.value()

            # ── Pre-parse G-code (needed to auto-size stock bounds) ────
            program = None
            if not self._gcode_file:
                self._set_status("Please select a G-code file", WARNING)
                return
            self.rb_gcode.setChecked(True)
            self._set_status("Parsing G-code...", WARNING)
            QApplication.processEvents()
            program = self._load_canonical_gcode(self._gcode_file)
            self._canonical_program = program
            self._gcode_moves = gcode_moves_from_canonical_moves(program.moves)
            self._display_gcode_moves = None  # no shift needed (see below)

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
                # When G-code is loaded, derive stock bounds directly from the
                # G-code cutting footprint so that the stock coordinate system
                # matches the G-code coordinate system with no translation.
                # This avoids any centroid-based mis-alignment for arcs / circles.
                if self._gcode_moves:
                    cut_pts = np.array(
                        [(m.x, m.y, m.z) for m in self._gcode_moves if not m.rapid],
                        dtype=float,
                    )
                    if len(cut_pts) > 0:
                        margin = r + res  # tool radius + one resolution cell
                        bounds = (
                            float(cut_pts[:, 0].min()) - margin,
                            float(cut_pts[:, 0].max()) + margin,
                            float(cut_pts[:, 1].min()) - margin,
                            float(cut_pts[:, 1].max()) + margin,
                            float(cut_pts[:, 2].min()) - margin,
                            float(cut_pts[:, 2].max()),
                        )
                        print(
                            f"[Stock auto-sized from G-code footprint] "
                            f"X=[{bounds[0]:.1f},{bounds[1]:.1f}] "
                            f"Y=[{bounds[2]:.1f},{bounds[3]:.1f}] "
                            f"Z=[{bounds[4]:.1f},{bounds[5]:.1f}]"
                        )
                    else:
                        w = self.sp_bx.value()
                        d = self.sp_by.value()
                        h = self.sp_bz.value()
                        bounds = (0.0, w, 0.0, d, 0.0, h)
                else:
                    w = self.sp_bx.value()
                    d = self.sp_by.value()
                    h = self.sp_bz.value()
                    bounds = (0.0, w, 0.0, d, 0.0, h)

                self._stock = TriDexelStock(bounds, res)
                self._stock.initialize_box_stock()

            # ── Build tool ────────────────────────────────────────────
            cutting_length = self.sp_cutting_length.value()
            overall_length = cutting_length + max(2.0 * r, 1.0)
            tool = BallEndMill(
                r,
                cutting_length=cutting_length,
                overall_length=overall_length,
            ) if self.cb_tool.currentIndex() == 0 else FlatEndMill(
                r,
                cutting_length=cutting_length,
                overall_length=overall_length,
            )
            method_idx = self.cb_sim_method.currentIndex()
            self._last_sim_method_idx = method_idx
            use_swept = method_idx == 2
            update_side_grids = method_idx == 1
            if use_swept:
                self._engine = SweptVolumeSimulationEngine(
                    self._stock,
                    tool,
                    radial_segments=6,
                    axial_segments=2,
                    use_envelope=False,
                    subdivide_moves=False,
                    legacy_z_topdown=True,
                    detect_collision=self.chk_collision.isChecked(),
                    collision_pose_samples=2,
                    collision_n_u=8,
                    collision_n_v=3,
                )
            else:
                self._engine = SimulationEngine(
                    self._stock,
                    tool,
                    update_side_grids=update_side_grids,
                )
            self._update_collision_status()

            # ── Build toolpath ────────────────────────────────────────
            # program & _gcode_moves already set above in pre-parse step.
            has_rotary = any(
                m.a is not None or m.b is not None or m.c is not None
                for m in self._gcode_moves
            )
            if use_swept:
                toolpath = all_pose_segments_from_gcode_moves(self._gcode_moves)
            elif has_rotary:
                toolpath = all_pose_segments_from_gcode_moves(self._gcode_moves)
            else:
                toolpath = self._toolpath_from_canonical(program)
            if not toolpath:
                self._set_status("No cutting moves found in G-code", WARNING)
                return
            # Stock was sized to match G-code coordinates; overlay uses the same moves.
            self._display_gcode_moves = self._gcode_moves

            # ── Launch ────────────────────────────────────────────────
            self._result_mesh = None
            self._display_mesh = None
            self._last_display_quality_idx = self._display_quality_index()
            self._mesh_worker_display_result = True
            self._set_data_state("Simulating live preview", "Not ready")
            self.btn_export.setEnabled(False)
            run_note = ""

            self._build_scene(bounds)

            delay_ms, frame_interval_ms, segment_batch = self._speed_settings(
                self.sl_speed.value()
            )
            self._worker = SimWorker(
                self._engine,
                toolpath,
                emit_every=1,
                delay_ms=delay_ms,
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
            if use_swept:
                method = "Swept Volume"
            elif update_side_grids:
                method = "Legacy Full Tri-Dexel"
            else:
                method = "Legacy Surface Fast"
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
        self._pause_five_axis_preview()
        self._render_timer.stop()
        self._pending_hmap = None
        self._pending_frame_dirty = False
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)
        self._set_status("Stopped", DANGER)

    def _reset(self):
        self._pause_five_axis_preview()
        self._render_timer.stop()
        self._pending_hmap = None
        self._pending_frame_dirty = False
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            if not self._worker.wait(5000):
                self._worker.terminate()
        if self._mesh_worker and self._mesh_worker.isRunning():
            if not self._mesh_worker.wait(5000):
                self._mesh_worker.terminate()
        self._surface            = None
        self._surface_actor      = None
        self._tool_actor         = None
        self._result_mesh        = None
        self._display_mesh       = None
        self._display_gcode_moves = None
        self._set_data_state("Not ready", "Not ready")
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
        self.lbl_status.setText(f"●  {text}")
        self.lbl_status.setStyleSheet(
            f"color:{color};font-weight:bold;background:transparent;border:none;"
        )

    def closeEvent(self, event):
        self._pause_five_axis_preview()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            if not self._worker.wait(5000):
                self._worker.terminate()
        if self._mesh_worker and self._mesh_worker.isRunning():
            if not self._mesh_worker.wait(5000):
                self._mesh_worker.terminate()
        self.plotter.close()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "IMACT.CAM.Simulation"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("IMACT CAM")
    app.setWindowIcon(app_icon())
    app.setStyle("Fusion")
    base_font = QFont("Segoe UI", 11)
    base_font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(base_font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
