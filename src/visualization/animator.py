from __future__ import annotations

import math
import numpy as np
import pyvista as pv

from ..stock.tri_dexel import TriDexelStock
from ..simulation.engine import SimulationEngine


class MachiningAnimator:
    """
    Real-time PyVista animation of tri-dexel material removal.

    Each VTK timer tick:
      1. Runs `segments_per_frame` toolpath segments through the engine
      2. Rebuilds the height-map surface mesh in-place (only Z values change)
      3. Translates the tool-sphere actor to the current tip position
      4. Updates the progress text overlay

    The PyVista window stays fully interactive (rotate/zoom) during simulation.
    """

    _BG_BOTTOM  = "#0f0f1a"
    _BG_TOP     = "#1a1a2e"
    _STOCK_COLOR = "#c8b89a"
    _TOOL_COLOR  = "#4fc3f7"
    _PATH_COLOR  = "#ff7043"
    _BOX_COLOR   = "#3a7bd5"

    def __init__(
        self,
        stock: TriDexelStock,
        engine: SimulationEngine,
        toolpath: list[tuple],       # list of (start_xyz, end_xyz)
        segments_per_frame: int = 2,
        fps: int = 12,
        title: str = "Tri-dexel Machining Simulation",
    ) -> None:
        self.stock = stock
        self.engine = engine
        self.toolpath = toolpath
        self.spf = segments_per_frame
        self._interval_ms = max(1, int(1000 / fps))
        self._title = title

        self._idx = 0
        self._total = len(toolpath)
        self._done = False

        # PyVista objects (set in _setup)
        self._pl: pv.Plotter | None = None
        self._surface: pv.StructuredGrid | None = None
        self._xx: np.ndarray | None = None
        self._yy: np.ndarray | None = None
        self._tool_actor = None
        self._text_actor = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _make_surface(self, zz: np.ndarray) -> pv.StructuredGrid:
        return pv.StructuredGrid(self._xx, self._yy, zz)

    def _setup(self) -> None:
        s = self.stock
        x_min, x_max, y_min, y_max, z_min, z_max = s.bounds

        # Pre-compute XY meshgrid (fixed for the whole simulation)
        x = np.linspace(x_min, x_max, s.nx)
        y = np.linspace(y_min, y_max, s.ny)
        self._xx, self._yy = np.meshgrid(x, y, indexing="ij")

        self._pl = pv.Plotter(title=self._title)
        self._pl.set_background(self._BG_BOTTOM, top=self._BG_TOP)

        # Wireframe bounding box
        box = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
        self._pl.add_mesh(
            box, style="wireframe", color=self._BOX_COLOR,
            line_width=1.5, opacity=0.4,
        )

        # Initial stock surface (flat top)
        zz_init = np.full((s.nx, s.ny), z_max)
        self._surface = self._make_surface(zz_init)
        self._pl.add_mesh(
            self._surface,
            color=self._STOCK_COLOR,
            smooth_shading=True,
            show_edges=False,
            ambient=0.15,
            diffuse=0.75,
            specular=0.2,
        )

        # Floor / base plate
        floor = pv.Plane(
            center=((x_min + x_max) / 2, (y_min + y_max) / 2, z_min),
            direction=(0, 0, 1),
            i_size=(x_max - x_min) * 1.1,
            j_size=(y_max - y_min) * 1.1,
        )
        self._pl.add_mesh(floor, color="#444455", opacity=0.4)

        # Tool-path centre-line (static overlay)
        if self.toolpath:
            pts = []
            for seg_s, seg_e in self.toolpath:
                pts.append(seg_s)
                pts.append(seg_e)
            line = pv.lines_from_points(np.array(pts, dtype=float))
            self._pl.add_mesh(line, color=self._PATH_COLOR, line_width=1.0, opacity=0.4)

        # Tool sphere (centred at world origin; position set via actor transform)
        r = self.engine.tool.radius
        tool_sphere = pv.Sphere(radius=r, theta_resolution=28, phi_resolution=28)
        self._tool_actor = self._pl.add_mesh(
            tool_sphere,
            color=self._TOOL_COLOR,
            opacity=0.80,
            smooth_shading=True,
            specular=0.6,
        )
        # Place at first segment start
        first_pos = list(self.toolpath[0][0]) if self.toolpath else [x_min, y_min, z_max]
        self._tool_actor.position = first_pos

        # Progress text (top-left corner)
        self._text_actor = self._pl.add_text(
            "0 %   [ 0 / 0 ]",
            position="upper_left",
            font_size=11,
            color="white",
            shadow=True,
        )

        # Info text (bottom)
        tool_name = type(self.engine.tool).__name__
        self._pl.add_text(
            f"Tool: {tool_name}  r={self.engine.tool.radius} mm  "
            f"| Resolution: {self.stock.resolution} mm  "
            f"| Grid: {s.nx}×{s.ny}",
            position="lower_left",
            font_size=9,
            color="#aaaacc",
        )

        self._pl.show_axes()
        # Isometric-ish camera
        cx, cy, cz = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
        cam_dist = max(x_max-x_min, y_max-y_min) * 1.8
        self._pl.camera_position = [
            (cx + cam_dist, cy - cam_dist * 0.6, cz + cam_dist * 0.9),
            (cx, cy, cz),
            (0, 0, 1),
        ]

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _on_timer(self, caller=None, event=None) -> None:
        if self._done:
            return

        # Simulate next batch of segments
        for _ in range(self.spf):
            if self._idx >= self._total:
                self._done = True
                break
            seg_s, seg_e = self.toolpath[self._idx]
            self.engine.simulate_move(seg_s, seg_e)
            self._idx += 1

        # Current tool tip position
        if self._idx < self._total:
            cur_pos = list(self.toolpath[self._idx][0])
        else:
            cur_pos = list(self.toolpath[-1][1])

        # Rebuild surface points through the property setter so VTK redraws.
        hmap = self.stock.z_grid.height_map()
        zz = np.nan_to_num(hmap, nan=self.stock.z_min)
        new_grid = self._make_surface(zz)
        self._surface.points = new_grid.points.copy()

        # Move tool actor
        self._tool_actor.position = cur_pos

        # Update progress label
        pct = self._idx * 100 // self._total
        self._text_actor.SetInput(
            f"{pct} %   [ {self._idx} / {self._total} ]"
        )
        self._pl.render()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._setup()
        max_ticks = math.ceil(self._total / max(1, self.spf)) + 10
        self._pl.add_timer_event(
            max_steps=max_ticks,
            duration=self._interval_ms,
            callback=self._on_timer,
        )
        self._pl.show()
