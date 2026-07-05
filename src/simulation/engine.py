from __future__ import annotations

import math
import sys
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from ..stock.tri_dexel import TriDexelStock
from ..tool.tool_geometry import ToolGeometry
from ..gcode.parser import GCodeMove

_Point3 = Tuple[float, float, float]


class SimulationEngine:
    """
    Core material-removal engine.

    For each tool position the engine updates all three dexel grids:

    Z-grid (rays // Z):
        For each dexel (i,j) in the XY footprint of the tool, call tool.z_cut(d, zt)
        to get the lowest Z the tool reaches; subtract [z_cut, z_max] from that ray.

    X-grid (rays // X, grid in YZ):
        For each dexel (j,k) in the YZ footprint, compute the tool's cross-section
        radius rz at height zk; if |dy| <= rz, subtract [xt-half_x, xt+half_x].

    Y-grid (rays // Y, grid in XZ):
        Symmetric to X-grid but for Y intervals.
    """

    def __init__(self, stock: TriDexelStock, tool: ToolGeometry) -> None:
        self.stock = stock
        self.tool = tool

    # ------------------------------------------------------------------
    # Single tool position
    # ------------------------------------------------------------------

    def apply_tool_at(self, xt: float, yt: float, zt: float) -> None:
        """Subtract tool volume from all three dexel grids at position (xt, yt, zt)."""
        self._update_z_grid(xt, yt, zt)
        self._update_x_grid(xt, yt, zt)
        self._update_y_grid(xt, yt, zt)

    def _update_z_grid(self, xt: float, yt: float, zt: float) -> None:
        tool = self.tool
        r = tool.radius
        g = self.stock.z_grid
        z_max = self.stock.bounds[5]

        # A dexel represents a finite XY cell, not an infinitesimal point at
        # the cell center. Include cells whose area intersects the tool
        # footprint; center-only tests leave small uncut stair-step islands at
        # arcs and inside corners.
        half_dx = g.dx * 0.5
        half_dy = g.dy * 0.5
        i0 = max(0, g.row_index(xt - r - half_dx))
        i1 = min(g.nx - 1, g.row_index(xt + r + half_dx))
        j0 = max(0, g.col_index(yt - r - half_dy))
        j1 = min(g.ny - 1, g.col_index(yt + r + half_dy))
        if i0 > i1 or j0 > j1:
            return

        ni, nj = i1 - i0 + 1, j1 - j0 + 1

        # Closest distance from tool center to each XY cell rectangle.
        xi = np.abs(np.array([g.row_center(i) for i in range(i0, i1 + 1)]) - xt)
        yj = np.abs(np.array([g.col_center(j) for j in range(j0, j1 + 1)]) - yt)
        dx = np.maximum(xi - half_dx, 0.0)
        dy = np.maximum(yj - half_dy, 0.0)
        d = np.sqrt(dx[:, np.newaxis] ** 2 + dy[np.newaxis, :] ** 2)  # (ni, nj)

        # Vectorised z_cut; NaN where outside footprint
        z_cuts = tool.z_cut_arr(d, zt)  # (ni, nj)

        # Skip cells where the cut does not improve (lower) the current surface.
        # For horizontal moves this eliminates the vast majority of subtract_at
        # calls: once a cell reaches its minimum z_cut (when the tool was
        # directly overhead), all subsequent positions produce shallower cuts.
        h_slice = g._height[i0:i1 + 1, j0:j1 + 1]
        improve = ~np.isnan(z_cuts) & (np.isnan(h_slice) | (z_cuts < h_slice))

        for li, lj in zip(*np.where(improve)):
            g.subtract_at(i0 + int(li), j0 + int(lj), float(z_cuts[li, lj]), z_max)

    def _update_x_grid(self, xt: float, yt: float, zt: float) -> None:
        """X-grid: rows=Y, cols=Z, depth=X."""
        tool = self.tool
        r = tool.radius
        g = self.stock.x_grid

        j0 = max(0, g.row_index(yt - r))
        j1 = min(g.nx - 1, g.row_index(yt + r))
        k0 = max(0, g.col_index(zt))
        k1 = g.ny - 1
        if j0 > j1:
            return

        nk = k1 - k0 + 1
        zk_arr  = np.array([g.col_center(k) for k in range(k0, k1 + 1)])
        valid, rz_arr = tool.cross_section_radius_arr(zk_arr, zt)
        rz2_arr = rz_arr * rz_arr  # avoid repeated squaring

        for j in range(j0, j1 + 1):
            dy  = g.row_center(j) - yt
            dy2 = dy * dy
            for lk in range(nk):
                if not valid[lk]:
                    continue
                rz2 = rz2_arr[lk]
                if dy2 > rz2:
                    continue
                half_x = math.sqrt(rz2 - dy2)
                g.rays[j][k0 + lk].subtract(xt - half_x, xt + half_x)

    def _update_y_grid(self, xt: float, yt: float, zt: float) -> None:
        """Y-grid: rows=X, cols=Z, depth=Y."""
        tool = self.tool
        r = tool.radius
        g = self.stock.y_grid

        i0 = max(0, g.row_index(xt - r))
        i1 = min(g.nx - 1, g.row_index(xt + r))
        k0 = max(0, g.col_index(zt))
        k1 = g.ny - 1
        if i0 > i1:
            return

        nk = k1 - k0 + 1
        zk_arr  = np.array([g.col_center(k) for k in range(k0, k1 + 1)])
        valid, rz_arr = tool.cross_section_radius_arr(zk_arr, zt)
        rz2_arr = rz_arr * rz_arr

        for i in range(i0, i1 + 1):
            dx  = g.row_center(i) - xt
            dx2 = dx * dx
            for lk in range(nk):
                if not valid[lk]:
                    continue
                rz2 = rz2_arr[lk]
                if dx2 > rz2:
                    continue
                half_y = math.sqrt(rz2 - dx2)
                g.rays[i][k0 + lk].subtract(yt - half_y, yt + half_y)

    # ------------------------------------------------------------------
    # Linear move
    # ------------------------------------------------------------------

    def simulate_move(
        self,
        start: _Point3,
        end: _Point3,
        step: Optional[float] = None,
    ) -> None:
        """Simulate material removal along a linear tool path from start to end.

        step defaults to half the stock resolution to guarantee full coverage.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        if dist < 1e-12:
            self.apply_tool_at(x1, y1, z1)
            return

        if step is None:
            step = self.stock.resolution  # one grid cell per step guarantees full coverage

        n = max(1, math.ceil(dist / step))
        for k in range(n + 1):
            t = k / n
            self.apply_tool_at(
                x1 + t * (x2 - x1),
                y1 + t * (y2 - y1),
                z1 + t * (z2 - z1),
            )

    def simulate_toolpath(
        self,
        toolpath: Iterable[tuple[_Point3, _Point3]],
        step: Optional[float] = None,
        progress_callback: Callable[[int, int, _Point3], None] | None = None,
        stop_callback: Callable[[], bool] | None = None,
        corner_angle_degrees: float = 5.0,
    ) -> None:
        """Simulate a continuous feed polyline with distance-based sampling.

        ``simulate_move`` samples every segment independently. That is correct
        but wasteful for CAM output that contains tens of thousands of tiny
        collinear or near-collinear segments. This path samples by accumulated
        arc length across connected feed segments, so runtime depends mainly on
        travelled distance / step instead of raw G-code segment count.
        Sharp direction changes are still sampled explicitly at the junction,
        which prevents small uncut islands around corners and arc-linearisation
        seams without falling back to per-segment sampling.
        """
        segments = list(toolpath)
        total = len(segments)
        if total == 0:
            return
        if step is None:
            step = self.stock.resolution
        step = max(float(step), 1e-9)

        carry = 0.0
        active = False
        current: _Point3 | None = None
        prev_dir: _Point3 | None = None
        corner_cos = math.cos(math.radians(max(0.0, corner_angle_degrees)))

        for idx, (start, end) in enumerate(segments, start=1):
            if stop_callback is not None and stop_callback():
                break
            x1, y1, z1 = start
            x2, y2, z2 = end
            if (
                not active
                or current is None
                or math.dist(current, start) > max(step, self.stock.resolution) * 2.0
            ):
                self.apply_tool_at(x1, y1, z1)
                carry = 0.0
                active = True
                prev_dir = None

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            travelled = 0.0
            if length <= 1e-12:
                self.apply_tool_at(x2, y2, z2)
                carry = 0.0
                prev_dir = None
            else:
                cur_dir: _Point3 = (dx / length, dy / length, dz / length)
                if prev_dir is not None:
                    dot = (
                        prev_dir[0] * cur_dir[0]
                        + prev_dir[1] * cur_dir[1]
                        + prev_dir[2] * cur_dir[2]
                    )
                    if max(-1.0, min(1.0, dot)) < corner_cos:
                        self.apply_tool_at(x1, y1, z1)
                        carry = 0.0

                needed = step - carry if carry > 1e-12 else step
                while travelled + needed <= length + 1e-12:
                    if stop_callback is not None and stop_callback():
                        return
                    travelled += needed
                    t = min(1.0, travelled / length)
                    self.apply_tool_at(
                        x1 + t * dx,
                        y1 + t * dy,
                        z1 + t * dz,
                    )
                    carry = 0.0
                    needed = step
                carry += max(0.0, length - travelled)
                # carry is always in [0, step) here (while-loop invariant).
                # If the segment endpoint is more than half a step from the
                # last trigger, sample it explicitly so that direction changes
                # at junctions between short G1 segments (e.g. linearised arcs)
                # never leave a gap larger than step/2.
                if carry > step * 0.5:
                    self.apply_tool_at(x2, y2, z2)
                    carry = 0.0
                prev_dir = cur_dir

            current = end
            if progress_callback is not None:
                progress_callback(idx, total, end)

        if current is not None:
            self.apply_tool_at(*current)

    # ------------------------------------------------------------------
    # G-code simulation
    # ------------------------------------------------------------------

    def simulate_gcode(
        self,
        moves: List[GCodeMove],
        progress: bool = True,
    ) -> None:
        """
        Execute a list of parsed G-code moves.

        Rapid moves (G0) are skipped for material removal but their end position
        is still tracked. Feed moves (G1) call simulate_move.
        """
        pos: _Point3 = (0.0, 0.0, self.stock.z_max)
        n = len(moves)
        interval = max(1, n // 20)

        for idx, mv in enumerate(moves):
            if progress and idx % interval == 0:
                pct = idx * 100 // n
                sys.stdout.write(f"\r  simulating {pct:3d}%  [{idx}/{n}]")
                sys.stdout.flush()

            new_pos: _Point3 = (mv.x, mv.y, mv.z)
            if not mv.rapid:
                self.simulate_move(pos, new_pos)
            pos = new_pos

        if progress:
            sys.stdout.write(f"\r  simulating 100%  [{n}/{n}]\n")
