from __future__ import annotations

import math
import sys
from typing import Callable, List, Optional, Tuple

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

        i0 = max(0, g.row_index(xt - r))
        i1 = min(g.nx - 1, g.row_index(xt + r))
        j0 = max(0, g.col_index(yt - r))
        j1 = min(g.ny - 1, g.col_index(yt + r))
        if i0 > i1 or j0 > j1:
            return

        ni, nj = i1 - i0 + 1, j1 - j0 + 1

        # Vectorised distance computation (one sqrt call for the whole footprint)
        xi = np.array([g.row_center(i) for i in range(i0, i1 + 1)]) - xt
        yj = np.array([g.col_center(j) for j in range(j0, j1 + 1)]) - yt
        d  = np.sqrt(xi[:, np.newaxis] ** 2 + yj[np.newaxis, :] ** 2)  # (ni, nj)

        # Vectorised z_cut; NaN where outside footprint
        z_cuts = tool.z_cut_arr(d, zt)  # (ni, nj)

        for li in range(ni):
            row = z_cuts[li]
            for lj in range(nj):
                zc = row[lj]
                if not np.isnan(zc):
                    g.subtract_at(i0 + li, j0 + lj, float(zc), z_max)

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
