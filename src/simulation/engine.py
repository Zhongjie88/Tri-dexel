from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from ..stock.tri_dexel import TriDexelStock
from ..tool.tool_geometry import BallEndMill, FlatEndMill, ToolGeometry
from ..gcode.parser import GCodeMove
from ..motion.pose import ToolPose
from .collision import CollisionEvent, detect_pose_collision

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

    def __init__(
        self,
        stock: TriDexelStock,
        tool: ToolGeometry,
        update_side_grids: bool = True,
        detect_collision: bool = False,
    ) -> None:
        self.stock = stock
        self.tool = tool
        self.update_side_grids = bool(update_side_grids)
        self.detect_collision = bool(detect_collision)
        self.collision_events: list[CollisionEvent] = []
        self._pose_cutting_surface_cache: dict[tuple[int, int], np.ndarray] = {}
        self._z_footprint_cache = self._build_z_footprint_cache()
        # Cache cross_section_radius_arr result for the most recent zt value.
        # X-grid and Y-grid share the same Z depth array when the stock is uniform,
        # so one computation serves both grids for the same tool position.
        # (zt, k0, col_centers_id) → (valid, rz, rz2)
        # Including k0 and col_centers identity prevents x-grid and y-grid
        # from sharing a stale cached slice when their depth ranges differ.
        self._csr_cache_key: tuple = (float("nan"), -1, -1)
        self._csr_cache_valid: np.ndarray | None = None
        self._csr_cache_rz: np.ndarray | None = None
        self._csr_cache_rz2: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Single tool position
    # ------------------------------------------------------------------

    def _build_z_footprint_cache(self) -> dict[str, np.ndarray] | None:
        """Precompute the XY footprint for common vertical legacy tools."""
        if not isinstance(self.tool, (FlatEndMill, BallEndMill)):
            return None

        g = self.stock.z_grid
        r = float(self.tool.radius)
        if r < max(g.dx, g.dy):
            return None
        half_dx = g.dx * 0.5
        half_dy = g.dy * 0.5
        max_di = int(math.ceil((r + half_dx) / g.dx)) + 1
        max_dj = int(math.ceil((r + half_dy) / g.dy)) + 1
        di_vals = np.arange(-max_di, max_di + 1, dtype=int)
        dj_vals = np.arange(-max_dj, max_dj + 1, dtype=int)
        di_grid, dj_grid = np.meshgrid(di_vals, dj_vals, indexing="ij")

        return {
            "di": di_grid.ravel().astype(int),
            "dj": dj_grid.ravel().astype(int),
            "kind": "flat" if isinstance(self.tool, FlatEndMill) else "ball",
        }

    def apply_tool_at(self, xt: float, yt: float, zt: float) -> None:
        """Subtract tool volume from all three dexel grids at position (xt, yt, zt)."""
        self._record_collision_at_pose(ToolPose((xt, yt, zt)))
        self._update_z_grid(xt, yt, zt)
        if self.update_side_grids:
            self._update_x_grid(xt, yt, zt)
            self._update_y_grid(xt, yt, zt)

    def _record_collision_at_pose(self, pose: ToolPose) -> None:
        if not self.detect_collision:
            return
        self.collision_events.extend(
            detect_pose_collision(
                self.stock,
                self.tool,
                pose,
                n_u=12,
                n_v=4,
                max_events=200,
            )
        )

    def check_collision_at_pose(self, pose: ToolPose) -> list[CollisionEvent]:
        return detect_pose_collision(self.stock, self.tool, pose)

    def _subtract_z_intervals(
        self,
        ii: np.ndarray,
        jj: np.ndarray,
        cut_los: np.ndarray,
        cut_hi: float,
    ) -> None:
        g = self.stock.z_grid
        cut_hi = min(float(cut_hi), float(self.stock.z_max))
        if len(ii) == 0:
            return
        valid = np.asarray(cut_los, dtype=float) < cut_hi
        if not np.any(valid):
            return
        ii = np.asarray(ii)[valid]
        jj = np.asarray(jj)[valid]
        cut_los = np.asarray(cut_los, dtype=float)[valid]
        if cut_hi >= float(self.stock.z_max) - 1e-12 and hasattr(g, "batch_subtract"):
            g.batch_subtract(ii, jj, cut_los, self.stock.z_max)
            return
        for i, j, cut_lo in zip(ii, jj, cut_los):
            g.subtract_at(int(i), int(j), float(cut_lo), cut_hi)

    def apply_tool_pose_at(
        self,
        pose: ToolPose,
        n_u: int = 48,
        n_v: int = 18,
    ) -> None:
        """Subtract an oriented tool pose using sampled cutting geometry.

        This is the legacy 5-axis path: it keeps the fast height-map workflow,
        but uses the A/B/C-derived ToolPose instead of assuming a vertical tool.
        The original analytic apply_tool_at remains unchanged for 3-axis use.
        """
        if float(np.linalg.norm(pose.axis - np.array([0.0, 0.0, 1.0]))) < 1e-9:
            self.apply_tool_at(
                float(pose.position[0]),
                float(pose.position[1]),
                float(pose.position[2]),
            )
            return

        self._record_collision_at_pose(pose)

        key = (int(n_u), int(n_v))
        local_points = self._pose_cutting_surface_cache.get(key)
        if local_points is None:
            surface = self.tool.sample_surface(
                n_u=n_u,
                n_v=n_v,
                include_non_cutting=False,
            )
            local_points = np.asarray(surface.points, dtype=float)
            self._pose_cutting_surface_cache[key] = local_points
        points = local_points @ pose.rotation.T + pose.position
        self._update_z_grid_from_surface_points(points)

    def _update_z_grid_from_surface_points(self, points: np.ndarray) -> None:
        g = self.stock.z_grid
        z_max = self.stock.bounds[5]
        half_dx = g.dx * 0.5
        half_dy = g.dy * 0.5

        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.size == 0:
            return

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        i = ((x - g.row_min) / g.dx).astype(int)
        j = ((y - g.col_min) / g.dy).astype(int)
        valid = (i >= 0) & (i < g.nx) & (j >= 0) & (j < g.ny)
        if not np.any(valid):
            return

        x = x[valid]
        y = y[valid]
        z = z[valid]
        i = i[valid]
        j = j[valid]

        ii_parts = [i]
        jj_parts = [j]
        z_parts = [z]

        # A sampled inclined surface can otherwise miss cells exactly on a
        # cell boundary. Touch immediate neighbours only when the point is
        # close to that boundary. Keep the same neighbour policy as before,
        # but batch the indexing work instead of using a Python dict per point.
        near_i = np.abs(x - g.row_centers[i]) > half_dx * 0.75
        if np.any(near_i):
            ii = i[near_i] + np.where(x[near_i] > g.row_centers[i[near_i]], 1, -1)
            ok = (ii >= 0) & (ii < g.nx)
            if np.any(ok):
                ii_parts.append(ii[ok])
                jj_parts.append(j[near_i][ok])
                z_parts.append(z[near_i][ok])

        near_j = np.abs(y - g.col_centers[j]) > half_dy * 0.75
        if np.any(near_j):
            jj = j[near_j] + np.where(y[near_j] > g.col_centers[j[near_j]], 1, -1)
            ok = (jj >= 0) & (jj < g.ny)
            if np.any(ok):
                ii_parts.append(i[near_j][ok])
                jj_parts.append(jj[ok])
                z_parts.append(z[near_j][ok])

        ii_all = np.concatenate(ii_parts)
        jj_all = np.concatenate(jj_parts)
        z_all = np.concatenate(z_parts)
        keys = ii_all * g.ny + jj_all
        order = np.argsort(keys)
        keys_sorted = keys[order]
        z_sorted = z_all[order]
        starts = np.r_[0, np.flatnonzero(np.diff(keys_sorted)) + 1]
        unique_keys = keys_sorted[starts]
        min_z = np.minimum.reduceat(z_sorted, starts)
        max_z = np.maximum.reduceat(z_sorted, starts)

        legacy_unbounded_cutter = math.isinf(float(self.tool.cutting_length))
        for key, z_cut, z_hi in zip(unique_keys, min_z, max_z):
            ii = int(key // g.ny)
            jj = int(key % g.ny)
            cut_hi = z_max if legacy_unbounded_cutter else min(float(z_hi), z_max)
            g.subtract_at(ii, jj, float(z_cut), cut_hi)

    def _update_z_grid(self, xt: float, yt: float, zt: float) -> None:
        if self._z_footprint_cache is not None:
            self._update_z_grid_cached(xt, yt, zt)
            return

        tool = self.tool
        r = tool.radius
        g = self.stock.z_grid
        z_max = self.stock.bounds[5]
        cut_hi = min(z_max, tool.cutting_top_z(zt))

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
        xi = np.abs(g.row_centers[i0:i1 + 1] - xt)
        yj = np.abs(g.col_centers[j0:j1 + 1] - yt)
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

        li_arr, lj_arr = np.where(improve)
        self._subtract_z_intervals(
            i0 + li_arr,
            j0 + lj_arr,
            z_cuts[li_arr, lj_arr],
            cut_hi,
        )

    def _update_z_grid_cached(self, xt: float, yt: float, zt: float) -> None:
        g = self.stock.z_grid
        z_max = self.stock.bounds[5]
        cut_hi = min(z_max, self.tool.cutting_top_z(zt))
        cache = self._z_footprint_cache
        assert cache is not None

        ci = g.row_index(xt)
        cj = g.col_index(yt)
        ii = ci + cache["di"]
        jj = cj + cache["dj"]
        valid = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny)
        if not np.any(valid):
            return

        ii = ii[valid]
        jj = jj[valid]
        dx = np.maximum(np.abs(g.row_centers[ii] - xt) - g.dx * 0.5, 0.0)
        dy = np.maximum(np.abs(g.col_centers[jj] - yt) - g.dy * 0.5, 0.0)
        d = np.sqrt(dx * dx + dy * dy)
        r = float(self.tool.radius)
        in_footprint = d <= r
        if not np.any(in_footprint):
            return

        ii = ii[in_footprint]
        jj = jj[in_footprint]
        d = d[in_footprint]
        if cache["kind"] == "flat":
            z_cuts = np.full(d.shape, float(zt), dtype=float)
        else:
            z_cuts = float(zt) + r - np.sqrt(np.maximum(0.0, r * r - d * d))
        h_vals = g._height[ii, jj]
        improve = ~np.isnan(z_cuts) & (np.isnan(h_vals) | (z_cuts < h_vals))
        if not np.any(improve):
            return

        self._subtract_z_intervals(ii[improve], jj[improve], z_cuts[improve], cut_hi)

    def _get_cross_section(self, zt: float, k0: int, k1: int, col_centers: np.ndarray):
        """Return (valid, rz_arr, rz2_arr) for depth slice [k0..k1].

        Cached per (zt, k0, id(col_centers)) so x-grid and y-grid with
        different depth ranges never share a stale cached result.
        """
        key = (zt, k0, id(col_centers))
        if key != self._csr_cache_key or self._csr_cache_valid is None:
            zk_arr = col_centers[k0:k1 + 1]
            valid, rz_arr = self.tool.cross_section_radius_arr(zk_arr, zt)
            # Guarantee contiguous layout so callers can pass directly to numba.
            self._csr_cache_key = key
            self._csr_cache_valid = np.ascontiguousarray(valid)
            self._csr_cache_rz = np.ascontiguousarray(rz_arr)
            self._csr_cache_rz2 = np.ascontiguousarray(rz_arr * rz_arr)
        return self._csr_cache_valid, self._csr_cache_rz, self._csr_cache_rz2

    def _update_side_grid_impl(
        self,
        g,
        cut_axis_val: float,
        perp_val: float,
        zt: float,
    ) -> None:
        """Shared implementation for X-grid and Y-grid updates.

        cut_axis_val — xt for X-grid, yt for Y-grid (the depth axis being subtracted)
        perp_val     — yt for X-grid, xt for Y-grid (the row/perpendicular axis)
        """
        r = self.tool.radius
        i0 = max(0, g.row_index(perp_val - r))
        i1 = min(g.nx - 1, g.row_index(perp_val + r))
        if i0 > i1:
            return
        k0 = max(0, g.col_index(zt))
        k1 = g.ny - 1
        valid, _rz_arr, rz2_arr = self._get_cross_section(zt, k0, k1, g.col_centers)
        if hasattr(g, "update_side_grid"):
            g.update_side_grid(i0, i1, k0, cut_axis_val, perp_val, valid, rz2_arr)
        else:
            nk = k1 - k0 + 1
            for i in range(i0, i1 + 1):
                dp2 = (g.row_centers[i] - perp_val) ** 2
                for lk in range(nk):
                    if not valid[lk]:
                        continue
                    rz2 = float(rz2_arr[lk])
                    if dp2 > rz2:
                        continue
                    half = math.sqrt(rz2 - dp2)
                    g.rays[i][k0 + lk].subtract(cut_axis_val - half, cut_axis_val + half)

    def _update_x_grid(self, xt: float, yt: float, zt: float) -> None:
        """X-grid: rows=Y, cols=Z, depth=X."""
        self._update_side_grid_impl(self.stock.x_grid, xt, yt, zt)

    def _update_y_grid(self, xt: float, yt: float, zt: float) -> None:
        """Y-grid: rows=X, cols=Z, depth=Y."""
        self._update_side_grid_impl(self.stock.y_grid, yt, xt, zt)

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

    def simulate_pose_move(
        self,
        start: ToolPose,
        end: ToolPose,
        step: Optional[float] = None,
    ) -> None:
        """Simulate a legacy move with oriented start/end ToolPose data."""
        dist = float(math.dist(start.position, end.position))
        if step is None:
            step = self.stock.resolution
        if dist < 1e-12:
            self.apply_tool_pose_at(end)
            return

        n = max(1, math.ceil(dist / max(float(step), 1e-9)))
        for k in range(n + 1):
            pose = start.interpolate(end, k / n)
            self.apply_tool_pose_at(pose)

    def simulate_pose_toolpath(
        self,
        toolpath: Iterable[tuple[ToolPose, ToolPose]],
        step: Optional[float] = None,
        progress_callback: Callable[[int, int, ToolPose], None] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> None:
        segments = list(toolpath)
        total = len(segments)
        if step is None:
            step = self.stock.resolution
        for idx, (start, end) in enumerate(segments, start=1):
            if stop_callback is not None and stop_callback():
                break
            self.simulate_pose_move(start, end, step=step)
            if progress_callback is not None:
                progress_callback(idx, total, end)

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
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Execute a list of parsed G-code moves.

        Rapid moves (G0) update position only. Feed moves (G1) cut material.
        progress_callback(current, total) fires after each move.
        stop_callback() halts simulation early if it returns True.
        """
        pos: _Point3 = (0.0, 0.0, self.stock.z_max)
        n = len(moves)

        for idx, mv in enumerate(moves):
            if stop_callback is not None and stop_callback():
                break
            new_pos: _Point3 = (mv.x, mv.y, mv.z)
            if not mv.rapid:
                self.simulate_move(pos, new_pos)
            pos = new_pos
            if progress_callback is not None:
                progress_callback(idx + 1, n)
