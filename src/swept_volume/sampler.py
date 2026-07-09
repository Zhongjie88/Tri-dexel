from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stock.z_dexel_grid import ZDexelGrid
from .builder import SweptVolume


@dataclass(frozen=True)
class DexelIntersection:
    i: int
    j: int
    cut_lo: float
    cut_hi: float
    normal: tuple[float, float, float]
    component_id: int | None = None


class SweptVolumeSampler:
    """Project swept-volume triangles onto a dexel grid."""

    def sample_grid(
        self,
        volume: SweptVolume,
        grid: ZDexelGrid,
        depth_min: float,
        depth_max: float,
        row_axis: int,
        col_axis: int,
        depth_axis: int,
    ) -> list[DexelIntersection]:
        if volume.n_tri == 0:
            return []

        # Triangle data is already in pre-stacked numpy arrays — no per-triangle
        # Python loop needed.
        verts = volume.vertices        # (n_tri, 3, 3)
        tri_normals = volume.normals   # (n_tri, 3)
        tri_comp_ids = volume.component_ids  # (n_tri,) int, -1 = None
        n_tri = volume.n_tri

        v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]  # (n_tri, 3)

        # Per-triangle AABB in (row, col) space.
        lo_row = np.minimum(np.minimum(v0[:, row_axis], v1[:, row_axis]), v2[:, row_axis])
        hi_row = np.maximum(np.maximum(v0[:, row_axis], v1[:, row_axis]), v2[:, row_axis])
        lo_col = np.minimum(np.minimum(v0[:, col_axis], v1[:, col_axis]), v2[:, col_axis])
        hi_col = np.maximum(np.maximum(v0[:, col_axis], v1[:, col_axis]), v2[:, col_axis])

        # Overall AABB → which grid cells to visit.
        lo_all = np.minimum(np.minimum(v0, v1), v2).min(axis=0)
        hi_all = np.maximum(np.maximum(v0, v1), v2).max(axis=0)
        i0 = grid.clamp_row(grid.row_index(lo_all[row_axis]))
        i1 = grid.clamp_row(grid.row_index(hi_all[row_axis]))
        j0 = grid.clamp_col(grid.col_index(lo_all[col_axis]))
        j1 = grid.clamp_col(grid.col_index(hi_all[col_axis]))
        if i0 > i1 or j0 > j1:
            return []

        # Möller-Trumbore precomputation: ray direction is fixed for every cell.
        direction = np.zeros(3, dtype=float)
        direction[depth_axis] = 1.0
        e1 = v1 - v0                             # (n_tri, 3)
        e2 = v2 - v0
        pvec = np.cross(direction, e2)           # (n_tri, 3)
        det = (e1 * pvec).sum(axis=1)            # (n_tri,)
        nondegenerate = np.abs(det) > 1e-9
        inv_det = np.where(nondegenerate, 1.0 / np.where(nondegenerate, det, 1.0), 0.0)

        n_rows = i1 - i0 + 1
        n_cols = j1 - j0 + 1
        row_vals = np.array([grid.row_center(i0 + ri) for ri in range(n_rows)])
        col_vals = np.array([grid.col_center(j0 + ci) for ci in range(n_cols)])
        half_drow = 0.5 * grid.dx
        half_dcol = 0.5 * grid.dy
        origin_depth = depth_min - 1.0

        # --- Vectorised cell-triangle overlap mask --------------------------------
        # cell_mask[ri, ci, k] = True if triangle k overlaps cell (i0+ri, j0+ci).
        # Broadcasting: (n_rows, 1, 1) op (1, 1, n_tri) → (n_rows, 1, n_tri)
        #               (1, n_cols, 1) op (1, 1, n_tri) → (1, n_cols, n_tri)
        # Combined &:   (n_rows, n_cols, n_tri)
        row_ok = (
            (lo_row[None, None, :] <= row_vals[:, None, None] + half_drow)
            & (hi_row[None, None, :] >= row_vals[:, None, None] - half_drow)
        )
        col_ok = (
            (lo_col[None, None, :] <= col_vals[None, :, None] + half_dcol)
            & (hi_col[None, None, :] >= col_vals[None, :, None] - half_dcol)
        )
        cell_mask = nondegenerate[None, None, :] & row_ok & col_ok  # (n_rows, n_cols, n_tri)

        # Only process cells that have at least one triangle candidate.
        ri_arr, ci_arr = np.nonzero(cell_mask.any(axis=2))
        n_active = len(ri_arr)
        if n_active == 0:
            return []

        # Build ray origins for all active cells at once: (n_active, 3).
        origins = np.zeros((n_active, 3), dtype=float)
        origins[:, row_axis] = row_vals[ri_arr]
        origins[:, col_axis] = col_vals[ci_arr]
        origins[:, depth_axis] = origin_depth

        # Batch Möller-Trumbore for all (active_cell × triangle) pairs.
        tvec = origins[:, None, :] - v0[None, :, :]              # (n_active, n_tri, 3)
        u = (tvec * pvec[None, :, :]).sum(axis=2) * inv_det[None, :]  # (n_active, n_tri)

        active_mask = cell_mask[ri_arr, ci_arr, :]                # (n_active, n_tri)
        u_ok = active_mask & (u >= -1e-9) & (u <= 1.0 + 1e-9)

        qvec = np.cross(tvec, e1[None, :, :])                     # (n_active, n_tri, 3)
        v_c = (direction * qvec).sum(axis=2) * inv_det[None, :]   # (n_active, n_tri)
        uv_ok = u_ok & (v_c >= -1e-9) & ((u + v_c) <= 1.0 + 1e-9)

        t_vals = (e2[None, :, :] * qvec).sum(axis=2) * inv_det[None, :]  # (n_active, n_tri)
        hit_ok = uv_ok & (t_vals >= -1e-9)

        depths = origin_depth + t_vals                                    # (n_active, n_tri)
        in_range = hit_ok & (depths >= depth_min) & (depths <= depth_max)

        # Iterate only over the few cells that actually have in-range hits.
        hits: list[DexelIntersection] = []
        for flat in np.where(in_range.any(axis=1))[0]:
            in_range_cell = in_range[flat]           # (n_tri,) bool
            valid_depths = depths[flat, in_range_cell]

            orig_idx = np.where(in_range_cell)[0]
            best_local = int(np.argmin(valid_depths))
            best_tri = int(orig_idx[best_local])
            best_normal = tri_normals[best_tri]
            raw_comp = int(tri_comp_ids[best_tri])
            best_comp_id = None if raw_comp < 0 else raw_comp

            depth_sorted = np.sort(valid_depths)
            n_d = len(depth_sorted)
            if n_d == 1:
                intervals: list[tuple[float, float]] = [(float(depth_sorted[0]), depth_max)]
            else:
                intervals = [
                    (float(depth_sorted[k]), float(depth_sorted[k + 1]))
                    for k in range(0, n_d - 1, 2)
                ]

            i_grid = int(i0 + ri_arr[flat])
            j_grid = int(j0 + ci_arr[flat])
            for cut_lo, cut_hi in intervals:
                if cut_hi <= cut_lo:
                    continue
                hits.append(
                    DexelIntersection(
                        i=i_grid,
                        j=j_grid,
                        cut_lo=max(depth_min, cut_lo),
                        cut_hi=min(depth_max, cut_hi),
                        normal=tuple(float(x) for x in best_normal),
                        component_id=best_comp_id,
                    )
                )
        return hits

    def sample_z_grid(
        self,
        volume: SweptVolume,
        grid: ZDexelGrid,
        depth_min: float,
        depth_max: float,
    ) -> list[DexelIntersection]:
        return self.sample_grid(
            volume, grid, depth_min, depth_max,
            row_axis=0, col_axis=1, depth_axis=2,
        )

    def sample_x_grid(
        self,
        volume: SweptVolume,
        grid: ZDexelGrid,
        depth_min: float,
        depth_max: float,
    ) -> list[DexelIntersection]:
        return self.sample_grid(
            volume, grid, depth_min, depth_max,
            row_axis=1, col_axis=2, depth_axis=0,
        )

    def sample_y_grid(
        self,
        volume: SweptVolume,
        grid: ZDexelGrid,
        depth_min: float,
        depth_max: float,
    ) -> list[DexelIntersection]:
        return self.sample_grid(
            volume, grid, depth_min, depth_max,
            row_axis=0, col_axis=2, depth_axis=1,
        )
