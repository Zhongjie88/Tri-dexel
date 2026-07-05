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
        if not volume.triangles:
            return []

        n_tri = len(volume.triangles)

        # Pre-stack all triangle data into contiguous numpy arrays (one per call).
        # Avoids repeated Python attribute access inside the hot cell loop.
        verts = np.empty((n_tri, 3, 3), dtype=float)
        tri_normals = np.empty((n_tri, 3), dtype=float)
        tri_comp_ids: list[int | None] = []
        for k, tri in enumerate(volume.triangles):
            verts[k] = tri.vertices       # (3, 3)
            tri_normals[k] = tri.normal
            tri_comp_ids.append(tri.component_id)
        v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]  # (n_tri, 3) views

        # Per-triangle AABB in (row, col) space — used to filter per cell
        lo_row = np.minimum(np.minimum(v0[:, row_axis], v1[:, row_axis]), v2[:, row_axis])
        hi_row = np.maximum(np.maximum(v0[:, row_axis], v1[:, row_axis]), v2[:, row_axis])
        lo_col = np.minimum(np.minimum(v0[:, col_axis], v1[:, col_axis]), v2[:, col_axis])
        hi_col = np.maximum(np.maximum(v0[:, col_axis], v1[:, col_axis]), v2[:, col_axis])

        # Overall AABB → which grid cells to visit
        lo_all = np.minimum(np.minimum(v0, v1), v2).min(axis=0)
        hi_all = np.maximum(np.maximum(v0, v1), v2).max(axis=0)
        i0 = grid.clamp_row(grid.row_index(lo_all[row_axis]))
        i1 = grid.clamp_row(grid.row_index(hi_all[row_axis]))
        j0 = grid.clamp_col(grid.col_index(lo_all[col_axis]))
        j1 = grid.clamp_col(grid.col_index(hi_all[col_axis]))
        if i0 > i1 or j0 > j1:
            return []

        # Möller-Trumbore precomputation: ray direction is the same for every cell.
        direction = np.zeros(3, dtype=float)
        direction[depth_axis] = 1.0
        e1 = v1 - v0                            # (n_tri, 3)
        e2 = v2 - v0                            # (n_tri, 3)
        pvec = np.cross(direction, e2)           # (n_tri, 3) — direction broadcast
        det = (e1 * pvec).sum(axis=1)            # (n_tri,)
        nondegenerate = np.abs(det) > 1e-9
        inv_det = np.where(nondegenerate, 1.0 / np.where(nondegenerate, det, 1.0), 0.0)

        origin_depth = depth_min - 1.0
        half_drow = 0.5 * grid.dx
        half_dcol = 0.5 * grid.dy
        hits: list[DexelIntersection] = []

        for i in range(i0, i1 + 1):
            row_val = grid.row_center(i)
            # Row-AABB pre-filter (reused for every j in this row)
            row_mask = nondegenerate & (lo_row <= row_val + half_drow) & (hi_row >= row_val - half_drow)
            if not row_mask.any():
                continue

            for j in range(j0, j1 + 1):
                col_val = grid.col_center(j)
                # Col-AABB filter narrows to triangles that overlap this cell
                mask = row_mask & (lo_col <= col_val + half_dcol) & (hi_col >= col_val - half_dcol)
                if not mask.any():
                    continue

                # Build ray origin for this (row, col) cell
                origin = np.zeros(3, dtype=float)
                origin[row_axis] = row_val
                origin[col_axis] = col_val
                origin[depth_axis] = origin_depth

                # Batch Möller-Trumbore: test all masked triangles simultaneously
                v0m  = v0[mask]                                             # (m, 3)
                tvec = origin - v0m                                          # (m, 3)
                u    = (tvec * pvec[mask]).sum(axis=1) * inv_det[mask]       # (m,)
                u_ok = (u >= -1e-9) & (u <= 1.0 + 1e-9)
                if not u_ok.any():
                    continue

                qvec  = np.cross(tvec, e1[mask])                             # (m, 3)
                v_c   = (direction * qvec).sum(axis=1) * inv_det[mask]        # (m,)
                uv_ok = u_ok & (v_c >= -1e-9) & ((u + v_c) <= 1.0 + 1e-9)
                if not uv_ok.any():
                    continue

                t_vals = (e2[mask] * qvec).sum(axis=1) * inv_det[mask]        # (m,)
                hit_ok = uv_ok & (t_vals >= -1e-9)
                if not hit_ok.any():
                    continue

                depths = origin_depth + t_vals[hit_ok]
                in_range = (depths >= depth_min) & (depths <= depth_max)
                if not in_range.any():
                    continue

                valid_depths = depths[in_range]

                # Map back to original triangle indices for normal / component_id
                mask_idx  = np.where(mask)[0]
                hit_idx   = np.where(hit_ok)[0]
                range_idx = np.where(in_range)[0]
                orig_idx  = mask_idx[hit_idx[range_idx]]
                best_local = int(np.argmin(valid_depths))
                best_tri   = int(orig_idx[best_local])
                best_normal   = tri_normals[best_tri]
                best_comp_id  = tri_comp_ids[best_tri]

                depth_sorted = np.sort(valid_depths)
                n_d = len(depth_sorted)
                if n_d == 1:
                    intervals: list[tuple[float, float]] = [(float(depth_sorted[0]), depth_max)]
                else:
                    intervals = [
                        (float(depth_sorted[k]), float(depth_sorted[k + 1]))
                        for k in range(0, n_d - 1, 2)
                    ]
                for cut_lo, cut_hi in intervals:
                    if cut_hi <= cut_lo:
                        continue
                    hits.append(
                        DexelIntersection(
                            i=i,
                            j=j,
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
            volume,
            grid,
            depth_min,
            depth_max,
            row_axis=0,
            col_axis=1,
            depth_axis=2,
        )

    def sample_x_grid(
        self,
        volume: SweptVolume,
        grid: ZDexelGrid,
        depth_min: float,
        depth_max: float,
    ) -> list[DexelIntersection]:
        return self.sample_grid(
            volume,
            grid,
            depth_min,
            depth_max,
            row_axis=1,
            col_axis=2,
            depth_axis=0,
        )

    def sample_y_grid(
        self,
        volume: SweptVolume,
        grid: ZDexelGrid,
        depth_min: float,
        depth_max: float,
    ) -> list[DexelIntersection]:
        return self.sample_grid(
            volume,
            grid,
            depth_min,
            depth_max,
            row_axis=0,
            col_axis=2,
            depth_axis=1,
        )
