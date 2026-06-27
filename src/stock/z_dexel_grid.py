from __future__ import annotations

import numpy as np

from .dexel_ray import DexelRay


class ZDexelGrid:
    """
    2D grid of dexel rays, all parallel to the same axis (the 'depth' axis).

    Used for all three orthogonal directions in the tri-dexel model:
      - Z-direction: rows=X cols=Y depth=Z  → ZDexelGrid(x_min,x_max, y_min,y_max, nx,ny)
      - X-direction: rows=Y cols=Z depth=X  → ZDexelGrid(y_min,y_max, z_min,z_max, ny,nz)
      - Y-direction: rows=X cols=Z depth=Y  → ZDexelGrid(x_min,x_max, z_min,z_max, nx,nz)

    rays[i][j] is the dexel at row i, column j.
    Cell centers: row_center(i) = row_min + (i+0.5)*d_row
                  col_center(j) = col_min + (j+0.5)*d_col
    """

    def __init__(
        self,
        row_min: float,
        row_max: float,
        col_min: float,
        col_max: float,
        n_row: int,
        n_col: int,
    ) -> None:
        self.row_min = row_min
        self.row_max = row_max
        self.col_min = col_min
        self.col_max = col_max
        self.nx = n_row  # kept as nx/ny for readability at call sites
        self.ny = n_col
        self.dx = (row_max - row_min) / n_row
        self.dy = (col_max - col_min) / n_col
        self.rays: list[list[DexelRay]] = [
            [DexelRay() for _ in range(n_col)] for _ in range(n_row)
        ]
        # Cached top-surface heights; updated incrementally by subtract_at().
        self._height: np.ndarray = np.full((n_row, n_col), np.nan)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_stock(self, depth_lo: float, depth_hi: float) -> None:
        """Fill every ray with a single solid interval [depth_lo, depth_hi]."""
        for row in self.rays:
            for ray in row:
                ray.set_solid(depth_lo, depth_hi)
        self._height[:] = depth_hi

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def row_center(self, i: int) -> float:
        return self.row_min + (i + 0.5) * self.dx

    def col_center(self, j: int) -> float:
        return self.col_min + (j + 0.5) * self.dy

    def row_index(self, v: float) -> int:
        return int((v - self.row_min) / self.dx)

    def col_index(self, v: float) -> int:
        return int((v - self.col_min) / self.dy)

    def clamp_row(self, i: int) -> int:
        return max(0, min(self.nx - 1, i))

    def clamp_col(self, j: int) -> int:
        return max(0, min(self.ny - 1, j))

    # ------------------------------------------------------------------
    # Fast update path (used by the simulation engine)
    # ------------------------------------------------------------------

    def subtract_at(self, i: int, j: int, cut_lo: float, cut_hi: float) -> None:
        """Subtract interval from ray (i,j) and update the height cache."""
        ray = self.rays[i][j]
        ray.subtract(cut_lo, cut_hi)
        top = ray.top()
        self._height[i, j] = top if top is not None else np.nan

    def sync_height_cache(self) -> None:
        """Rebuild the height cache from current ray state (O(nx*ny) one-time cost).

        Call this after bulk-writing ray.intervals directly (e.g. STL import).
        """
        for i in range(self.nx):
            for j in range(self.ny):
                top = self.rays[i][j].top()
                self._height[i, j] = top if top is not None else np.nan

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def height_map(self) -> np.ndarray:
        """Top-surface depth values for every ray; NaN where ray is empty.

        Returns a copy of the internally cached array — O(nx*ny) copy only.
        """
        return self._height.copy()

    def to_dense_voxel(
        self, depth_min: float, depth_max: float, n_depth: int
    ) -> np.ndarray:
        """Convert to boolean voxel array of shape [n_row, n_col, n_depth].

        voxel[i, j, k] is True if depth slice k falls inside material at ray (i,j).
        """
        d_step = (depth_max - depth_min) / n_depth
        voxels = np.zeros((self.nx, self.ny, n_depth), dtype=bool)
        for i in range(self.nx):
            for j in range(self.ny):
                for lo, hi in self.rays[i][j].intervals:
                    k_lo = max(0, int((lo - depth_min) / d_step))
                    k_hi = min(n_depth, int(np.ceil((hi - depth_min) / d_step)))
                    voxels[i, j, k_lo:k_hi] = True
        return voxels
