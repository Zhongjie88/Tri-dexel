from __future__ import annotations

import numpy as np

from .z_dexel_grid import ZDexelGrid


class TriDexelStock:
    """
    Tri-dexel stock model: three orthogonal dexel grids (X, Y, Z ray directions).

    Tri-dexel representation gives accurate material description for 5-axis cuts:
      - z_grid: rays // Z axis, grid in XY plane  → stores Z intervals
      - x_grid: rays // X axis, grid in YZ plane  → stores X intervals
      - y_grid: rays // Y axis, grid in XZ plane  → stores Y intervals

    A voxel is material iff it is inside ALL three grids (intersection semantics).
    This eliminates the 'fin' artefacts that single-direction dexels produce on
    near-vertical surfaces.

    Parameters
    ----------
    bounds : (x_min, x_max, y_min, y_max, z_min, z_max)
    resolution : uniform cell size in mm
    """

    def __init__(self, bounds: tuple[float, ...], resolution: float) -> None:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        self.bounds = bounds
        self.resolution = resolution

        nx = max(1, round((x_max - x_min) / resolution))
        ny = max(1, round((y_max - y_min) / resolution))
        nz = max(1, round((z_max - z_min) / resolution))
        self.nx, self.ny, self.nz = nx, ny, nz

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max

        # Z-direction: rays // Z, indexed by (i=X, j=Y)
        self.z_grid = ZDexelGrid(x_min, x_max, y_min, y_max, nx, ny)
        # X-direction: rays // X, indexed by (j=Y, k=Z)
        self.x_grid = ZDexelGrid(y_min, y_max, z_min, z_max, ny, nz)
        # Y-direction: rays // Y, indexed by (i=X, k=Z)
        self.y_grid = ZDexelGrid(x_min, x_max, z_min, z_max, nx, nz)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_box_stock(self) -> None:
        """Fill all three grids with the full bounding-box block."""
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        self.z_grid.initialize_stock(z_min, z_max)
        self.x_grid.initialize_stock(x_min, x_max)
        self.y_grid.initialize_stock(y_min, y_max)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Return True when all three dexel grids still contain this point."""
        if not (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        ):
            return False

        zi = self.z_grid.clamp_row(self.z_grid.row_index(x))
        zj = self.z_grid.clamp_col(self.z_grid.col_index(y))
        xi = self.x_grid.clamp_row(self.x_grid.row_index(y))
        xj = self.x_grid.clamp_col(self.x_grid.col_index(z))
        yi = self.y_grid.clamp_row(self.y_grid.row_index(x))
        yj = self.y_grid.clamp_col(self.y_grid.col_index(z))

        return (
            self.z_grid.rays[zi][zj].contains(z)
            and self.x_grid.rays[xi][xj].contains(x)
            and self.y_grid.rays[yi][yj].contains(y)
        )

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def to_voxel_grid(self) -> np.ndarray:
        """Reconstruct a dense boolean voxel grid [nx, ny, nz] via intersection.

        Slower path – call after simulation to get the full solid for Marching Cubes.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        nx, ny, nz = self.nx, self.ny, self.nz

        # Each grid contributes [n_row, n_col, n_depth]
        vox_z = self.z_grid.to_dense_voxel(z_min, z_max, nz)          # [nx, ny, nz]
        vox_x = self.x_grid.to_dense_voxel(x_min, x_max, nx)          # [ny, nz, nx]
        vox_x = np.transpose(vox_x, (2, 0, 1))                         # [nx, ny, nz]
        vox_y = self.y_grid.to_dense_voxel(y_min, y_max, ny)          # [nx, nz, ny]
        vox_y = np.transpose(vox_y, (0, 2, 1))                         # [nx, ny, nz]

        return vox_z & vox_x & vox_y
