from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Maximum number of intervals per dexel ray.
# For 3-axis machining typical values are 1-3; 32 is a safe upper bound.
MAX_INTERVALS: int = 32

try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _nb_subtract_one(ivs, n, cut_lo, cut_hi):
        if cut_lo >= cut_hi:
            return n
        cap = ivs.shape[0]
        write = 0
        changed = False
        for k in range(n):
            lo = ivs[k, 0]
            hi = ivs[k, 1]
            if hi <= cut_lo or lo >= cut_hi:
                if write < cap:
                    ivs[write, 0] = lo
                    ivs[write, 1] = hi
                write += 1
            else:
                changed = True
                if lo < cut_lo and write < cap:
                    ivs[write, 0] = lo
                    ivs[write, 1] = cut_lo
                    write += 1
                if hi > cut_hi and write < cap:
                    ivs[write, 0] = cut_hi
                    ivs[write, 1] = hi
                    write += 1
        return write if changed else n

    @_numba.njit(cache=True)
    def _nb_update_side(
        intervals_flat,  # (n_rows*n_cols, MAX_IV, 2) float64
        n_intervals,     # (n_rows*n_cols,)           int32
        row_centers,     # (n_rows,)                  float64
        j0, j1,          # inclusive row range
        k0,              # first col index (depth lower bound)
        cut_axis_val,    # xt (x-grid) or yt (y-grid)
        perp_val,        # yt (x-grid) or xt (y-grid)
        valid,           # (nk,) bool
        rz2_arr,         # (nk,) float64
        n_cols,          # column stride (g.ny)
    ):
        """JIT inner loop for one side-grid update pass."""
        nk = len(rz2_arr)
        for j in range(j0, j1 + 1):
            dp = row_centers[j] - perp_val
            dp2 = dp * dp
            for lk in range(nk):
                if not valid[lk]:
                    continue
                rz2 = rz2_arr[lk]
                if dp2 > rz2:
                    continue
                half = math.sqrt(rz2 - dp2)
                ray_idx = j * n_cols + (k0 + lk)
                n = n_intervals[ray_idx]
                n_intervals[ray_idx] = _nb_subtract_one(
                    intervals_flat[ray_idx],
                    n,
                    cut_axis_val - half,
                    cut_axis_val + half,
                )

    @_numba.njit(cache=True)
    def _nb_dense_voxel(
        intervals_flat,  # (n_rays, MAX_IV, 2)
        n_intervals,     # (n_rays,)
        n_rows, n_cols,
        depth_min, depth_max, n_depth,
        voxels,          # (n_rows, n_cols, n_depth) bool — pre-allocated
    ):
        inv = n_depth / (depth_max - depth_min)
        for i in range(n_rows):
            for j in range(n_cols):
                ray_idx = i * n_cols + j
                n = n_intervals[ray_idx]
                for iv in range(n):
                    lo = intervals_flat[ray_idx, iv, 0]
                    hi = intervals_flat[ray_idx, iv, 1]
                    k_lo = int((lo - depth_min) * inv)
                    k_hi = int(math.ceil((hi - depth_min) * inv))
                    if k_lo < 0:
                        k_lo = 0
                    if k_hi > n_depth:
                        k_hi = n_depth
                    for kk in range(k_lo, k_hi):
                        voxels[i, j, kk] = True

    @_numba.njit(cache=True)
    def _nb_batch_subtract_z(
        intervals,    # (n_rays, MAX_IV, 2) float64
        n_intervals,  # (n_rays,) int32
        height,       # (n_row, n_col) float64 — z-grid height cache
        ii,           # (n_cuts,) int64 — row indices
        jj,           # (n_cuts,) int64 — col indices
        z_cuts,       # (n_cuts,) float64
        z_max,        # float64
        ny,           # int — column stride
    ):
        """Batch Z-grid subtract with inline height-cache update."""
        n_cuts = len(ii)
        for idx in range(n_cuts):
            i = ii[idx]
            j = jj[idx]
            z_cut = z_cuts[idx]
            ray_idx = i * ny + j
            n = n_intervals[ray_idx]
            n_new = _nb_subtract_one(intervals[ray_idx], n, z_cut, z_max)
            n_intervals[ray_idx] = n_new
            if n_new > 0:
                height[i, j] = intervals[ray_idx, n_new - 1, 1]
            else:
                height[i, j] = math.nan

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False
    _nb_subtract_one = None       # type: ignore[assignment]
    _nb_update_side = None        # type: ignore[assignment]
    _nb_dense_voxel = None        # type: ignore[assignment]
    _nb_batch_subtract_z = None   # type: ignore[assignment]


def _py_subtract_one(ivs: np.ndarray, n: int, cut_lo: float, cut_hi: float) -> int:
    """Pure-Python fallback subtract for when numba is unavailable."""
    if cut_lo >= cut_hi:
        return n
    write = 0
    changed = False
    for k in range(n):
        lo = float(ivs[k, 0])
        hi = float(ivs[k, 1])
        if hi <= cut_lo or lo >= cut_hi:
            ivs[write, 0] = lo
            ivs[write, 1] = hi
            write += 1
        else:
            changed = True
            if lo < cut_lo:
                ivs[write, 0] = lo
                ivs[write, 1] = cut_lo
                write += 1
            if hi > cut_hi:
                ivs[write, 0] = cut_hi
                ivs[write, 1] = hi
                write += 1
    return write if changed else n


class NumpyDexelGrid:
    """
    Numpy/numba-backed dexel grid for X and Y side grids.

    Replaces ``ZDexelGrid(track_height=False)`` with a pure-numpy interval
    representation.  The hot inner loops for ``_update_x_grid`` and
    ``_update_y_grid`` are JIT-compiled by numba (if available), eliminating
    all Python-object overhead for the per-ray subtract calls.

    API is a drop-in superset of ``ZDexelGrid`` for the methods actually used
    on X/Y grids (no height tracking is performed).
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
        self.row_min = float(row_min)
        self.row_max = float(row_max)
        self.col_min = float(col_min)
        self.col_max = float(col_max)
        self.nx = int(n_row)
        self.ny = int(n_col)
        self.dx = (row_max - row_min) / n_row
        self.dy = (col_max - col_min) / n_col
        self.row_centers: np.ndarray = (
            row_min + (np.arange(n_row, dtype=np.float64) + 0.5) * self.dx
        )
        self.col_centers: np.ndarray = (
            col_min + (np.arange(n_col, dtype=np.float64) + 0.5) * self.dy
        )
        self._track_height = False

        n_rays = n_row * n_col
        self._intervals = np.zeros((n_rays, MAX_INTERVALS, 2), dtype=np.float64)
        self._n_intervals = np.zeros(n_rays, dtype=np.int32)
        # Sparse dict for surface metadata; populated only by paths that pass metadata.
        self._surface_metadata: dict[int, list] = {}

        self._rays_proxy: Optional[_RaysProxy] = None

    # ------------------------------------------------------------------
    # Coordinate helpers (same as ZDexelGrid)
    # ------------------------------------------------------------------

    def row_center(self, i: int) -> float:
        return float(self.row_centers[i])

    def col_center(self, j: int) -> float:
        return float(self.col_centers[j])

    def row_index(self, v: float) -> int:
        return int((v - self.row_min) / self.dx)

    def col_index(self, v: float) -> int:
        return int((v - self.col_min) / self.dy)

    def clamp_row(self, i: int) -> int:
        return max(0, min(self.nx - 1, i))

    def clamp_col(self, j: int) -> int:
        return max(0, min(self.ny - 1, j))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_stock(self, depth_lo: float, depth_hi: float) -> None:
        self._intervals[:, 0, 0] = depth_lo
        self._intervals[:, 0, 1] = depth_hi
        self._n_intervals[:] = 1
        self._surface_metadata.clear()
        self._rays_proxy = None  # invalidate cached proxy

    # ------------------------------------------------------------------
    # Hot path: full side-grid update (called by SimulationEngine)
    # ------------------------------------------------------------------

    def update_side_grid(
        self,
        j0: int,
        j1: int,
        k0: int,
        cut_axis_val: float,
        perp_val: float,
        valid: np.ndarray,
        rz2_arr: np.ndarray,
    ) -> None:
        """Subtract tool footprint from the inner j×k loop.

        This is the numba-accelerated hot path.  Both ``valid`` and
        ``rz2_arr`` must be 1-D float64/bool arrays of length ``k1-k0+1``.
        """
        if _HAS_NUMBA:
            _nb_update_side(
                self._intervals,
                self._n_intervals,
                self.row_centers,
                j0, j1, k0,
                float(cut_axis_val),
                float(perp_val),
                valid, rz2_arr,
                self.ny,
            )
        else:
            nk = len(rz2_arr)
            for j in range(j0, j1 + 1):
                dp = self.row_centers[j] - perp_val
                dp2 = dp * dp
                for lk in range(nk):
                    if not valid[lk]:
                        continue
                    rz2 = float(rz2_arr[lk])
                    if dp2 > rz2:
                        continue
                    half = math.sqrt(rz2 - dp2)
                    ray_idx = j * self.ny + (k0 + lk)
                    n = int(self._n_intervals[ray_idx])
                    self._n_intervals[ray_idx] = _py_subtract_one(
                        self._intervals[ray_idx], n,
                        float(cut_axis_val) - half,
                        float(cut_axis_val) + half,
                    )

    # ------------------------------------------------------------------
    # Per-ray subtract (used by SweptVolume engine and proxy writes)
    # ------------------------------------------------------------------

    def subtract_at(
        self,
        i: int,
        j: int,
        cut_lo: float,
        cut_hi: float,
        metadata=None,
    ) -> None:
        ray_idx = i * self.ny + j
        n = int(self._n_intervals[ray_idx])
        if _HAS_NUMBA:
            self._n_intervals[ray_idx] = _nb_subtract_one(
                self._intervals[ray_idx], n, cut_lo, cut_hi
            )
        else:
            self._n_intervals[ray_idx] = _py_subtract_one(
                self._intervals[ray_idx], n, cut_lo, cut_hi
            )
        if metadata is not None:
            if ray_idx not in self._surface_metadata:
                self._surface_metadata[ray_idx] = []
            self._surface_metadata[ray_idx].append(metadata)

    # ------------------------------------------------------------------
    # Voxelization
    # ------------------------------------------------------------------

    def to_dense_voxel(
        self, depth_min: float, depth_max: float, n_depth: int
    ) -> np.ndarray:
        voxels = np.zeros((self.nx, self.ny, n_depth), dtype=bool)
        if _HAS_NUMBA:
            _nb_dense_voxel(
                self._intervals, self._n_intervals,
                self.nx, self.ny,
                float(depth_min), float(depth_max), int(n_depth),
                voxels,
            )
        else:
            d_step = (depth_max - depth_min) / n_depth
            for i in range(self.nx):
                for j in range(self.ny):
                    ray_idx = i * self.ny + j
                    n = int(self._n_intervals[ray_idx])
                    for iv in range(n):
                        lo = float(self._intervals[ray_idx, iv, 0])
                        hi = float(self._intervals[ray_idx, iv, 1])
                        k_lo = max(0, int((lo - depth_min) / d_step))
                        k_hi = min(n_depth, math.ceil((hi - depth_min) / d_step))
                        voxels[i, j, k_lo:k_hi] = True
        return voxels

    # ------------------------------------------------------------------
    # No-op height tracking (keeps API compatible with ZDexelGrid)
    # ------------------------------------------------------------------

    def height_map(self) -> np.ndarray:
        return np.full((self.nx, self.ny), np.nan)

    def sync_height_cache(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Backward-compatible rays proxy (read path for contains_point etc.)
    # ------------------------------------------------------------------

    @property
    def rays(self) -> "_RaysProxy":
        if self._rays_proxy is None:
            self._rays_proxy = _RaysProxy(
                self._intervals, self._n_intervals, self._surface_metadata,
                self.nx, self.ny,
            )
        return self._rays_proxy


# ---------------------------------------------------------------------------
# Lightweight proxy objects for backward-compatible read access
# ---------------------------------------------------------------------------

class _RaysProxy:
    __slots__ = ("_ivs", "_niv", "_meta", "_nx", "_ny")

    def __init__(
        self,
        intervals: np.ndarray,
        n_intervals: np.ndarray,
        surface_metadata: dict,
        nx: int,
        ny: int,
    ) -> None:
        self._ivs = intervals
        self._niv = n_intervals
        self._meta = surface_metadata
        self._nx = nx
        self._ny = ny

    def __getitem__(self, i: int) -> "_RowProxy":
        if i < 0 or i >= self._nx:
            raise IndexError(i)
        return _RowProxy(self._ivs, self._niv, self._meta, i, self._ny)

    def __len__(self) -> int:
        return self._nx

    def __iter__(self):
        for i in range(self._nx):
            yield self[i]


class _RowProxy:
    __slots__ = ("_ivs", "_niv", "_meta", "_row", "_ny")

    def __init__(
        self,
        intervals: np.ndarray,
        n_intervals: np.ndarray,
        surface_metadata: dict,
        row: int,
        ny: int,
    ) -> None:
        self._ivs = intervals
        self._niv = n_intervals
        self._meta = surface_metadata
        self._row = row
        self._ny = ny

    def __getitem__(self, j: int) -> "_RayView":
        if j < 0 or j >= self._ny:
            raise IndexError(j)
        return _RayView(self._ivs, self._niv, self._meta, self._row * self._ny + j)

    def __len__(self) -> int:
        return self._ny

    def __iter__(self):
        for j in range(self._ny):
            yield self[j]


class _RayView:
    """Read/write view of a single ray stored in a NumpyDexelGrid."""

    __slots__ = ("_ivs", "_niv", "_meta", "_idx")

    def __init__(
        self,
        intervals: np.ndarray,
        n_intervals: np.ndarray,
        surface_metadata: dict,
        ray_idx: int,
    ) -> None:
        self._ivs = intervals
        self._niv = n_intervals
        self._meta = surface_metadata
        self._idx = ray_idx

    @property
    def surface_metadata(self) -> list:
        return self._meta.get(self._idx, [])

    @property
    def intervals(self) -> list:
        n = int(self._niv[self._idx])
        return [
            (float(self._ivs[self._idx, k, 0]), float(self._ivs[self._idx, k, 1]))
            for k in range(n)
        ]

    @intervals.setter
    def intervals(self, value: list) -> None:
        n = len(value)
        if n > MAX_INTERVALS:
            raise ValueError(f"Too many intervals: {n} > MAX_INTERVALS={MAX_INTERVALS}")
        for k, (lo, hi) in enumerate(value):
            self._ivs[self._idx, k, 0] = lo
            self._ivs[self._idx, k, 1] = hi
        self._niv[self._idx] = n

    def set_solid(self, lo: float, hi: float) -> None:
        self._ivs[self._idx, 0, 0] = lo
        self._ivs[self._idx, 0, 1] = hi
        self._niv[self._idx] = 1

    def subtract(self, cut_lo: float, cut_hi: float, metadata=None) -> None:
        n = int(self._niv[self._idx])
        if _HAS_NUMBA:
            self._niv[self._idx] = _nb_subtract_one(
                self._ivs[self._idx], n, cut_lo, cut_hi
            )
        else:
            self._niv[self._idx] = _py_subtract_one(
                self._ivs[self._idx], n, cut_lo, cut_hi
            )

    def contains(self, value: float) -> bool:
        n = int(self._niv[self._idx])
        for k in range(n):
            lo = float(self._ivs[self._idx, k, 0])
            if lo > value:
                return False  # intervals are sorted; nothing later can match
            if value <= float(self._ivs[self._idx, k, 1]):
                return True
        return False

    def top(self):
        n = int(self._niv[self._idx])
        if n == 0:
            return None
        return float(self._ivs[self._idx, n - 1, 1])

    def bottom(self):
        n = int(self._niv[self._idx])
        if n == 0:
            return None
        return float(self._ivs[self._idx, 0, 0])

    def is_empty(self) -> bool:
        return int(self._niv[self._idx]) == 0

    def __repr__(self) -> str:
        return f"_RayView({self.intervals})"


# ---------------------------------------------------------------------------
# Z-direction variant: adds height-map tracking to NumpyDexelGrid
# ---------------------------------------------------------------------------

class NumpyZDexelGrid(NumpyDexelGrid):
    """
    NumpyDexelGrid with an incremental height-map cache.

    Used as the Z-grid in TriDexelStock.  The ``_height`` array mirrors
    ``ZDexelGrid._height`` so the simulation engine can read it directly via
    ``g._height[ii, jj]`` without any API change.

    The hot subtract path (``batch_subtract``) runs entirely inside a single
    numba JIT call, eliminating all Python-object overhead from the inner loop.
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
        super().__init__(row_min, row_max, col_min, col_max, n_row, n_col)
        self._height: np.ndarray = np.full((n_row, n_col), np.nan)
        self._track_height = True

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_stock(self, depth_lo: float, depth_hi: float) -> None:
        super().initialize_stock(depth_lo, depth_hi)
        self._height[:] = depth_hi

    # ------------------------------------------------------------------
    # Per-ray subtract (single-cell path; also used by proxy writes)
    # ------------------------------------------------------------------

    def subtract_at(
        self,
        i: int,
        j: int,
        cut_lo: float,
        cut_hi: float,
        metadata=None,
    ) -> None:
        super().subtract_at(i, j, cut_lo, cut_hi, metadata)
        ray_idx = i * self.ny + j
        n = int(self._n_intervals[ray_idx])
        self._height[i, j] = (
            float(self._intervals[ray_idx, n - 1, 1]) if n > 0 else np.nan
        )

    # ------------------------------------------------------------------
    # Batch subtract — the main hot path called by SimulationEngine
    # ------------------------------------------------------------------

    def batch_subtract(
        self,
        ii: np.ndarray,
        jj: np.ndarray,
        z_cuts: np.ndarray,
        z_max: float,
    ) -> None:
        """JIT-compiled batch subtract with inline height-cache update.

        Eliminates the Python for-loop that was the largest remaining hotspot
        in the Z-grid update path.
        """
        if len(ii) == 0:
            return
        ii64 = np.asarray(ii, dtype=np.int64)
        jj64 = np.asarray(jj, dtype=np.int64)
        zc64 = np.ascontiguousarray(z_cuts, dtype=np.float64)
        if _HAS_NUMBA:
            _nb_batch_subtract_z(
                self._intervals,
                self._n_intervals,
                self._height,
                ii64,
                jj64,
                zc64,
                float(z_max),
                self.ny,
            )
        else:
            for i, j, z_cut in zip(ii64, jj64, zc64):
                self.subtract_at(int(i), int(j), float(z_cut), float(z_max))

    # ------------------------------------------------------------------
    # Height-map access (replaces ZDexelGrid.height_map)
    # ------------------------------------------------------------------

    def height_map(self) -> np.ndarray:
        return self._height.copy()

    def sync_height_cache(self) -> None:
        """Rebuild height cache from numpy interval arrays (called after bulk writes)."""
        for i in range(self.nx):
            for j in range(self.ny):
                ray_idx = i * self.ny + j
                n = int(self._n_intervals[ray_idx])
                self._height[i, j] = (
                    float(self._intervals[ray_idx, n - 1, 1]) if n > 0 else np.nan
                )
