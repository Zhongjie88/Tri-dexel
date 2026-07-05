from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Triangle:
    vertices: np.ndarray
    normal: np.ndarray
    source: str = "surface"
    component_id: int | None = None

    def __init__(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        normal: np.ndarray | None = None,
        source: str = "surface",
        component_id: int | None = None,
    ) -> None:
        verts = np.asarray([v0, v1, v2], dtype=float)
        if verts.shape != (3, 3):
            raise ValueError("triangle vertices must have shape (3, 3)")
        if normal is None:
            normal_arr = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        else:
            normal_arr = np.asarray(normal, dtype=float)
        n_norm = np.linalg.norm(normal_arr)
        if n_norm > 1e-12:
            normal_arr = normal_arr / n_norm
        object.__setattr__(self, "vertices", verts)
        object.__setattr__(self, "normal", normal_arr)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "_bounds_lo", verts.min(axis=0))
        object.__setattr__(self, "_bounds_hi", verts.max(axis=0))

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._bounds_lo, self._bounds_hi

    def ray_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        eps: float = 1e-9,
    ) -> tuple[float, np.ndarray] | None:
        """Return distance and interpolated normal for a ray hit, if any."""
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        v0, v1, v2 = self.vertices
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = np.cross(d, e2)
        det = float(np.dot(e1, pvec))
        if abs(det) < eps:
            return None
        inv_det = 1.0 / det
        tvec = o - v0
        u = float(np.dot(tvec, pvec) * inv_det)
        if u < -eps or u > 1.0 + eps:
            return None
        qvec = np.cross(tvec, e1)
        v = float(np.dot(d, qvec) * inv_det)
        if v < -eps or u + v > 1.0 + eps:
            return None
        t = float(np.dot(e2, qvec) * inv_det)
        if t < -eps:
            return None
        return t, self.normal.copy()
