from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

COMPONENT_BOTTOM = 0
COMPONENT_SIDE = 1
COMPONENT_BALL = 2
COMPONENT_CORNER = 3
COMPONENT_SHANK = 4
COMPONENT_HOLDER = 5


class ToolComponentRole(Enum):
    """Engineering role of a tool component in machining simulation."""

    CUTTING = "cutting"
    COLLISION_ONLY = "collision_only"
    VISUAL_ONLY = "visual_only"


@dataclass(frozen=True)
class ToolComponent:
    """Semantic role of a sampled tool surface component."""

    component_id: int
    name: str
    surface_role: str
    cutting: bool
    collision: bool
    role: ToolComponentRole = ToolComponentRole.CUTTING


TOOL_COMPONENTS = {
    COMPONENT_BOTTOM: ToolComponent(
        COMPONENT_BOTTOM,
        name="bottom",
        surface_role="bottom",
        cutting=True,
        collision=True,
        role=ToolComponentRole.CUTTING,
    ),
    COMPONENT_SIDE: ToolComponent(
        COMPONENT_SIDE,
        name="side",
        surface_role="side",
        cutting=True,
        collision=True,
        role=ToolComponentRole.CUTTING,
    ),
    COMPONENT_BALL: ToolComponent(
        COMPONENT_BALL,
        name="ball_tip",
        surface_role="ball",
        cutting=True,
        collision=True,
        role=ToolComponentRole.CUTTING,
    ),
    COMPONENT_CORNER: ToolComponent(
        COMPONENT_CORNER,
        name="corner",
        surface_role="corner",
        cutting=True,
        collision=True,
        role=ToolComponentRole.CUTTING,
    ),
    COMPONENT_SHANK: ToolComponent(
        COMPONENT_SHANK,
        name="shank",
        surface_role="shank",
        cutting=False,
        collision=True,
        role=ToolComponentRole.COLLISION_ONLY,
    ),
    COMPONENT_HOLDER: ToolComponent(
        COMPONENT_HOLDER,
        name="holder",
        surface_role="holder",
        cutting=False,
        collision=True,
        role=ToolComponentRole.COLLISION_ONLY,
    ),
}


def component_for_id(component_id: int) -> ToolComponent:
    try:
        return TOOL_COMPONENTS[int(component_id)]
    except KeyError as exc:
        raise ValueError(f"unknown tool component id: {component_id}") from exc


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


@dataclass(frozen=True)
class ToolSurfaceSample:
    """Sampled tool surface with local parameters for each point."""

    points: np.ndarray
    normals: np.ndarray
    component_ids: np.ndarray
    parameters: np.ndarray
    cutting_mask: np.ndarray
    collision_mask: np.ndarray

    def __iter__(self):
        yield self.points
        yield self.normals
        yield self.component_ids

    def filtered(
        self,
        *,
        cutting: bool | None = None,
        collision: bool | None = None,
    ) -> "ToolSurfaceSample":
        mask = np.ones(self.points.shape[0], dtype=bool)
        if cutting is not None:
            mask &= self.cutting_mask == bool(cutting)
        if collision is not None:
            mask &= self.collision_mask == bool(collision)
        return ToolSurfaceSample(
            points=self.points[mask].copy(),
            normals=self.normals[mask].copy(),
            component_ids=self.component_ids[mask].copy(),
            parameters=self.parameters[mask].copy(),
            cutting_mask=self.cutting_mask[mask].copy(),
            collision_mask=self.collision_mask[mask].copy(),
        )

    def cutting_surface(self) -> "ToolSurfaceSample":
        return self.filtered(cutting=True)

    def collision_surface(self) -> "ToolSurfaceSample":
        return self.filtered(collision=True)


class ToolGeometry(ABC):
    """
    Abstract tool geometry.

    All tools are assumed to be axis-symmetric, rotating about Z (the tool axis).
    'tip_z' is the Z coordinate of the lowest point of the tool.

    Two queries drive the simulation:
      z_cut(d, tip_z)               — for Z-dexel updates
      cross_section_radius(zk, tip_z) — for X/Y-dexel updates
    """

    @property
    @abstractmethod
    def radius(self) -> float:
        """Maximum radial extent of the cutting envelope."""

    @property
    def cutting_length(self) -> float:
        """Effective axial cutting length measured from the tool tip along +Z."""
        return math.inf

    @property
    def overall_length(self) -> float:
        """Overall modelled tool length measured from the tool tip along +Z."""
        return self.cutting_length

    def cutting_z_range(self, tip_z: float) -> tuple[float, float]:
        """World Z range of cutting geometry for a vertical tool pose."""
        return float(tip_z), float(tip_z) + float(self.cutting_length)

    def cutting_top_z(self, tip_z: float) -> float:
        """Highest world Z reached by cutting geometry for a vertical pose."""
        return self.cutting_z_range(tip_z)[1]

    def get_cutting_components(self) -> tuple[ToolComponent, ...]:
        return tuple(component for component in TOOL_COMPONENTS.values() if component.cutting)

    def get_collision_components(self) -> tuple[ToolComponent, ...]:
        return tuple(component for component in TOOL_COMPONENTS.values() if component.collision)

    @abstractmethod
    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        """
        Z coordinate of the lowest tool surface at radial offset d from the axis.
        Returns None if d is outside the tool's cutting footprint.

        Material above z_cut at this radial offset is removed.
        """

    @abstractmethod
    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        """
        Cutting radius of the tool at height zk.
        Returns None if zk is below the tool tip (no tool material there).

        Used for X- and Y-dexel updates: at a given height, the tool cuts a
        circular cross section of this radius.
        """

    # ------------------------------------------------------------------
    # Vectorised helpers (override in subclasses for speed)
    # ------------------------------------------------------------------

    def z_cut_arr(self, d_arr: np.ndarray, tip_z: float) -> np.ndarray:
        """Vectorised z_cut; NaN where d is outside the tool footprint."""
        out = np.full(d_arr.shape, np.nan)
        for idx in np.ndindex(d_arr.shape):
            zc = self.z_cut(float(d_arr[idx]), tip_z)
            if zc is not None:
                out[idx] = zc
        return out

    def cross_section_radius_arr(
        self, zk_arr: np.ndarray, tip_z: float
    ) -> tuple:
        """Vectorised cross_section_radius.

        Returns (valid, rz) where both are 1-D arrays of length len(zk_arr).
        valid[k] is True when zk_arr[k] is within the tool body.
        rz[k] is the cross-section radius (0 where not valid).
        """
        n = len(zk_arr)
        valid = np.zeros(n, dtype=bool)
        rz    = np.zeros(n, dtype=float)
        for k, zk in enumerate(zk_arr):
            r = self.cross_section_radius(float(zk), tip_z)
            if r is not None:
                valid[k] = True
                rz[k]    = r
        return valid, rz

    # ------------------------------------------------------------------
    # Surface sampling for swept-volume based simulation
    # ------------------------------------------------------------------

    def sample_surface(
        self,
        n_u: int = 32,
        n_v: int = 16,
        include_shank: bool = True,
        include_non_cutting: bool = True,
        **legacy_kwargs,
    ) -> ToolSurfaceSample | dict[str, np.ndarray]:
        raise NotImplementedError

    def transform_surface(self, pose, **sample_kwargs):
        surface = self.sample_surface(**sample_kwargs)
        if isinstance(surface, dict):
            return {
                "points": pose.transform_points(surface["points"]),
                "normals": pose.transform_normals(surface["normals"]),
                "component_ids": surface.get("component_ids"),
            }
        normals_world = pose.transform_normals(surface.normals)
        lengths = np.linalg.norm(normals_world, axis=1)
        valid = lengths > 1e-12
        normals_world[valid] /= lengths[valid, np.newaxis]
        return ToolSurfaceSample(
            points=pose.transform_points(surface.points),
            normals=normals_world,
            component_ids=surface.component_ids.copy(),
            parameters=surface.parameters.copy(),
            cutting_mask=surface.cutting_mask.copy(),
            collision_mask=surface.collision_mask.copy(),
        )

    @staticmethod
    def _validate_surface(
        points: np.ndarray,
        normals: np.ndarray,
        component_ids: np.ndarray,
        parameters: np.ndarray,
        include_non_cutting: bool = True,
    ) -> ToolSurfaceSample:
        points = np.asarray(points, dtype=float)
        normals = np.asarray(normals, dtype=float)
        component_ids = np.asarray(component_ids, dtype=int)
        parameters = np.asarray(parameters, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if normals.shape != points.shape:
            raise ValueError("normals must have the same shape as points")
        if component_ids.shape != (points.shape[0],):
            raise ValueError("component_ids must have shape (N,)")
        if parameters.shape != (points.shape[0], 2):
            raise ValueError("parameters must have shape (N, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points contain NaN or Inf")
        if not np.all(np.isfinite(normals)):
            raise ValueError("normals contain NaN or Inf")
        if not np.all(np.isfinite(parameters)):
            raise ValueError("parameters contain NaN or Inf")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("surface normals must be non-zero")
        normals = normals / lengths[:, np.newaxis]
        cutting_mask = np.array(
            [component_for_id(cid).cutting for cid in component_ids],
            dtype=bool,
        )
        collision_mask = np.array(
            [component_for_id(cid).collision for cid in component_ids],
            dtype=bool,
        )
        if not include_non_cutting:
            mask = cutting_mask
            points = points[mask]
            normals = normals[mask]
            component_ids = component_ids[mask]
            parameters = parameters[mask]
            cutting_mask = cutting_mask[mask]
            collision_mask = collision_mask[mask]
        return ToolSurfaceSample(
            points,
            normals,
            component_ids,
            parameters,
            cutting_mask,
            collision_mask,
        )


# ---------------------------------------------------------------------------
# Concrete tool types
# ---------------------------------------------------------------------------


class FlatEndMill(ToolGeometry):
    """
    Flat-bottom end mill (square end).

    Cutting geometry:
      - Flat disk of radius r at z = tip_z
      - Cylindrical shank above
    """

    def __init__(
        self,
        radius: float,
        height: float | None = None,
        cutting_length: float | None = None,
        overall_length: float | None = None,
        shank_diameter: float | None = None,
    ) -> None:
        self._radius = _positive_float(radius, "radius")
        if cutting_length is None and height is None:
            # Backward compatibility: older callers only supplied a radius and
            # expected the vertical analytical cutter to remove material up to
            # the current stock top.  Keep that behaviour until a flute length
            # is explicitly supplied.
            self._cutting_length = math.inf
            self._height = 2.0 * self._radius
        else:
            if cutting_length is None:
                cutting_length = height
            self._cutting_length = _positive_float(cutting_length, "cutting_length")
            self._height = self._cutting_length  # backward-compatible alias
        if overall_length is None:
            overall_length = self._height
        self._overall_length = _positive_float(overall_length, "overall_length")
        if math.isfinite(self._cutting_length) and self._overall_length < self._cutting_length:
            raise ValueError("overall_length must be >= cutting_length")
        if shank_diameter is None:
            self._shank_radius = self._radius
        else:
            self._shank_radius = 0.5 * _positive_float(shank_diameter, "shank_diameter")

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def cutting_length(self) -> float:
        return self._cutting_length

    @property
    def overall_length(self) -> float:
        return self._overall_length

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        if d > self._radius:
            return None
        return tip_z

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        if zk < tip_z or zk > tip_z + self._cutting_length:
            return None
        return self._radius

    def z_cut_arr(self, d_arr: np.ndarray, tip_z: float) -> np.ndarray:
        out = np.full(d_arr.shape, np.nan)
        out[d_arr <= self._radius] = tip_z
        return out

    def cross_section_radius_arr(self, zk_arr: np.ndarray, tip_z: float) -> tuple:
        valid = (zk_arr >= tip_z) & (zk_arr <= tip_z + self._cutting_length)
        rz    = np.where(valid, self._radius, 0.0)
        return valid, rz

    def sample_surface(
        self,
        n_u: int = 32,
        n_v: int = 16,
        include_shank: bool = True,
        include_non_cutting: bool = True,
        **legacy_kwargs,
    ) -> ToolSurfaceSample | dict[str, np.ndarray]:
        if "radial_segments" in legacy_kwargs or "axial_segments" in legacy_kwargs:
            radial_segments = legacy_kwargs.get("radial_segments", n_u)
            axial_segments = legacy_kwargs.get("axial_segments", n_v)
            height = legacy_kwargs.get("height", self._height)
            if not include_non_cutting:
                height = min(float(height), self._height)
            return self._legacy_ring_surface(radial_segments, axial_segments, height)

        n_u = max(3, int(n_u))
        n_v = max(1, int(n_v))
        angles = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)
        rho_values = np.linspace(0.0, self._radius, n_v + 1)
        z_values = np.linspace(0.0, self._height, n_v + 1)

        points = []
        normals = []
        component_ids = []
        parameters = []

        for rho in rho_values:
            for a in angles:
                points.append((rho * math.cos(a), rho * math.sin(a), 0.0))
                normals.append((0.0, 0.0, -1.0))
                component_ids.append(COMPONENT_BOTTOM)
                parameters.append((a, rho))

        for z in z_values:
            for a in angles:
                ca = math.cos(a)
                sa = math.sin(a)
                points.append((self._radius * ca, self._radius * sa, z))
                normals.append((ca, sa, 0.0))
                component_ids.append(COMPONENT_SIDE)
                parameters.append((a, z))

        if include_shank and self._overall_length > self._height:
            z_values = np.linspace(self._height, self._overall_length, n_v + 1)
            for z in z_values:
                for a in angles:
                    ca = math.cos(a)
                    sa = math.sin(a)
                    points.append((self._shank_radius * ca, self._shank_radius * sa, z))
                    normals.append((ca, sa, 0.0))
                    component_ids.append(COMPONENT_SHANK)
                    parameters.append((a, z))

        return self._validate_surface(
            np.asarray(points),
            np.asarray(normals),
            np.asarray(component_ids),
            np.asarray(parameters),
            include_non_cutting=include_non_cutting,
        )

    def _legacy_ring_surface(
        self,
        radial_segments: int,
        axial_segments: int,
        height: float,
    ) -> dict[str, np.ndarray]:
        radial_segments = max(3, int(radial_segments))
        axial_segments = max(1, int(axial_segments))
        angles = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False)
        z_values = np.linspace(0.0, float(height), axial_segments + 1)
        points = np.zeros((radial_segments, axial_segments + 1, 3), dtype=float)
        normals = np.zeros_like(points)
        component_ids = np.full(
            (radial_segments, axial_segments + 1),
            COMPONENT_SIDE,
            dtype=int,
        )
        for i, a in enumerate(angles):
            ca = math.cos(a)
            sa = math.sin(a)
            points[i, :, 0] = self._radius * ca
            points[i, :, 1] = self._radius * sa
            points[i, :, 2] = z_values
            normals[i, :, 0] = ca
            normals[i, :, 1] = sa
        return {"points": points, "normals": normals, "component_ids": component_ids}


class BallEndMill(ToolGeometry):
    """
    Ball-nose end mill.

    Cutting geometry:
      - Hemisphere of radius r, center at (0, 0, tip_z + r)
      - Tip at (0, 0, tip_z)
      - Cylindrical shank above equator (z > tip_z + r)
    """

    def __init__(
        self,
        radius: float,
        height: float | None = None,
        cutting_length: float | None = None,
        overall_length: float | None = None,
        shank_diameter: float | None = None,
    ) -> None:
        self._radius = _positive_float(radius, "radius")
        self._height = (
            _positive_float(height, "height")
            if height is not None
            else 2.0 * self._radius
        )
        if cutting_length is None:
            # Backward compatibility for legacy analytic side-grid cutting:
            # old BallEndMill(radius) behaved as if the cylindrical section
            # above the ball could cut indefinitely.  Explicit cutting_length
            # enables machining-aware flute limits.
            self._cutting_length = math.inf
            self._surface_cutting_length = self._radius
        else:
            self._cutting_length = max(
                self._radius,
                _positive_float(cutting_length, "cutting_length"),
            )
            self._surface_cutting_length = self._cutting_length
        if overall_length is None:
            overall_length = self._radius + self._height
        self._overall_length = _positive_float(overall_length, "overall_length")
        if math.isfinite(self._cutting_length) and self._overall_length < self._cutting_length:
            raise ValueError("overall_length must be >= cutting_length")
        if shank_diameter is None:
            self._shank_radius = self._radius
        else:
            self._shank_radius = 0.5 * _positive_float(shank_diameter, "shank_diameter")

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def cutting_length(self) -> float:
        return self._cutting_length

    @property
    def overall_length(self) -> float:
        return self._overall_length

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        r = self._radius
        if d > r:
            return None
        # z_cut at radial offset d: lowest point of sphere surface
        return tip_z + r - math.sqrt(r * r - d * d)

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        r = self._radius
        if zk < tip_z:
            return None
        if zk <= tip_z + r:
            # Ball zone: cross-section radius shrinks toward tip
            val = r * r - (zk - tip_z - r) ** 2
            return math.sqrt(max(0.0, val))
        if zk <= tip_z + self._cutting_length:
            # Optional side flute above the ball equator.
            return r
        return None

    def z_cut_arr(self, d_arr: np.ndarray, tip_z: float) -> np.ndarray:
        r = self._radius
        in_foot = d_arr <= r
        d2 = d_arr * d_arr
        out = np.full(d_arr.shape, np.nan)
        out[in_foot] = tip_z + r - np.sqrt(np.maximum(0.0, r*r - d2[in_foot]))
        return out

    def cross_section_radius_arr(self, zk_arr: np.ndarray, tip_z: float) -> tuple:
        r = self._radius
        ball_zone  = (zk_arr >= tip_z) & (zk_arr <= tip_z + r)
        side_zone = (zk_arr > tip_z + r) & (zk_arr <= tip_z + self._cutting_length)
        valid = ball_zone | side_zone
        rz = np.zeros(len(zk_arr), dtype=float)
        if np.any(ball_zone):
            dz = zk_arr[ball_zone] - (tip_z + r)
            rz[ball_zone] = np.sqrt(np.maximum(0.0, r*r - dz*dz))
        rz[side_zone] = r
        return valid, rz

    def sample_surface(
        self,
        n_u: int = 32,
        n_v: int = 16,
        include_shank: bool = True,
        include_non_cutting: bool = True,
        **legacy_kwargs,
    ) -> ToolSurfaceSample | dict[str, np.ndarray]:
        if "radial_segments" in legacy_kwargs or "axial_segments" in legacy_kwargs:
            radial_segments = legacy_kwargs.get("radial_segments", n_u)
            axial_segments = legacy_kwargs.get("axial_segments", n_v)
            height = legacy_kwargs.get("height", self._overall_length)
            if not include_non_cutting:
                height = self._surface_cutting_length
            return self._legacy_ring_surface(radial_segments, axial_segments, height)

        n_u = max(3, int(n_u))
        n_v = max(2, int(n_v))
        angles = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)
        theta_values = np.linspace(0.0, 0.5 * math.pi, n_v + 1)

        points = []
        normals = []
        component_ids = []
        parameters = []

        for theta in theta_values:
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            for phi in angles:
                x = self._radius * sin_t * math.cos(phi)
                y = self._radius * sin_t * math.sin(phi)
                z = self._radius - self._radius * cos_t
                points.append((x, y, z))
                normals.append((x, y, z - self._radius))
                component_ids.append(COMPONENT_BALL)
                parameters.append((phi, theta))

        if self._surface_cutting_length > self._radius:
            z_values = np.linspace(self._radius, self._surface_cutting_length, n_v + 1)
            for z in z_values:
                for phi in angles:
                    ca = math.cos(phi)
                    sa = math.sin(phi)
                    points.append((self._radius * ca, self._radius * sa, z))
                    normals.append((ca, sa, 0.0))
                    component_ids.append(COMPONENT_SIDE)
                    parameters.append((phi, z))

        if include_shank and self._overall_length > self._surface_cutting_length:
            z_values = np.linspace(self._surface_cutting_length, self._overall_length, n_v + 1)
            for z in z_values:
                for phi in angles:
                    ca = math.cos(phi)
                    sa = math.sin(phi)
                    points.append((self._shank_radius * ca, self._shank_radius * sa, z))
                    normals.append((ca, sa, 0.0))
                    component_ids.append(COMPONENT_SHANK)
                    parameters.append((phi, z))

        return self._validate_surface(
            np.asarray(points),
            np.asarray(normals),
            np.asarray(component_ids),
            np.asarray(parameters),
            include_non_cutting=include_non_cutting,
        )

    def _legacy_ring_surface(
        self,
        radial_segments: int,
        axial_segments: int,
        height: float,
    ) -> dict[str, np.ndarray]:
        radial_segments = max(3, int(radial_segments))
        axial_segments = max(2, int(axial_segments))
        angles = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False)
        z_values = np.linspace(0.0, float(height), axial_segments + 1)
        points = np.zeros((radial_segments, axial_segments + 1, 3), dtype=float)
        normals = np.zeros_like(points)
        component_ids = np.full(
            (radial_segments, axial_segments + 1),
            COMPONENT_SHANK,
            dtype=int,
        )
        for j, z in enumerate(z_values):
            rr = self.cross_section_radius(float(z), 0.0)
            if rr is None:
                rr = 0.0
            for i, a in enumerate(angles):
                ca = math.cos(a)
                sa = math.sin(a)
                points[i, j] = (rr * ca, rr * sa, z)
                if z <= self._radius:
                    component_ids[i, j] = COMPONENT_BALL
                    n = np.array([rr * ca, rr * sa, z - self._radius])
                    n_norm = np.linalg.norm(n)
                    normals[i, j] = n / n_norm if n_norm > 1e-12 else (0.0, 0.0, -1.0)
                else:
                    component_ids[i, j] = (
                        COMPONENT_SIDE
                        if z <= self._cutting_length
                        else COMPONENT_SHANK
                    )
                    normals[i, j] = (ca, sa, 0.0)
        return {"points": points, "normals": normals, "component_ids": component_ids}


class BullNoseEndMill(ToolGeometry):
    """
    Bull-nose (toroidal) end mill: flat center with rounded corner.

    radius       — outer radius of the tool
    corner_radius — radius of the corner fillet
    flat_radius   = radius - corner_radius (flat region)
    """

    def __init__(
        self,
        radius: float,
        corner_radius: float,
        height: float | None = None,
        shank_height: float | None = None,
    ) -> None:
        radius = _positive_float(radius, "radius")
        corner_radius = _positive_float(corner_radius, "corner_radius")
        if corner_radius >= radius:
            raise ValueError("corner_radius must be less than radius")
        self._radius = radius
        self._cr = corner_radius
        self._flat_r = radius - corner_radius
        self._height = (
            _positive_float(height, "height")
            if height is not None
            else 2.0 * radius
        )
        self._shank_height = (
            _positive_float(shank_height, "shank_height")
            if shank_height is not None
            else 2.0 * radius
        )

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def cutting_length(self) -> float:
        return self._cr + self._height

    @property
    def overall_length(self) -> float:
        return self._cr + self._height + self._shank_height

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        r, cr, fr = self._radius, self._cr, self._flat_r
        if d > r:
            return None
        if d <= fr:
            return tip_z
        # Corner fillet: torus cross-section
        offset = d - fr
        return tip_z + cr - math.sqrt(max(0.0, cr * cr - offset * offset))

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        r, cr, fr = self._radius, self._cr, self._flat_r
        if zk < tip_z:
            return None
        if zk > tip_z + self.cutting_length:
            return None
        if zk <= tip_z + cr:
            # Corner fillet zone: outer boundary shrinks
            dz = zk - tip_z
            outer = fr + math.sqrt(max(0.0, cr * cr - (dz - cr) ** 2))
            return outer
        return r

    def sample_surface(
        self,
        n_u: int = 32,
        n_v: int = 16,
        include_shank: bool = True,
        include_non_cutting: bool = True,
        **legacy_kwargs,
    ) -> ToolSurfaceSample | dict[str, np.ndarray]:
        n_u = max(3, int(n_u))
        n_v = max(2, int(n_v))
        angles = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)

        points = []
        normals = []
        component_ids = []
        parameters = []

        rho_values = np.linspace(0.0, self._flat_r, n_v + 1)
        for rho in rho_values:
            for phi in angles:
                points.append((rho * math.cos(phi), rho * math.sin(phi), 0.0))
                normals.append((0.0, 0.0, -1.0))
                component_ids.append(COMPONENT_BOTTOM)
                parameters.append((phi, rho))

        beta_values = np.linspace(0.0, 0.5 * math.pi, n_v + 1)
        for beta in beta_values:
            radial = self._flat_r + self._cr * math.sin(beta)
            z = self._cr - self._cr * math.cos(beta)
            nr = math.sin(beta)
            nz = -math.cos(beta)
            for phi in angles:
                ca = math.cos(phi)
                sa = math.sin(phi)
                points.append((radial * ca, radial * sa, z))
                normals.append((nr * ca, nr * sa, nz))
                component_ids.append(COMPONENT_CORNER)
                parameters.append((phi, beta))

        z_values = np.linspace(self._cr, self._cr + self._height, n_v + 1)
        for z in z_values:
            for phi in angles:
                ca = math.cos(phi)
                sa = math.sin(phi)
                points.append((self._radius * ca, self._radius * sa, z))
                normals.append((ca, sa, 0.0))
                component_ids.append(COMPONENT_SIDE)
                parameters.append((phi, z))

        if include_shank:
            z0 = self._cr + self._height
            z1 = z0 + self._shank_height
            for z in np.linspace(z0, z1, n_v + 1):
                for phi in angles:
                    ca = math.cos(phi)
                    sa = math.sin(phi)
                    points.append((self._radius * ca, self._radius * sa, z))
                    normals.append((ca, sa, 0.0))
                    component_ids.append(COMPONENT_SHANK)
                    parameters.append((phi, z))

        return self._validate_surface(
            np.asarray(points),
            np.asarray(normals),
            np.asarray(component_ids),
            np.asarray(parameters),
            include_non_cutting=include_non_cutting,
        )


class TaperTool(ToolGeometry):
    """Conical/tapered cutter with optional corner and shank surfaces."""

    def __init__(
        self,
        bottom_radius: float,
        top_radius: float,
        height: float,
        corner_radius: float = 0.0,
        shank_height: float | None = None,
    ) -> None:
        self._bottom_radius = _positive_float(bottom_radius, "bottom_radius")
        self._top_radius = _positive_float(top_radius, "top_radius")
        self._height = _positive_float(height, "height")
        corner_radius = float(corner_radius)
        if not math.isfinite(corner_radius) or corner_radius < 0.0:
            raise ValueError("corner_radius must be a finite non-negative number")
        if corner_radius >= self._bottom_radius:
            raise ValueError("corner_radius must be smaller than bottom_radius")
        self._corner_radius = corner_radius
        self._shank_height = (
            _positive_float(shank_height, "shank_height")
            if shank_height is not None
            else 2.0 * self._top_radius
        )

    @property
    def radius(self) -> float:
        return max(self._bottom_radius, self._top_radius)

    @property
    def cutting_length(self) -> float:
        return self._height

    @property
    def overall_length(self) -> float:
        return self._height + self._shank_height

    def _radius_at_local_z(self, z: float) -> float:
        t = min(1.0, max(0.0, z / self._height))
        return self._bottom_radius + (self._top_radius - self._bottom_radius) * t

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        if d > self.radius:
            return None
        if d <= self._bottom_radius:
            return tip_z
        if self._top_radius <= self._bottom_radius:
            return None
        t = (d - self._bottom_radius) / (self._top_radius - self._bottom_radius)
        if 0.0 <= t <= 1.0:
            return tip_z + t * self._height
        return None

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        if zk < tip_z:
            return None
        local_z = zk - tip_z
        if local_z <= self._height:
            return self._radius_at_local_z(local_z)
        return None

    def sample_surface(
        self,
        n_u: int = 32,
        n_v: int = 16,
        include_shank: bool = True,
        include_non_cutting: bool = True,
        **legacy_kwargs,
    ) -> ToolSurfaceSample | dict[str, np.ndarray]:
        n_u = max(3, int(n_u))
        n_v = max(2, int(n_v))
        angles = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)

        points = []
        normals = []
        component_ids = []
        parameters = []

        bottom_flat_radius = max(0.0, self._bottom_radius - self._corner_radius)
        for rho in np.linspace(0.0, bottom_flat_radius, n_v + 1):
            for phi in angles:
                points.append((rho * math.cos(phi), rho * math.sin(phi), 0.0))
                normals.append((0.0, 0.0, -1.0))
                component_ids.append(COMPONENT_BOTTOM)
                parameters.append((phi, rho))

        if self._corner_radius > 0.0:
            for beta in np.linspace(0.0, 0.5 * math.pi, n_v + 1):
                radial = bottom_flat_radius + self._corner_radius * math.sin(beta)
                z = self._corner_radius - self._corner_radius * math.cos(beta)
                nr = math.sin(beta)
                nz = -math.cos(beta)
                for phi in angles:
                    ca = math.cos(phi)
                    sa = math.sin(phi)
                    points.append((radial * ca, radial * sa, z))
                    normals.append((nr * ca, nr * sa, nz))
                    component_ids.append(COMPONENT_CORNER)
                    parameters.append((phi, beta))

        dr = self._top_radius - self._bottom_radius
        slope_len = math.hypot(dr, self._height)
        normal_r = self._height / slope_len
        normal_z = -dr / slope_len
        for z in np.linspace(0.0, self._height, n_v + 1):
            radial = self._radius_at_local_z(float(z))
            for phi in angles:
                ca = math.cos(phi)
                sa = math.sin(phi)
                points.append((radial * ca, radial * sa, z))
                normals.append((normal_r * ca, normal_r * sa, normal_z))
                component_ids.append(COMPONENT_SIDE)
                parameters.append((phi, z))

        if include_shank:
            z0 = self._height
            z1 = z0 + self._shank_height
            for z in np.linspace(z0, z1, n_v + 1):
                for phi in angles:
                    ca = math.cos(phi)
                    sa = math.sin(phi)
                    points.append((self._top_radius * ca, self._top_radius * sa, z))
                    normals.append((ca, sa, 0.0))
                    component_ids.append(COMPONENT_SHANK)
                    parameters.append((phi, z))

        return self._validate_surface(
            np.asarray(points),
            np.asarray(normals),
            np.asarray(component_ids),
            np.asarray(parameters),
            include_non_cutting=include_non_cutting,
        )


class ToolHolder(ToolGeometry):
    """Cylindrical holder/shank geometry for collision and gouging checks."""

    def __init__(self, radius: float, height: float, z_offset: float = 0.0) -> None:
        self._radius = _positive_float(radius, "radius")
        self._height = _positive_float(height, "height")
        z_offset = float(z_offset)
        if not math.isfinite(z_offset):
            raise ValueError("z_offset must be finite")
        self._z_offset = z_offset

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def cutting_length(self) -> float:
        return 0.0

    @property
    def overall_length(self) -> float:
        return self._z_offset + self._height

    def cutting_z_range(self, tip_z: float) -> tuple[float, float]:
        return float(tip_z), float(tip_z)

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        return None

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        local_z = zk - tip_z
        if self._z_offset <= local_z <= self._z_offset + self._height:
            return self._radius
        return None

    def sample_surface(
        self,
        n_u: int = 32,
        n_v: int = 16,
        include_shank: bool = True,
        include_non_cutting: bool = True,
        **legacy_kwargs,
    ) -> ToolSurfaceSample | dict[str, np.ndarray]:
        n_u = max(3, int(n_u))
        n_v = max(1, int(n_v))
        angles = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)
        z_values = np.linspace(self._z_offset, self._z_offset + self._height, n_v + 1)

        points = []
        normals = []
        component_ids = []
        parameters = []
        for z in z_values:
            for phi in angles:
                ca = math.cos(phi)
                sa = math.sin(phi)
                points.append((self._radius * ca, self._radius * sa, z))
                normals.append((ca, sa, 0.0))
                component_ids.append(COMPONENT_HOLDER)
                parameters.append((phi, z))

        return self._validate_surface(
            np.asarray(points),
            np.asarray(normals),
            np.asarray(component_ids),
            np.asarray(parameters),
            include_non_cutting=include_non_cutting,
        )
