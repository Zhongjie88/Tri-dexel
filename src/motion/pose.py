from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_vec3(value: Iterable[float]) -> np.ndarray:
    arr = np.asarray(tuple(value), dtype=float)
    if arr.shape != (3,):
        raise ValueError("expected a 3D vector")
    return arr


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    u, _, vt = np.linalg.svd(rotation)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


@dataclass(frozen=True)
class ToolPose:
    """Rigid tool pose: local tool coordinates to world coordinates.

    Project convention:
    - ``position`` is the programmed tool tip point.
    - local +Z is the tool axis from the tip toward the shank/holder.
    - ``axis`` returns that local +Z direction in world coordinates.
    """

    position: np.ndarray
    rotation: np.ndarray
    feed: float | None
    line_no: int | None
    tool_id: str | None
    motion_type: str | None
    source_line: str | None
    segment_index: int | None
    segment_count: int | None
    arc_center: tuple[float, float] | None
    arc_radius: float | None
    arc_direction: str | None

    def __init__(
        self,
        position: Iterable[float],
        rotation: Iterable[Iterable[float]] | None = None,
        feed: float | None = None,
        line_no: int | None = None,
        tool_id: str | None = None,
        motion_type: str | None = None,
        source_line: str | None = None,
        segment_index: int | None = None,
        segment_count: int | None = None,
        arc_center: tuple[float, float] | None = None,
        arc_radius: float | None = None,
        arc_direction: str | None = None,
    ) -> None:
        object.__setattr__(self, "position", _as_vec3(position))
        if rotation is None:
            r = np.eye(3, dtype=float)
        else:
            r = _orthonormalize(np.asarray(rotation, dtype=float))
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "feed", feed)
        object.__setattr__(self, "line_no", line_no)
        object.__setattr__(self, "tool_id", tool_id)
        object.__setattr__(self, "motion_type", motion_type)
        object.__setattr__(self, "source_line", source_line)
        object.__setattr__(self, "segment_index", segment_index)
        object.__setattr__(self, "segment_count", segment_count)
        object.__setattr__(self, "arc_center", arc_center)
        object.__setattr__(self, "arc_radius", arc_radius)
        object.__setattr__(self, "arc_direction", arc_direction)

    @classmethod
    def from_axis(
        cls,
        position: Iterable[float],
        axis: Iterable[float],
        feed: float | None = None,
        line_no: int | None = None,
        tool_id: str | None = None,
        motion_type: str | None = None,
        source_line: str | None = None,
        segment_index: int | None = None,
        segment_count: int | None = None,
        arc_center: tuple[float, float] | None = None,
        arc_radius: float | None = None,
        arc_direction: str | None = None,
    ) -> "ToolPose":
        """Build a pose whose local +Z axis points along ``axis``."""
        z_axis = _as_vec3(axis)
        norm = np.linalg.norm(z_axis)
        if norm < 1e-12:
            raise ValueError("axis must be non-zero")
        z_axis = z_axis / norm

        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, z_axis))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(ref, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return cls(
            position,
            np.column_stack((x_axis, y_axis, z_axis)),
            feed=feed,
            line_no=line_no,
            tool_id=tool_id,
            motion_type=motion_type,
            source_line=source_line,
            segment_index=segment_index,
            segment_count=segment_count,
            arc_center=arc_center,
            arc_radius=arc_radius,
            arc_direction=arc_direction,
        )

    @property
    def axis(self) -> np.ndarray:
        return self.rotation[:, 2].copy()

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        return pts @ self.rotation.T + self.position

    def transform_normals(self, normals: np.ndarray) -> np.ndarray:
        n = np.asarray(normals, dtype=float)
        return n @ self.rotation.T

    def interpolate(self, other: "ToolPose", t: float) -> "ToolPose":
        """Linear position interpolation with re-orthonormalized rotation blend."""
        t = float(t)
        pos = self.position * (1.0 - t) + other.position * t
        rot = _orthonormalize(self.rotation * (1.0 - t) + other.rotation * t)
        feed = other.feed if other.feed is not None else self.feed
        line_no = other.line_no if other.line_no is not None else self.line_no
        tool_id = other.tool_id if other.tool_id is not None else self.tool_id
        motion_type = (
            other.motion_type if other.motion_type is not None else self.motion_type
        )
        source_line = (
            other.source_line if other.source_line is not None else self.source_line
        )
        segment_index = (
            other.segment_index
            if other.segment_index is not None
            else self.segment_index
        )
        segment_count = (
            other.segment_count
            if other.segment_count is not None
            else self.segment_count
        )
        arc_center = other.arc_center if other.arc_center is not None else self.arc_center
        arc_radius = other.arc_radius if other.arc_radius is not None else self.arc_radius
        arc_direction = (
            other.arc_direction
            if other.arc_direction is not None
            else self.arc_direction
        )
        return ToolPose(
            pos,
            rot,
            feed=feed,
            line_no=line_no,
            tool_id=tool_id,
            motion_type=motion_type,
            source_line=source_line,
            segment_index=segment_index,
            segment_count=segment_count,
            arc_center=arc_center,
            arc_radius=arc_radius,
            arc_direction=arc_direction,
        )
