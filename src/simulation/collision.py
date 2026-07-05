from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..motion.pose import ToolPose
from ..stock.surface_metadata import component_labels
from ..stock.tri_dexel import TriDexelStock
from ..tool.tool_geometry import ToolGeometry, ToolSurfaceSample


@dataclass(frozen=True)
class CollisionEvent:
    """A sampled non-cutting tool surface point intersecting remaining stock."""

    position: tuple[float, float, float]
    component_id: int | None
    component_name: str | None
    surface_role: str | None
    line_no: int | None = None
    motion_type: str | None = None
    source_line: str | None = None
    tool_id: str | None = None
    sample_index: int | None = None


def _surface_from_tool(
    tool: ToolGeometry,
    pose: ToolPose,
    n_u: int,
    n_v: int,
) -> ToolSurfaceSample:
    surface = tool.transform_surface(
        pose,
        n_u=n_u,
        n_v=n_v,
        include_shank=True,
        include_non_cutting=True,
    )
    if not isinstance(surface, ToolSurfaceSample):
        raise TypeError("collision detection requires ToolSurfaceSample output")
    return surface


def detect_pose_collision(
    stock: TriDexelStock,
    tool: ToolGeometry,
    pose: ToolPose,
    *,
    n_u: int = 12,
    n_v: int = 4,
    max_events: int = 200,
) -> list[CollisionEvent]:
    """Sample non-cutting collision geometry and test it against current stock."""
    surface = _surface_from_tool(tool, pose, n_u=n_u, n_v=n_v)
    collision_surface = surface.filtered(cutting=False, collision=True)
    if collision_surface.points.size == 0:
        return []

    events: list[CollisionEvent] = []
    for sample_index, (point, component_id) in enumerate(
        zip(collision_surface.points, collision_surface.component_ids)
    ):
        x, y, z = (float(point[0]), float(point[1]), float(point[2]))
        if not stock.contains_point(x, y, z):
            continue

        cid = int(component_id) if np.isfinite(component_id) else None
        component_name, surface_role = component_labels(cid)
        events.append(
            CollisionEvent(
                position=(x, y, z),
                component_id=cid,
                component_name=component_name,
                surface_role=surface_role,
                line_no=pose.line_no,
                motion_type=pose.motion_type,
                source_line=pose.source_line,
                tool_id=pose.tool_id,
                sample_index=sample_index,
            )
        )
        if len(events) >= max_events:
            break
    return events


def detect_segment_collision(
    stock: TriDexelStock,
    tool: ToolGeometry,
    start: ToolPose,
    end: ToolPose,
    *,
    pose_samples: int = 2,
    n_u: int = 12,
    n_v: int = 4,
    max_events: int = 200,
) -> list[CollisionEvent]:
    """Check sampled poses along a segment for non-cutting stock collision."""
    pose_samples = max(1, int(pose_samples))
    events: list[CollisionEvent] = []
    for index in range(pose_samples + 1):
        t = index / pose_samples
        pose = start.interpolate(end, t)
        remaining = max_events - len(events)
        if remaining <= 0:
            break
        events.extend(
            detect_pose_collision(
                stock,
                tool,
                pose,
                n_u=n_u,
                n_v=n_v,
                max_events=remaining,
            )
        )
    return events


def collision_summary(events: Iterable[CollisionEvent]) -> dict[str, int]:
    """Count collision events by component name for diagnostics."""
    summary: dict[str, int] = {}
    for event in events:
        key = event.component_name or "unknown"
        summary[key] = summary.get(key, 0) + 1
    return summary
