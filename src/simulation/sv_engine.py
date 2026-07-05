from __future__ import annotations

import math
from typing import Iterable, Tuple

from ..gcode.parser import GCodeMove
from ..motion.gcode import tool_pose_segments_from_gcode_moves
from ..motion.pose import ToolPose
from ..stock.surface_metadata import SurfaceMetadata, component_labels
from ..stock.tri_dexel import TriDexelStock
from ..swept_volume.builder import SweptVolumeBuilder
from ..swept_volume.sampler import SweptVolumeSampler
from ..tool.tool_geometry import ToolGeometry
from .collision import CollisionEvent, detect_pose_collision, detect_segment_collision
from .engine import SimulationEngine

_Point3 = Tuple[float, float, float]


class SweptVolumeSimulationEngine:
    """VWP-lite simulation path layered beside the existing tri-dexel engine."""

    def __init__(
        self,
        stock: TriDexelStock,
        tool: ToolGeometry,
        radial_segments: int = 24,
        axial_segments: int = 8,
        use_envelope: bool = False,
        envelope_eps: float = 0.05,
        subdivide_moves: bool = True,
        legacy_z_topdown: bool = False,
        active_cutting_only: bool = True,
        detect_collision: bool = False,
        collision_pose_samples: int = 2,
        collision_n_u: int = 12,
        collision_n_v: int = 4,
        max_collision_events_per_segment: int = 200,
    ) -> None:
        self.stock = stock
        self.tool = tool
        self.builder = SweptVolumeBuilder(
            tool,
            radial_segments,
            axial_segments,
            use_envelope=use_envelope,
            envelope_eps=envelope_eps,
            active_cutting_only=active_cutting_only,
        )
        self.sampler = SweptVolumeSampler()
        self.subdivide_moves = subdivide_moves
        self.legacy_z_topdown = legacy_z_topdown
        self._legacy_engine = SimulationEngine(stock, tool) if legacy_z_topdown else None
        self.detect_collision = detect_collision
        self.collision_pose_samples = max(1, int(collision_pose_samples))
        self.collision_n_u = max(3, int(collision_n_u))
        self.collision_n_v = max(1, int(collision_n_v))
        self.max_collision_events_per_segment = max(
            1,
            int(max_collision_events_per_segment),
        )
        self.collision_events: list[CollisionEvent] = []

    def check_collision_at_pose(self, pose: ToolPose) -> list[CollisionEvent]:
        return detect_pose_collision(
            self.stock,
            self.tool,
            pose,
            n_u=self.collision_n_u,
            n_v=self.collision_n_v,
            max_events=self.max_collision_events_per_segment,
        )

    def check_collision_between(
        self,
        start: ToolPose,
        end: ToolPose,
    ) -> list[CollisionEvent]:
        return detect_segment_collision(
            self.stock,
            self.tool,
            start,
            end,
            pose_samples=self.collision_pose_samples,
            n_u=self.collision_n_u,
            n_v=self.collision_n_v,
            max_events=self.max_collision_events_per_segment,
        )

    def subtract_swept_volume(
        self,
        start: ToolPose,
        end: ToolPose,
        pose_samples: int = 2,
    ) -> int:
        volume = self.builder.build_between(start, end, samples=pose_samples)
        total_hits = 0
        grid_jobs = []
        if not self.legacy_z_topdown:
            grid_jobs.append(
                (
                    "z",
                    self.stock.z_grid,
                    self.stock.z_min,
                    self.stock.z_max,
                    self.sampler.sample_z_grid,
                )
            )
        grid_jobs.extend(
            [
                (
                "x",
                self.stock.x_grid,
                self.stock.x_min,
                self.stock.x_max,
                self.sampler.sample_x_grid,
                ),
                (
                "y",
                self.stock.y_grid,
                self.stock.y_min,
                self.stock.y_max,
                self.sampler.sample_y_grid,
                ),
            ]
        )
        for grid_name, grid, depth_min, depth_max, sample_fn in grid_jobs:
            hits = sample_fn(volume, grid, depth_min, depth_max)
            total_hits += len(hits)
            for hit in hits:
                component_name, surface_role = component_labels(hit.component_id)
                grid.subtract_at(
                    hit.i,
                    hit.j,
                    hit.cut_lo,
                    hit.cut_hi,
                    metadata=SurfaceMetadata(
                        cut_range=(hit.cut_lo, hit.cut_hi),
                        normal=hit.normal,
                        source="swept_volume",
                        grid=grid_name,
                        component_id=hit.component_id,
                        component_name=component_name,
                        surface_role=surface_role,
                        tool_id=end.tool_id or start.tool_id,
                        line_no=end.line_no or start.line_no,
                        motion_type=end.motion_type or start.motion_type,
                        source_line=end.source_line or start.source_line,
                        segment_index=end.segment_index or start.segment_index,
                        segment_count=end.segment_count or start.segment_count,
                        arc_center=end.arc_center or start.arc_center,
                        arc_radius=end.arc_radius or start.arc_radius,
                        arc_direction=end.arc_direction or start.arc_direction,
                    ),
                )
        return total_hits

    def simulate_move(
        self,
        start: _Point3,
        end: _Point3,
        step: float | None = None,
    ) -> int:
        if self.legacy_z_topdown:
            self._simulate_legacy_z_move(start, end, step=step)
        x1, y1, z1 = start
        x2, y2, z2 = end
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        if step is None:
            step = self.stock.resolution
        n = max(1, math.ceil(dist / step)) if self.subdivide_moves else 1
        total_hits = 0
        prev = ToolPose(start)
        for k in range(1, n + 1):
            t = k / n
            cur = ToolPose(
                (
                    x1 + (x2 - x1) * t,
                    y1 + (y2 - y1) * t,
                    z1 + (z2 - z1) * t,
                )
            )
            if self.detect_collision:
                self.collision_events.extend(self.check_collision_between(prev, cur))
            total_hits += self.subtract_swept_volume(prev, cur)
            prev = cur
        return total_hits

    def simulate_pose_move(
        self,
        start: ToolPose,
        end: ToolPose,
        step: float | None = None,
    ) -> int:
        """Simulate one oriented tool motion segment."""
        if self.legacy_z_topdown:
            self._simulate_legacy_z_move(
                tuple(float(v) for v in start.position),
                tuple(float(v) for v in end.position),
                step=step,
            )
        dist = float(math.dist(start.position, end.position))
        if step is None:
            step = self.stock.resolution
        n = max(1, math.ceil(dist / step)) if self.subdivide_moves else 1
        total_hits = 0
        prev = start
        for k in range(1, n + 1):
            cur = end if k == n else start.interpolate(end, k / n)
            if self.detect_collision:
                self.collision_events.extend(self.check_collision_between(prev, cur))
            total_hits += self.subtract_swept_volume(prev, cur)
            prev = cur
        return total_hits

    def _simulate_legacy_z_move(
        self,
        start: _Point3,
        end: _Point3,
        step: float | None = None,
    ) -> None:
        if self._legacy_engine is None:
            return
        x1, y1, z1 = start
        x2, y2, z2 = end
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        if step is None:
            step = self.stock.resolution
        n = max(1, math.ceil(dist / step))
        for k in range(n + 1):
            t = k / n
            self._legacy_engine._update_z_grid(
                x1 + t * (x2 - x1),
                y1 + t * (y2 - y1),
                z1 + t * (z2 - z1),
            )

    def simulate_poses(self, poses: Iterable[ToolPose]) -> int:
        pose_list = list(poses)
        total_hits = 0
        for start, end in zip(pose_list[:-1], pose_list[1:]):
            total_hits += self.simulate_pose_move(start, end)
        return total_hits

    def simulate_gcode(
        self,
        moves: Iterable[GCodeMove],
        initial_position: _Point3 | None = None,
        tool_id: str | None = None,
    ) -> int:
        """Simulate parsed G-code moves while preserving source metadata."""
        if initial_position is None:
            initial_position = (0.0, 0.0, self.stock.z_max)
        total_hits = 0
        for start, end in tool_pose_segments_from_gcode_moves(moves, tool_id=tool_id):
            total_hits += self.simulate_pose_move(start, end)
        return total_hits
