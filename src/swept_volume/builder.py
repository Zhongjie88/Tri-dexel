from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..motion.pose import ToolPose
from ..tool.tool_geometry import ToolGeometry
from .envelope import select_translation_envelope_between_poses
from .triangle import Triangle


@dataclass(frozen=True)
class SweptVolume:
    triangles: tuple[Triangle, ...]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.triangles:
            nan = np.full(3, np.nan)
            return nan, nan
        lo = np.min([tri.bounds[0] for tri in self.triangles], axis=0)
        hi = np.max([tri.bounds[1] for tri in self.triangles], axis=0)
        return lo, hi


class SweptVolumeBuilder:
    """Build a light triangle swept surface between sampled tool poses."""

    def __init__(
        self,
        tool: ToolGeometry,
        radial_segments: int = 24,
        axial_segments: int = 8,
        use_envelope: bool = False,
        envelope_eps: float = 0.05,
        active_cutting_only: bool = True,
    ) -> None:
        self.tool = tool
        self.radial_segments = radial_segments
        self.axial_segments = axial_segments
        self.use_envelope = use_envelope
        self.envelope_eps = envelope_eps
        self.active_cutting_only = active_cutting_only

    def build_between(
        self,
        start: ToolPose,
        end: ToolPose,
        samples: int = 2,
    ) -> SweptVolume:
        if self.use_envelope:
            return self.build_translation_envelope_between(start, end)
        poses = [
            start.interpolate(end, t)
            for t in np.linspace(0.0, 1.0, max(2, int(samples)))
        ]
        return self.build_from_poses(poses)

    def build_translation_envelope_between(
        self,
        start: ToolPose,
        end: ToolPose,
    ) -> SweptVolume:
        """Build a 3-axis translational swept surface from envelope candidates."""
        envelope = select_translation_envelope_between_poses(
            self.tool,
            start,
            end,
            eps=self.envelope_eps,
            n_u=self.radial_segments,
            n_v=self.axial_segments,
            include_shank=True,
            include_non_cutting=not self.active_cutting_only,
        )
        if envelope.count < 2:
            return SweptVolume(())

        p0 = envelope.points
        p1 = p0 + (end.position - start.position)
        n0 = envelope.normals
        triangles: list[Triangle] = []

        groups: dict[int, list[int]] = {}
        for local_idx, component in enumerate(envelope.component_ids):
            groups.setdefault(int(component), []).append(local_idx)

        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            for a, b in zip(group_indices[:-1], group_indices[1:]):
                component_id = int(envelope.component_ids[a])
                triangles.append(
                    Triangle(
                        p0[a],
                        p1[a],
                        p1[b],
                        normal=n0[a] + n0[b],
                        source="translation_envelope",
                        component_id=component_id,
                    )
                )
                triangles.append(
                    Triangle(
                        p0[a],
                        p1[b],
                        p0[b],
                        normal=n0[a] + n0[b],
                        source="translation_envelope",
                        component_id=component_id,
                    )
                )
        return SweptVolume(tuple(triangles))

    def build_from_poses(self, poses: Iterable[ToolPose]) -> SweptVolume:
        pose_list = list(poses)
        if len(pose_list) < 2:
            raise ValueError("at least two poses are required")

        rings = []
        normals = []
        component_ids = []
        for pose in pose_list:
            surface = self.tool.transform_surface(
                pose,
                radial_segments=self.radial_segments,
                axial_segments=self.axial_segments,
                include_non_cutting=not self.active_cutting_only,
            )
            rings.append(surface["points"])
            normals.append(surface["normals"])
            ids = surface.get("component_ids")
            if ids is None:
                ids = np.full(surface["points"].shape[:2], -1, dtype=int)
            component_ids.append(ids)

        triangles: list[Triangle] = []
        n_ring, n_col, _ = rings[0].shape
        for a in range(len(rings) - 1):
            p0 = rings[a]
            p1 = rings[a + 1]
            n0 = normals[a]
            n1 = normals[a + 1]
            c0_ids = component_ids[a]
            c1_ids = component_ids[a + 1]
            for i in range(n_ring):
                ni = (i + 1) % n_ring
                for j in range(n_col - 1):
                    component_id = int(c0_ids[i, j]) if c0_ids[i, j] >= 0 else None
                    triangles.append(
                        Triangle(
                            p0[i, j],
                            p1[i, j],
                            p1[ni, j],
                            normal=n0[i, j] + n1[i, j] + n1[ni, j],
                            source="swept",
                            component_id=component_id,
                        )
                    )
                    triangles.append(
                        Triangle(
                            p0[i, j],
                            p1[ni, j],
                            p0[ni, j],
                            normal=n0[i, j] + n1[ni, j] + n0[ni, j],
                            source="swept",
                            component_id=component_id,
                        )
                    )
                for j_cap in (0, n_col - 1):
                    component_id = int(c0_ids[i, j_cap]) if c0_ids[i, j_cap] >= 0 else None
                    c0 = p0[:, j_cap].mean(axis=0)
                    c1 = p1[:, j_cap].mean(axis=0)
                    triangles.append(
                        Triangle(
                            c0,
                            c1,
                            p1[i, j_cap],
                            source="swept_cap",
                            component_id=component_id,
                        )
                    )
                    triangles.append(
                        Triangle(
                            c0,
                            p1[i, j_cap],
                            p0[i, j_cap],
                            source="swept_cap",
                            component_id=component_id,
                        )
                    )
            for p, n, source, c_ids in (
                (p0, n0, "swept_start_cap", c0_ids),
                (p1, n1, "swept_end_cap", c1_ids),
            ):
                for i in range(n_ring):
                    ni = (i + 1) % n_ring
                    for j in range(n_col - 1):
                        component_id = int(c_ids[i, j]) if c_ids[i, j] >= 0 else None
                        triangles.append(
                            Triangle(
                                p[i, j],
                                p[ni, j],
                                p[ni, j + 1],
                                normal=n[i, j] + n[ni, j] + n[ni, j + 1],
                                source=source,
                                component_id=component_id,
                            )
                        )
                        triangles.append(
                            Triangle(
                                p[i, j],
                                p[ni, j + 1],
                                p[i, j + 1],
                                normal=n[i, j] + n[ni, j + 1] + n[i, j + 1],
                                source=source,
                                component_id=component_id,
                            )
                        )
                for j in range(n_col):
                    component_id = int(c_ids[0, j]) if c_ids[0, j] >= 0 else None
                    center = p[:, j].mean(axis=0)
                    for i in range(n_ring):
                        ni = (i + 1) % n_ring
                        triangles.append(
                            Triangle(
                                center,
                                p[i, j],
                                p[ni, j],
                                normal=n[i, j] + n[ni, j],
                                source=source,
                                component_id=component_id,
                            )
                        )
        return SweptVolume(tuple(triangles))
