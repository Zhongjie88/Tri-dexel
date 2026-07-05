from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..motion.pose import ToolPose
from ..tool.tool_geometry import ToolGeometry, ToolSurfaceSample


@dataclass(frozen=True)
class EnvelopeSample:
    """Tool surface points selected by an envelope condition."""

    points: np.ndarray
    normals: np.ndarray
    component_ids: np.ndarray
    parameters: np.ndarray
    scores: np.ndarray
    indices: np.ndarray
    motion_direction: np.ndarray

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector")
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return v / norm


def select_translation_envelope(
    surface: ToolSurfaceSample,
    motion_direction: np.ndarray,
    eps: float = 0.05,
    include_front_back: bool = False,
) -> EnvelopeSample:
    """Select 3-axis translational swept-envelope candidates.

    For pure translation, points on the swept envelope satisfy n dot tau = 0.
    ``eps`` is a sampling tolerance because the tool surface is discretized.
    """
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    tau = _unit_vector(motion_direction, "motion_direction")
    scores = surface.normals @ tau
    if include_front_back:
        mask = np.ones(surface.points.shape[0], dtype=bool)
    else:
        mask = np.abs(scores) <= eps
    indices = np.nonzero(mask)[0]
    return EnvelopeSample(
        points=surface.points[mask].copy(),
        normals=surface.normals[mask].copy(),
        component_ids=surface.component_ids[mask].copy(),
        parameters=surface.parameters[mask].copy(),
        scores=scores[mask].copy(),
        indices=indices,
        motion_direction=tau,
    )


def select_translation_envelope_between_poses(
    tool: ToolGeometry,
    pose0: ToolPose,
    pose1: ToolPose,
    eps: float = 0.05,
    n_u: int = 64,
    n_v: int = 24,
    include_shank: bool = True,
    include_non_cutting: bool = True,
) -> EnvelopeSample:
    """Sample a tool at pose0 and select envelope candidates toward pose1."""
    delta = pose1.position - pose0.position
    tau = _unit_vector(delta, "pose motion")
    surface = tool.transform_surface(
        pose0,
        n_u=n_u,
        n_v=n_v,
        include_shank=include_shank,
        include_non_cutting=include_non_cutting,
    )
    if isinstance(surface, dict):
        raise TypeError("translation envelope selector requires ToolSurfaceSample")
    return select_translation_envelope(surface, tau, eps=eps)
