import numpy as np
import pytest

from src.motion.pose import ToolPose
from src.swept_volume.builder import SweptVolumeBuilder
from src.swept_volume.envelope import (
    select_translation_envelope,
    select_translation_envelope_between_poses,
)
from src.tool.tool_geometry import (
    COMPONENT_BALL,
    COMPONENT_SIDE,
    BallEndMill,
    FlatEndMill,
)


def test_translation_envelope_selects_side_points_perpendicular_to_motion():
    surface = FlatEndMill(5.0).sample_surface(n_u=64, n_v=8)

    envelope = select_translation_envelope(
        surface,
        np.array([1.0, 0.0, 0.0]),
        eps=0.02,
    )

    assert envelope.count > 0
    assert np.all(np.abs(envelope.scores) <= 0.02 + 1e-12)
    assert COMPONENT_SIDE in envelope.component_ids
    side_normals = envelope.normals[envelope.component_ids == COMPONENT_SIDE]
    assert np.allclose(side_normals[:, 0], 0.0, atol=0.02)


def test_translation_envelope_between_poses_uses_world_motion_direction():
    tool = FlatEndMill(5.0)
    pose0 = ToolPose((0.0, 0.0, 0.0))
    pose1 = ToolPose((10.0, 0.0, 0.0))

    envelope = select_translation_envelope_between_poses(
        tool,
        pose0,
        pose1,
        eps=0.02,
        n_u=64,
        n_v=8,
    )

    assert envelope.count > 0
    assert envelope.motion_direction == pytest.approx((1.0, 0.0, 0.0))
    assert np.all(np.abs(envelope.scores) <= 0.02 + 1e-12)


def test_ball_translation_envelope_preserves_component_and_uv():
    surface = BallEndMill(5.0).sample_surface(
        n_u=64,
        n_v=16,
        include_shank=False,
    )

    envelope = select_translation_envelope(
        surface,
        np.array([1.0, 0.0, 0.0]),
        eps=0.04,
    )

    assert envelope.count > 0
    assert set(envelope.component_ids.tolist()) == {COMPONENT_BALL}
    assert envelope.parameters.shape == (envelope.count, 2)
    assert np.all(np.abs(envelope.scores) <= 0.04 + 1e-12)


def test_translation_envelope_rejects_zero_motion():
    surface = FlatEndMill(5.0).sample_surface(n_u=16, n_v=4)

    with pytest.raises(ValueError):
        select_translation_envelope(surface, np.array([0.0, 0.0, 0.0]))


def test_translation_envelope_rejects_zero_pose_delta():
    tool = FlatEndMill(5.0)
    pose = ToolPose((1.0, 2.0, 3.0))

    with pytest.raises(ValueError):
        select_translation_envelope_between_poses(tool, pose, pose)


def test_swept_volume_builder_can_use_translation_envelope_mode():
    tool = FlatEndMill(5.0)
    start = ToolPose((0.0, 0.0, 0.0))
    end = ToolPose((10.0, 0.0, 0.0))
    all_surface = SweptVolumeBuilder(
        tool,
        radial_segments=32,
        axial_segments=8,
    ).build_between(start, end)
    envelope = SweptVolumeBuilder(
        tool,
        radial_segments=32,
        axial_segments=8,
        use_envelope=True,
        envelope_eps=0.02,
    ).build_between(start, end)

    assert envelope.triangles
    assert len(envelope.triangles) < len(all_surface.triangles)
    assert {tri.source for tri in envelope.triangles} == {"translation_envelope"}


def test_swept_volume_builder_default_still_uses_all_surface_mode():
    tool = FlatEndMill(5.0)
    volume = SweptVolumeBuilder(
        tool,
        radial_segments=12,
        axial_segments=3,
    ).build_between(ToolPose((0.0, 0.0, 0.0)), ToolPose((5.0, 0.0, 0.0)))

    assert volume.triangles
    assert "swept" in {tri.source for tri in volume.triangles}


def test_swept_volume_builder_excludes_ball_shank_by_default():
    tool = BallEndMill(5.0, height=20.0)
    volume = SweptVolumeBuilder(
        tool,
        radial_segments=12,
        axial_segments=4,
    ).build_between(ToolPose((0.0, 0.0, 0.0)), ToolPose((10.0, 0.0, 0.0)))

    lo, hi = volume.bounds
    assert volume.triangles
    assert lo[2] == pytest.approx(0.0)
    assert hi[2] == pytest.approx(5.0)


def test_swept_volume_builder_can_include_non_cutting_geometry():
    tool = BallEndMill(5.0, height=20.0)
    volume = SweptVolumeBuilder(
        tool,
        radial_segments=12,
        axial_segments=4,
        active_cutting_only=False,
    ).build_between(ToolPose((0.0, 0.0, 0.0)), ToolPose((10.0, 0.0, 0.0)))

    _, hi = volume.bounds
    assert hi[2] > 5.0
