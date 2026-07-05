import numpy as np
import pytest

from src.motion.pose import ToolPose
from src import tool as tool_pkg
from src.tool.tool_geometry import (
    COMPONENT_BALL,
    COMPONENT_BOTTOM,
    COMPONENT_CORNER,
    COMPONENT_HOLDER,
    COMPONENT_SHANK,
    COMPONENT_SIDE,
    TOOL_COMPONENTS,
    BallEndMill,
    BullNoseEndMill,
    FlatEndMill,
    TaperTool,
    ToolComponent,
    ToolHolder,
    ToolSurfaceSample,
    component_for_id,
)


def _assert_surface_valid(points, normals, component_ids):
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert normals.shape == points.shape
    assert component_ids.shape == (points.shape[0],)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(normals))
    assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0)


def _assert_sample_valid(sample: ToolSurfaceSample):
    _assert_surface_valid(sample.points, sample.normals, sample.component_ids)
    assert sample.parameters.shape == (sample.points.shape[0], 2)
    assert sample.cutting_mask.shape == (sample.points.shape[0],)
    assert sample.collision_mask.shape == (sample.points.shape[0],)
    assert np.all(np.isfinite(sample.parameters))


def test_flat_end_mill_surface_shapes():
    sample = FlatEndMill(5.0).sample_surface(n_u=16, n_v=8)
    points, normals, component_ids = sample

    _assert_surface_valid(points, normals, component_ids)
    _assert_sample_valid(sample)


def test_flat_end_mill_normals_are_unit_vectors():
    _, normals, _ = FlatEndMill(5.0).sample_surface(n_u=16, n_v=8)

    assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0)


def test_flat_end_mill_components_exist():
    points, normals, component_ids = FlatEndMill(5.0, height=12.0).sample_surface(
        n_u=16,
        n_v=8,
    )

    assert COMPONENT_BOTTOM in component_ids
    assert COMPONENT_SIDE in component_ids
    assert np.all(points[component_ids == COMPONENT_BOTTOM, 2] == pytest.approx(0.0))
    assert np.allclose(
        normals[component_ids == COMPONENT_BOTTOM],
        np.array([0.0, 0.0, -1.0]),
    )


def test_ball_end_mill_surface_shapes():
    sample = BallEndMill(5.0).sample_surface(n_u=16, n_v=8)
    points, normals, component_ids = sample

    _assert_surface_valid(points, normals, component_ids)
    _assert_sample_valid(sample)


def test_ball_end_mill_tip_and_equator_z():
    radius = 5.0
    points, _, component_ids = BallEndMill(radius).sample_surface(
        n_u=16,
        n_v=8,
        include_shank=False,
    )
    ball_points = points[component_ids == COMPONENT_BALL]

    assert np.min(points[:, 2]) == pytest.approx(0.0)
    assert np.max(ball_points[:, 2]) == pytest.approx(radius)


def test_ball_end_mill_components_exist():
    _, _, component_ids = BallEndMill(5.0).sample_surface(
        n_u=16,
        n_v=8,
        include_shank=True,
    )

    assert COMPONENT_BALL in component_ids
    assert COMPONENT_SHANK in component_ids


def test_tool_components_define_cutting_and_collision_roles():
    assert isinstance(component_for_id(COMPONENT_BALL), ToolComponent)
    assert component_for_id(COMPONENT_BALL).cutting is True
    assert component_for_id(COMPONENT_SIDE).cutting is True
    assert component_for_id(COMPONENT_SHANK).cutting is False
    assert component_for_id(COMPONENT_HOLDER).cutting is False
    assert all(component.collision for component in TOOL_COMPONENTS.values())


def test_ball_end_mill_cutting_surface_excludes_shank():
    sample = BallEndMill(5.0).sample_surface(
        n_u=16,
        n_v=8,
        include_shank=True,
        include_non_cutting=False,
    )

    _assert_sample_valid(sample)
    assert set(sample.component_ids.tolist()) == {COMPONENT_BALL}
    assert np.all(sample.cutting_mask)
    assert np.all(sample.collision_mask)


def test_surface_sample_filter_helpers_preserve_roles():
    sample = BallEndMill(5.0).sample_surface(n_u=16, n_v=8, include_shank=True)

    cutting = sample.cutting_surface()
    collision = sample.collision_surface()

    assert set(cutting.component_ids.tolist()) == {COMPONENT_BALL}
    assert COMPONENT_SHANK in collision.component_ids
    assert np.all(cutting.cutting_mask)
    assert np.all(collision.collision_mask)


def test_transform_surface_translation_only():
    tool = FlatEndMill(5.0)
    sample = tool.sample_surface(n_u=12, n_v=4)
    points, normals, component_ids = sample
    pose = ToolPose((10.0, 20.0, 30.0))

    world_sample = tool.transform_surface(
        pose,
        n_u=12,
        n_v=4,
    )
    points_world, normals_world, component_ids_world = world_sample

    assert np.allclose(points_world - points, np.array([10.0, 20.0, 30.0]))
    assert np.allclose(normals_world, normals)
    assert np.array_equal(component_ids_world, component_ids)
    assert np.allclose(world_sample.parameters, sample.parameters)


def test_surface_parameters_are_component_local_uv():
    flat_sample = FlatEndMill(5.0, height=12.0).sample_surface(n_u=8, n_v=4)
    bottom_uv = flat_sample.parameters[
        flat_sample.component_ids == COMPONENT_BOTTOM
    ]
    side_uv = flat_sample.parameters[flat_sample.component_ids == COMPONENT_SIDE]

    assert np.min(bottom_uv[:, 1]) == pytest.approx(0.0)
    assert np.max(bottom_uv[:, 1]) == pytest.approx(5.0)
    assert np.min(side_uv[:, 1]) == pytest.approx(0.0)
    assert np.max(side_uv[:, 1]) == pytest.approx(12.0)

    ball_sample = BallEndMill(5.0, height=10.0).sample_surface(n_u=8, n_v=4)
    ball_uv = ball_sample.parameters[ball_sample.component_ids == COMPONENT_BALL]
    shank_uv = ball_sample.parameters[
        ball_sample.component_ids == COMPONENT_SHANK
    ]

    assert np.min(ball_uv[:, 1]) == pytest.approx(0.0)
    assert np.max(ball_uv[:, 1]) == pytest.approx(0.5 * np.pi)
    assert np.min(shank_uv[:, 1]) == pytest.approx(5.0)
    assert np.max(shank_uv[:, 1]) == pytest.approx(15.0)


def test_existing_z_cut_and_cross_section_radius_still_work():
    flat = FlatEndMill(5.0)
    ball = BallEndMill(5.0)

    assert flat.z_cut(0.0, 10.0) == pytest.approx(10.0)
    assert flat.z_cut(6.0, 10.0) is None
    assert flat.cross_section_radius(10.0, 10.0) == pytest.approx(5.0)
    assert flat.cross_section_radius(9.0, 10.0) is None

    assert ball.z_cut(0.0, 10.0) == pytest.approx(10.0)
    assert ball.z_cut(5.0, 10.0) == pytest.approx(15.0)
    assert ball.z_cut(6.0, 10.0) is None
    assert ball.cross_section_radius(10.0, 10.0) == pytest.approx(0.0)
    assert ball.cross_section_radius(15.0, 10.0) == pytest.approx(5.0)


def test_invalid_tool_dimensions_are_rejected():
    with pytest.raises(ValueError):
        FlatEndMill(0.0)
    with pytest.raises(ValueError):
        FlatEndMill(5.0, height=0.0)
    with pytest.raises(ValueError):
        BallEndMill(-1.0)
    with pytest.raises(ValueError):
        BallEndMill(5.0, height=float("nan"))
    with pytest.raises(ValueError):
        BullNoseEndMill(5.0, corner_radius=5.0)
    with pytest.raises(ValueError):
        BullNoseEndMill(5.0, corner_radius=-1.0)
    with pytest.raises(ValueError):
        TaperTool(bottom_radius=0.0, top_radius=5.0, height=10.0)
    with pytest.raises(ValueError):
        TaperTool(bottom_radius=2.0, top_radius=5.0, height=0.0)
    with pytest.raises(ValueError):
        TaperTool(bottom_radius=2.0, top_radius=5.0, height=10.0, corner_radius=-0.1)
    with pytest.raises(ValueError):
        ToolHolder(radius=0.0, height=10.0)
    with pytest.raises(ValueError):
        ToolHolder(radius=5.0, height=float("inf"))


def test_bullnose_surface_sampling_has_expected_components():
    sample = BullNoseEndMill(5.0, corner_radius=1.0, height=8.0).sample_surface(
        n_u=16,
        n_v=8,
        include_shank=True,
    )

    _assert_sample_valid(sample)
    assert COMPONENT_BOTTOM in sample.component_ids
    assert COMPONENT_CORNER in sample.component_ids
    assert COMPONENT_SIDE in sample.component_ids
    assert COMPONENT_SHANK in sample.component_ids
    assert np.min(sample.points[:, 2]) == pytest.approx(0.0)


def test_taper_tool_surface_sampling_has_expected_components():
    sample = TaperTool(
        bottom_radius=2.0,
        top_radius=5.0,
        height=10.0,
        corner_radius=0.5,
    ).sample_surface(n_u=16, n_v=8, include_shank=True)

    _assert_sample_valid(sample)
    assert COMPONENT_BOTTOM in sample.component_ids
    assert COMPONENT_CORNER in sample.component_ids
    assert COMPONENT_SIDE in sample.component_ids
    assert COMPONENT_SHANK in sample.component_ids
    side_points = sample.points[sample.component_ids == COMPONENT_SIDE]
    assert np.min(side_points[:, 2]) == pytest.approx(0.0)
    assert np.max(side_points[:, 2]) == pytest.approx(10.0)


def test_tool_holder_surface_sampling_is_collision_geometry():
    holder = ToolHolder(radius=8.0, height=20.0, z_offset=15.0)
    sample = holder.sample_surface(n_u=16, n_v=4)

    _assert_sample_valid(sample)
    assert set(sample.component_ids.tolist()) == {COMPONENT_HOLDER}
    assert not np.any(sample.cutting_mask)
    assert np.all(sample.collision_mask)
    assert np.min(sample.points[:, 2]) == pytest.approx(15.0)
    assert np.max(sample.points[:, 2]) == pytest.approx(35.0)
    assert holder.z_cut(0.0, 0.0) is None
    assert holder.cross_section_radius(20.0, 0.0) == pytest.approx(8.0)


def test_component_ids_are_exported_from_tool_package():
    assert tool_pkg.COMPONENT_BOTTOM == COMPONENT_BOTTOM
    assert tool_pkg.COMPONENT_SIDE == COMPONENT_SIDE
    assert tool_pkg.COMPONENT_BALL == COMPONENT_BALL
    assert tool_pkg.COMPONENT_CORNER == COMPONENT_CORNER
    assert tool_pkg.COMPONENT_HOLDER == COMPONENT_HOLDER
    assert tool_pkg.COMPONENT_SHANK == COMPONENT_SHANK
    assert tool_pkg.TOOL_COMPONENTS is TOOL_COMPONENTS
    assert tool_pkg.component_for_id(COMPONENT_BALL).name == "ball_tip"
    assert tool_pkg.TaperTool is TaperTool
    assert tool_pkg.ToolHolder is ToolHolder
