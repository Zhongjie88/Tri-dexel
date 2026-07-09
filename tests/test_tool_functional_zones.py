import pytest

from src.motion.pose import ToolPose
from src.simulation.engine import SimulationEngine
from src.swept_volume.builder import SweptVolumeBuilder
from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import (
    COMPONENT_BALL,
    COMPONENT_HOLDER,
    COMPONENT_SHANK,
    COMPONENT_SIDE,
    BallEndMill,
    FlatEndMill,
    ToolComponentRole,
    ToolHolder,
    component_for_id,
)


def _stock(bounds=(0.0, 10.0, 0.0, 10.0, 0.0, 20.0)):
    stock = TriDexelStock(bounds, 1.0)
    stock.initialize_box_stock()
    return stock


def _z_ray_material(stock, x=5.0, y=5.0):
    i = stock.z_grid.row_index(x)
    j = stock.z_grid.col_index(y)
    return sum(hi - lo for lo, hi in stock.z_grid.rays[i][j].intervals)


def test_component_roles_are_explicit():
    assert component_for_id(COMPONENT_SIDE).role is ToolComponentRole.CUTTING
    assert component_for_id(COMPONENT_BALL).role is ToolComponentRole.CUTTING
    assert component_for_id(COMPONENT_SHANK).role is ToolComponentRole.COLLISION_ONLY
    assert component_for_id(COMPONENT_HOLDER).role is ToolComponentRole.COLLISION_ONLY


def test_flat_end_mill_cutting_length_limits_material_removal():
    stock = _stock()
    tool = FlatEndMill(radius=1.0, cutting_length=2.0, overall_length=8.0)
    engine = SimulationEngine(stock, tool, update_side_grids=False)

    before = _z_ray_material(stock)
    engine.apply_tool_at(5.0, 5.0, 15.0)
    after = _z_ray_material(stock)

    assert before - after == pytest.approx(2.0)
    assert stock.z_grid.rays[5][5].intervals[-1] == pytest.approx((17.0, 20.0))


def test_flat_end_mill_shank_contact_warns_without_cutting():
    stock = _stock(bounds=(0.0, 10.0, 0.0, 10.0, 12.0, 20.0))
    tool = FlatEndMill(radius=1.0, cutting_length=2.0, overall_length=8.0)
    engine = SimulationEngine(
        stock,
        tool,
        update_side_grids=False,
        detect_collision=True,
    )

    before = _z_ray_material(stock)
    engine.apply_tool_at(5.0, 5.0, 8.0)
    after = _z_ray_material(stock)

    assert after == pytest.approx(before)
    assert engine.collision_events
    assert any(event.component_id == COMPONENT_SHANK for event in engine.collision_events)


def test_holder_contact_warns_without_material_removal():
    stock = _stock(bounds=(0.0, 10.0, 0.0, 10.0, 12.0, 20.0))
    holder = ToolHolder(radius=1.5, height=8.0, z_offset=4.0)
    engine = SimulationEngine(
        stock,
        holder,
        update_side_grids=False,
        detect_collision=True,
    )

    before = _z_ray_material(stock)
    engine.apply_tool_at(5.0, 5.0, 8.0)
    after = _z_ray_material(stock)

    assert after == pytest.approx(before)
    assert engine.collision_events
    assert any(event.component_id == COMPONENT_HOLDER for event in engine.collision_events)


def test_ball_end_mill_ball_tip_cuts_but_shank_only_contact_does_not():
    stock = _stock()
    tool = BallEndMill(radius=2.0, cutting_length=2.0, overall_length=8.0)
    engine = SimulationEngine(stock, tool, update_side_grids=False)

    before = _z_ray_material(stock)
    engine.apply_tool_at(5.0, 5.0, 12.0)
    assert _z_ray_material(stock) < before

    stock = _stock(bounds=(0.0, 10.0, 0.0, 10.0, 15.0, 20.0))
    engine = SimulationEngine(
        stock,
        tool,
        update_side_grids=False,
        detect_collision=True,
    )
    before = _z_ray_material(stock)
    engine.apply_tool_at(5.0, 5.0, 10.0)

    assert _z_ray_material(stock) == pytest.approx(before)
    assert any(event.component_id == COMPONENT_SHANK for event in engine.collision_events)


def test_swept_volume_builder_filters_collision_only_components():
    tool = BallEndMill(radius=2.0, cutting_length=2.0, overall_length=8.0)
    start = ToolPose((0.0, 0.0, 0.0))
    end = ToolPose((2.0, 0.0, 0.0))

    active = SweptVolumeBuilder(
        tool,
        radial_segments=8,
        axial_segments=3,
        active_cutting_only=True,
    ).build_between(start, end, samples=2)
    all_components = SweptVolumeBuilder(
        tool,
        radial_segments=8,
        axial_segments=3,
        active_cutting_only=False,
    ).build_between(start, end, samples=2)

    assert COMPONENT_SHANK not in set(active.component_ids.tolist())
    assert COMPONENT_SHANK in set(all_components.component_ids.tolist())
