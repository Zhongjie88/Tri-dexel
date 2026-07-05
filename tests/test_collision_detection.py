from src.motion.pose import ToolPose
from src.simulation.collision import collision_summary
from src.simulation.sv_engine import SweptVolumeSimulationEngine
from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import BallEndMill, COMPONENT_SHANK, FlatEndMill


def _box_stock():
    stock = TriDexelStock((0.0, 10.0, 0.0, 10.0, 0.0, 10.0), 1.0)
    stock.initialize_box_stock()
    return stock


def test_tri_dexel_stock_contains_point_uses_all_three_grids():
    stock = _box_stock()

    assert stock.contains_point(5.0, 5.0, 5.0)
    assert not stock.contains_point(-1.0, 5.0, 5.0)

    stock.z_grid.subtract_at(5, 5, 4.0, 6.0)
    assert not stock.contains_point(5.5, 5.5, 5.0)


def test_collision_detection_reports_non_cutting_shank_inside_stock():
    stock = _box_stock()
    engine = SweptVolumeSimulationEngine(
        stock,
        BallEndMill(2.0, height=8.0),
        detect_collision=True,
        collision_n_u=12,
        collision_n_v=4,
    )

    events = engine.check_collision_at_pose(
        ToolPose(
            (5.0, 5.0, 0.0),
            line_no=7,
            tool_id="BALL4",
            motion_type="G1",
            source_line="N7 G1 X5 Y5 Z0",
        )
    )

    assert events
    assert any(event.component_id == COMPONENT_SHANK for event in events)
    assert any(event.component_name == "shank" for event in events)
    assert any(event.line_no == 7 for event in events)
    assert collision_summary(events)["shank"] > 0


def test_collision_detection_ignores_cutting_only_flat_tool_surface():
    stock = _box_stock()
    engine = SweptVolumeSimulationEngine(
        stock,
        FlatEndMill(2.0),
        detect_collision=True,
    )

    events = engine.check_collision_at_pose(ToolPose((5.0, 5.0, 5.0)))

    assert events == []


def test_sv_engine_only_accumulates_collision_events_when_enabled():
    stock = _box_stock()
    disabled = SweptVolumeSimulationEngine(
        stock,
        BallEndMill(2.0, height=8.0),
        radial_segments=8,
        axial_segments=2,
        subdivide_moves=False,
        detect_collision=False,
    )

    disabled.simulate_move((4.0, 5.0, 0.0), (6.0, 5.0, 0.0), step=10.0)
    assert disabled.collision_events == []

    stock = _box_stock()
    enabled = SweptVolumeSimulationEngine(
        stock,
        BallEndMill(2.0, height=8.0),
        radial_segments=8,
        axial_segments=2,
        subdivide_moves=False,
        detect_collision=True,
        collision_n_u=8,
        collision_n_v=2,
    )

    enabled.simulate_move((4.0, 5.0, 0.0), (6.0, 5.0, 0.0), step=10.0)
    assert enabled.collision_events
    assert all(event.component_name == "shank" for event in enabled.collision_events)
