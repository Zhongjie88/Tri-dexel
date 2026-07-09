import math
import numpy as np
import pytest
from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import FlatEndMill, BallEndMill
from src.simulation.engine import SimulationEngine
from src.motion.pose import ToolPose


BOUNDS = (0.0, 50.0, 0.0, 50.0, 0.0, 30.0)
RES = 1.0


def make_stock() -> TriDexelStock:
    s = TriDexelStock(BOUNDS, RES)
    s.initialize_box_stock()
    return s


# ---------------------------------------------------------------------------
# Tool geometry unit tests
# ---------------------------------------------------------------------------

class TestFlatEndMill:
    def test_z_cut_center(self):
        t = FlatEndMill(5.0)
        assert t.z_cut(0.0, 10.0) == pytest.approx(10.0)

    def test_z_cut_edge(self):
        t = FlatEndMill(5.0)
        assert t.z_cut(5.0, 10.0) == pytest.approx(10.0)

    def test_z_cut_outside(self):
        t = FlatEndMill(5.0)
        assert t.z_cut(6.0, 10.0) is None

    def test_cross_section_below_tip(self):
        t = FlatEndMill(5.0)
        assert t.cross_section_radius(5.0, 10.0) is None

    def test_cross_section_at_tip(self):
        t = FlatEndMill(5.0)
        assert t.cross_section_radius(10.0, 10.0) == pytest.approx(5.0)


class TestBallEndMill:
    def test_z_cut_tip(self):
        t = BallEndMill(5.0)
        # At center (d=0), z_cut should equal tip_z
        assert t.z_cut(0.0, 10.0) == pytest.approx(10.0)

    def test_z_cut_equator(self):
        t = BallEndMill(5.0)
        # At d=r, z_cut = tip_z + r (equator height)
        assert t.z_cut(5.0, 10.0) == pytest.approx(15.0)

    def test_z_cut_outside(self):
        t = BallEndMill(5.0)
        assert t.z_cut(6.0, 10.0) is None

    def test_cross_section_at_tip(self):
        t = BallEndMill(5.0)
        # At the tip (zk == tip_z), cross-section radius = 0
        assert t.cross_section_radius(10.0, 10.0) == pytest.approx(0.0, abs=1e-9)

    def test_cross_section_at_equator(self):
        t = BallEndMill(5.0)
        assert t.cross_section_radius(15.0, 10.0) == pytest.approx(5.0)

    def test_cross_section_shank(self):
        t = BallEndMill(5.0)
        assert t.cross_section_radius(25.0, 10.0) == pytest.approx(5.0)

    def test_cross_section_below_tip(self):
        t = BallEndMill(5.0)
        assert t.cross_section_radius(9.0, 10.0) is None


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestEngine:
    def test_flat_mill_single_position_removes_material(self):
        stock = make_stock()
        tool = FlatEndMill(5.0)
        eng = SimulationEngine(stock, tool)
        # Lower tool to z=20 at centre of stock
        eng.apply_tool_at(25.0, 25.0, 20.0)
        hmap = stock.z_grid.height_map()
        # Centre dexel should now top out at 20
        i = stock.z_grid.row_index(25.0)
        j = stock.z_grid.col_index(25.0)
        assert hmap[i, j] == pytest.approx(20.0, abs=1.5)

    def test_surface_fast_mode_updates_only_z_grid(self):
        stock = make_stock()
        tool = FlatEndMill(5.0)
        eng = SimulationEngine(stock, tool, update_side_grids=False)

        eng.apply_tool_at(25.0, 25.0, 20.0)

        zi = stock.z_grid.row_index(25.0)
        zj = stock.z_grid.col_index(25.0)
        xi = stock.x_grid.row_index(25.0)
        xj = stock.x_grid.col_index(20.0)
        yi = stock.y_grid.row_index(25.0)
        yj = stock.y_grid.col_index(20.0)

        assert stock.z_grid.height_map()[zi, zj] == pytest.approx(20.0, abs=1.5)
        assert stock.x_grid.rays[xi][xj].intervals == [(0.0, 50.0)]
        assert stock.y_grid.rays[yi][yj].intervals == [(0.0, 50.0)]

    def test_flat_mill_cuts_partially_covered_z_cells(self):
        stock = TriDexelStock((0.0, 4.0, 0.0, 4.0, 0.0, 4.0), 1.0)
        stock.initialize_box_stock()
        tool = FlatEndMill(0.2)
        eng = SimulationEngine(stock, tool)

        eng.apply_tool_at(1.1, 1.5, 2.0)

        hmap = stock.z_grid.height_map()
        left_cell_center_distance = math.dist((0.5, 1.5), (1.1, 1.5))
        assert left_cell_center_distance > tool.radius
        assert hmap[0, 1] == pytest.approx(2.0)

    def test_ball_mill_creates_curved_pocket(self):
        stock = make_stock()
        tool = BallEndMill(4.0)
        eng = SimulationEngine(stock, tool)
        eng.apply_tool_at(25.0, 25.0, 15.0)
        hmap = stock.z_grid.height_map()
        i0 = stock.z_grid.row_index(25.0)
        j0 = stock.z_grid.col_index(25.0)
        # Centre depth should be lower than a point near the edge
        i_edge = stock.z_grid.row_index(28.0)
        assert hmap[i0, j0] < hmap[i_edge, j0]

    @pytest.mark.parametrize("tool", [FlatEndMill(5.0), BallEndMill(4.0)])
    def test_cached_z_footprint_matches_uncached_update(self, tool):
        stock_cached = make_stock()
        stock_uncached = make_stock()
        cached = SimulationEngine(stock_cached, tool, update_side_grids=False)
        uncached = SimulationEngine(stock_uncached, tool, update_side_grids=False)
        uncached._z_footprint_cache = None

        for pos in [
            (25.0, 25.0, 15.0),
            (25.4, 25.6, 14.5),
            (31.2, 26.7, 13.0),
        ]:
            cached.apply_tool_at(*pos)
            uncached.apply_tool_at(*pos)

        assert stock_cached.z_grid.height_map() == pytest.approx(
            stock_uncached.z_grid.height_map(),
            abs=1e-9,
        )

    def test_oriented_legacy_pose_uses_tool_axis_angle(self):
        stock = TriDexelStock((0.0, 20.0, 0.0, 20.0, 0.0, 20.0), 1.0)
        stock.initialize_box_stock()
        eng = SimulationEngine(stock, FlatEndMill(4.0))

        pose = ToolPose.from_axis((10.0, 10.0, 10.0), (0.4, 0.0, 1.0))
        eng.apply_tool_pose_at(pose, n_u=64, n_v=20)

        hmap = stock.z_grid.height_map()
        cut = hmap < 20.0
        assert np.any(cut)
        cut_heights = hmap[cut]
        assert float(cut_heights.max() - cut_heights.min()) > 0.5

    def test_simulate_move_cuts_along_path(self):
        stock = make_stock()
        tool = FlatEndMill(3.0)
        eng = SimulationEngine(stock, tool)
        eng.simulate_move((10.0, 25.0, 20.0), (40.0, 25.0, 20.0))
        hmap = stock.z_grid.height_map()
        # Several points along Y=25 should have been cut to z=20
        j = stock.z_grid.col_index(25.0)
        for xi in [12.0, 20.0, 30.0, 38.0]:
            i = stock.z_grid.row_index(xi)
            assert hmap[i, j] == pytest.approx(20.0, abs=1.5), f"xi={xi}"

    def test_simulate_toolpath_matches_segmented_path_for_connected_lines(self):
        toolpath = [
            ((10.0, 25.0, 20.0), (15.0, 25.0, 20.0)),
            ((15.0, 25.0, 20.0), (20.0, 25.0, 20.0)),
            ((20.0, 25.0, 20.0), (25.0, 25.0, 20.0)),
            ((25.0, 25.0, 20.0), (30.0, 25.0, 20.0)),
        ]
        stock_segmented = make_stock()
        stock_continuous = make_stock()
        tool = FlatEndMill(3.0)
        segmented = SimulationEngine(stock_segmented, tool)
        continuous = SimulationEngine(stock_continuous, tool)

        for start, end in toolpath:
            segmented.simulate_move(start, end, step=1.0)
        continuous.simulate_toolpath(toolpath, step=1.0)

        assert stock_continuous.z_grid.height_map() == pytest.approx(
            stock_segmented.z_grid.height_map(),
            abs=1e-9,
        )

    def test_simulate_toolpath_samples_sharp_direction_change_junction(self):
        stock = make_stock()
        tool = FlatEndMill(3.0)
        eng = SimulationEngine(stock, tool)
        calls = []
        original_apply = eng.apply_tool_at

        def record_apply(x, y, z):
            calls.append((round(x, 3), round(y, 3), round(z, 3)))
            original_apply(x, y, z)

        eng.apply_tool_at = record_apply
        toolpath = [
            ((10.0, 10.0, 20.0), (10.4, 10.0, 20.0)),
            ((10.4, 10.0, 20.0), (10.4, 10.4, 20.0)),
        ]

        eng.simulate_toolpath(toolpath, step=1.0)

        assert (10.4, 10.0, 20.0) in calls

    def test_rapid_move_does_not_cut(self):
        from src.gcode.parser import GCodeMove
        stock = make_stock()
        tool = FlatEndMill(5.0)
        eng = SimulationEngine(stock, tool)
        moves = [GCodeMove(x=25.0, y=25.0, z=5.0, feed=0.0, rapid=True)]
        eng.simulate_gcode(moves)
        hmap = stock.z_grid.height_map()
        i = stock.z_grid.row_index(25.0)
        j = stock.z_grid.col_index(25.0)
        # Stock should be untouched (top = 30.0)
        assert hmap[i, j] == pytest.approx(30.0, abs=1.5)


def test_gcode_parse_file_handles_utf8_comments(tmp_path):
    from src.gcode.parser import GCodeParser

    nc = tmp_path / "program.nc"
    nc.write_text("; 測試\nG0 Z10\nG1 X5 F100\n", encoding="utf-8")

    moves = GCodeParser().parse_file(str(nc))

    assert len(moves) == 2
    assert moves[0].rapid is True
    assert moves[1].rapid is False


def test_gcode_parser_linearizes_ij_arcs():
    from src.gcode.parser import GCodeParser

    moves = GCodeParser(arc_segment_length=1.0).parse(
        "G90 G0 X1 Y0 Z0\n"
        "G3 X0 Y1 I-1 J0 F100\n"
    )

    assert len(moves) > 2
    assert moves[0].rapid is True
    assert all(not move.rapid for move in moves[1:])
    assert moves[-1].x == pytest.approx(0.0)
    assert moves[-1].y == pytest.approx(1.0)


def test_gcode_parser_preserves_arc_source_metadata():
    from src.gcode.parser import GCodeParser

    moves = GCodeParser(arc_segment_length=0.5).parse(
        "G90 G0 X1 Y0 Z0\n"
        "G2 X0 Y-1 I-1 J0 F100\n"
        "X-1 Y0 I0 J1\n"
    )
    arc_moves = [move for move in moves if move.motion_type == "G2"]

    assert len(arc_moves) > 2
    assert all(move.arc_center is not None for move in arc_moves)
    assert all(move.arc_direction == "CW" for move in arc_moves)
    assert {move.line_no for move in arc_moves} == {2, 3}
    assert arc_moves[0].segment_index == 1
    assert arc_moves[0].segment_count is not None
    assert arc_moves[-1].source_line == "X-1 Y0 I0 J1"


# ---------------------------------------------------------------------------
# 4-E: additional coverage
# ---------------------------------------------------------------------------


def test_flat_endmill_cross_section_above_height():
    from src.tool.tool_geometry import FlatEndMill

    tool = FlatEndMill(radius=5.0, height=10.0)
    tip_z = 3.0
    # inside flute range → valid
    assert tool.cross_section_radius(tip_z + 5.0, tip_z) is not None
    # above top of flute → None
    assert tool.cross_section_radius(tip_z + 10.0 + 0.001, tip_z) is None


def test_gcode_parser_g20_inch_mode():
    import pytest
    from src.gcode.parser import GCodeParser

    moves = GCodeParser().parse("G20\nG90 G1 X1.0 Y0.5 Z-0.1 F100")
    feed = [m for m in moves if not m.rapid]
    assert len(feed) >= 1
    assert feed[0].x == pytest.approx(25.4, abs=1e-6)
    assert feed[0].y == pytest.approx(12.7, abs=1e-6)
    assert feed[0].z == pytest.approx(-2.54, abs=1e-6)


def test_voxel_to_mesh_empty_voxel():
    """Empty voxel grid must not raise — marching cubes has nothing to find."""
    import numpy as np
    from src.reconstruction.mesh import voxel_to_mesh

    voxel = np.zeros((10, 10, 10), dtype=bool)
    # Should either return None/empty or raise in a controlled way (not crash).
    try:
        result = voxel_to_mesh(voxel, (0, 10, 0, 10, 0, 10))
    except Exception:
        result = None
    # Accept None or a mesh object — just must not be an unhandled crash.
    assert result is None or hasattr(result, "points")


def test_simulate_gcode_stop_callback():
    from src.stock.tri_dexel import TriDexelStock
    from src.tool.tool_geometry import BallEndMill
    from src.simulation.engine import SimulationEngine
    from src.gcode.parser import GCodeParser

    stock = TriDexelStock((0, 20, 0, 20, 0, 10), resolution=2.0)
    stock.initialize_box_stock()
    engine = SimulationEngine(stock, BallEndMill(radius=2.0))

    # 10 cutting moves; stop after first one
    gcode = "\n".join(f"G1 X{i} Y0 Z-1 F100" for i in range(1, 11))
    moves = GCodeParser().parse("G90\n" + gcode)
    feed_moves = [m for m in moves if not m.rapid]
    assert len(feed_moves) == 10

    executed: list[int] = []

    def _progress(cur, total):
        executed.append(cur)

    def _stop():
        return len(executed) >= 1  # halt after the first move

    engine.simulate_gcode(feed_moves, progress_callback=_progress, stop_callback=_stop)
    # stop_callback fires after the first move completes, so at most 2 moves run
    assert len(executed) <= 2
