import math
import pytest
from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import FlatEndMill, BallEndMill
from src.simulation.engine import SimulationEngine


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

    def test_rapid_move_does_not_cut(self):
        from src.gcode.parser import GCodeMove
        stock = make_stock()
        tool = FlatEndMill(5.0)
        eng = SimulationEngine(stock, tool)
        moves = [GCodeMove(x=25.0, y=25.0, z=5.0, feed=0.0, rapid=True)]
        eng.simulate_gcode(moves, progress=False)
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
