import pytest

from src.gcode import CanonicalMotion, GCodeParser, SiemensSinumerikParser
from src.motion.gcode import gcode_move_from_canonical


def test_parser_outputs_canonical_moves_with_start_end_and_arc_metadata():
    program = GCodeParser(arc_segment_length=0.5).parse_canonical(
        "G90 G0 X1 Y0 Z0\n"
        "G3 X0 Y1 I-1 J0 F100\n"
    )

    assert program.controller == "generic"
    assert program.moves[0].motion == CanonicalMotion.RAPID
    assert program.moves[0].start == pytest.approx((0.0, 0.0, 0.0))
    assert program.moves[0].end == pytest.approx((1.0, 0.0, 0.0))

    arc_moves = [move for move in program.moves if move.motion == CanonicalMotion.ARC]
    assert arc_moves
    assert arc_moves[0].raw_motion == "G3"
    assert arc_moves[0].plane == "G17"
    assert arc_moves[0].arc_direction == "CCW"
    assert arc_moves[0].arc_center == pytest.approx((0.0, 0.0, 0.0))
    assert arc_moves[-1].end == pytest.approx((0.0, 1.0, 0.0))


def test_parser_linearizes_g17_radius_arcs():
    program = GCodeParser(arc_segment_length=0.25).parse_canonical(
        "G90 G0 X1 Y0 Z0\n"
        "G2 X0 Y-1 R1 F100\n"
    )

    arc_moves = [move for move in program.moves if move.motion == CanonicalMotion.ARC]

    assert len(arc_moves) > 1
    assert arc_moves[0].arc_center == pytest.approx((0.0, 0.0, 0.0))
    assert arc_moves[0].arc_radius == pytest.approx(1.0)
    assert arc_moves[0].arc_direction == "CW"
    assert arc_moves[-1].end == pytest.approx((0.0, -1.0, 0.0))


def test_siemens_parser_marks_controller_and_keeps_mpf_source_line():
    program = SiemensSinumerikParser(arc_segment_length=1.0).parse_canonical(
        "N10 G0 X0 Y0 Z50\n"
        "N20 G1 X10 Y0 Z20 F500\n"
    )

    assert program.controller == "siemens"
    assert program.moves[0].controller == "siemens"
    assert program.moves[1].motion == CanonicalMotion.LINEAR
    assert program.moves[1].source_line == "N20 G1 X10 Y0 Z20 F500"
    assert program.moves[1].feed == pytest.approx(500.0)


def test_non_g17_arc_is_preserved_with_warning_in_canonical_program():
    program = GCodeParser().parse_canonical(
        "G18 G0 X1 Z0\n"
        "G2 X0 Z1 I-1 K0 F100\n"
    )

    assert program.warnings
    assert "only G17 I/J and R arcs are linearized" in program.warnings[0]
    assert program.moves[-1].motion == CanonicalMotion.ARC
    assert program.moves[-1].plane == "G18"
    assert program.moves[-1].warnings


def test_parser_resets_modal_state_between_parse_calls():
    parser = GCodeParser()

    first = parser.parse("G91 G1 X5 F100\n")
    second = parser.parse("G1 X5 F100\n")

    assert first[-1].x == pytest.approx(5.0)
    assert second[-1].x == pytest.approx(5.0)
    assert second[-1].start_x == pytest.approx(0.0)


def test_canonical_move_converts_back_to_legacy_gcode_move_shape():
    program = SiemensSinumerikParser().parse_canonical(
        "N10 G0 X0 Y0 Z10\n"
        "N20 G1 X5 Y0 Z5 F200\n"
    )

    move = gcode_move_from_canonical(program.moves[-1])

    assert move.x == pytest.approx(5.0)
    assert move.start_x == pytest.approx(0.0)
    assert move.motion_type == "G1"
    assert move.controller == "siemens"
    assert move.source_line == "N20 G1 X5 Y0 Z5 F200"


def test_siemens_parser_preserves_traori_ac_rotary_words():
    program = SiemensSinumerikParser().parse_canonical(
        "TRAORI\n"
        "G0 X-113.7995 Y-116.0991 Z40. A-11.2981 C=DC(267.417)\n"
        "G1 X-120.4977 Y-92.8071 Z1.2842 A-11.381 C=DC(266.4691) F1200\n"
        "X-121.0 Y-91.0 Z1.0\n"
    )

    assert len(program.moves) == 3
    assert program.moves[0].end_rotary[0] == pytest.approx(-11.2981)
    assert program.moves[0].end_rotary[1] is None
    assert program.moves[0].end_rotary[2] == pytest.approx(267.417)
    assert program.moves[1].start_rotary[0] == pytest.approx(-11.2981)
    assert program.moves[1].start_rotary[2] == pytest.approx(267.417)
    assert program.moves[1].end_rotary[0] == pytest.approx(-11.381)
    assert program.moves[1].end_rotary[2] == pytest.approx(266.4691)
    assert program.moves[2].end_rotary[0] == pytest.approx(-11.381)
    assert program.moves[2].end_rotary[2] == pytest.approx(266.4691)

    legacy = gcode_move_from_canonical(program.moves[1])
    assert legacy.a == pytest.approx(-11.381)
    assert legacy.c == pytest.approx(266.4691)
    assert legacy.start_a == pytest.approx(-11.2981)


def test_siemens_parser_accepts_modal_function_numbers_like_reference_fiveaxis():
    program = SiemensSinumerikParser().parse_canonical(
        "TRAORI\n"
        "G1 X.5 Y-.25 Z40. A-.6232 B=AC(2.) C=CIC(267.417) F1000\n"
        "X1. Y0. Z.9961 C=CAC(268.)\n"
        "TRAFOOF\n"
    )

    assert len(program.moves) == 2
    assert program.moves[0].end == pytest.approx((0.5, -0.25, 40.0))
    assert program.moves[0].end_rotary[0] == pytest.approx(-0.6232)
    assert program.moves[0].end_rotary[1] == pytest.approx(2.0)
    assert program.moves[0].end_rotary[2] == pytest.approx(267.417)
    assert program.moves[1].end == pytest.approx((1.0, 0.0, 0.9961))
    assert program.moves[1].end_rotary[2] == pytest.approx(268.0)
