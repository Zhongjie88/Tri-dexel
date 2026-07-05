import numpy as np
import pytest

from src.gcode.parser import GCodeMove
from src.motion.gcode import (
    rotation_from_abc,
    rotation_from_ac,
    tool_pose_from_gcode_move,
    tool_poses_from_gcode_moves,
)
from src.motion.pose import ToolPose


def test_tool_pose_metadata_is_optional_and_preserved():
    pose = ToolPose(
        (1.0, 2.0, 3.0),
        feed=1200.0,
        line_no=42,
        tool_id="T1",
        motion_type="G1",
    )

    assert pose.axis == pytest.approx((0.0, 0.0, 1.0))
    assert pose.feed == pytest.approx(1200.0)
    assert pose.line_no == 42
    assert pose.tool_id == "T1"
    assert pose.motion_type == "G1"


def test_tool_pose_from_axis_accepts_motion_metadata():
    pose = ToolPose.from_axis(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        feed=800.0,
        line_no=7,
        tool_id="BALL",
        motion_type="G2",
    )

    assert pose.axis == pytest.approx((1.0, 0.0, 0.0))
    assert pose.feed == pytest.approx(800.0)
    assert pose.line_no == 7
    assert pose.tool_id == "BALL"
    assert pose.motion_type == "G2"


def test_tool_pose_interpolate_carries_later_motion_metadata():
    start = ToolPose((0.0, 0.0, 0.0), feed=100.0, line_no=1, tool_id="T0")
    end = ToolPose(
        (10.0, 0.0, 0.0),
        feed=200.0,
        line_no=2,
        tool_id="T1",
        motion_type="G1",
    )

    mid = start.interpolate(end, 0.25)

    assert mid.position == pytest.approx((2.5, 0.0, 0.0))
    assert np.linalg.norm(mid.axis) == pytest.approx(1.0)
    assert mid.feed == pytest.approx(200.0)
    assert mid.line_no == 2
    assert mid.tool_id == "T1"
    assert mid.motion_type == "G1"


def test_gcode_move_converts_to_metadata_rich_tool_pose():
    move = GCodeMove(
        x=1.0,
        y=2.0,
        z=3.0,
        feed=500.0,
        rapid=False,
        line_no=12,
        motion_type="G2",
        source_line="N12 G2 X1 Y2 I3 J4",
        segment_index=2,
        segment_count=5,
        arc_center=(3.0, 4.0),
        arc_radius=10.0,
        arc_direction="CW",
    )

    pose = tool_pose_from_gcode_move(move, tool_id="T12")

    assert pose.position == pytest.approx((1.0, 2.0, 3.0))
    assert pose.feed == pytest.approx(500.0)
    assert pose.line_no == 12
    assert pose.tool_id == "T12"
    assert pose.motion_type == "G2"
    assert pose.source_line == "N12 G2 X1 Y2 I3 J4"
    assert pose.segment_index == 2
    assert pose.segment_count == 5
    assert pose.arc_center == (3.0, 4.0)
    assert pose.arc_radius == pytest.approx(10.0)
    assert pose.arc_direction == "CW"


def test_gcode_moves_convert_to_tool_pose_endpoints():
    moves = [
        GCodeMove(0.0, 0.0, 5.0, 0.0, rapid=True, motion_type="G0"),
        GCodeMove(1.0, 0.0, 5.0, 100.0, rapid=False, motion_type="G1"),
    ]

    poses = tool_poses_from_gcode_moves(moves, tool_id="FLAT")

    assert len(poses) == 2
    assert poses[0].motion_type == "G0"
    assert poses[1].motion_type == "G1"
    assert all(pose.tool_id == "FLAT" for pose in poses)


def test_rotary_ac_words_drive_tool_pose_orientation():
    move = GCodeMove(
        x=1.0,
        y=2.0,
        z=3.0,
        feed=500.0,
        rapid=False,
        motion_type="G1",
        a=-11.2981,
        c=267.417,
    )

    pose = tool_pose_from_gcode_move(move)

    assert pose.rotation == pytest.approx(rotation_from_ac(-11.2981, 267.417))
    assert pose.axis[2] < 1.0
    assert np.linalg.norm(pose.axis) == pytest.approx(1.0)


def test_rotary_abc_orientation_matches_reference_formula():
    move = GCodeMove(
        x=0.0,
        y=0.0,
        z=0.0,
        feed=0.0,
        rapid=False,
        a=10.0,
        b=20.0,
        c=30.0,
    )

    pose = tool_pose_from_gcode_move(move)

    assert pose.rotation == pytest.approx(rotation_from_abc(10.0, 20.0, 30.0))
    assert pose.axis == pytest.approx((0.378522, 0.018028, 0.925417), abs=1e-6)
