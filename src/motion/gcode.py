from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..gcode.canonical import CanonicalMotion, CanonicalMove
from ..gcode.parser import GCodeMove
from .pose import ToolPose


def rotation_from_abc(
    a_deg: float | None,
    b_deg: float | None,
    c_deg: float | None,
) -> np.ndarray:
    """Build a Siemens-style A/B/C tool orientation matrix.

    This follows the reference fiveaxis.py convention:
    local tool +Z is mapped by Rz(C) * Ry(B) * Rx(A). Exact production use
    should still be calibrated against the real machine's CYCLE800 definition.
    """
    a = math.radians(a_deg or 0.0)
    b = math.radians(b_deg or 0.0)
    c = math.radians(c_deg or 0.0)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ca, -sa],
            [0.0, sa, ca],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cb, 0.0, sb],
            [0.0, 1.0, 0.0],
            [-sb, 0.0, cb],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [cc, -sc, 0.0],
            [sc, cc, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rz @ ry @ rx


def rotation_from_ac(a_deg: float | None, c_deg: float | None) -> np.ndarray:
    """Backward-compatible A/C wrapper for older callers."""
    return rotation_from_abc(a_deg, None, c_deg)


def gcode_move_from_canonical(move: CanonicalMove) -> GCodeMove:
    """Convert a canonical motion endpoint back to the legacy GCodeMove shape."""
    if move.raw_motion is not None:
        motion_type = move.raw_motion
    elif move.motion == CanonicalMotion.RAPID:
        motion_type = "G0"
    elif move.motion == CanonicalMotion.ARC:
        motion_type = "G2" if move.arc_direction == "CW" else "G3"
    elif move.motion == CanonicalMotion.LINEAR:
        motion_type = "G1"
    else:
        motion_type = None

    arc_center = None
    if move.arc_center is not None:
        arc_center = (move.arc_center[0], move.arc_center[1])

    return GCodeMove(
        x=move.end[0],
        y=move.end[1],
        z=move.end[2],
        feed=move.feed,
        rapid=move.rapid,
        line_no=move.line_no,
        motion_type=motion_type,
        source_line=move.source_line,
        segment_index=move.segment_index,
        segment_count=move.segment_count,
        arc_center=arc_center,
        arc_radius=move.arc_radius,
        arc_direction=move.arc_direction,
        plane=move.plane,
        controller=move.controller,
        warnings=move.warnings,
        start_x=move.start[0],
        start_y=move.start[1],
        start_z=move.start[2],
        start_a=move.start_rotary[0] if move.start_rotary is not None else None,
        start_b=move.start_rotary[1] if move.start_rotary is not None else None,
        start_c=move.start_rotary[2] if move.start_rotary is not None else None,
        a=move.end_rotary[0] if move.end_rotary is not None else None,
        b=move.end_rotary[1] if move.end_rotary is not None else None,
        c=move.end_rotary[2] if move.end_rotary is not None else None,
    )


def tool_pose_from_gcode_move(
    move: GCodeMove,
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> ToolPose:
    """Convert a parsed G-code move endpoint into a metadata-rich tool pose."""
    motion_type = move.motion_type or ("G0" if move.rapid else "G1")
    if move.a is not None or move.b is not None or move.c is not None:
        return ToolPose(
            (move.x, move.y, move.z),
            rotation_from_abc(move.a, move.b, move.c),
            feed=move.feed,
            line_no=move.line_no,
            tool_id=tool_id,
            motion_type=motion_type,
            source_line=move.source_line,
            segment_index=move.segment_index,
            segment_count=move.segment_count,
            arc_center=move.arc_center,
            arc_radius=move.arc_radius,
            arc_direction=move.arc_direction,
        )
    return ToolPose.from_axis(
        (move.x, move.y, move.z),
        axis,
        feed=move.feed,
        line_no=move.line_no,
        tool_id=tool_id,
        motion_type=motion_type,
        source_line=move.source_line,
        segment_index=move.segment_index,
        segment_count=move.segment_count,
        arc_center=move.arc_center,
        arc_radius=move.arc_radius,
        arc_direction=move.arc_direction,
    )


def tool_pose_start_from_gcode_move(
    move: GCodeMove,
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> ToolPose:
    """Convert a parsed G-code move start point into a ToolPose."""
    start_move = GCodeMove(
        x=move.start_x,
        y=move.start_y,
        z=move.start_z,
        feed=move.feed,
        rapid=move.rapid,
        line_no=move.line_no,
        motion_type=move.motion_type,
        source_line=move.source_line,
        segment_index=move.segment_index,
        segment_count=move.segment_count,
        arc_center=move.arc_center,
        arc_radius=move.arc_radius,
        arc_direction=move.arc_direction,
        plane=move.plane,
        controller=move.controller,
        warnings=move.warnings,
        start_x=move.start_x,
        start_y=move.start_y,
        start_z=move.start_z,
        a=move.start_a,
        b=move.start_b,
        c=move.start_c,
        start_a=move.start_a,
        start_b=move.start_b,
        start_c=move.start_c,
    )
    return tool_pose_from_gcode_move(start_move, axis=axis, tool_id=tool_id)


def tool_pose_from_canonical_move(
    move: CanonicalMove,
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> ToolPose:
    """Convert a canonical move endpoint into a metadata-rich tool pose."""
    return tool_pose_from_gcode_move(
        gcode_move_from_canonical(move),
        axis=axis,
        tool_id=tool_id,
    )


def tool_poses_from_gcode_moves(
    moves: Iterable[GCodeMove],
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> list[ToolPose]:
    """Convert parsed G-code moves to ToolPose endpoints."""
    return [
        tool_pose_from_gcode_move(move, axis=axis, tool_id=tool_id)
        for move in moves
    ]


def tool_pose_segments_from_gcode_moves(
    moves: Iterable[GCodeMove],
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> list[tuple[ToolPose, ToolPose]]:
    """Convert cutting G-code moves into oriented start/end ToolPose segments."""
    segments: list[tuple[ToolPose, ToolPose]] = []
    for move in moves:
        if move.rapid:
            continue
        segments.append(
            (
                tool_pose_start_from_gcode_move(move, axis=axis, tool_id=tool_id),
                tool_pose_from_gcode_move(move, axis=axis, tool_id=tool_id),
            )
        )
    return segments


def all_pose_segments_from_gcode_moves(
    moves: Iterable[GCodeMove],
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> list[tuple[ToolPose, ToolPose]]:
    """Return start/end ToolPose pairs for ALL moves including G0 rapid.

    Unlike tool_pose_segments_from_gcode_moves, rapid moves are included so
    the display can follow the full path.  The caller checks
    ``seg_end.motion_type == "G0"`` to skip material removal for those.
    """
    segments: list[tuple[ToolPose, ToolPose]] = []
    for move in moves:
        segments.append(
            (
                tool_pose_start_from_gcode_move(move, axis=axis, tool_id=tool_id),
                tool_pose_from_gcode_move(move, axis=axis, tool_id=tool_id),
            )
        )
    return segments


def gcode_moves_from_canonical_moves(
    moves: Iterable[CanonicalMove],
) -> list[GCodeMove]:
    return [gcode_move_from_canonical(move) for move in moves]


def tool_poses_from_canonical_moves(
    moves: Iterable[CanonicalMove],
    axis: Iterable[float] = (0.0, 0.0, 1.0),
    tool_id: str | None = None,
) -> list[ToolPose]:
    return [
        tool_pose_from_canonical_move(move, axis=axis, tool_id=tool_id)
        for move in moves
    ]
