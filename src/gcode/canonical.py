from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


Point3 = Tuple[float, float, float]


class CanonicalMotion(str, Enum):
    RAPID = "RAPID"
    LINEAR = "LINEAR"
    ARC = "ARC"
    DWELL = "DWELL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CanonicalMove:
    """Controller-neutral motion command for simulation and verification."""

    motion: CanonicalMotion
    start: Point3
    end: Point3
    feed: float = 0.0
    plane: str = "G17"
    line_no: int | None = None
    source_line: str | None = None
    controller: str = "generic"
    raw_motion: str | None = None
    segment_index: int | None = None
    segment_count: int | None = None
    arc_center: Point3 | None = None
    arc_radius: float | None = None
    arc_direction: str | None = None
    start_rotary: tuple[float | None, float | None, float | None] | None = None
    end_rotary: tuple[float | None, float | None, float | None] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rapid(self) -> bool:
        return self.motion == CanonicalMotion.RAPID


@dataclass(frozen=True)
class CanonicalProgram:
    moves: tuple[CanonicalMove, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    controller: str = "generic"

    def cutting_moves(self) -> tuple[CanonicalMove, ...]:
        return tuple(move for move in self.moves if not move.rapid)
