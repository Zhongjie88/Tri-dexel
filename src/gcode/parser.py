from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GCodeMove:
    x: float
    y: float
    z: float
    feed: float
    rapid: bool


class GCodeParser:
    """
    Minimal G-code parser for 3-axis CNC programs.

    Handles: G0/G00 (rapid), G1/G01 (feed), G2/G03 (arc – linearised).
    Ignores canned cycles, tool changes, and coolant commands.
    Modal positions are tracked so partial XYZ lines are resolved correctly.
    """

    _WORD_RE = re.compile(r"([A-Z])([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)")

    def __init__(self, arc_segment_length: float = 1.0) -> None:
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._feed = 0.0
        self._modal_motion = 0  # 0=G0, 1=G1, 2=G2, 3=G3
        self._absolute = True
        self.arc_segment_length = arc_segment_length

    def _parse_words(self, line: str) -> dict[str, float]:
        line = re.split(r"[;(%]", line)[0].upper().strip()
        return {m.group(1): float(m.group(2)) for m in self._WORD_RE.finditer(line)}

    def parse(self, gcode: str) -> List[GCodeMove]:
        moves: List[GCodeMove] = []
        for line in gcode.splitlines():
            line_moves = self._process_line(line)
            if line_moves:
                moves.extend(line_moves)
        return moves

    def parse_file(self, path: str) -> List[GCodeMove]:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            return self.parse(fh.read())

    def _g_codes(self, line: str) -> list[int]:
        clean = re.split(r"[;(%]", line)[0].upper().strip()
        return [int(float(g)) for g in re.findall(r"G([+-]?\d*\.?\d+)", clean)]

    def _process_line(self, line: str) -> list[GCodeMove]:
        words = self._parse_words(line)
        if not words:
            return []

        # Update feed rate
        if "F" in words:
            self._feed = words["F"]

        for g in self._g_codes(line):
            if g in (0, 1, 2, 3):
                self._modal_motion = g
            elif g == 90:
                self._absolute = True
            elif g == 91:
                self._absolute = False

        # Only emit a move when XYZ coordinates are present
        has_motion = any(k in words for k in "XYZ")
        if not has_motion:
            return []

        if self._absolute:
            new_x = words.get("X", self._x)
            new_y = words.get("Y", self._y)
            new_z = words.get("Z", self._z)
        else:
            new_x = self._x + words.get("X", 0.0)
            new_y = self._y + words.get("Y", 0.0)
            new_z = self._z + words.get("Z", 0.0)

        if self._modal_motion in (2, 3):
            moves = self._linearize_arc(words, new_x, new_y, new_z)
        else:
            moves = [
                GCodeMove(
                    x=new_x,
                    y=new_y,
                    z=new_z,
                    feed=self._feed,
                    rapid=(self._modal_motion == 0),
                )
            ]

        self._x, self._y, self._z = new_x, new_y, new_z
        return moves

    def _linearize_arc(
        self,
        words: dict[str, float],
        end_x: float,
        end_y: float,
        end_z: float,
    ) -> list[GCodeMove]:
        """Linearise a G2/G3 XY-plane arc using incremental I/J centre offsets."""
        if "I" not in words and "J" not in words:
            return [GCodeMove(end_x, end_y, end_z, self._feed, rapid=False)]

        sx, sy, sz = self._x, self._y, self._z
        cx = sx + words.get("I", 0.0)
        cy = sy + words.get("J", 0.0)
        r = math.hypot(sx - cx, sy - cy)
        if r < 1e-9:
            return [GCodeMove(end_x, end_y, end_z, self._feed, rapid=False)]

        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(end_y - cy, end_x - cx)

        if self._modal_motion == 2:  # G2 clockwise
            sweep = a1 - a0
            if sweep >= 0.0:
                sweep -= 2.0 * math.pi
        else:  # G3 counter-clockwise
            sweep = a1 - a0
            if sweep <= 0.0:
                sweep += 2.0 * math.pi

        arc_len = abs(sweep) * r
        n = max(1, int(math.ceil(arc_len / max(0.05, self.arc_segment_length))))

        moves: list[GCodeMove] = []
        for k in range(1, n + 1):
            t = k / n
            a = a0 + sweep * t
            z = sz + (end_z - sz) * t
            moves.append(
                GCodeMove(
                    x=cx + r * math.cos(a),
                    y=cy + r * math.sin(a),
                    z=z,
                    feed=self._feed,
                    rapid=False,
                )
            )
        return moves
