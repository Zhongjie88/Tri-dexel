from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .canonical import CanonicalMotion, CanonicalMove, CanonicalProgram


@dataclass
class GCodeMove:
    x: float
    y: float
    z: float
    feed: float
    rapid: bool
    line_no: Optional[int] = None
    motion_type: Optional[str] = None
    source_line: Optional[str] = None
    segment_index: Optional[int] = None
    segment_count: Optional[int] = None
    arc_center: Optional[Tuple[float, float]] = None
    arc_radius: Optional[float] = None
    arc_direction: Optional[str] = None
    plane: str = "G17"
    controller: str = "generic"
    warnings: Tuple[str, ...] = ()
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    start_a: Optional[float] = None
    start_b: Optional[float] = None
    start_c: Optional[float] = None


class GCodeParser:
    """
    Minimal G-code parser for 3-axis CNC programs.

    Handles: G0/G00 (rapid), G1/G01 (feed), G2/G03 (arc – linearised).
    Ignores canned cycles, tool changes, and coolant commands.
    Modal positions are tracked so partial XYZ lines are resolved correctly.
    """

    _WORD_RE = re.compile(
        r"([A-Z])\s*=?\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)"
    )
    _SINUMERIK_RE = re.compile(
        r"([ABC])\s*=\s*(?:DC|ACP|ACN|AC|IC|CIC|CAC)\s*\(\s*"
        r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[Ee][+-]?\d+)?)\s*\)"
    )
    _SINUMERIK_FUNCS = frozenset(("DC", "ACP", "ACN", "AC", "IC", "CIC", "CAC"))

    def __init__(self, arc_segment_length: float = 1.0, controller: str = "generic") -> None:
        self.arc_segment_length = arc_segment_length
        self.controller = controller
        self._reset_state()

    def _reset_state(self) -> None:
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._a: Optional[float] = None
        self._b: Optional[float] = None
        self._c: Optional[float] = None
        self._feed = 0.0
        self._modal_motion = 0  # 0=G0, 1=G1, 2=G2, 3=G3
        self._absolute = True
        self._plane = "G17"
        self.warnings: list[str] = []

    def _strip_comments(self, line: str) -> str:
        """Strip comments while preserving Siemens axis syntax like C=DC(...)."""
        out: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch in ";%":
                break
            if ch == "(":
                prefix = "".join(out).rstrip().upper()
                if any(prefix.endswith(kw) for kw in self._SINUMERIK_FUNCS):
                    out.append(ch)
                    i += 1
                    continue
                depth = 1
                i += 1
                while i < len(line) and depth:
                    if line[i] == "(":
                        depth += 1
                    elif line[i] == ")":
                        depth -= 1
                    i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _parse_words(self, line: str) -> dict[str, float]:
        clean = self._strip_comments(line).upper().strip()
        words = {m.group(1): float(m.group(2)) for m in self._WORD_RE.finditer(clean)}
        for m in self._SINUMERIK_RE.finditer(clean):
            words[m.group(1)] = float(m.group(2))
        return words

    def parse(self, gcode: str) -> List[GCodeMove]:
        self._reset_state()
        moves: List[GCodeMove] = []
        for line_no, line in enumerate(gcode.splitlines(), start=1):
            line_moves = self._process_line(line, line_no)
            if line_moves:
                moves.extend(line_moves)
        return moves

    def parse_file(self, path: str) -> List[GCodeMove]:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            return self.parse(fh.read())

    def parse_canonical(self, gcode: str) -> CanonicalProgram:
        moves = tuple(self._move_to_canonical(move) for move in self.parse(gcode))
        return CanonicalProgram(
            moves=moves,
            warnings=tuple(self.warnings),
            controller=self.controller,
        )

    def parse_file_canonical(self, path: str) -> CanonicalProgram:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            return self.parse_canonical(fh.read())

    def _g_codes(self, line: str) -> list[int]:
        clean = self._strip_comments(line).upper().strip()
        return [int(float(g)) for g in re.findall(r"G([+-]?\d*\.?\d+)", clean)]

    def _process_line(self, line: str, line_no: Optional[int] = None) -> list[GCodeMove]:
        words = self._parse_words(line)
        if not words:
            return []

        # Update feed rate
        if "F" in words:
            self._feed = words["F"]

        for g in self._g_codes(line):
            if g in (0, 1, 2, 3):
                self._modal_motion = g
            elif g in (17, 18, 19):
                self._plane = f"G{g}"
            elif g == 90:
                self._absolute = True
            elif g == 91:
                self._absolute = False

        # Emit motion for XYZ travel or pure rotary reorientation blocks.
        has_motion = any(k in words for k in "XYZABC")
        if not has_motion:
            return []

        if self._absolute:
            new_x = words.get("X", self._x)
            new_y = words.get("Y", self._y)
            new_z = words.get("Z", self._z)
            new_a = words.get("A", self._a)
            new_b = words.get("B", self._b)
            new_c = words.get("C", self._c)
        else:
            new_x = self._x + words.get("X", 0.0)
            new_y = self._y + words.get("Y", 0.0)
            new_z = self._z + words.get("Z", 0.0)
            new_a = self._add_optional_axis(self._a, words.get("A"))
            new_b = self._add_optional_axis(self._b, words.get("B"))
            new_c = self._add_optional_axis(self._c, words.get("C"))

        if self._modal_motion in (2, 3):
            moves = self._linearize_arc(
                words,
                new_x,
                new_y,
                new_z,
                new_a,
                new_b,
                new_c,
                line_no,
                line,
            )
        else:
            motion_type = f"G{self._modal_motion}"
            moves = [
                GCodeMove(
                    start_x=self._x,
                    start_y=self._y,
                    start_z=self._z,
                    x=new_x,
                    y=new_y,
                    z=new_z,
                    feed=self._feed,
                    rapid=(self._modal_motion == 0),
                    line_no=line_no,
                    motion_type=motion_type,
                    source_line=line.strip(),
                    segment_index=1,
                    segment_count=1,
                    plane=self._plane,
                    controller=self.controller,
                    a=new_a,
                    b=new_b,
                    c=new_c,
                    start_a=self._a,
                    start_b=self._b,
                    start_c=self._c,
                )
            ]

        self._x, self._y, self._z = new_x, new_y, new_z
        self._a, self._b, self._c = new_a, new_b, new_c
        return moves

    @staticmethod
    def _add_optional_axis(current: Optional[float], delta: Optional[float]) -> Optional[float]:
        if delta is None:
            return current
        return (current or 0.0) + delta

    @staticmethod
    def _interpolate_optional_axis(
        start: Optional[float],
        end: Optional[float],
        t: float,
    ) -> Optional[float]:
        if start is None:
            return end
        if end is None:
            return start
        return start + (end - start) * t

    def _arc_sweep(self, sx: float, sy: float, end_x: float, end_y: float, cx: float, cy: float) -> tuple[str, float]:
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(end_y - cy, end_x - cx)

        if self._modal_motion == 2:  # G2 clockwise
            arc_direction = "CW"
            sweep = a1 - a0
            if sweep >= 0.0:
                sweep -= 2.0 * math.pi
        else:  # G3 counter-clockwise
            arc_direction = "CCW"
            sweep = a1 - a0
            if sweep <= 0.0:
                sweep += 2.0 * math.pi
        return arc_direction, sweep

    def _arc_center_from_radius(
        self,
        sx: float,
        sy: float,
        end_x: float,
        end_y: float,
        radius_word: float,
    ) -> Optional[tuple[float, float, float]]:
        r = abs(radius_word)
        dx = end_x - sx
        dy = end_y - sy
        chord = math.hypot(dx, dy)
        if r < 1e-9 or chord < 1e-9 or chord > 2.0 * r + 1e-7:
            return None

        mx = (sx + end_x) * 0.5
        my = (sy + end_y) * 0.5
        h = math.sqrt(max(0.0, r * r - (chord * 0.5) ** 2))
        ux = -dy / chord
        uy = dx / chord
        candidates = [
            (mx + ux * h, my + uy * h),
            (mx - ux * h, my - uy * h),
        ]

        want_major = radius_word < 0.0
        best = None
        for cx, cy in candidates:
            _, sweep = self._arc_sweep(sx, sy, end_x, end_y, cx, cy)
            is_major = abs(sweep) > math.pi + 1e-9
            if is_major == want_major:
                return cx, cy, r
            if best is None:
                best = (cx, cy, r)
        return best

    def _linearize_arc(
        self,
        words: dict[str, float],
        end_x: float,
        end_y: float,
        end_z: float,
        end_a: Optional[float] = None,
        end_b: Optional[float] = None,
        end_c: Optional[float] = None,
        line_no: Optional[int] = None,
        source_line: str = "",
    ) -> list[GCodeMove]:
        """Linearise a G2/G3 XY-plane arc using I/J center offsets or R radius."""
        motion_type = f"G{self._modal_motion}"
        if self._plane != "G17":
            warning = (
                f"line {line_no}: {motion_type} in {self._plane} is preserved as a "
                "linear endpoint; only G17 I/J and R arcs are linearized"
            )
            self.warnings.append(warning)
            return [
                GCodeMove(
                    start_x=self._x,
                    start_y=self._y,
                    start_z=self._z,
                    x=end_x,
                    y=end_y,
                    z=end_z,
                    feed=self._feed,
                    rapid=False,
                    line_no=line_no,
                    motion_type=motion_type,
                    source_line=source_line.strip(),
                    segment_index=1,
                    segment_count=1,
                    plane=self._plane,
                    controller=self.controller,
                    warnings=(warning,),
                    a=end_a,
                    b=end_b,
                    c=end_c,
                    start_a=self._a,
                    start_b=self._b,
                    start_c=self._c,
                )
            ]
        sx, sy, sz = self._x, self._y, self._z
        if "I" in words or "J" in words:
            cx = sx + words.get("I", 0.0)
            cy = sy + words.get("J", 0.0)
            r = math.hypot(sx - cx, sy - cy)
        elif "R" in words:
            center = self._arc_center_from_radius(sx, sy, end_x, end_y, words["R"])
            if center is None:
                warning = (
                    f"line {line_no}: {motion_type} R arc cannot be resolved and "
                    "is preserved as a linear endpoint"
                )
                self.warnings.append(warning)
                return [
                    GCodeMove(
                        start_x=self._x,
                        start_y=self._y,
                        start_z=self._z,
                        x=end_x,
                        y=end_y,
                        z=end_z,
                        feed=self._feed,
                        rapid=False,
                        line_no=line_no,
                        motion_type=motion_type,
                        source_line=source_line.strip(),
                        segment_index=1,
                        segment_count=1,
                        plane=self._plane,
                        controller=self.controller,
                        warnings=(warning,),
                        a=end_a,
                        b=end_b,
                        c=end_c,
                        start_a=self._a,
                        start_b=self._b,
                        start_c=self._c,
                    )
                ]
            cx, cy, r = center
        else:
            return [
                GCodeMove(
                    start_x=self._x,
                    start_y=self._y,
                    start_z=self._z,
                    x=end_x,
                    y=end_y,
                    z=end_z,
                    feed=self._feed,
                    rapid=False,
                    line_no=line_no,
                    motion_type=motion_type,
                    source_line=source_line.strip(),
                    segment_index=1,
                    segment_count=1,
                    plane=self._plane,
                    controller=self.controller,
                    a=end_a,
                    b=end_b,
                    c=end_c,
                    start_a=self._a,
                    start_b=self._b,
                    start_c=self._c,
                )
            ]

        if r < 1e-9:
            return [
                GCodeMove(
                    start_x=self._x,
                    start_y=self._y,
                    start_z=self._z,
                    x=end_x,
                    y=end_y,
                    z=end_z,
                    feed=self._feed,
                    rapid=False,
                    line_no=line_no,
                    motion_type=motion_type,
                    source_line=source_line.strip(),
                    segment_index=1,
                    segment_count=1,
                    plane=self._plane,
                    controller=self.controller,
                    a=end_a,
                    b=end_b,
                    c=end_c,
                    start_a=self._a,
                    start_b=self._b,
                    start_c=self._c,
                )
            ]

        arc_direction, sweep = self._arc_sweep(sx, sy, end_x, end_y, cx, cy)
        a0 = math.atan2(sy - cy, sx - cx)

        arc_len = abs(sweep) * r
        n = max(1, int(math.ceil(arc_len / max(0.05, self.arc_segment_length))))

        moves: list[GCodeMove] = []
        seg_start_x, seg_start_y, seg_start_z = sx, sy, sz
        seg_start_a, seg_start_b, seg_start_c = self._a, self._b, self._c
        for k in range(1, n + 1):
            t = k / n
            a = a0 + sweep * t
            z = sz + (end_z - sz) * t
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            seg_end_a = self._interpolate_optional_axis(self._a, end_a, t)
            seg_end_b = self._interpolate_optional_axis(self._b, end_b, t)
            seg_end_c = self._interpolate_optional_axis(self._c, end_c, t)
            moves.append(
                GCodeMove(
                    start_x=seg_start_x,
                    start_y=seg_start_y,
                    start_z=seg_start_z,
                    x=x,
                    y=y,
                    z=z,
                    feed=self._feed,
                    rapid=False,
                    line_no=line_no,
                    motion_type=motion_type,
                    source_line=source_line.strip(),
                    segment_index=k,
                    segment_count=n,
                    arc_center=(cx, cy),
                    arc_radius=r,
                    arc_direction=arc_direction,
                    plane=self._plane,
                    controller=self.controller,
                    a=seg_end_a,
                    b=seg_end_b,
                    c=seg_end_c,
                    start_a=seg_start_a,
                    start_b=seg_start_b,
                    start_c=seg_start_c,
                )
            )
            seg_start_x, seg_start_y, seg_start_z = x, y, z
            seg_start_a, seg_start_b, seg_start_c = seg_end_a, seg_end_b, seg_end_c
        return moves

    def _move_to_canonical(self, move: GCodeMove) -> CanonicalMove:
        if move.rapid:
            motion = CanonicalMotion.RAPID
        elif move.motion_type in ("G2", "G3"):
            motion = CanonicalMotion.ARC
        elif move.motion_type == "G1":
            motion = CanonicalMotion.LINEAR
        else:
            motion = CanonicalMotion.UNKNOWN

        arc_center = None
        if move.arc_center is not None:
            cx, cy = move.arc_center
            arc_center = (cx, cy, move.start_z)

        return CanonicalMove(
            motion=motion,
            start=(move.start_x, move.start_y, move.start_z),
            end=(move.x, move.y, move.z),
            feed=move.feed,
            plane=move.plane,
            line_no=move.line_no,
            source_line=move.source_line,
            controller=move.controller,
            raw_motion=move.motion_type,
            segment_index=move.segment_index,
            segment_count=move.segment_count,
            arc_center=arc_center,
            arc_radius=move.arc_radius,
            arc_direction=move.arc_direction,
            start_rotary=(move.start_a, move.start_b, move.start_c),
            end_rotary=(move.a, move.b, move.c),
            warnings=move.warnings,
        )
