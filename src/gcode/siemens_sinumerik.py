from __future__ import annotations

from .parser import GCodeParser


class SiemensSinumerikParser(GCodeParser):
    """SINUMERIK MPF adapter.

    Tracks TRAORI/TRAFOOF blocks so A/C rotary values are only used as
    tool-orientation when five-axis transformation is active.
    CYCLE800 and other cycles are silently skipped.
    """

    def __init__(self, arc_segment_length: float = 1.0) -> None:
        super().__init__(arc_segment_length=arc_segment_length, controller="siemens")
        self._in_traori = False

    def _reset_state(self) -> None:
        super()._reset_state()
        self._in_traori = False

    def _process_line(self, line: str, line_no=None):
        # Strip inline comment and N-word block number, then check keywords.
        stripped = line.strip().split(";")[0].strip().upper()
        # Remove leading N-number (e.g. "N10 TRAORI" → "TRAORI")
        if stripped and stripped[0] == "N" and len(stripped) > 1 and stripped[1].isdigit():
            parts = stripped.split(None, 1)
            stripped = parts[1] if len(parts) > 1 else ""

        if stripped.startswith("TRAORI"):
            self._in_traori = True
            return []
        if stripped.startswith("TRAFOOF"):
            self._in_traori = False
            self._a = None
            self._b = None
            self._c = None
            return []
        if stripped.startswith("CYCLE"):
            return []

        moves = super()._process_line(line, line_no)
        if not self._in_traori:
            for m in moves:
                m.a = None
                m.b = None
                m.c = None
                m.start_a = None
                m.start_b = None
                m.start_c = None
        return moves
