from __future__ import annotations
import bisect
from typing import Any, Callable, List, Mapping, Optional, Tuple

from .surface_metadata import SurfaceMetadata, make_surface_metadata

Interval = Tuple[float, float]


class DexelRay:
    """
    Single dexel ray: sorted, non-overlapping material intervals along one axis.

    Example: [(0.0, 10.0), (15.0, 20.0)] means material exists at [0,10] and [15,20],
    with a void (hole or slot) between 10 and 15.
    """

    __slots__ = ("intervals", "surface_metadata", "version", "_on_change")

    def __init__(self, on_change: Optional[Callable[["DexelRay"], None]] = None) -> None:
        self.intervals: List[Interval] = []
        self.surface_metadata: List[SurfaceMetadata] = []
        self.version = 0
        self._on_change = on_change

    def _mark_changed(self) -> None:
        self.version += 1
        if self._on_change is not None:
            self._on_change(self)

    def set_solid(self, lo: float, hi: float) -> None:
        """Initialize ray as a single solid block from lo to hi."""
        self.intervals = [(lo, hi)]
        self.surface_metadata.clear()
        self._mark_changed()

    def is_empty(self) -> bool:
        return len(self.intervals) == 0

    def subtract(
        self,
        cut_lo: float,
        cut_hi: float,
        metadata: Optional[Mapping[str, Any] | SurfaceMetadata] = None,
    ) -> None:
        """Remove material in range [cut_lo, cut_hi]. No-op if range is degenerate."""
        if cut_lo >= cut_hi:
            return
        result: List[Interval] = []
        changed = False
        for lo, hi in self.intervals:
            if hi <= cut_lo or lo >= cut_hi:
                result.append((lo, hi))
                continue
            # This interval overlaps the cut — material is removed.
            changed = True
            if lo < cut_lo:
                result.append((lo, cut_lo))
            if hi > cut_hi:
                result.append((cut_hi, hi))
        if changed:
            self.intervals = result
            if metadata is not None:
                self.surface_metadata.append(make_surface_metadata((cut_lo, cut_hi), metadata))
            self._mark_changed()

    def union(self, add_lo: float, add_hi: float) -> None:
        """Add material in [add_lo, add_hi], merging overlapping intervals."""
        if add_lo >= add_hi:
            return
        bisect.insort(self.intervals, (add_lo, add_hi))
        merged: List[Interval] = [self.intervals[0]]
        for lo, hi in self.intervals[1:]:
            if lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        if merged != self.intervals:
            self.intervals = merged
            self._mark_changed()

    def top(self) -> Optional[float]:
        """Highest material boundary, or None if empty."""
        return self.intervals[-1][1] if self.intervals else None

    def bottom(self) -> Optional[float]:
        """Lowest material boundary, or None if empty."""
        return self.intervals[0][0] if self.intervals else None

    def contains(self, value: float) -> bool:
        for lo, hi in self.intervals:
            if lo > value:
                return False  # intervals are sorted; nothing later can match
            if value <= hi:
                return True
        return False

    def __repr__(self) -> str:
        return f"DexelRay({self.intervals})"
