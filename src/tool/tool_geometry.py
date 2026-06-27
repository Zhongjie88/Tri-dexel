from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class ToolGeometry(ABC):
    """
    Abstract tool geometry.

    All tools are assumed to be axis-symmetric, rotating about Z (the tool axis).
    'tip_z' is the Z coordinate of the lowest point of the tool.

    Two queries drive the simulation:
      z_cut(d, tip_z)               — for Z-dexel updates
      cross_section_radius(zk, tip_z) — for X/Y-dexel updates
    """

    @property
    @abstractmethod
    def radius(self) -> float:
        """Maximum radial extent of the cutting envelope."""

    @abstractmethod
    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        """
        Z coordinate of the lowest tool surface at radial offset d from the axis.
        Returns None if d is outside the tool's cutting footprint.

        Material above z_cut at this radial offset is removed.
        """

    @abstractmethod
    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        """
        Cutting radius of the tool at height zk.
        Returns None if zk is below the tool tip (no tool material there).

        Used for X- and Y-dexel updates: at a given height, the tool cuts a
        circular cross section of this radius.
        """

    # ------------------------------------------------------------------
    # Vectorised helpers (override in subclasses for speed)
    # ------------------------------------------------------------------

    def z_cut_arr(self, d_arr: np.ndarray, tip_z: float) -> np.ndarray:
        """Vectorised z_cut; NaN where d is outside the tool footprint."""
        out = np.full(d_arr.shape, np.nan)
        for idx in np.ndindex(d_arr.shape):
            zc = self.z_cut(float(d_arr[idx]), tip_z)
            if zc is not None:
                out[idx] = zc
        return out

    def cross_section_radius_arr(
        self, zk_arr: np.ndarray, tip_z: float
    ) -> tuple:
        """Vectorised cross_section_radius.

        Returns (valid, rz) where both are 1-D arrays of length len(zk_arr).
        valid[k] is True when zk_arr[k] is within the tool body.
        rz[k] is the cross-section radius (0 where not valid).
        """
        n = len(zk_arr)
        valid = np.zeros(n, dtype=bool)
        rz    = np.zeros(n, dtype=float)
        for k, zk in enumerate(zk_arr):
            r = self.cross_section_radius(float(zk), tip_z)
            if r is not None:
                valid[k] = True
                rz[k]    = r
        return valid, rz


# ---------------------------------------------------------------------------
# Concrete tool types
# ---------------------------------------------------------------------------


class FlatEndMill(ToolGeometry):
    """
    Flat-bottom end mill (square end).

    Cutting geometry:
      - Flat disk of radius r at z = tip_z
      - Cylindrical shank above
    """

    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        if d > self._radius:
            return None
        return tip_z

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        if zk < tip_z:
            return None
        return self._radius

    def z_cut_arr(self, d_arr: np.ndarray, tip_z: float) -> np.ndarray:
        out = np.full(d_arr.shape, np.nan)
        out[d_arr <= self._radius] = tip_z
        return out

    def cross_section_radius_arr(self, zk_arr: np.ndarray, tip_z: float) -> tuple:
        valid = zk_arr >= tip_z
        rz    = np.where(valid, self._radius, 0.0)
        return valid, rz


class BallEndMill(ToolGeometry):
    """
    Ball-nose end mill.

    Cutting geometry:
      - Hemisphere of radius r, center at (0, 0, tip_z + r)
      - Tip at (0, 0, tip_z)
      - Cylindrical shank above equator (z > tip_z + r)
    """

    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        r = self._radius
        if d > r:
            return None
        # z_cut at radial offset d: lowest point of sphere surface
        return tip_z + r - math.sqrt(r * r - d * d)

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        r = self._radius
        if zk < tip_z:
            return None
        if zk <= tip_z + r:
            # Ball zone: cross-section radius shrinks toward tip
            val = r * r - (zk - tip_z - r) ** 2
            return math.sqrt(max(0.0, val))
        # Shank zone: full radius
        return r

    def z_cut_arr(self, d_arr: np.ndarray, tip_z: float) -> np.ndarray:
        r = self._radius
        in_foot = d_arr <= r
        d2 = d_arr * d_arr
        out = np.full(d_arr.shape, np.nan)
        out[in_foot] = tip_z + r - np.sqrt(np.maximum(0.0, r*r - d2[in_foot]))
        return out

    def cross_section_radius_arr(self, zk_arr: np.ndarray, tip_z: float) -> tuple:
        r = self._radius
        ball_zone  = (zk_arr >= tip_z) & (zk_arr <= tip_z + r)
        shank_zone = zk_arr > tip_z + r
        valid = ball_zone | shank_zone
        rz = np.zeros(len(zk_arr), dtype=float)
        if np.any(ball_zone):
            dz = zk_arr[ball_zone] - (tip_z + r)
            rz[ball_zone] = np.sqrt(np.maximum(0.0, r*r - dz*dz))
        rz[shank_zone] = r
        return valid, rz


class BullNoseEndMill(ToolGeometry):
    """
    Bull-nose (toroidal) end mill: flat center with rounded corner.

    radius       — outer radius of the tool
    corner_radius — radius of the corner fillet
    flat_radius   = radius - corner_radius (flat region)
    """

    def __init__(self, radius: float, corner_radius: float) -> None:
        if corner_radius >= radius:
            raise ValueError("corner_radius must be less than radius")
        self._radius = radius
        self._cr = corner_radius
        self._flat_r = radius - corner_radius

    @property
    def radius(self) -> float:
        return self._radius

    def z_cut(self, d: float, tip_z: float) -> Optional[float]:
        r, cr, fr = self._radius, self._cr, self._flat_r
        if d > r:
            return None
        if d <= fr:
            return tip_z
        # Corner fillet: torus cross-section
        offset = d - fr
        return tip_z + cr - math.sqrt(max(0.0, cr * cr - offset * offset))

    def cross_section_radius(self, zk: float, tip_z: float) -> Optional[float]:
        r, cr, fr = self._radius, self._cr, self._flat_r
        if zk < tip_z:
            return None
        if zk <= tip_z + cr:
            # Corner fillet zone: outer boundary shrinks
            dz = zk - tip_z
            outer = fr + math.sqrt(max(0.0, cr * cr - (dz - cr) ** 2))
            return outer
        return r
