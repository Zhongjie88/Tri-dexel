from __future__ import annotations

from typing import Any, Optional

try:
    import pyvista as pv
    _PYVISTA = True
except ImportError:
    _PYVISTA = False


class StockRenderer:
    """
    Thin PyVista wrapper for displaying machined stock results.

    Usage
    -----
    r = StockRenderer(title="Pocket demo")
    r.add_stock_wireframe(bounds)
    r.add_mesh(surface_mesh, color="#d4b896")
    r.show()
    """

    # Aesthetic defaults that approximate the grey "in-progress" look of
    # commercial CAM verification software.
    DEFAULT_STOCK_COLOR = "#c8c8c8"
    DEFAULT_BG_COLOR = "#2b2b2b"

    def __init__(self, title: str = "Tri-dexel Simulation", off_screen: bool = False) -> None:
        if not _PYVISTA:
            raise ImportError("pyvista is required for visualization")
        self.pl = pv.Plotter(title=title, off_screen=off_screen)
        self.pl.set_background(self.DEFAULT_BG_COLOR)

    def add_mesh(
        self,
        mesh: Any,
        color: str = DEFAULT_STOCK_COLOR,
        opacity: float = 1.0,
        smooth_shading: bool = True,
        **kwargs: Any,
    ) -> None:
        self.pl.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            smooth_shading=smooth_shading,
            **kwargs,
        )

    def add_stock_wireframe(self, bounds: tuple[float, ...], color: str = "#4488ff") -> None:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        box = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
        self.pl.add_mesh(box, style="wireframe", color=color, line_width=2, opacity=0.4)

    def add_tool_path(self, points: list[tuple[float, float, float]], color: str = "#ff4444") -> None:
        """Visualise the tool centre-line trajectory."""
        import numpy as np
        pts = np.array(points, dtype=float)
        if len(pts) < 2:
            return
        lines = pv.lines_from_points(pts)
        self.pl.add_mesh(lines, color=color, line_width=1.5)

    def show(self, axes: bool = True, bounds: bool = True) -> None:
        if axes:
            self.pl.show_axes()
        if bounds:
            self.pl.show_bounds(minor_ticks=True, font_size=8)
        self.pl.show()

    def screenshot(self, path: str, transparent: bool = False) -> None:
        self.pl.screenshot(path, transparent_background=transparent)
        print(f"Screenshot saved → {path}")
