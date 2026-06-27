"""
demo_animated.py — real-time machining animation using tri-dexel simulation.

Usage
-----
  python -m examples.demo_animated                   # default pocket
  python -m examples.demo_animated --tool flat       # flat end mill
  python -m examples.demo_animated --res 0.5         # higher resolution (slower)
  python -m examples.demo_animated --gcode examples/sample.nc

Controls (PyVista window)
  Left-drag   rotate
  Right-drag  zoom
  Middle-drag pan
  R           reset camera
  Q / Esc     quit
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import BallEndMill, FlatEndMill
from src.simulation.engine import SimulationEngine
from src.visualization.animator import MachiningAnimator


# ---------------------------------------------------------------------------
# Toolpath generators
# ---------------------------------------------------------------------------

def serpentine_pocket(
    x_start=10.0, x_end=90.0,
    y_start=10.0, y_end=90.0,
    z_cut=35.0,
    step_over=6.0,
    retract_z=50.0,
):
    """Serpentine (zig-zag) pocket toolpath."""
    segments = []
    y = y_start
    direction = 1
    while y <= y_end + 1e-6:
        xs = x_start if direction == 1 else x_end
        xe = x_end   if direction == 1 else x_start
        segments.append(((xs, y, retract_z), (xs, y, z_cut)))  # plunge
        segments.append(((xs, y, z_cut),     (xe, y, z_cut)))  # cut
        segments.append(((xe, y, z_cut),     (xe, y, retract_z)))  # retract
        y += step_over
        direction *= -1
    return segments


def contour_pocket(
    cx=50.0, cy=50.0,
    r_start=35.0, r_end=5.0,
    z_cut=35.0,
    step_over=5.0,
    n_points=72,
):
    """Spiral contour pocket (circular passes shrinking inward)."""
    import math
    segments = []
    r = r_start
    while r >= r_end - 1e-6:
        pts = []
        for k in range(n_points + 1):
            angle = 2 * math.pi * k / n_points
            pts.append((
                cx + r * math.cos(angle),
                cy + r * math.sin(angle),
                z_cut,
            ))
        # plunge to first point
        segments.append(((pts[0][0], pts[0][1], z_cut + 15), pts[0]))
        for k in range(len(pts) - 1):
            segments.append((pts[k], pts[k + 1]))
        r -= step_over
    return segments


# ---------------------------------------------------------------------------
# G-code toolpath
# ---------------------------------------------------------------------------

def gcode_toolpath(nc_path):
    from src.gcode.parser import GCodeParser
    parser = GCodeParser()
    moves = parser.parse_file(nc_path)
    segments = []
    prev = None
    for m in moves:
        cur = (m.x, m.y, m.z)
        if prev is not None and not m.rapid:
            segments.append((prev, cur))
        prev = cur
    return segments, moves


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Tri-dexel real-time machining animation")
    ap.add_argument("--gcode", metavar="FILE", help="NC file to simulate")
    ap.add_argument("--tool",  choices=["ball", "flat"], default="ball")
    ap.add_argument("--radius", type=float, default=5.0, help="tool radius mm")
    ap.add_argument("--res",   type=float, default=1.0,  help="grid resolution mm")
    ap.add_argument("--spf",   type=int,   default=2,    help="segments per animation frame")
    ap.add_argument("--fps",   type=int,   default=12,   help="target animation fps")
    ap.add_argument("--spiral", action="store_true",      help="use spiral contour pocket")
    args = ap.parse_args()

    # --- build toolpath + bounds ---
    if args.gcode:
        print(f"Parsing {args.gcode} ...")
        segments, moves = gcode_toolpath(args.gcode)
        xs = [m.x for m in moves]
        ys = [m.y for m in moves]
        zs = [m.z for m in moves]
        margin = 10.0
        bounds = (
            min(xs) - margin, max(xs) + margin,
            min(ys) - margin, max(ys) + margin,
            min(zs) - 5.0,   min(zs) + 40.0,
        )
    elif args.spiral:
        segments = contour_pocket(z_cut=35.0, step_over=5.0)
        bounds = (0.0, 100.0, 0.0, 100.0, 0.0, 50.0)
    else:
        segments = serpentine_pocket(step_over=6.0, z_cut=35.0)
        bounds = (0.0, 100.0, 0.0, 100.0, 0.0, 50.0)

    print(f"Toolpath: {len(segments)} segments")

    # --- stock + engine ---
    print(f"Initialising stock  ({args.res} mm resolution) ...")
    stock = TriDexelStock(bounds, args.res)
    stock.initialize_box_stock()

    tool = BallEndMill(args.radius) if args.tool == "ball" else FlatEndMill(args.radius)
    engine = SimulationEngine(stock, tool)

    print(f"Tool  : {type(tool).__name__}  r={args.radius} mm")
    print(f"Grid  : {stock.nx} × {stock.ny} × {stock.nz}")
    print(f"Frame : {args.spf} segment(s)/frame  @  {args.fps} fps target")
    print()
    print("Opening animation window …")
    print("  Rotate: left-drag | Zoom: right-drag | Pan: middle-drag | Quit: Q")
    print()

    animator = MachiningAnimator(
        stock=stock,
        engine=engine,
        toolpath=segments,
        segments_per_frame=args.spf,
        fps=args.fps,
        title="Tri-dexel Real-time Machining",
    )
    animator.run()


if __name__ == "__main__":
    main()
