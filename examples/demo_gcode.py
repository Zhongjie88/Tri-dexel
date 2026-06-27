"""
demo_gcode.py — simulate a .nc file and display the result.

Usage:
    python -m examples.demo_gcode examples/sample.nc --resolution 0.5

The script auto-detects the stock bounding box from the G-code extents
and adds a small margin so the initial block is visible.
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gcode.parser import GCodeParser
from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import FlatEndMill, BallEndMill
from src.simulation.engine import SimulationEngine
from src.reconstruction.mesh import height_map_to_surface
from src.visualization.renderer import StockRenderer


def infer_bounds(moves, margin=5.0, z_stock_height=30.0):
    xs = [m.x for m in moves]
    ys = [m.y for m in moves]
    zs = [m.z for m in moves]
    z_min = min(zs)
    return (
        min(xs) - margin, max(xs) + margin,
        min(ys) - margin, max(ys) + margin,
        z_min - 2.0, z_min + z_stock_height,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nc_file", nargs="?", default="examples/sample.nc")
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=5.0, help="tool radius mm")
    ap.add_argument("--tool", choices=["flat", "ball"], default="ball")
    args = ap.parse_args()

    print(f"Parsing {args.nc_file} …")
    parser = GCodeParser()
    moves = parser.parse_file(args.nc_file)
    feed_moves = [m for m in moves if not m.rapid]
    print(f"  {len(moves)} total moves, {len(feed_moves)} cutting moves")

    bounds = infer_bounds(moves)
    print(f"  inferred bounds: {tuple(round(b,1) for b in bounds)}")

    stock = TriDexelStock(bounds, args.resolution)
    stock.initialize_box_stock()

    tool = BallEndMill(args.radius) if args.tool == "ball" else FlatEndMill(args.radius)
    print(f"  tool: {type(tool).__name__} r={args.radius} mm")

    engine = SimulationEngine(stock, tool)
    print("Simulating …")
    t0 = time.perf_counter()
    engine.simulate_gcode(moves, progress=True)
    print(f"Done in {time.perf_counter() - t0:.1f} s")

    hmap = stock.z_grid.height_map()
    mesh = height_map_to_surface(hmap, bounds[0], bounds[1], bounds[2], bounds[3])

    renderer = StockRenderer(title=f"NC: {os.path.basename(args.nc_file)}")
    renderer.add_stock_wireframe(bounds)
    renderer.add_mesh(mesh)
    renderer.show()


if __name__ == "__main__":
    main()
