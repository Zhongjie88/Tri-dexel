"""
demo_pocket.py — serpentine pocket milling demo using tri-dexel simulation.

Stock   : 100 × 100 × 50 mm
Tool    : 10 mm diameter ball-end mill
Strategy: 3-axis serpentine pocket at z=35mm
Resolution: 0.5 mm  (200×200×100 dexels per direction)

Run:
    python -m examples.demo_pocket
"""

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stock.tri_dexel import TriDexelStock
from src.tool.tool_geometry import BallEndMill
from src.simulation.engine import SimulationEngine
from src.reconstruction.mesh import height_map_to_surface, voxel_to_mesh, add_stock_box
from src.visualization.renderer import StockRenderer


def build_serpentine_pocket(
    x_start=10.0, x_end=90.0,
    y_start=10.0, y_end=90.0,
    z_cut=35.0,
    step_over=4.0,
):
    """Generate (start, end) pairs for a serpentine pocket toolpath."""
    moves = []
    y = y_start
    direction = 1
    while y <= y_end + 1e-6:
        xs = x_start if direction == 1 else x_end
        xe = x_end   if direction == 1 else x_start
        moves.append(((xs, y, z_cut + 15), (xs, y, z_cut)))  # plunge
        moves.append(((xs, y, z_cut), (xe, y, z_cut)))        # cut
        moves.append(((xe, y, z_cut), (xe, y, z_cut + 15)))  # retract
        y += step_over
        direction *= -1
    return moves


def main(use_tri_dexel_mesh: bool = False):
    bounds = (0.0, 100.0, 0.0, 100.0, 0.0, 50.0)
    resolution = 0.5

    print("Initialising tri-dexel stock …")
    stock = TriDexelStock(bounds, resolution)
    stock.initialize_box_stock()

    tool = BallEndMill(radius=5.0)
    engine = SimulationEngine(stock, tool)

    toolpath = build_serpentine_pocket(step_over=4.0, z_cut=35.0)
    n_moves = len(toolpath)
    print(f"Simulating {n_moves} linear segments …")
    t0 = time.perf_counter()
    for idx, (start, end) in enumerate(toolpath):
        engine.simulate_move(start, end)
        pct = (idx + 1) * 100 // n_moves
        sys.stdout.write(f"\r  {pct:3d}%  [{idx+1}/{n_moves}]")
        sys.stdout.flush()
    print(f"\nSimulation done in {time.perf_counter() - t0:.1f} s")

    print("Building mesh …")
    if use_tri_dexel_mesh:
        print("  (full tri-dexel → marching cubes – may take a moment)")
        voxels = stock.to_voxel_grid()
        mesh = voxel_to_mesh(voxels, bounds, resolution)
    else:
        hmap = stock.z_grid.height_map()
        x_min, x_max, y_min, y_max = bounds[:4]
        mesh = height_map_to_surface(hmap, x_min, x_max, y_min, y_max)

    print("Rendering …")
    renderer = StockRenderer(title="Tri-dexel pocket demo")
    renderer.add_stock_wireframe(bounds)
    renderer.add_mesh(mesh, color="#c8b89a")

    # Overlay toolpath centreline
    path_pts = []
    for (s, e) in toolpath:
        path_pts.append(s)
        path_pts.append(e)
    renderer.add_tool_path(path_pts)

    renderer.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tri", action="store_true", help="use tri-dexel marching-cubes mesh")
    args = ap.parse_args()
    main(use_tri_dexel_mesh=args.tri)
