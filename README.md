# Tri-dexel NC Machining Simulator

A Python-based CNC machining simulation system using the **tri-dexel** material representation for accurate 3-axis (and extensible to 5-axis) material removal.

## Features

- Tri-dexel stock model (three orthogonal dexel grids) for accurate vertical walls and near-vertical surfaces
- G-code parser: G0/G1 (rapid/feed), G2/G3 (arc linearisation), G17/18/19 (planes), G20/G21 (inch/mm), G90/G91 (absolute/incremental)
- Tool types: Ball-end mill, Flat-end mill, Bull-nose end mill, Taper tool
- Real-time height-map preview during simulation
- Final marching-cubes mesh reconstruction from the full tri-dexel voxel grid
- PyQt6 + PyVista GUI with adjustable speed slider

## Requirements

- Anaconda or Miniconda on Windows
- The project setup creates a `tridexel-occ` conda environment with Python 3.10
- OpenCascade / `pythonocc-core` is installed from conda-forge by `setup.bat`
- Python runtime packages are installed from `requirements.txt`

## Quick Start

First-time setup:

```bat
setup.bat
```

Then launch the application:

```bat
run.bat
```

Or manually:

```bash
conda create -n tridexel-occ -c conda-forge python=3.10 pip pythonocc-core
conda activate tridexel-occ
pip install -r requirements.txt
python app.py
```

## Usage

1. **Stock** — choose a box (enter Width × Depth × Height) or load an STL/OBJ/PLY file
2. **Tool path** — select a built-in demo (Serpentine Pocket or Spiral Contour) or load an NC/G-code file
3. **Tool** — select Ball-end Mill or Flat-end Mill and set the radius
4. **Resolution** — smaller values give higher fidelity but slower simulation
5. Click **Start Simulation**

## Project Structure

```
app.py                    GUI entry point
src/
  gcode/parser.py         G-code parser
  simulation/engine.py    Material removal engine
  stock/                  Tri-dexel stock model
    tri_dexel.py          TriDexelStock (3 orthogonal grids)
    numpy_grid.py         Numpy/numba-backed dexel grids
    dexel_ray.py          Single dexel ray (material intervals)
  tool/tool_geometry.py   Tool geometry definitions
  reconstruction/mesh.py  Height-map and voxel-to-mesh reconstruction
examples/                 Standalone demo scripts
tests/                    pytest test suite
```

## Running Tests

```bash
pip install pytest
pytest
```
