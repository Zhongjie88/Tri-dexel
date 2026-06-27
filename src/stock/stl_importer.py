"""
stl_importer.py
Load an external mesh file (.STL, .STEP, .OBJ, …) and initialise a
TriDexelStock from it.

Public API
----------
  load_mesh(path)                         → pv.PolyData
  initialize_stock_from_mesh(stock, mesh) → None  (mutates stock in-place)
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(path: str):
    """
    Load a mesh file and return a cleaned pv.PolyData.

    Formats supported by PyVista/VTK natively:
        .stl  .obj  .ply  .vtk  .vtp  and many others.

    STEP / STP files require pythonocc-core:
        conda install -c conda-forge pythonocc-core
    or convert to STL first (FreeCAD, Fusion 360, online converters).
    """
    import pyvista as pv

    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("step", "stp"):
        return _load_step(path)

    mesh = pv.read(path)
    if not hasattr(mesh, "n_points") or mesh.n_points == 0:
        raise RuntimeError(f"Could not read mesh from: {path}")
    return mesh.triangulate().clean()


def _load_step(path: str):
    """
    Convert a STEP file to pv.PolyData.

    Tries loaders in order of installation weight:
      1. gmsh          (pip install gmsh)          ← lightweight, recommended
      2. pythonocc-core (conda install pythonocc-core) ← full OCC
    Raises RuntimeError with install instructions if neither is available.
    """
    errors: list[str] = []

    # ── 1. Try gmsh (pip install gmsh) ───────────────────────────────
    try:
        return _load_step_gmsh(path)
    except ImportError as exc:
        errors.append(f"gmsh is not installed: {exc}")
    except Exception as exc:
        errors.append(f"gmsh could not mesh this STEP file: {exc}")

    # ── 2. Try pythonocc-core ─────────────────────────────────────────
    try:
        return _load_step_occ(path)
    except ImportError as exc:
        errors.append(f"pythonocc-core is not installed: {exc}")
    except Exception as exc:
        errors.append(f"pythonocc-core could not read this STEP file: {exc}")

    details = "\n".join(f"  - {msg}" for msg in errors)

    raise RuntimeError(
        "STEP import failed.\n\n"
        f"{details}\n\n"
        "Recommended fixes:\n"
        "  1. Export the CAD model as STL and import the STL.\n"
        "  2. Try a simpler STEP export setting, such as AP214/AP242.\n"
        "  3. Install pythonocc-core for a full OpenCascade STEP reader."
    )


def _load_step_gmsh(path: str):
    """STEP → pv.PolyData via gmsh (pip install gmsh)."""
    import gmsh          # noqa: F401 — ImportError propagates to caller
    import pyvista as pv

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)   # suppress console output
        gmsh.model.add("step_model")
        gmsh.merge(path)
        gmsh.model.occ.synchronize()

        # Generate a surface triangulation; characteristic length controls density.
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        elem_types, _, elem_nodes = gmsh.model.mesh.getElements()

        verts = node_coords.reshape(-1, 3)
        # Map 1-based gmsh node tags → 0-based array indices
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        faces = []
        for etype, enodes in zip(elem_types, elem_nodes):
            if int(etype) == 2:          # element type 2 = 3-node triangle
                tris = enodes.reshape(-1, 3)
                for tri in tris:
                    faces += [3,
                              tag_to_idx[int(tri[0])],
                              tag_to_idx[int(tri[1])],
                              tag_to_idx[int(tri[2])]]
    finally:
        gmsh.finalize()

    if not faces:
        raise RuntimeError("gmsh produced no triangles from this STEP file.")

    mesh = pv.PolyData(verts, np.array(faces, dtype=int))
    return mesh.triangulate().clean()


def _load_step_occ(path: str):
    """STEP → pv.PolyData via pythonocc-core."""
    from OCC.Extend.DataExchange import read_step_file   # noqa: F401
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    import pyvista as pv

    shape = read_step_file(path)
    BRepMesh_IncrementalMesh(shape, 0.1).Perform()

    verts, faces_list, offset = [], [], 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        tri = BRep_Tool.Triangulation(face, face.Location())
        if tri is not None:
            for i in range(1, tri.NbNodes() + 1):
                n = tri.Node(i)
                verts.append([n.X(), n.Y(), n.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                faces_list += [3, offset + a - 1, offset + b - 1, offset + c - 1]
            offset += tri.NbNodes()
        explorer.Next()

    if not verts:
        raise RuntimeError("No geometry found in STEP file.")

    mesh = pv.PolyData(np.array(verts, dtype=float), np.array(faces_list, dtype=int))
    return mesh.triangulate().clean()


# ---------------------------------------------------------------------------
# TriDexelStock initialisation
# ---------------------------------------------------------------------------

def _voxelize_compat(mesh, resolution: float):
    """
    Thin wrapper that handles PyVista's voxelize API across versions:

      >= 0.46  mesh.voxelize(spacing=...)          (DataSetFilters method)
      0.44-0.45  mesh.voxelize(density=..., check_surface=False)
      < 0.44   pv.voxelize(mesh, density=..., check_surface=False)
    """
    # ── PyVista >= 0.46: spacing= keyword, method on mesh ────────────
    try:
        return mesh.voxelize(spacing=resolution)
    except TypeError:
        pass   # wrong kwargs, try next

    # ── PyVista 0.44-0.45: density= keyword, method on mesh ──────────
    try:
        return mesh.voxelize(density=resolution, check_surface=False)
    except (AttributeError, TypeError):
        pass

    # ── PyVista < 0.44: module-level function ─────────────────────────
    import pyvista as pv
    return pv.voxelize(mesh, density=resolution, check_surface=False)


def initialize_stock_from_mesh(stock, mesh) -> None:
    """
    Fill all three dexel grids of *stock* from a closed surface mesh.

    Algorithm
    ---------
    1. Voxelise the mesh at stock.resolution using PyVista.
    2. Map voxel cell-centres to boolean ndarray [nx, ny, nz].
    3. Scan each axis direction to derive per-ray material intervals.

    Parameters
    ----------
    stock : TriDexelStock   Already constructed; will be written in-place.
    mesh  : pv.PolyData     Closed solid mesh (the feedstock shape).
    """
    import pyvista as pv

    mesh = mesh.triangulate().clean()

    vox = _voxelize_compat(mesh, stock.resolution)
    if vox.n_cells == 0:
        raise RuntimeError(
            "Voxelization returned no cells.\n"
            "Ensure the mesh is a closed solid and the resolution is small "
            "enough relative to the mesh size."
        )

    nx, ny, nz = stock.nx, stock.ny, stock.nz
    res = stock.resolution
    x_min, y_min, z_min = stock.x_min, stock.y_min, stock.z_min

    # Map cell centres to voxel indices (vectorised)
    centres = vox.cell_centers().points                          # (N, 3)
    ii = np.floor((centres[:, 0] - x_min) / res).astype(int)
    jj = np.floor((centres[:, 1] - y_min) / res).astype(int)
    kk = np.floor((centres[:, 2] - z_min) / res).astype(int)
    valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
    voxels = np.zeros((nx, ny, nz), dtype=bool)
    voxels[ii[valid], jj[valid], kk[valid]] = True

    _init_dexels_from_voxels(stock, voxels)


def _init_dexels_from_voxels(stock, voxels: np.ndarray) -> None:
    """Write all three dexel grids from a boolean voxel array [nx, ny, nz]."""
    nx, ny, nz = stock.nx, stock.ny, stock.nz
    res = stock.resolution
    x_min, y_min, z_min = stock.x_min, stock.y_min, stock.z_min

    # Z-grid: one ray per (i, j) scanning along k (depth = Z)
    for i in range(nx):
        for j in range(ny):
            stock.z_grid.rays[i][j].intervals = _col_intervals(
                voxels[i, j, :], z_min, res
            )

    # X-grid: one ray per (j, k) scanning along i (depth = X)
    for j in range(ny):
        for k in range(nz):
            stock.x_grid.rays[j][k].intervals = _col_intervals(
                voxels[:, j, k], x_min, res
            )

    # Y-grid: one ray per (i, k) scanning along j (depth = Y)
    for i in range(nx):
        for k in range(nz):
            stock.y_grid.rays[i][k].intervals = _col_intervals(
                voxels[i, :, k], y_min, res
            )

    # Rebuild Z-grid height cache (rays were written directly above)
    stock.z_grid.sync_height_cache()


def _col_intervals(arr: np.ndarray, origin: float, resolution: float) -> list:
    """Convert a 1D bool array into sorted (lo, hi) material intervals."""
    intervals: list = []
    in_mat = False
    lo = 0.0
    n = len(arr)
    for k in range(n):
        pos = origin + k * resolution
        if arr[k] and not in_mat:
            lo = pos
            in_mat = True
        elif not arr[k] and in_mat:
            intervals.append((lo, pos))
            in_mat = False
    if in_mat:
        intervals.append((lo, origin + n * resolution))
    return intervals
