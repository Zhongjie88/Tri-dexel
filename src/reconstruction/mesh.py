from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
    _PYVISTA = True
except ImportError:
    _PYVISTA = False

try:
    from skimage.measure import marching_cubes as _mc
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


def height_map_to_surface(
    height_map: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> "pv.StructuredGrid":
    """
    Convert a Z-dexel height map to a PyVista StructuredGrid surface.

    This is the fast path for display: O(nx*ny) with no Marching Cubes needed.
    The output is a watertight top surface – suitable for real-time preview.
    """
    if not _PYVISTA:
        raise ImportError("pyvista is required for mesh reconstruction")

    nx, ny = height_map.shape
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.nan_to_num(height_map, nan=0.0)

    return pv.StructuredGrid(xx, yy, zz)


def voxel_to_mesh(
    voxels: np.ndarray,
    bounds: tuple[float, ...],
    resolution: float,
) -> "pv.PolyData":
    """
    Convert a dense boolean voxel grid to a triangle mesh via Marching Cubes.

    Uses scikit-image marching_cubes for the iso-surface at level=0.5.
    The resulting mesh has smooth normals and correct Z-direction orientation.

    This is the high-quality path – slower than height_map_to_surface but
    captures overhangs, holes, and vertical walls (the advantage of tri-dexel).
    """
    if not _PYVISTA:
        raise ImportError("pyvista is required for mesh reconstruction")
    if not _SKIMAGE:
        raise ImportError("scikit-image is required for voxel_to_mesh")

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Pad with False so Marching Cubes produces a closed surface at the stock boundary
    padded = np.pad(voxels.astype(np.float32), 1, mode="constant", constant_values=0.0)

    verts, faces, normals, _ = _mc(
        padded,
        level=0.5,
        spacing=(resolution, resolution, resolution),
    )

    # Voxels are cell-centered samples. After adding one false padding cell,
    # the material boundary between padded index 0 (outside) and 1 (first
    # material cell) is emitted by marching cubes at coordinate 0.5. Therefore
    # the world offset must place that coordinate at the stock minimum.
    half = 0.5 * resolution
    verts[:, 0] += x_min - half
    verts[:, 1] += y_min - half
    verts[:, 2] += z_min - half

    faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(verts, faces_pv)
    mesh.compute_normals(inplace=True)
    return mesh


def add_stock_box(
    bounds: tuple[float, ...],
) -> "pv.PolyData":
    """Return wireframe bounding box for the initial stock."""
    if not _PYVISTA:
        raise ImportError("pyvista is required")
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
