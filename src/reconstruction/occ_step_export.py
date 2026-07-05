from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StepExportResult:
    input_triangles: int
    output_triangles: int
    wrote_solid: bool


@dataclass(frozen=True)
class SurfaceStepExportResult:
    source_shape: tuple[int, int]
    output_shape: tuple[int, int]


def prepare_mesh_for_step(mesh, max_triangles: int = 50_000):
    """Return a cleaned, triangulated, size-limited mesh for CAD STEP export."""
    max_triangles = max(100, int(max_triangles))
    tri_mesh = mesh.triangulate().clean()
    input_triangles = int(tri_mesh.n_cells)

    if input_triangles > max_triangles:
        reduction = 1.0 - (max_triangles / input_triangles)
        try:
            tri_mesh = tri_mesh.decimate_pro(
                reduction,
                preserve_topology=True,
                splitting=False,
                boundary_vertex_deletion=False,
            )
        except Exception:
            tri_mesh = tri_mesh.decimate(reduction, volume_preservation=True)
        tri_mesh = tri_mesh.triangulate().clean()

    return tri_mesh, input_triangles


def export_mesh_as_occ_step(
    mesh,
    path: str,
    *,
    max_triangles: int = 50_000,
    sewing_tolerance: float = 0.05,
) -> StepExportResult:
    """Export a simulation mesh as an OpenCascade AP214 faceted STEP file.

    The result is a faceted CAD B-Rep, not a fitted analytic surface model.
    This is the practical SolidWorks-compatible path for tri-dexel output:
    limit triangle count, build OCC faces, sew them into a shell, attempt a
    solid, then let OpenCascade's STEP writer produce the Part 21 file.
    """
    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_Sewing,
        )
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Interface import Interface_Static
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
        from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SHELL
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import TopoDS_Compound, topods
        from OCC.Core.gp import gp_Pnt
    except ImportError as exc:
        raise RuntimeError(
            "STEP export requires pythonocc-core / OpenCascade. "
            "Start the app from the tridexel-occ environment."
        ) from exc

    tri_mesh, input_triangles = prepare_mesh_for_step(mesh, max_triangles=max_triangles)
    points = tri_mesh.points
    raw_faces = tri_mesh.faces.reshape((-1, 4))
    triangles = [
        (int(face[1]), int(face[2]), int(face[3]))
        for face in raw_faces
        if int(face[0]) == 3
    ]
    if not triangles:
        raise RuntimeError("No triangle faces are available for STEP export.")

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    sewing = BRepBuilderAPI_Sewing(max(float(sewing_tolerance), 1e-5))

    face_count = 0
    for ids in triangles:
        poly = BRepBuilderAPI_MakePolygon()
        for idx in ids:
            x, y, z = points[idx]
            poly.Add(gp_Pnt(float(x), float(y), float(z)))
        poly.Close()
        if not poly.IsDone():
            continue
        made_face = BRepBuilderAPI_MakeFace(poly.Wire())
        if made_face.IsDone():
            occ_face = made_face.Face()
            builder.Add(compound, occ_face)
            sewing.Add(occ_face)
            face_count += 1

    if face_count == 0:
        raise RuntimeError("OpenCascade could not build faces from the mesh.")

    sewing.Perform()
    shape = sewing.SewedShape()
    if shape.IsNull():
        shape = compound

    wrote_solid = False
    shell = None
    if not shape.IsNull() and shape.ShapeType() == TopAbs_SHELL:
        shell = topods.Shell(shape)
    elif not shape.IsNull() and shape.ShapeType() == TopAbs_COMPOUND:
        explorer = TopExp_Explorer(shape, TopAbs_SHELL)
        if explorer.More():
            shell = topods.Shell(explorer.Current())

    if shell is not None:
        solid_maker = BRepBuilderAPI_MakeSolid()
        solid_maker.Add(shell)
        if solid_maker.IsDone():
            solid = solid_maker.Solid()
            if not solid.IsNull():
                shape = solid
                wrote_solid = True

    try:
        from OCC.Core.ShapeFix import ShapeFix_Shape

        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        fixed = fixer.Shape()
        if not fixed.IsNull():
            shape = fixed
    except Exception:
        pass

    Interface_Static.SetCVal("write.step.schema", "AP214")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(Path(path)))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OpenCascade failed to write STEP file, status={status}.")

    return StepExportResult(
        input_triangles=input_triangles,
        output_triangles=len(triangles),
        wrote_solid=wrote_solid,
    )


def export_zmap_surface_as_occ_step(
    stock,
    path: str,
    *,
    max_grid: int = 80,
    tolerance: float = 0.1,
) -> SurfaceStepExportResult:
    """Export only the machined top surface as a compact OpenCascade STEP file.

    This is the preferred SolidWorks workflow when the downstream use only
    needs the machined surface, not a full voxel solid. The Z-dexel height map
    is downsampled and fitted as one B-Spline surface, so the file is orders of
    magnitude smaller than a faceted B-Rep made from every triangle.
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.GeomAbs import GeomAbs_C1, GeomAbs_C2
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Interface import Interface_Static
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
        from OCC.Core.TColgp import TColgp_Array2OfPnt
        from OCC.Core.gp import gp_Pnt
    except ImportError as exc:
        raise RuntimeError(
            "STEP export requires pythonocc-core / OpenCascade. "
            "Start the app from the tridexel-occ environment."
        ) from exc

    z_grid = stock.z_grid
    h_map = z_grid.height_map()
    if h_map.size == 0 or np.all(np.isnan(h_map)):
        raise RuntimeError("No Z-map surface is available for STEP export.")

    source_shape = tuple(int(v) for v in h_map.shape)
    max_grid = max(8, int(max_grid))
    sx = max(1, int(np.ceil(h_map.shape[0] / max_grid)))
    sy = max(1, int(np.ceil(h_map.shape[1] / max_grid)))
    h_sub = h_map[::sx, ::sy].copy()

    # Ensure the last stock boundary is represented after stride downsampling.
    if (h_map.shape[0] - 1) % sx != 0:
        h_sub = np.vstack([h_sub, h_map[-1:, ::sy]])
    if (h_map.shape[1] - 1) % sy != 0:
        last_col = h_map[::sx, -1:]
        if h_sub.shape[0] != last_col.shape[0]:
            last_col = np.vstack([last_col, h_map[-1:, -1:]])
        h_sub = np.hstack([h_sub, last_col])

    fill = float(np.nanmin(h_map))
    np.nan_to_num(h_sub, nan=fill, copy=False)
    nx, ny = h_sub.shape

    x_values = np.linspace(float(z_grid.row_min), float(z_grid.row_max), nx)
    y_values = np.linspace(float(z_grid.col_min), float(z_grid.col_max), ny)
    points = TColgp_Array2OfPnt(1, nx, 1, ny)
    for i, x in enumerate(x_values, start=1):
        for j, y in enumerate(y_values, start=1):
            points.SetValue(i, j, gp_Pnt(float(x), float(y), float(h_sub[i - 1, j - 1])))

    approx = GeomAPI_PointsToBSplineSurface(points, 3, 8, GeomAbs_C2, float(tolerance))
    if not approx.IsDone():
        approx = GeomAPI_PointsToBSplineSurface(points, 3, 8, GeomAbs_C1, float(tolerance) * 2.0)
    if not approx.IsDone():
        raise RuntimeError("OpenCascade could not fit a B-Spline surface from the Z-map.")

    face = BRepBuilderAPI_MakeFace(approx.Surface(), max(float(tolerance), 1e-4))
    if not face.IsDone():
        raise RuntimeError("OpenCascade could not create a face from the fitted surface.")

    Interface_Static.SetCVal("write.step.schema", "AP214")
    writer = STEPControl_Writer()
    writer.Transfer(face.Face(), STEPControl_AsIs)
    status = writer.Write(str(Path(path)))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OpenCascade failed to write STEP surface, status={status}.")

    return SurfaceStepExportResult(source_shape=source_shape, output_shape=(nx, ny))
