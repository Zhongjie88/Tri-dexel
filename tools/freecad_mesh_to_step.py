from __future__ import annotations

import sys

import FreeCAD
import Import
import Mesh
import Part


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: FreeCADCmd freecad_mesh_to_step.py input.stl output.step")
        return 2

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    mesh = Mesh.Mesh(in_path)
    if mesh.CountFacets == 0:
        raise RuntimeError("input mesh has no facets")

    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, 0.1)
    shape = shape.removeSplitter()

    export_shape = shape
    try:
        shell = Part.Shell(shape.Faces)
        if shell.isClosed():
            export_shape = Part.Solid(shell).removeSplitter()
        else:
            export_shape = shell
    except Exception:
        export_shape = shape

    doc = FreeCAD.newDocument("tri_dexel_export")
    obj = doc.addObject("Part::Feature", "machined_stock")
    obj.Shape = export_shape
    doc.recompute()
    Import.export([obj], out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
