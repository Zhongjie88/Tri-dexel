import pytest

from src.reconstruction.occ_step_export import (
    export_zmap_surface_as_occ_step,
    prepare_mesh_for_step,
)
from src.stock.tri_dexel import TriDexelStock


def test_prepare_mesh_for_step_limits_triangle_count():
    pv = pytest.importorskip("pyvista")

    mesh = pv.Sphere(theta_resolution=64, phi_resolution=64).triangulate()
    reduced, original = prepare_mesh_for_step(mesh, max_triangles=500)

    assert original > 500
    assert reduced.n_cells <= original
    assert reduced.n_cells <= 650


def test_export_zmap_surface_step_writes_compact_surface(tmp_path):
    pytest.importorskip("OCC")

    stock = TriDexelStock((0.0, 10.0, 0.0, 10.0, 0.0, 5.0), 1.0)
    stock.initialize_box_stock()
    stock.z_grid.rays[5][5].subtract(2.0, 5.0)
    path = tmp_path / "surface.step"

    result = export_zmap_surface_as_occ_step(stock, str(path), max_grid=12)

    assert path.exists()
    assert path.stat().st_size > 0
    assert result.source_shape == (10, 10)
    assert result.output_shape[0] <= 12
    assert result.output_shape[1] <= 12
