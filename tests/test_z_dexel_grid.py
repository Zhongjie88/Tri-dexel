import numpy as np
import pytest
from src.stock.z_dexel_grid import ZDexelGrid


def make_grid(nx=10, ny=10) -> ZDexelGrid:
    return ZDexelGrid(0.0, 10.0, 0.0, 10.0, nx, ny)


def test_initialize_stock():
    g = make_grid()
    g.initialize_stock(0.0, 50.0)
    for row in g.rays:
        for ray in row:
            assert ray.intervals == [(0.0, 50.0)]


def test_height_map_full():
    g = make_grid()
    g.initialize_stock(0.0, 50.0)
    hmap = g.height_map()
    assert hmap.shape == (10, 10)
    assert np.all(hmap == 50.0)


def test_height_map_after_subtract():
    g = make_grid(nx=4, ny=4)
    g.initialize_stock(0.0, 50.0)
    g.rays[1][2].subtract(30.0, 100.0)
    hmap = g.height_map()
    assert hmap[1, 2] == pytest.approx(30.0)
    assert hmap[0, 0] == pytest.approx(50.0)


def test_height_map_empty_ray():
    g = make_grid(nx=2, ny=2)
    g.initialize_stock(0.0, 50.0)
    g.rays[0][0].subtract(0.0, 50.0)
    hmap = g.height_map()
    assert np.isnan(hmap[0, 0])
    assert hmap[1, 1] == pytest.approx(50.0)


def test_to_dense_voxel_shape():
    g = make_grid(nx=5, ny=5)
    g.initialize_stock(0.0, 10.0)
    v = g.to_dense_voxel(0.0, 10.0, 10)
    assert v.shape == (5, 5, 10)
    assert v.dtype == bool


def test_to_dense_voxel_values():
    g = make_grid(nx=2, ny=2)
    g.initialize_stock(0.0, 10.0)
    g.rays[0][0].subtract(5.0, 100.0)  # cut top half of ray (0,0)
    v = g.to_dense_voxel(0.0, 10.0, 10)
    # ray (0,0): material at [0,5], voxels 0..4 should be True, 5..9 False
    assert all(v[0, 0, k] for k in range(5))
    assert not any(v[0, 0, k] for k in range(5, 10))
    # ray (1,1): full material
    assert all(v[1, 1, :])


def test_coordinate_helpers():
    g = ZDexelGrid(0.0, 100.0, 0.0, 200.0, 10, 20)
    assert g.row_center(0) == pytest.approx(5.0)
    assert g.col_center(0) == pytest.approx(5.0)
    assert g.row_index(55.0) == 5
    assert g.col_index(105.0) == 10
