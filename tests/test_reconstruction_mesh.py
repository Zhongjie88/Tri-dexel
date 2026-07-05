import numpy as np
import pytest

from src.reconstruction.mesh import voxel_to_mesh


def test_voxel_to_mesh_preserves_cell_centered_bounds():
    voxels = np.ones((1, 1, 1), dtype=bool)

    try:
        mesh = voxel_to_mesh(voxels, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), 1.0)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert mesh.bounds == pytest.approx((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
