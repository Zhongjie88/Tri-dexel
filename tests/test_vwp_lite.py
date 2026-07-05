import numpy as np
import pytest

from src.gcode.parser import GCodeMove
from src.motion.pose import ToolPose
from src.simulation.sv_engine import SweptVolumeSimulationEngine
from src.stock.dexel_ray import DexelRay
from src.stock.tri_dexel import TriDexelStock
from src.swept_volume.builder import SweptVolumeBuilder
from src.swept_volume.sampler import SweptVolumeSampler
from src.swept_volume.triangle import Triangle
from src.tool.tool_geometry import FlatEndMill
from src.tool.tool_geometry import COMPONENT_SIDE


def test_tool_pose_transforms_points_and_axis():
    pose = ToolPose.from_axis((1.0, 2.0, 3.0), (1.0, 0.0, 0.0))

    assert pose.axis == pytest.approx((1.0, 0.0, 0.0))
    assert pose.transform_points(np.array([[0.0, 0.0, 0.0]])).shape == (1, 3)
    assert pose.transform_points(np.array([[0.0, 0.0, 0.0]]))[0] == pytest.approx(
        (1.0, 2.0, 3.0)
    )


def test_flat_endmill_surface_sampling_is_headless():
    tool = FlatEndMill(2.0)

    surface = tool.sample_surface(radial_segments=12, axial_segments=4)

    assert surface["points"].shape == (12, 5, 3)
    assert surface["normals"].shape == (12, 5, 3)
    assert np.linalg.norm(surface["normals"][0, 0]) == pytest.approx(1.0)


def test_triangle_ray_intersection_returns_distance_and_normal():
    tri = Triangle(
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    )

    hit = tri.ray_intersection(
        np.array([0.25, 0.25, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    assert hit is not None
    distance, normal = hit
    assert distance == pytest.approx(1.0)
    assert np.linalg.norm(normal) == pytest.approx(1.0)


def test_triangle_preserves_component_id():
    tri = Triangle(
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        component_id=COMPONENT_SIDE,
    )

    assert tri.component_id == COMPONENT_SIDE


def test_swept_volume_builder_and_sampler_find_z_grid_hits():
    stock = TriDexelStock((0.0, 20.0, 0.0, 20.0, 0.0, 30.0), 1.0)
    stock.initialize_box_stock()
    tool = FlatEndMill(3.0)
    builder = SweptVolumeBuilder(tool, radial_segments=12, axial_segments=3)
    volume = builder.build_between(
        ToolPose((5.0, 10.0, 24.0)),
        ToolPose((15.0, 10.0, 24.0)),
    )

    hits = SweptVolumeSampler().sample_z_grid(
        volume,
        stock.z_grid,
        stock.z_min,
        stock.z_max,
    )

    assert volume.triangles
    assert hits
    assert all(hit.cut_lo <= hit.cut_hi for hit in hits)
    assert any(hit.component_id == COMPONENT_SIDE for hit in hits)


def test_swept_volume_sampler_finds_x_y_z_grid_hits():
    stock = TriDexelStock((0.0, 20.0, 0.0, 20.0, 0.0, 30.0), 1.0)
    stock.initialize_box_stock()
    tool = FlatEndMill(4.0)
    volume = SweptVolumeBuilder(tool, radial_segments=16, axial_segments=4).build_between(
        ToolPose((5.0, 10.0, 20.0)),
        ToolPose((15.0, 10.0, 20.0)),
    )
    sampler = SweptVolumeSampler()

    z_hits = sampler.sample_z_grid(volume, stock.z_grid, stock.z_min, stock.z_max)
    x_hits = sampler.sample_x_grid(volume, stock.x_grid, stock.x_min, stock.x_max)
    y_hits = sampler.sample_y_grid(volume, stock.y_grid, stock.y_min, stock.y_max)

    assert z_hits
    assert x_hits
    assert y_hits


def test_sv_engine_subtracts_material_and_records_surface_metadata():
    stock = TriDexelStock((0.0, 20.0, 0.0, 20.0, 0.0, 30.0), 1.0)
    stock.initialize_box_stock()
    engine = SweptVolumeSimulationEngine(
        stock,
        FlatEndMill(5.0),
        radial_segments=12,
        axial_segments=3,
    )

    hits = engine.simulate_move((5.0, 10.0, 20.0), (15.0, 10.0, 20.0), step=5.0)

    hmap = stock.z_grid.height_map()
    assert hits > 0
    assert np.nanmin(hmap) < stock.z_max
    assert any(
        ray.surface_metadata
        for row in stock.z_grid.rays
        for ray in row
    )
    assert any(
        metadata.get("grid") == "x"
        for row in stock.x_grid.rays
        for ray in row
        for metadata in ray.surface_metadata
    )
    assert any(
        metadata.get("grid") == "y"
        for row in stock.y_grid.rays
        for ray in row
        for metadata in ray.surface_metadata
    )
    assert any(
        metadata.get("component_id") == COMPONENT_SIDE
        for row in stock.x_grid.rays
        for ray in row
        for metadata in ray.surface_metadata
    )
    assert any(
        metadata.get("component_name") == "side"
        and metadata.get("surface_role") == "side"
        for row in stock.x_grid.rays
        for ray in row
        for metadata in ray.surface_metadata
    )


def test_sv_engine_simulate_gcode_preserves_gcode_metadata():
    stock = TriDexelStock((0.0, 20.0, 0.0, 20.0, 0.0, 30.0), 1.0)
    stock.initialize_box_stock()
    engine = SweptVolumeSimulationEngine(
        stock,
        FlatEndMill(4.0),
        radial_segments=12,
        axial_segments=3,
    )
    moves = [
        GCodeMove(5.0, 10.0, 20.0, 0.0, rapid=True, line_no=1, motion_type="G0"),
        GCodeMove(
            15.0,
            10.0,
            20.0,
            100.0,
            rapid=False,
            line_no=2,
            motion_type="G1",
            source_line="N2 G1 X15 Y10 Z20 F100",
            segment_index=1,
            segment_count=1,
        ),
    ]

    hits = engine.simulate_gcode(moves, tool_id="FLAT4")

    metadata = [
        item
        for row in stock.z_grid.rays
        for ray in row
        for item in ray.surface_metadata
    ]
    assert hits > 0
    assert metadata
    assert any(item.get("line_no") == 2 for item in metadata)
    assert any(item.get("motion_type") == "G1" for item in metadata)
    assert any(item.get("tool_id") == "FLAT4" for item in metadata)
    assert any(item.get("source_line") == "N2 G1 X15 Y10 Z20 F100" for item in metadata)


def test_sv_engine_simulate_gcode_uses_rotary_tool_pose_orientation():
    stock = TriDexelStock((0.0, 20.0, 0.0, 20.0, 0.0, 30.0), 1.0)
    stock.initialize_box_stock()
    engine = SweptVolumeSimulationEngine(
        stock,
        FlatEndMill(4.0),
        subdivide_moves=False,
    )
    recorded_axes = []

    def record_subtract(start, end, pose_samples=2):
        recorded_axes.append((start.axis.copy(), end.axis.copy()))
        return 1

    engine.subtract_swept_volume = record_subtract
    moves = [
        GCodeMove(
            5.0,
            10.0,
            20.0,
            0.0,
            rapid=True,
            line_no=1,
            motion_type="G0",
            a=0.0,
            c=0.0,
        ),
        GCodeMove(
            15.0,
            10.0,
            20.0,
            100.0,
            rapid=False,
            line_no=2,
            motion_type="G1",
            start_x=5.0,
            start_y=10.0,
            start_z=20.0,
            start_a=0.0,
            start_c=0.0,
            a=-10.0,
            c=45.0,
        ),
    ]

    hits = engine.simulate_gcode(moves)

    assert hits == 1
    assert recorded_axes
    start_axis, end_axis = recorded_axes[0]
    assert start_axis == pytest.approx((0.0, 0.0, 1.0))
    assert end_axis[2] < 1.0


def test_dexel_ray_metadata_is_optional_and_non_breaking():
    ray = DexelRay()
    ray.set_solid(0.0, 10.0)

    ray.subtract(2.0, 4.0)
    ray.subtract(6.0, 8.0, metadata={"normal": (0.0, 0.0, 1.0)})

    assert ray.intervals == [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]
    assert ray.surface_metadata == [
        {"range": (6.0, 8.0), "normal": (0.0, 0.0, 1.0)}
    ]
