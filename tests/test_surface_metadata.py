from src.stock.dexel_ray import DexelRay
from src.stock.surface_metadata import (
    SurfaceMetadata,
    component_labels,
    make_surface_metadata,
)
from src.tool.tool_geometry import COMPONENT_BALL, COMPONENT_SIDE


def test_surface_metadata_behaves_like_a_mapping():
    metadata = SurfaceMetadata(
        cut_range=(1.0, 2.0),
        normal=(0.0, 0.0, 1.0),
        source="swept_volume",
        grid="z",
        line_no=12,
    )

    assert metadata["range"] == (1.0, 2.0)
    assert metadata.get("grid") == "z"
    assert dict(metadata) == {
        "range": (1.0, 2.0),
        "normal": (0.0, 0.0, 1.0),
        "source": "swept_volume",
        "grid": "z",
        "line_no": 12,
    }


def test_make_surface_metadata_preserves_legacy_dict_shape():
    metadata = make_surface_metadata(
        (1.0, 2.0),
        {"normal": (1.0, 0.0, 0.0), "grid": "x"},
    )

    assert metadata == {
        "range": (1.0, 2.0),
        "normal": (1.0, 0.0, 0.0),
        "grid": "x",
    }


def test_make_surface_metadata_expands_component_labels():
    metadata = make_surface_metadata(
        (1.0, 2.0),
        {"component_id": COMPONENT_BALL},
    )

    assert metadata["component_id"] == COMPONENT_BALL
    assert metadata["component_name"] == "ball_tip"
    assert metadata["surface_role"] == "ball"


def test_component_labels_returns_readable_component_metadata():
    assert component_labels(COMPONENT_SIDE) == ("side", "side")
    assert component_labels(None) == (None, None)


def test_dexel_ray_stores_structured_metadata_without_breaking_dict_comparison():
    ray = DexelRay()
    ray.set_solid(0.0, 10.0)

    ray.subtract(3.0, 5.0, metadata={"normal": (0.0, 1.0, 0.0)})

    assert isinstance(ray.surface_metadata[0], SurfaceMetadata)
    assert ray.surface_metadata == [
        {"range": (3.0, 5.0), "normal": (0.0, 1.0, 0.0)}
    ]
