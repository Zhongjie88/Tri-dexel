import pytest
from src.stock.dexel_ray import DexelRay


def test_initial_empty():
    r = DexelRay()
    assert r.is_empty()
    assert r.top() is None
    assert r.bottom() is None


def test_set_solid():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    assert not r.is_empty()
    assert r.top() == 10.0
    assert r.bottom() == 0.0
    assert r.intervals == [(0.0, 10.0)]


def test_subtract_all():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    r.subtract(0.0, 10.0)
    assert r.is_empty()


def test_subtract_top_portion():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    r.subtract(7.0, 20.0)
    assert r.intervals == [(0.0, 7.0)]
    assert r.top() == 7.0


def test_subtract_bottom_portion():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    r.subtract(-5.0, 3.0)
    assert r.intervals == [(3.0, 10.0)]
    assert r.bottom() == 3.0


def test_subtract_middle_splits_interval():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    r.subtract(3.0, 7.0)
    assert r.intervals == [(0.0, 3.0), (7.0, 10.0)]


def test_subtract_no_overlap():
    r = DexelRay()
    r.set_solid(0.0, 5.0)
    r.subtract(6.0, 10.0)
    assert r.intervals == [(0.0, 5.0)]


def test_subtract_degenerate_noop():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    r.subtract(5.0, 5.0)  # zero-width cut
    assert r.intervals == [(0.0, 10.0)]


def test_union_merge():
    r = DexelRay()
    r.set_solid(0.0, 5.0)
    r.union(4.0, 10.0)
    assert r.intervals == [(0.0, 10.0)]


def test_union_disjoint():
    r = DexelRay()
    r.set_solid(0.0, 3.0)
    r.union(6.0, 9.0)
    assert r.intervals == [(0.0, 3.0), (6.0, 9.0)]


def test_contains():
    r = DexelRay()
    r.set_solid(0.0, 10.0)
    r.subtract(4.0, 6.0)
    assert r.contains(2.0)
    assert r.contains(8.0)
    assert not r.contains(5.0)


def test_sequential_subtracts():
    r = DexelRay()
    r.set_solid(0.0, 20.0)
    r.subtract(15.0, 25.0)
    r.subtract(5.0, 10.0)
    assert r.intervals == [(0.0, 5.0), (10.0, 15.0)]
