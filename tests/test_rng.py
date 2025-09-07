# tests/test_rng.py
import numpy as np
import importlib

def fresh_rng_module():
    # Re-import to reset any state that previous tests may have changed
    from learnify.core import rng as lrng
    importlib.reload(lrng)
    return lrng

def test_seed_reproducibility():
    rng = fresh_rng_module()
    rng.seed(42)
    a1 = rng.rand(5)
    rng.seed(42)
    a2 = rng.rand(5)
    assert np.allclose(a1, a2), "Seeding should reproduce the same sequence"

def test_independence_from_numpy_global():
    rng = fresh_rng_module()
    rng.seed(7)
    seq1 = rng.normal(size=4)

    # Change NumPy's global RNG; Learnify's rng should be unaffected
    np.random.seed(123)
    seq2 = rng.normal(size=4)

    # Now reseed and ensure we can reproduce the *first* draw exactly
    rng.seed(7)
    seq1_re = rng.normal(size=4)

    assert not np.allclose(seq1, seq2), "Sequence should advance regardless of np.random.seed"
    assert np.allclose(seq1, seq1_re), "Learnify RNG must be independent of NumPy global RNG"

def test_rand_shape_and_range():
    rng = fresh_rng_module()
    rng.seed(0)
    x = rng.rand(3, 2, 1)
    assert x.shape == (3, 2, 1)
    assert (x >= 0).all() and (x < 1).all(), "rand should be in [0, 1)"

def test_normal_shape_and_stats():
    rng = fresh_rng_module()
    rng.seed(0)
    x = rng.normal(loc=0.0, scale=1.0, size=10_000)
    # Rough sanity checks (not a statistical test)
    assert x.shape == (10_000,)
    assert abs(x.mean()) < 0.1
    assert 0.8 < x.std() < 1.2

def test_integers_bounds_and_shape():
    rng = fresh_rng_module()
    rng.seed(123)
    x = rng.integers(5, 10, size=(4, 3))
    assert x.shape == (4, 3)
    assert (x >= 5).all() and (x < 10).all()

def test_choice_basic_and_replace_false():
    rng = fresh_rng_module()
    rng.seed(999)
    a = np.arange(100)

    # with replacement
    x = rng.choice(a, size=20, replace=True)
    assert x.shape == (20,)
    assert np.isin(x, a).all()

    # without replacement -> all unique, correct size
    y = rng.choice(a, size=20, replace=False)
    assert y.shape == (20,)
    assert len(np.unique(y)) == 20
    assert np.isin(y, a).all()

def test_reseed_resets_all_streams():
    rng = fresh_rng_module()
    rng.seed(123)
    u1 = rng.rand(4)
    n1 = rng.normal(size=4)
    i1 = rng.integers(0, 100, size=4)

    rng.seed(123)
    u2 = rng.rand(4)
    n2 = rng.normal(size=4)
    i2 = rng.integers(0, 100, size=4)

    assert np.allclose(u1, u2)
    assert np.allclose(n1, n2)
    assert np.array_equal(i1, i2)

def test_types_and_dtypes():
    rng = fresh_rng_module()
    rng.seed(1)
    u = rng.rand(3)
    n = rng.normal(size=3)
    i = rng.integers(0, 10, size=3)
    assert u.dtype.kind == "f"
    assert n.dtype.kind == "f"
    assert i.dtype.kind in ("i", "u")