# learnify/core/rng.py
import numpy as np

# private RNG instance
_rng = np.random.default_rng()

def seed(s: int):
    """Set the seed for reproducibility."""
    global _rng
    _rng = np.random.default_rng(s)

def rand(*shape):
    """Uniform random numbers in [0, 1]"""
    return _rng.random(shape)

def normal(loc=0.0, scale=1.0, size=None):
    """Draw samples from a normal (Gaussian) distribution."""
    return _rng.normal(loc, scale, size)

def choice(a, size=None, replace=True, p=None):
    """Randomly choose elements from a sequence or array."""
    return _rng.choice(a, size=size, replace=replace, p=p)

def integers(low, high=None, size=None):
    """Random integers from low (inclusive) to high (exclusive)."""
    return _rng.integers(low, high=high, size=size)