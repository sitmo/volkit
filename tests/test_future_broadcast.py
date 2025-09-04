# test_black_broadcast.py
import numpy as np
import pytest
from scipy.stats import norm

from volkit import price_euro_future

def _direct_black_price(F, K, T, r, sigma, cp):
    """Direct formula used as an oracle in tests (independent of black_price)."""
    F = np.asarray(F, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    r = np.asarray(r, float)
    sigma = np.asarray(sigma, float)
    cp = np.asarray(cp, int)

    # Broadcast all to a common shape
    shape = np.broadcast(F, K, T, r, sigma, cp).shape
    F, K, T, r, sigma, cp = (np.broadcast_to(x, shape) for x in (F, K, T, r, sigma, cp))

    sqrtT = np.sqrt(np.maximum(T, 0.0))
    a = np.maximum(sigma * sqrtT, 1e-16)
    d1 = np.log(F / K) / a + 0.5 * a
    d2 = d1 - a
    ert = np.exp(-r * T)
    return ert * (cp * F * norm.cdf(cp * d1) - cp * K * norm.cdf(cp * d2))


def _mk_alternating_signs(n):
    s = np.ones(n, dtype=int)
    s[1::2] = -1
    return s


@pytest.mark.parametrize("case", [
    # Case 1: 2D broadcast (shape -> (3,4)), T & r scalars, cp as vector (3,1)
    {
        "F": lambda: np.array([[90.0], [100.0], [110.0]]),        # (3,1)
        "K": lambda: np.array([[80., 90., 100., 120.]]),          # (1,4)
        "T": lambda: 0.75,                                        # scalar
        "r": lambda: 0.02,                                        # scalar
        "sigma": lambda: np.array([[0.15], [0.20], [0.25]]),      # (3,1)
        "cp": lambda: np.array([[1], [-1], [1]]),                 # (3,1)
    },
    # Case 2: Same shapes, cp scalar
    {
        "F": lambda: np.array([[90.0], [100.0], [110.0]]),
        "K": lambda: np.array([[80., 90., 100., 120.]]),
        "T": lambda: 1.25,
        "r": lambda: 0.00,
        "sigma": lambda: np.array([[0.10], [0.20], [0.30]]),
        "cp": lambda: 1,  # scalar call
    },
    # Case 3: 1D vectors (shape -> (5,)), cp alternating +1/-1, tiny T to hit stability path
    {
        "F": lambda: np.linspace(80, 120, 5),
        "K": lambda: np.linspace(70, 130, 5),
        "T": lambda: np.full(5, 1e-10),  # tiny T
        "r": lambda: 0.01,
        "sigma": lambda: np.linspace(0.05, 0.25, 5),
        "cp": lambda: _mk_alternating_signs(5),
    },
    # Case 4: 3D broadcast (2,3,4) from (2,1,1) x (1,3,1) x (1,1,4), mixed scalars/arrays
    {
        "F": lambda: np.array([[[95.0]], [[105.0]]]),             # (2,1,1)
        "K": lambda: np.array([[[90.0],[100.0],[110.0]]]),        # (1,3,1)
        "T": lambda: np.array([[[0.5, 1.0, 1.5, 2.0]]]),          # (1,1,4)
        "r": lambda: np.array([[[0.0]]]),                         # (1,1,1)
        "sigma": lambda: 0.2,                                     # scalar
        "cp": lambda: np.array([[[1]], [[-1]]]),                  # (2,1,1)
    },
    # Case 5: F scalar, others vectors (shape -> (6,)), cp vector
    {
        "F": lambda: 100.0,                                       # scalar
        "K": lambda: np.linspace(80, 120, 6),
        "T": lambda: np.linspace(0.1, 1.5, 6),
        "r": lambda: 0.03,                                        # scalar
        "sigma": lambda: np.linspace(0.12, 0.35, 6),
        "cp": lambda: _mk_alternating_signs(6),
    },
])
def test_black_price_broadcast_matches_direct(case):
    # Build inputs
    F, K, T, r, sigma, cp = (case[k]() for k in ["F", "K", "T", "r", "sigma", "cp"])

    # Direct oracle on fully expanded arrays
    expected = _direct_black_price(F, K, T, r, sigma, cp)

    # Check shape and values for black_price with the original (possibly scalar) inputs
    out = price_euro_future(F, K, T, r, sigma, cp)
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    # Also verify that calling with explicit broadcasted arrays gives the same result
    shape = np.broadcast(F, K, T, r, sigma, cp).shape
    F2, K2, T2, r2, sigma2, cp2 = (np.broadcast_to(np.asarray(x), shape) for x in (F, K, T, r, sigma, cp))
    out2 = price_euro_future(F2, K2, T2, r2, sigma2, cp2)
    assert out2.shape == expected.shape
    np.testing.assert_allclose(out2, expected, rtol=1e-12, atol=1e-12)


def test_black_price_raises_on_non_broadcastable():
    # Intentionally incompatible shapes: (3,) with (2,)
    F = np.array([100.0, 101.0, 102.0])
    K = np.array([100.0, 99.0])
    T = 1.0
    r = 0.01
    sigma = 0.2
    cp = 1
    with pytest.raises(ValueError):
        # Numpy will raise via internal broadcasting when trying to do log(F/K)
        _ = price_euro_future(F, K, T, r, sigma, cp)
