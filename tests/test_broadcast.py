# tests/test_black_broadcast.py
import numpy as np
import pytest

from volkit import (
    price_euro_future as price_fn,
    delta_euro_future,
    gamma_euro_future,
    vega_euro_future,
    theta_euro_future,
    rho_euro_future,
    dual_delta_euro_future,
    vanna_euro_future,
    vomma_euro_future,
    lambda_euro_future,
    implied_vol_euro_future,
)

# ------------------------------
# Broadcast cases (no tiny-T for IV)
# ------------------------------
BROADCAST_CASES = [
    # (3,4) from (3,1) x (1,4), cp as (3,1), T & r scalars
    {
        "F": lambda: np.array([[90.0], [100.0], [110.0]]),  # (3,1)
        "K": lambda: np.array([[80.0, 90.0, 100.0, 120.0]]),  # (1,4)
        "T": lambda: 0.75,  # scalar
        "r": lambda: 0.02,  # scalar
        "sigma": lambda: np.array([[0.15], [0.20], [0.25]]),  # (3,1)
        "cp": lambda: np.array([[1], [-1], [1]]),  # (3,1)
    },
    # Same shapes, cp scalar
    {
        "F": lambda: np.array([[90.0], [100.0], [110.0]]),
        "K": lambda: np.array([[80.0, 90.0, 100.0, 120.0]]),
        "T": lambda: 1.25,
        "r": lambda: 0.00,
        "sigma": lambda: np.array([[0.10], [0.20], [0.30]]),
        "cp": lambda: 1,
    },
    # 3D broadcast (2,3,4) from (2,1,1) x (1,3,1) x (1,1,4), cp (2,1,1)
    {
        "F": lambda: np.array([[[95.0]], [[105.0]]]),  # (2,1,1)
        "K": lambda: np.array([[[90.0], [100.0], [110.0]]]),  # (1,3,1)
        "T": lambda: np.array([[[0.5, 1.0, 1.5, 2.0]]]),  # (1,1,4)
        "r": lambda: np.array([[[0.0]]]),  # (1,1,1)
        "sigma": lambda: 0.2,  # scalar
        "cp": lambda: np.array([[[1]], [[-1]]]),  # (2,1,1)
    },
    # (6,) vectors, cp alternating
    {
        "F": lambda: 100.0,  # scalar
        "K": lambda: np.linspace(80, 120, 6),
        "T": lambda: np.linspace(0.1, 1.5, 6),
        "r": lambda: 0.03,  # scalar
        "sigma": lambda: np.linspace(0.12, 0.35, 6),
        "cp": lambda: np.array([1, -1, 1, -1, 1, -1]),
    },
]

# For IV we avoid pathological tiny T / extreme params
IV_CASE_IDS = [0, 1, 2, 3]


def _broadcast_equivalence_check(fn, F, K, T, r, sigma, cp, rtol=1e-12, atol=1e-12):
    """Assert that vectorized output equals element-wise scalar calls."""
    # Compute vectorized
    out = fn(F, K, T, r, sigma, cp)

    # Compute broadcasted arguments
    shape = np.broadcast(F, K, T, r, sigma, cp).shape
    F2, K2, T2, R2, S2, CP2 = (
        np.broadcast_to(np.asarray(F, float), shape),
        np.broadcast_to(np.asarray(K, float), shape),
        np.broadcast_to(np.asarray(T, float), shape),
        np.broadcast_to(np.asarray(r, float), shape),
        np.broadcast_to(np.asarray(sigma, float), shape),
        np.broadcast_to(np.asarray(cp, int), shape),
    )

    assert out.shape == shape

    # Element-wise scalar calls
    expected = np.empty(shape, dtype=float)
    for idx in np.ndindex(shape):
        expected[idx] = float(
            fn(F2[idx], K2[idx], T2[idx], R2[idx], S2[idx], int(CP2[idx]))
        )

    np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol)


# ------------------------------
# Price broadcasting
# ------------------------------
@pytest.mark.parametrize("case", BROADCAST_CASES)
def test_price_broadcast_shape_and_equivalence(case):
    F, K, T, r, sigma, cp = (case[k]() for k in ["F", "K", "T", "r", "sigma", "cp"])
    _broadcast_equivalence_check(price_fn, F, K, T, r, sigma, cp)


def test_price_raises_on_non_broadcastable():
    # (3,) vs (2,) should error when doing operations internally
    F = np.array([100.0, 101.0, 102.0])
    K = np.array([100.0, 99.0])
    with pytest.raises(ValueError):
        _ = price_fn(F, K, 1.0, 0.01, 0.2, 1)


# ------------------------------
# Greeks broadcasting
# ------------------------------
@pytest.mark.parametrize("case", BROADCAST_CASES)
@pytest.mark.parametrize(
    "fn",
    [
        delta_euro_future,
        gamma_euro_future,
        vega_euro_future,
        theta_euro_future,
        rho_euro_future,
        dual_delta_euro_future,
        vanna_euro_future,
        vomma_euro_future,
        lambda_euro_future,
    ],
)
def test_greeks_broadcast_shape_and_equivalence(case, fn):
    F, K, T, r, sigma, cp = (case[k]() for k in ["F", "K", "T", "r", "sigma", "cp"])
    # Greeks can be tiny; use slightly looser abs tol but still strict
    _broadcast_equivalence_check(fn, F, K, T, r, sigma, cp, rtol=1e-11, atol=1e-11)


# ------------------------------
# IV broadcasting
# ------------------------------
@pytest.mark.parametrize("case_id", IV_CASE_IDS)
def test_iv_broadcast_shape_and_equivalence(case_id):
    case = BROADCAST_CASES[case_id]
    F, K, T, r, sigma_true, cp = (
        case[k]() for k in ["F", "K", "T", "r", "sigma", "cp"]
    )

    # Ensure sigma_true not too small; keep it moderate and positive
    sigma_true = np.asarray(sigma_true, float)
    sigma_true = np.maximum(sigma_true, 0.08)

    # Generate prices and recover IVs
    C = price_fn(F, K, T, r, sigma_true, cp)
    iv = implied_vol_euro_future(C, F, K, T, r, cp)

    # Shape check
    shape = np.broadcast(F, K, T, r, sigma_true, cp).shape
    assert iv.shape == shape

    # Element-wise scalar equivalence (each element is solved independently)
    F2, K2, T2, R2, S2, CP2 = (
        np.broadcast_to(np.asarray(F, float), shape),
        np.broadcast_to(np.asarray(K, float), shape),
        np.broadcast_to(np.asarray(T, float), shape),
        np.broadcast_to(np.asarray(r, float), shape),
        np.broadcast_to(np.asarray(sigma_true, float), shape),
        np.broadcast_to(np.asarray(cp, int), shape),
    )
    expected = np.empty(shape, dtype=float)
    for idx in np.ndindex(shape):
        C_ = float(price_fn(F2[idx], K2[idx], T2[idx], R2[idx], S2[idx], int(CP2[idx])))
        expected[idx] = float(
            implied_vol_euro_future(
                C_,
                float(F2[idx]),
                float(K2[idx]),
                float(T2[idx]),
                float(R2[idx]),
                int(CP2[idx]),
            )
        )

    # Equivalence (same solver on same scalar inputs)
    np.testing.assert_allclose(iv, expected, rtol=0, atol=5e-9)

    # And equality to the true sigma used to generate prices
    target = np.broadcast_to(np.asarray(sigma_true, float), shape)
    np.testing.assert_allclose(iv, target, rtol=0, atol=5e-7)
