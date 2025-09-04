# tests/test_iv_and_utils.py
import numpy as np
import pytest
import warnings

from volkit.future import _parse_cp, price_euro_future, iv_euro_future

def test_parse_cp_values_and_broadcast():
    cp = _parse_cp(["c", "put", +1, -1, "CALL", "P"])
    np.testing.assert_array_equal(cp, np.array([1, -1, 1, -1, 1, -1]))
    b = _parse_cp("call", target_shape=(2,3))
    assert b.shape == (2,3)
    assert np.all(b == 1)

def test_price_and_iv_roundtrip_calls_and_puts():
    F, K, T, r, sigma = 100.0, np.array([80.0, 100.0, 120.0]), 0.75, 0.01, 0.25
    df = np.exp(-r*T)
    C_call = price_euro_future(F, K, T, r, sigma, "call")
    C_put  = price_euro_future(F, K, T, r, sigma, "put")
    iv_call = iv_euro_future(C_call, F, K, T, r, "call")
    iv_put  = iv_euro_future(C_put,  F, K, T, r, "put")
    np.testing.assert_allclose(iv_call, sigma, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(iv_put,  sigma, rtol=2e-6, atol=2e-8)
    # parity sanity (not a test for equality here, just non-pathological values)
    assert np.all(C_call >= df*np.maximum(F-K,0))

def test_iv_at_cap_returns_inf():
    F, K, T, r = 100.0, 80.0, 1.0, 0.02
    df = np.exp(-r*T)
    C_cap = df * F  # call cap
    iv = iv_euro_future(C_cap, F, K, T, r, "call")
    assert np.isinf(iv)

def test_iv_t0_returns_zero_when_intrinsic():
    F, K, T, r, sigma = 100.0, 90.0, 0.0, 0.00, 0.30
    df = np.exp(-r*T)
    C_intr = df * max(F-K, 0.0)
    iv = iv_euro_future(C_intr, F, K, T, r, "call")
    assert iv == 0.0

def test_iv_below_noarb_is_nan():
    F, K, T, r = 100.0, 90.0, 0.5, 0.01
    df = np.exp(-r*T)
    C = df * max(F-K, 0.0) - 1e-4  # slightly below intrinsic
    iv = iv_euro_future(C, F, K, T, r, "call")
    assert np.isnan(iv)

def test_iv_requires_bracket_growth():
    # Make a very high vol so initial hi=1.0 must grow
    F, K, T, r, sigma_true = 100.0, 100.0, 1.0, 0.01, 3.0
    C = price_euro_future(F, K, T, r, sigma_true, "call")
    iv = iv_euro_future(C, F, K, T, r, "call")
    np.testing.assert_allclose(iv, sigma_true, rtol=1e-4, atol=1e-6)

def test_cp_vector_broadcast_in_iv():
    F = np.array([100.0, 100.0])
    K = np.array([90.0, 110.0])
    T, r, sigma = 0.5, 0.01, 0.2
    C = price_euro_future(F, K, T, r, sigma, ["call", "put"])
    iv = iv_euro_future(C, F, K, T, r, ["call", "put"])
    np.testing.assert_allclose(iv, sigma, rtol=2e-6, atol=2e-8)
