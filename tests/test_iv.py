# tests/test_iv_and_utils.py
import numpy as np
import pytest

from volkit.future import price_euro_future, iv_euro_future


def test_price_and_iv_roundtrip_calls_and_puts():
    F, K, T, r, sigma = 100.0, np.array([80.0, 100.0, 120.0]), 0.75, 0.01, 0.25
    df = np.exp(-r * T)
    C_call = price_euro_future(F, K, T, r, sigma, "call")
    C_put = price_euro_future(F, K, T, r, sigma, "put")
    iv_call = iv_euro_future(C_call, F, K, T, r, "call")
    iv_put = iv_euro_future(C_put, F, K, T, r, "put")
    np.testing.assert_allclose(iv_call, sigma, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(iv_put, sigma, rtol=2e-6, atol=2e-8)
    # parity sanity (non-pathological)
    assert np.all(C_call >= df * np.maximum(F - K, 0))


def test_iv_at_cap_returns_inf():
    F, K, T, r = 100.0, 80.0, 1.0, 0.02
    df = np.exp(-r * T)
    C_cap = df * F  # theoretical call cap
    iv = iv_euro_future(C_cap, F, K, T, r, "call")
    assert np.isinf(iv)


def test_iv_t0_returns_zero_when_intrinsic():
    F, K, T, r = 100.0, 90.0, 0.0, 0.00
    df = np.exp(-r * T)
    C_intr = df * max(F - K, 0.0)
    iv = iv_euro_future(C_intr, F, K, T, r, "call")
    assert iv == 0.0


def test_iv_below_noarb_is_nan():
    F, K, T, r = 100.0, 90.0, 0.5, 0.01
    df = np.exp(-r * T)
    C = df * max(F - K, 0.0) - 1e-4  # slightly below intrinsic
    iv = iv_euro_future(C, F, K, T, r, "call")
    assert np.isnan(iv)


def test_cp_vector_broadcast_in_iv():
    F = np.array([100.0, 100.0])
    K = np.array([90.0, 110.0])
    T, r, sigma = 0.5, 0.01, 0.2
    C = price_euro_future(F, K, T, r, sigma, ["call", "put"])
    iv = iv_euro_future(C, F, K, T, r, ["call", "put"])
    np.testing.assert_allclose(iv, sigma, rtol=2e-6, atol=2e-8)


def test_iv_bisection_early_tight_break():
    # Loose tolerances so 'tight' can be satisfied early (doesn't touch growth/polish paths)
    F, K, T, r, sigma_true = 100.0, 100.0, 0.75, 0.01, 0.20
    C = price_euro_future(F, K, T, r, sigma_true, "call")
    iv = iv_euro_future(C, F, K, T, r, "call", tol=1e-3, price_tol=1e-4, max_iter=100)
    np.testing.assert_allclose(iv, sigma_true, rtol=1e-6, atol=1e-8)


def test_iv_bisection_runs_to_max_iter_without_tight():
    # Strict tolerances so 'tight' stays False; small max_iter (doesn't touch growth/polish)
    F, K, T, r, sigma_true = 100.0, 100.0, 0.5, 0.01, 0.20
    C = price_euro_future(F, K, T, r, sigma_true, "call")
    iv = iv_euro_future(C, F, K, T, r, "call", tol=0.0, price_tol=0.0, max_iter=2)
    # Coarse accuracy expected
    np.testing.assert_allclose(iv, sigma_true, rtol=5e-2, atol=5e-3)


def test_cp_invalid_integer_raises_valueerror():
    # triggers: "cp integer must be +1 or -1"
    with pytest.raises(ValueError, match=r"cp integer must be \+1 or -1"):
        price_euro_future(100.0, 100.0, 1.0, 0.01, 0.2, cp=np.int64(2))


def test_cp_invalid_string_raises_valueerror():
    # triggers: "cp must be +1/-1 or 'c'/'call'/'p'/'put'"
    with pytest.raises(ValueError, match=r"cp must be \+1/-1 or 'c'/'call'/'p'/'put'"):
        price_euro_future(100.0, 100.0, 1.0, 0.01, 0.2, cp="xx")


def test_iv_returns_out_when_all_fail_after_growth():
    # Price generated from a volatility far above sigma_max, so after growth
    # the price at hi is still < C -> fail=True, solve becomes all False -> early return `out`.
    F, K, T, r, sigma_true = 100.0, 100.0, 1.0, 0.01, 2.0
    C = price_euro_future(F, K, T, r, sigma_true, "call")
    iv = iv_euro_future(C, F, K, T, r, "call", sigma_max=0.3)
    assert np.isnan(iv)
