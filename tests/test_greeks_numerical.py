import numpy as np
import pytest

from volkit.future import (
    price_euro_future,
    delta_euro_future,
    gamma_euro_future,
    vega_euro_future,
    theta_euro_future,
    rho_euro_future,
    dual_delta_euro_future,
    vanna_euro_future,
    vomma_euro_future,
    lambda_euro_future,
)


# ---------- numerical diffs ----------
def _price(F, K, T, r, sigma, cp):
    return price_euro_future(F, K, T, r, sigma, cp)


def n_delta(F, K, T, r, sigma, cp, h):
    return (_price(F + h, K, T, r, sigma, cp) - _price(F - h, K, T, r, sigma, cp)) / (
        2 * h
    )


def n_gamma(F, K, T, r, sigma, cp, h):
    return (
        _price(F + h, K, T, r, sigma, cp)
        - 2 * _price(F, K, T, r, sigma, cp)
        + _price(F - h, K, T, r, sigma, cp)
    ) / (h * h)


def n_vega(F, K, T, r, sigma, cp, k):
    return (_price(F, K, T, r, sigma + k, cp) - _price(F, K, T, r, sigma - k, cp)) / (
        2 * k
    )


def n_theta_calendar(F, K, T, r, sigma, cp, t):
    # calendar theta: dV/dt = - dV/dT
    t = min(t, 0.5 * T)
    return -(_price(F, K, T + t, r, sigma, cp) - _price(F, K, T - t, r, sigma, cp)) / (
        2 * t
    )


def n_rho(F, K, T, r, sigma, cp, q):
    return (_price(F, K, T, r + q, sigma, cp) - _price(F, K, T, r - q, sigma, cp)) / (
        2 * q
    )


def n_dual_delta(F, K, T, r, sigma, cp, k):
    return (_price(F, K + k, T, r, sigma, cp) - _price(F, K - k, T, r, sigma, cp)) / (
        2 * k
    )


def n_vanna(F, K, T, r, sigma, cp, h, k):
    return (
        _price(F + h, K, T, r, sigma + k, cp)
        - _price(F + h, K, T, r, sigma - k, cp)
        - _price(F - h, K, T, r, sigma + k, cp)
        + _price(F - h, K, T, r, sigma - k, cp)
    ) / (4 * h * k)


def n_vomma(F, K, T, r, sigma, cp, k):
    return (
        _price(F, K, T, r, sigma + k, cp)
        - 2 * _price(F, K, T, r, sigma, cp)
        + _price(F, K, T, r, sigma - k, cp)
    ) / (k * k)


def n_lambda(F, K, T, r, sigma, cp, h):
    Vp = _price(F * (1 + h), K, T, r, sigma, cp)
    Vm = _price(F * (1 - h), K, T, r, sigma, cp)
    return (np.log(Vp) - np.log(Vm)) / (2 * np.log(1 + h))  # d log V / d log F


# ---------- parameter grid ----------
CASES = [
    dict(F=100.0, K=100.0, T=0.5, r=0.02, sigma=0.20, cp="call"),
    dict(F=100.0, K=120.0, T=0.5, r=0.01, sigma=0.25, cp="call"),
    dict(F=100.0, K=80.0, T=0.5, r=0.00, sigma=0.15, cp="put"),
    dict(F=100.0, K=100.0, T=0.1, r=0.03, sigma=0.30, cp="call"),
]


@pytest.mark.parametrize("p", CASES)
def test_delta(p):
    h = max(1e-4 * p["F"], 1e-4)
    ana = delta_euro_future(**p)
    num = n_delta(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], h)
    np.testing.assert_allclose(ana, num, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("p", CASES)
def test_gamma(p):
    h = max(1e-3 * p["F"], 1e-3)
    ana = gamma_euro_future(**p)
    num = n_gamma(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], h)
    np.testing.assert_allclose(ana, num, rtol=5e-5, atol=5e-8)


@pytest.mark.parametrize("p", CASES)
def test_vega(p):
    k = max(1e-4 * p["sigma"], 1e-5)
    ana = vega_euro_future(**p)
    num = n_vega(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], k)
    np.testing.assert_allclose(ana, num, rtol=3e-6, atol=3e-8)


@pytest.mark.parametrize("p", CASES)
def test_theta_calendar(p):
    # slightly larger step + looser tolerance for robustness
    t = max(1e-4 * max(p["T"], 1.0), 1e-5)
    ana = theta_euro_future(**p)  # calendar theta (dV/dt)
    num = n_theta_calendar(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], t)
    np.testing.assert_allclose(ana, num, rtol=5e-2, atol=5e-4)


@pytest.mark.parametrize("p", CASES)
def test_rho(p):
    q = max(1e-4 * max(abs(p["r"]), 1.0), 1e-5)
    ana = rho_euro_future(**p)
    num = n_rho(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], q)
    np.testing.assert_allclose(ana, num, rtol=3e-6, atol=3e-8)


@pytest.mark.parametrize("p", CASES)
def test_dual_delta(p):
    k = max(1e-4 * p["K"], 1e-3)
    ana = dual_delta_euro_future(**p)
    num = n_dual_delta(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], k)
    np.testing.assert_allclose(ana, num, rtol=3e-6, atol=3e-8)


@pytest.mark.parametrize("p", CASES)
def test_vanna(p):
    h = max(1e-3 * p["F"], 1e-3)
    k = max(1e-3 * p["sigma"], 1e-4)
    ana = vanna_euro_future(**p)
    num = n_vanna(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], h, k)
    np.testing.assert_allclose(ana, num, rtol=8e-5, atol=1e-6)


@pytest.mark.parametrize("p", CASES)
def test_vomma(p):
    k = max(1e-3 * p["sigma"], 1e-4)
    ana = vomma_euro_future(**p)
    num = n_vomma(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], k)
    np.testing.assert_allclose(ana, num, rtol=8e-5, atol=1e-6)


@pytest.mark.parametrize("p", CASES)
def test_lambda(p):
    h = 1e-3
    ana = lambda_euro_future(**p)
    num = n_lambda(p["F"], p["K"], p["T"], p["r"], p["sigma"], p["cp"], h)
    np.testing.assert_allclose(ana, num, rtol=1e-3, atol=2e-6)
