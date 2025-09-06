"""
European options on futures (Black, 1976).

This module prices and computes Greeks for **European** options on **futures**,
using the Black (1976) model (a lognormal model on the futures price).

Conventions
-----------
- All inputs are **broadcastable** NumPy arrays; scalars work too.
- `T` is time to maturity in **years** (e.g., 0.5 = 6 months).
- `r` is the **continuously compounded** risk-free rate (annualized).
- `sigma` is **Black volatility** per √year (e.g., 0.20 for 20%).
- `cp` (call/put) accepts any of: `+1`, `-1`, `"c"`, `"p"`, `"call"`, `"put"`.
  It broadcasts to the common shape.
- Returns have the broadcasted shape of inputs.
- Numerical safety: we floor `sigma*sqrt(T)` at a tiny value (1e-16).

Notes
-----
- Vega is returned per **1.00** vol (not per 1%); divide by 100 if desired.
- `rho` assumes the futures price is independent of the discount rate.
"""

import numpy as np
from scipy.stats import norm

__all__ = [
    "price_euro_future",
    "delta_euro_future",
    "gamma_euro_future",
    "vega_euro_future",
    "theta_euro_future",
    "rho_euro_future",
    "dual_delta_euro_future",
    "vanna_euro_future",
    "vomma_euro_future",
    "lambda_euro_future",
]

# ---------- utils ----------


def _broadcast_shape(*xs):
    return np.broadcast(*[np.asarray(x) for x in xs]).shape


def _parse_cp(cp, target_shape=None):
    """
    Normalize call/put flags to {+1, -1} and optionally broadcast.

    Parameters
    ----------
    cp : {int, str, array-like}
        Call/put indicator(s). Accepts +1/-1, "c"/"p", "call"/"put"
        (case-insensitive). Arrays are allowed and will be elementwise parsed.
    target_shape : tuple, optional
        If given, the result is broadcast to this shape.

    Returns
    -------
    numpy.ndarray
        Integer array of +1 (call) or -1 (put), broadcast if `target_shape` set.

    Raises
    ------
    ValueError
        If a value cannot be parsed as call or put.
    """
    arr = np.asarray(cp, dtype=object)

    def _one(v):
        if isinstance(v, (int, np.integer)):
            if v in (1, -1):
                return int(v)
            raise ValueError("cp integer must be +1 or -1")
        s = str(v).strip().lower()
        if s in ("1", "+1", "c", "call"):
            return 1
        if s in ("-1", "p", "put"):
            return -1
        raise ValueError("cp must be +1/-1 or 'c'/'call'/'p'/'put'")

    if arr.ndim == 0:
        out = np.array(_one(arr.item()), dtype=int)
    else:
        vec = np.vectorize(_one, otypes=[int])
        out = vec(arr)

    if target_shape is not None:
        out = np.broadcast_to(out, target_shape)
    return out


def _prep(F, K, T, r, sigma, cp):
    """
    Common pre-computation and broadcasting for Black-76 on futures.

    Returns
    -------
    tuple
        (F, K, T, r, sigma, cp, sqrtT, a, d1, d2, df)
        where a = sigma*sqrtT (floored), df = exp(-r*T).
    """
    F = np.asarray(F, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    r = np.asarray(r, float)
    sigma = np.asarray(sigma, float)
    cp = _parse_cp(cp)  # should return +1 for call, -1 for put; keeps original shape

    # Broadcast ALL arrays together. This will yield (5,2) for your example.
    F, K, T, r, sigma, cp = np.broadcast_arrays(F, K, T, r, sigma, cp)

    sqrtT = np.sqrt(np.maximum(T, 0.0))
    a = np.maximum(sigma * sqrtT, 1e-16)  # avoid division by 0
    d1 = np.log(F / K) / a + 0.5 * a  # (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrtT)
    d2 = d1 - a
    df = np.exp(-r * T)
    return F, K, T, r, sigma, cp, sqrtT, a, d1, d2, df


# ---------- price ----------


def price_euro_future(F, K, T, r, sigma, cp=1):
    """
    Price of a European option on a futures contract (Black-76).

    Parameters
    ----------
    F : float or array-like
        Futures price today.
    K : float or array-like
        Strike price.
    T : float or array-like
        Time to maturity in years.
    r : float or array-like
        Continuously-compounded risk-free rate (annualized).
    sigma : float or array-like
        Black volatility per √year (e.g., 0.20 for 20%).
    cp : {+1, -1, 'c', 'p', 'call', 'put'}, optional
        Call/put flag (broadcastable). Default is +1 (call).

    Returns
    -------
    numpy.ndarray
        Option present value(s), broadcasted to the common shape.
    """
    F, K, T, r, sigma, cp, _, _, d1, d2, df = _prep(F, K, T, r, sigma, cp)
    return df * (cp * F * norm.cdf(cp * d1) - cp * K * norm.cdf(cp * d2))


# ---------- greeks ----------


def delta_euro_future(F, K, T, r, sigma, cp=1):
    """
    Futures delta ∂V/∂F.

    Returns
    -------
    numpy.ndarray
        Delta with broadcasted shape.
    """
    F, K, T, r, sigma, cp, _, _, d1, _, df = _prep(F, K, T, r, sigma, cp)
    return df * cp * norm.cdf(cp * d1)


def gamma_euro_future(F, K, T, r, sigma, cp=1):
    """
    Futures gamma ∂²V/∂F².

    Returns
    -------
    numpy.ndarray
        Gamma with broadcasted shape.
    """
    F, K, T, r, sigma, cp, _, a, d1, _, df = _prep(F, K, T, r, sigma, cp)
    return df * norm.pdf(d1) / (F * a)


def vega_euro_future(F, K, T, r, sigma, cp=1):
    """
    Vega ∂V/∂σ (per 1.00 volatility, not per 1%).

    Returns
    -------
    numpy.ndarray
        Vega with broadcasted shape.
    """
    F, K, T, r, sigma, cp, sqrtT, _, d1, _, df = _prep(F, K, T, r, sigma, cp)
    return df * F * norm.pdf(d1) * sqrtT


def theta_euro_future(F, K, T, r, sigma, cp=1):
    """
    Theta ∂V/∂T (calendar time theta, holding F constant).

    Returns
    -------
    numpy.ndarray
        Theta with broadcasted shape (per year).
    """
    F, K, T, r, sigma, cp, sqrtT, _, d1, _, df = _prep(F, K, T, r, sigma, cp)
    time_decay = -df * (F * norm.pdf(d1) * sigma) / (2.0 * np.maximum(sqrtT, 1e-16))
    carry = +r * price_euro_future(F, K, T, r, sigma, cp)
    return time_decay + carry


def rho_euro_future(F, K, T, r, sigma, cp=1):
    """
    Rho ∂V/∂r (assumes F independent of r).

    Returns
    -------
    numpy.ndarray
        Rho with broadcasted shape.
    """
    V = price_euro_future(F, K, T, r, sigma, cp)
    return -T * V


def dual_delta_euro_future(F, K, T, r, sigma, cp=1):
    """
    Strike sensitivity ∂V/∂K (a.k.a. dual delta).

    Returns
    -------
    numpy.ndarray
        Dual delta with broadcasted shape.
    """
    F, K, T, r, sigma, cp, _, _, _, d2, df = _prep(F, K, T, r, sigma, cp)
    return -df * cp * norm.cdf(cp * d2)


def vanna_euro_future(F, K, T, r, sigma, cp=1):
    """
    Vanna ∂²V/(∂F ∂σ).

    Returns
    -------
    numpy.ndarray
        Vanna with broadcasted shape.
    """
    F, K, T, r, sigma, cp, _, _, d1, d2, df = _prep(F, K, T, r, sigma, cp)
    return df * norm.pdf(d1) * (-d2 / np.maximum(sigma, 1e-16))


def vomma_euro_future(F, K, T, r, sigma, cp=1):
    """
    Vomma (volga) ∂²V/∂σ².

    Returns
    -------
    numpy.ndarray
        Vomma with broadcasted shape.
    """
    F, K, T, r, sigma, cp, sqrtT, _, d1, d2, df = _prep(F, K, T, r, sigma, cp)
    vega = df * F * norm.pdf(d1) * sqrtT
    return vega * d1 * d2 / np.maximum(sigma, 1e-16)


def lambda_euro_future(F, K, T, r, sigma, cp=1):
    """
    Elasticity (leverage) λ = (F/V) * delta.

    Returns
    -------
    numpy.ndarray
        Elasticity with broadcasted shape.
    """
    V = price_euro_future(F, K, T, r, sigma, cp)
    return (np.asarray(F, float) / np.maximum(V, 1e-16)) * delta_euro_future(
        F, K, T, r, sigma, cp
    )
