import numpy as np
from volkit import price_euro_future, vega_euro_future
from volkit.pricing.future import _broadcast_shape, _parse_cp


def implied_vol_euro_future(
    C, F, K, T, r, cp=1, tol=1e-8, max_iter=100, price_tol=1e-10, sigma_max=10.0
):
    """
    Implied Black volatility for European options on futures (bisection).

    Guardrails include no-arbitrage checks, cap detection, and one Newton polish.

    Parameters
    ----------
    C : float or array-like
        Observed option **price** (present value).
    F, K, T, r, cp
        As in :func:`price_euro_future`.
    tol : float, optional
        Tolerance on volatility bracket width. Default is 1e-8 (per-year vol).
    max_iter : int, optional
        Maximum bisection iterations. Default is 100.
    price_tol : float, optional
        Tolerance on price mismatch when declaring convergence. Default 1e-10.
    sigma_max : float, optional
        Upper bound during bracket growth (fail if still too cheap). Default 10.0.

    Returns
    -------
    numpy.ndarray
        Implied vol array. Special values:
        - `np.inf` if price is at the theoretical cap,
        - `np.nan` on failures (e.g., price outside no-arb bounds).

    Notes
    -----
    - No-arb lower bound: `PV >= DF * max(cp*(F-K), 0)`.
    - No-arb upper cap: `PV <= DF * (F if call else K)`.
    """
    C = np.asarray(C, float)
    F = np.asarray(F, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    r = np.asarray(r, float)

    shape = _broadcast_shape(C, F, K, T, r)
    C, F, K, T, r = (np.broadcast_to(x, shape) for x in (C, F, K, T, r))
    cp = _parse_cp(cp, target_shape=shape)

    df = np.exp(-r * T)
    intrinsic = df * np.maximum(cp * (F - K), 0.0)
    cap = df * np.where(cp == 1, F, K)

    out = np.full(shape, np.nan)
    t0 = T <= 0
    out[t0 & np.isclose(C, intrinsic, atol=price_tol)] = 0.0
    valid = ~t0

    below = (C < intrinsic - price_tol) & valid
    above = (C > cap + price_tol) & valid
    at_cap = np.isclose(C, cap, atol=price_tol) & valid
    out[at_cap] = np.inf
    solve = valid & ~(below | above | at_cap)

    if not np.any(solve):
        return out

    lo = np.full(shape, 1e-8)
    hi = np.full(shape, 1.0)

    # grow hi
    grow = solve.copy()
    it = 0
    while np.any(grow) and it < 50:
        p = price_euro_future(F, K, T, r, hi, cp)
        need = (p < C - price_tol) & grow
        hi = np.where(need, np.minimum(hi * 2.0, sigma_max), hi)
        done = (~need) | (hi >= sigma_max - 1e-12)
        grow = grow & ~done
        it += 1

    # fail if still too cheap
    fail = solve & (price_euro_future(F, K, T, r, hi, cp) < C - price_tol)
    out[fail] = np.nan
    solve &= ~fail
    if not np.any(solve):
        return out

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = price_euro_future(F, K, T, r, mid, cp)
        too_low = price < C
        lo = np.where(too_low & solve, mid, lo)
        hi = np.where((~too_low) & solve, mid, hi)
        tight = (np.abs(hi - lo) < tol) & (np.abs(price - C) < price_tol)
        if np.all(~solve | tight):
            break

    iv = 0.5 * (lo + hi)

    # Newton polish where vega is informative
    vega = vega_euro_future(F, K, T, r, iv, cp)
    good = solve & (vega > 1e-12)
    # Always compute; no-ops where ~good
    p_all = price_euro_future(F, K, T, r, iv, cp)
    step_all = (p_all - C) / np.maximum(vega, 1e-16)
    iv = np.where(good, np.clip(iv - step_all, 0.0, sigma_max), iv)

    out[solve] = iv[solve]
    return out
