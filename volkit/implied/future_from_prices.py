# file: implied_future_from_option_prices.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "ImpliedFutureResult",
    "implied_future_from_option_prices",
]

from .future_res import ImpliedFutureResult


# --------------------------- helpers (private) ---------------------------------


def _as_np1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={arr.shape}")
    return arr


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _count_distinct(x: np.ndarray, eps: float) -> int:
    if x.size == 0:
        return 0
    xs = np.sort(x)
    jumps = np.diff(xs) > eps
    return int(1 + jumps.sum())


def _constrained_ols_line(
    K: np.ndarray, y: np.ndarray, *, eps: float
) -> Tuple[float, float]:
    """Return (a, b) minimizing ||y - (a + bK)||^2 with *slope constraint* b ∈ [-1, 0]."""
    if K.size != y.size or K.size < 2:
        raise ValueError("Need at least two points for OLS line")

    K_mean = K.mean()
    y_mean = y.mean()
    Kc = K - K_mean
    yc = y - y_mean
    varK = float(Kc.dot(Kc))
    if varK <= eps:
        b = 0.0  # flat; implies D≈0
        a = y_mean
        return a, b
    b_ols = float(Kc.dot(yc) / varK)
    b = float(np.clip(b_ols, -1.0, 0.0))
    a = float((y - b * K).mean())
    return a, b


def _mad(x: np.ndarray, eps: float) -> float:
    if x.size == 0:
        return eps
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(eps, 1.4826 * mad))


def _mad_threshold(resid: np.ndarray, mult: float, eps: float) -> float:
    return float(max(5 * eps, mult * _mad(resid, eps)))


def _feasible_D_band_from_pairwise_slopes(
    K: np.ndarray,
    y: np.ndarray,
    *,
    q_low: float,
    q_high: float,
    clip_D: Tuple[float, float],
    eps: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return a robust D band from pairwise slopes s_ij = (y_i - y_j)/(K_i - K_j)
    via quantiles, with a single clamp to canonical clip bounds.
    """
    n = K.size
    if n < 2:
        return None, None

    slopes = []
    for i in range(n - 1):
        dK = K[i + 1 :] - K[i]
        m = np.abs(dK) > eps
        if not m.any():
            continue
        dy = y[i + 1 :] - y[i]
        slopes.append(dy[m] / dK[m])

    if not slopes:
        return None, None

    s_all = np.concatenate(slopes)
    s_all = s_all[np.isfinite(s_all)]
    if s_all.size == 0:
        return None, None

    # Quantiles in one pass; canonicalize order
    lo_q, hi_q = np.quantile(s_all, (q_low, q_high))
    s_lo, s_hi = sorted((float(lo_q), float(hi_q)))  # s_lo <= s_hi

    # D = -s ⇒ order flips
    D_lo_raw, D_hi_raw = -s_hi, -s_lo  # D_lo_raw <= D_hi_raw

    # Canonicalize clip bounds and clamp once
    c_lo, c_hi = clip_D if clip_D[0] <= clip_D[1] else (clip_D[1], clip_D[0])
    D_min = max(c_lo, D_lo_raw)
    D_max = min(c_hi, D_hi_raw)

    return D_min, D_max


def _forward_band_from_D_band(
    K: np.ndarray,
    y: np.ndarray,
    D_min: float,
    D_max: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Intersect per-strike forward intervals induced by [D_min, D_max]."""
    if K.size == 0:
        return None, None
    F_lo = -np.inf
    F_hi = np.inf
    for Ki, yi in zip(K, y):
        f1 = Ki + yi / D_min
        f2 = Ki + yi / D_max
        lo_i = min(f1, f2)
        hi_i = max(f1, f2)
        F_lo = max(F_lo, lo_i)
        F_hi = min(F_hi, hi_i)
    return float(F_lo), float(F_hi)


def _scatter_back_mask(
    base_mask: np.ndarray, idx: np.ndarray, inlier_local: np.ndarray
) -> np.ndarray:
    out = np.zeros_like(base_mask, dtype=bool)
    out_idx = idx[inlier_local]
    out[out_idx] = True
    return out


# ------------------------------ public API -------------------------------------


def implied_future_from_option_prices(
    K: np.ndarray,
    call: np.ndarray,
    put: np.ndarray,
    *,
    # Common knobs
    eps: float = 1e-12,
    plot: bool = False,
    ax=None,
    # Legacy/initial implementation knobs
    ransac: bool = False,
    slope_quantiles: Tuple[float, float] = (0.25, 0.75),
    clip_D: Tuple[float, float] = (1e-6, 1.0),
    trim_mad_mult: float = 3.5,
    max_trim_iters: int = 5,
    # Newer optional knobs (kept optional for backward compatibility in callers)
    tol_sigma: Optional[float] = None,
    rel_tol: Optional[float] = None,
    abs_tol: Optional[float] = None,
) -> Tuple[Optional[ImpliedFutureResult], np.ndarray]:
    """Infer forward ``F`` and discount factor ``D`` from *single* option prices.

    Returns
    -------
    (result, valid_mask)
        result : ImpliedFutureResult or None
            ``None`` if fewer than 2 distinct strikes (after filtering) or no
            feasible band could be established.
        valid_mask : ndarray of bool, shape (len(K),)
            In ORIGINAL input order. True if a strike was used in the solution.

    Notes
    -----
    • This mirrors the quote-based API. Intervals (F_bid/F_ask, D_min/D_max)
      are data-implied feasibility bands.
    • Supports both the initial MAD-trim flow and (optionally) sigma/relative
      thresholds via tol_sigma/rel_tol/abs_tol if provided.
    """
    K = _as_np1d(K, "K")
    C = _as_np1d(call, "call")
    P = _as_np1d(put, "put")
    if not (len(K) == len(C) == len(P)):
        raise ValueError("K, call, put must have the same length")

    n = len(K)
    finite_mask = _finite_mask(K, C, P)
    if finite_mask.sum() < 2:
        return None, np.zeros(n, dtype=bool)

    # Canonicalize discount clip bounds ONCE for this call
    c_lo, c_hi = clip_D if clip_D[0] <= clip_D[1] else (clip_D[1], clip_D[0])

    idx = np.nonzero(finite_mask)[0]
    Kv = K[idx]
    yv = (C - P)[idx]

    if _count_distinct(Kv, eps) < 2:
        return None, np.zeros(n, dtype=bool)

    # Robust trim loop around constrained OLS fit of y = a + bK, b∈[-1,0]
    inlier = np.ones_like(Kv, dtype=bool)

    def fit(Kx, yx):
        a, b = _constrained_ols_line(Kx, yx, eps=eps)
        return a, b

    a_hat, b_hat = fit(Kv[inlier], yv[inlier])

    for _ in range(max_trim_iters):
        resid = yv - (a_hat + b_hat * Kv)
        if tol_sigma is not None or rel_tol is not None or abs_tol is not None:
            # Sigma/relative thresholding
            scale = _mad(resid[inlier], eps)
            # conservative fallback scale
            if not np.isfinite(scale) or scale <= eps:
                scale = max(np.median(np.abs(resid[inlier])), eps)
            sig = tol_sigma if tol_sigma is not None else trim_mad_mult
            a_tol = abs_tol if abs_tol is not None else 0.0
            r_tol = rel_tol if rel_tol is not None else 0.0
            thresh = sig * scale
            allow = np.abs(resid) <= (
                thresh + a_tol + r_tol * np.maximum(np.abs(yv), eps)
            )
        else:
            # Legacy MAD trimming
            thresh = _mad_threshold(resid[inlier], mult=trim_mad_mult, eps=eps)
            allow = np.abs(resid) <= thresh

        new_inlier = allow
        if new_inlier.sum() < 2:
            break
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier
        a_hat, b_hat = fit(Kv[inlier], yv[inlier])

    if inlier.sum() < 2 or _count_distinct(Kv[inlier], eps) < 2:
        return None, np.zeros(n, dtype=bool)

    # Final fit
    a_hat, b_hat = fit(Kv[inlier], yv[inlier])
    D_hat = -b_hat
    D_hat = float(np.clip(D_hat, c_lo, c_hi))
    if not np.isfinite(D_hat) or D_hat <= 0.0:
        return None, _scatter_back_mask(finite_mask, idx, inlier)
    F_hat = a_hat / D_hat

    # D interval from pairwise slopes on inliers
    q_low, q_high = slope_quantiles
    D_min, D_max = _feasible_D_band_from_pairwise_slopes(
        Kv[inlier], yv[inlier], q_low=q_low, q_high=q_high, clip_D=(c_lo, c_hi), eps=eps
    )

    # Map D band to F band
    F_lo, F_hi = _forward_band_from_D_band(Kv[inlier], yv[inlier], D_min, D_max)

    # --- Containment fixes against roundoff ---
    tiny = max(1e-12, 5 * eps)
    # Keep D_hat inside [D_min, D_max], preserve order and respect canonical clip
    D_min = max(c_lo, min(D_min, D_hat + tiny))
    D_max = min(c_hi, max(D_max, D_hat - tiny))
    if D_min > D_max:  # extremely rare numerical flip; collapse to point
        mid = float(np.clip(D_hat, c_lo, c_hi))
        D_min = D_max = mid

    if F_lo is None or F_hi is None or F_lo > F_hi:
        # fallback: contain the point estimate
        F_lo, F_hi = F_hat - tiny, F_hat + tiny
    else:
        F_lo = float(min(F_lo, F_hat + tiny))
        F_hi = float(max(F_hi, F_hat - tiny))

    # Guarantee F-band contains F_hat
    F_lo = min(F_lo, F_hat)
    F_hi = max(F_hi, F_hat)

    result = ImpliedFutureResult(
        F=float(F_hat),
        F_bid=float(F_lo),
        F_ask=float(F_hi),
        D=float(D_hat),
        D_min=float(D_min),
        D_max=float(D_max),
    )

    out_mask = _scatter_back_mask(finite_mask, idx, inlier)

    # Optional plot kept minimal; best-effort and silent on errors.
    if plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            ax_ = ax if ax is not None else plt.gca()
            Kp = Kv
            yp = yv
            ax_.scatter(Kp[~inlier], yp[~inlier], marker="x", label="outliers")
            ax_.scatter(Kp[inlier], yp[inlier], s=16, label="inliers", alpha=0.85)
            kr = np.linspace(Kp.min(), Kp.max(), 64)
            yr = a_hat + (-D_hat) * kr
            ax_.plot(kr, yr, label=f"fit: D={D_hat:.6f}, F={F_hat:.6f}")
            ax_.legend(loc="best")
        except Exception:
            pass

    return result, out_mask
