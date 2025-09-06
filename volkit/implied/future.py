# volkit/implied/future.py

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import numpy as np

from volkit.implied.future_res import ImpliedFutureResult
from volkit.implied.future_plot import implied_future_from_option_quotes_plot


def implied_future_from_option_quotes(
    K: np.ndarray,
    call_bid: np.ndarray,
    call_ask: np.ndarray,
    put_bid: np.ndarray,
    put_ask: np.ndarray,
    *,
    eps: float = 1e-12,
    plot=False,
    ax=None,
) -> Tuple[Optional[ImpliedFutureResult], np.ndarray]:
    """
    Robust single-expiry inference with minimal exclusions + max-width forward band.

    Returns:
      (result, valid_mask)
        result: ImpliedFutureResult or None if no feasible 2+ strike subset
        valid_mask: bool array of shape (len(K),) in the ORIGINAL input order.
                    True for strikes kept by the minimal-exclusion solution,
                    False otherwise (including non-finite rows).

    Notes:
      - If 'result' is None, 'valid_mask' still marks the largest-overlap subset
        the routine found (it may have size 0 or 1, which is why result=None).
    """
    n0 = len(K)
    valid_mask_full = np.zeros(n0, dtype=bool)  # will fill at the end

    # -- 0) Basic filtering: need finite quotes on all four sides
    K0 = np.asarray(K, float)
    Cb0 = np.asarray(call_bid, float)
    Ca0 = np.asarray(call_ask, float)
    Pb0 = np.asarray(put_bid, float)
    Pa0 = np.asarray(put_ask, float)

    finite = (
        np.isfinite(K0)
        & np.isfinite(Cb0)
        & np.isfinite(Ca0)
        & np.isfinite(Pb0)
        & np.isfinite(Pa0)
    )
    if finite.sum() < 2:
        # no feasible 2+ strike subset
        return None, valid_mask_full

    # Extract finite rows and remember mapping to original indices
    idx_finite = np.nonzero(finite)[0]  # original -> finite
    K = K0[idx_finite]
    Cb = Cb0[idx_finite]
    Ca = Ca0[idx_finite]
    Pb = Pb0[idx_finite]
    Pa = Pa0[idx_finite]

    # Require at least two *distinct* strikes
    if np.unique(K).size < 2:
        return None, valid_mask_full

    # Sort by strike within the finite subset; keep mapping back to original
    order = np.argsort(K)
    K, Cb, Ca, Pb, Pa = K[order], Cb[order], Ca[order], Pb[order], Pa[order]
    # map: sorted-index -> original-index
    idx_sorted_to_orig = idx_finite[order]

    # Build L,U so J_i(D) = [D*K_i + L_i, D*K_i + U_i]
    L = Cb - Pa
    U = Ca - Pb

    # -- 1) Candidate discounts where endpoint order can change
    Ds = _critical_discounts(K, L, U, eps=eps)
    if Ds.size == 0:
        return None, valid_mask_full
    eval_Ds = _with_midpoints(Ds)

    # -- 2) Step 1: minimal exclusions via maximum overlap depth
    best_depth = -1
    candidate_subsets: Dict[Tuple[int, ...], List[float]] = (
        {}
    )  # subset (in sorted space) -> list of D
    for D in eval_Ds:
        a, b = _intervals_at_D(K, L, U, D)
        depth, x_star, keep_idx_sorted = _max_overlap_subset_at_D(a, b, eps=eps)
        if depth > best_depth:
            best_depth = depth
            candidate_subsets.clear()
        if depth == best_depth:
            key = tuple(keep_idx_sorted.tolist())
            candidate_subsets.setdefault(key, []).append(D)

    # if nothing to keep or only 1 strike overlaps anywhere, return None with mask showing the best we found
    if best_depth < 2:
        # mark the (unique) best subset if any
        if candidate_subsets:
            # take the first key deterministically (they are all size best_depth)
            key0 = next(iter(candidate_subsets.keys()))
            keep_orig = idx_sorted_to_orig[np.array(key0, dtype=int)]
            valid_mask_full[keep_orig] = True
        return None, valid_mask_full

    # -- 3) Step 2: among tied subsets, maximize forward width
    best = None  # (width, D_star, F_bid, F_ask, D_min, D_max, subset_key_sorted)
    for key_sorted in candidate_subsets.keys():
        S_sorted = np.array(key_sorted, dtype=int)
        D_interval = _feasible_D_interval_for_subset(K, L, U, S_sorted, eps=eps)
        if D_interval is None:
            continue
        D_lo, D_hi = D_interval
        D_cands = _width_candidate_discounts_for_subset(
            K, L, U, S_sorted, D_lo, D_hi, eps=eps
        )
        if D_cands.size == 0:
            continue

        w_best = -np.inf
        pick = None
        for D in D_cands:
            F_bid, F_ask = _forward_band_for_subset_at_D(K, L, U, S_sorted, D)
            w = F_ask - F_bid
            if w > w_best + 1e-15 or (
                abs(w - w_best) <= 1e-15 and (pick is None or D < pick[0])
            ):
                w_best = w
                pick = (D, F_bid, F_ask)

        if pick is None:
            continue

        D_star, F_bid_star, F_ask_star = pick
        if (
            (best is None)
            or (w_best > best[0] + 1e-15)
            or (abs(w_best - best[0]) <= 1e-15 and D_star < best[1])
        ):
            # Compute D_min/D_max for the chosen subset (already have it)
            best = (w_best, D_star, F_bid_star, F_ask_star, D_lo, D_hi, key_sorted)

    if best is None:
        # No feasible width candidates inside feasible interval (rare numeric corner)
        return None, valid_mask_full

    # -- 4) Map kept indices back to ORIGINAL order and build mask
    _, D_star, F_bid_star, F_ask_star, D_lo, D_hi, key_sorted = best
    keep_orig = idx_sorted_to_orig[np.array(key_sorted, dtype=int)]
    valid_mask_full[keep_orig] = True

    F_star = 0.5 * (F_bid_star + F_ask_star)
    result = ImpliedFutureResult(
        F=F_star, F_bid=F_bid_star, F_ask=F_ask_star, D_min=D_lo, D_max=D_hi
    )
    if plot:
        implied_future_from_option_quotes_plot(
            K0, Cb0, Ca0, Pb0, Pa0, result, valid_mask_full, ax
        )
    return result, valid_mask_full


# ----------------------------
# Helpers
# ----------------------------


def _intervals_at_D(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, D: float
) -> Tuple[np.ndarray, np.ndarray]:
    a = D * K + L
    b = D * K + U
    return a, b


def _critical_discounts(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    K = np.asarray(K)
    L = np.asarray(L)
    U = np.asarray(U)
    n = len(K)
    cand: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = K[i] - K[j]
            if abs(denom) < eps:
                continue
            D1 = (L[j] - L[i]) / denom  # a_i = a_j
            D2 = (U[j] - U[i]) / denom  # b_i = b_j
            D3 = (U[j] - L[i]) / denom  # a_i = b_j
            for D in (D1, D2, D3):
                if np.isfinite(D) and (D > 0):
                    cand.append(float(D))
    if not cand:
        return np.array([], dtype=float)
    return np.array(sorted(set(cand)), dtype=float)


def _with_midpoints(Ds: np.ndarray) -> np.ndarray:
    if Ds.size == 0:
        return Ds
    mids = (Ds[:-1] + Ds[1:]) * 0.5
    out = np.concatenate([Ds, mids])
    out.sort()
    mask = np.ones_like(out, dtype=bool)
    mask[1:] = np.abs(np.diff(out)) > 1e-15
    return out[mask]


def _max_overlap_subset_at_D(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-12
) -> Tuple[int, float, np.ndarray]:
    n = len(a)
    events: List[Tuple[float, int, int]] = []
    for i in range(n):
        ai, bi = a[i], b[i]
        if not (np.isfinite(ai) and np.isfinite(bi) and ai <= bi + eps):
            continue
        events.append((ai, +1, i))
        events.append((bi, -1, i))
    if not events:
        return 0, np.nan, np.array([], dtype=int)

    events.sort(key=lambda t: (t[0], -t[1]))  # starts before ends at same x

    best_depth = -1
    x_star = events[0][0]
    active = 0
    for x, typ, _ in events:
        if typ == +1:
            active += 1
            if active > best_depth:
                best_depth = active
                x_star = x
        else:
            active -= 1

    S = np.where((a <= x_star + eps) & (b + eps >= x_star))[0]
    return best_depth, x_star, S


def _feasible_D_interval_for_subset(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, S: np.ndarray, eps: float = 1e-12
) -> Optional[Tuple[float, float]]:
    Ks, Ls, Us = K[S], L[S], U[S]
    if np.unique(Ks).size < 2:
        return None
    D_lo = 0.0
    D_hi = np.inf
    for p in range(len(S)):
        for q in range(len(S)):
            if Ks[q] <= Ks[p] + eps:
                continue
            denom = Ks[q] - Ks[p]
            lo = (Ls[p] - Us[q]) / denom
            hi = (Us[p] - Ls[q]) / denom
            if np.isfinite(lo):
                D_lo = max(D_lo, lo)
            if np.isfinite(hi):
                D_hi = min(D_hi, hi)
    D_lo = max(D_lo, eps)
    if not np.isfinite(D_hi) or (D_lo > D_hi + 1e-15):
        return None
    return float(D_lo), float(D_hi)


def _width_candidate_discounts_for_subset(
    K: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    D_lo: float,
    D_hi: float,
    eps: float = 1e-12,
) -> np.ndarray:
    Ks, Ls, Us = K[S], L[S], U[S]
    cand: List[float] = [D_lo, D_hi]
    for p in range(len(S)):
        for q in range(p + 1, len(S)):
            denom = Ks[p] - Ks[q]
            if abs(denom) < eps:
                continue
            D1 = (Ls[q] - Ls[p]) / denom  # a_p = a_q
            D2 = (Us[q] - Us[p]) / denom  # b_p = b_q
            for D in (D1, D2):
                if np.isfinite(D) and (D_lo - 1e-14 <= D <= D_hi + 1e-14) and D > 0:
                    cand.append(float(D))
    Ds = np.array(sorted(set(cand)), dtype=float)
    if Ds.size >= 2:
        mids = (Ds[:-1] + Ds[1:]) * 0.5
        Ds = np.unique(np.concatenate([Ds, mids]))
        Ds.sort()
    return Ds


def _forward_band_for_subset_at_D(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, S: np.ndarray, D: float
) -> Tuple[float, float]:
    Ks, Ls, Us = K[S], L[S], U[S]
    a = D * Ks + Ls
    b = D * Ks + Us
    X_low = float(np.max(a))
    X_high = float(np.min(b))
    return X_low / D, X_high / D
