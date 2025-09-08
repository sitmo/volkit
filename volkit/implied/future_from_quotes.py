# volkit/implied/future.py

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Iterable, DefaultDict
from collections import defaultdict
import numpy as np

from volkit.implied.future_res import ImpliedFutureResult
from volkit.implied.future_from_quotes_plot import (
    implied_future_from_option_quotes_plot,
)


# ======================================================================
# Public API
# ======================================================================


def implied_future_from_option_quotes(
    K: np.ndarray,
    call_bid: np.ndarray,
    call_ask: np.ndarray,
    put_bid: np.ndarray,
    put_ask: np.ndarray,
    *,
    eps: float = 1e-12,
    plot: bool = False,
    ax=None,
) -> Tuple[Optional[ImpliedFutureResult], np.ndarray]:
    """
    Robust single-expiry inference with minimal exclusions + max-width forward band.

    Returns
    -------
    (result, valid_mask)
        result : ImpliedFutureResult or None
            No feasible subset if fewer than 2 distinct strikes.
        valid_mask : ndarray of bool, shape (len(K),)
            In ORIGINAL input order. True for strikes used by the solution;
            False otherwise.

    Notes
    -----
    - If result is None, valid_mask may still mark one "best" row *only* in the case
      of Ds == ∅ *and* there exist ≥2 distinct strikes among valid rows. If all valid
      rows share the same strike (e.g., duplicates-only situation), the mask is all False.
    - Duplicate-K subsets are rejected when selecting the final subset (they do not
      define a forward band).
    - Selection across depths: we try the highest feasible depth first, then fall back
      to lower depths (≥2). Within a depth, we choose the subset/D that maximizes width.
    """

    # ----------------------------
    # 0) Basic filtering & ordering
    # ----------------------------
    n0 = len(K)
    mask_out = np.zeros(n0, dtype=bool)

    K0, Cb0, Ca0, Pb0, Pa0, idx_sorted_to_orig = _filter_and_sort_finite(
        K, call_bid, call_ask, put_bid, put_ask
    )
    if K0.size < 2:
        return None, mask_out

    # At least two *distinct* strikes present in finite slice?
    if np.unique(K0).size < 2:
        return None, mask_out

    L, U = _make_LU(Cb0, Ca0, Pb0, Pa0)
    # Quick validity flags at row-level (independent of D)
    row_valid = (U >= L + eps) & np.isfinite(L) & np.isfinite(U)

    # -------------------------------------------
    # 1) Candidate discount grid (event locations)
    # -------------------------------------------
    Ds = _critical_discounts(K0, L, U, eps=eps)

    # If NO positive candidate discounts:
    # Mark narrowest valid singleton (only if ≥2 distinct K among valid rows),
    # else leave mask empty.
    if Ds.size == 0:
        idx_single = _narrowest_valid_singleton_index(K0, L, U, row_valid, eps)
        if idx_single is not None:
            mask_out[idx_sorted_to_orig[idx_single]] = True
        return None, mask_out

    eval_Ds = _with_midpoints(Ds)

    # --------------------------------------------------------------------
    # 2) Collect ALL max-overlap subsets grouped by DEPTH across all eval_Ds
    # --------------------------------------------------------------------
    depths_to_subsets, subset_to_Ds = _collect_subsets_by_depth(
        K0, L, U, row_valid, eval_Ds, eps=eps
    )
    if not depths_to_subsets:
        # No valid intervals anywhere
        return None, mask_out

    # ------------------------------------------------------------
    # 3) Choose subset with fallback: highest depth -> lower depth
    #    (reject duplicate-K subsets; require feasible D interval)
    # ------------------------------------------------------------
    choice = _choose_subset_with_fallback(
        K0, L, U, depths_to_subsets, subset_to_Ds, eps=eps
    )
    if choice is None:
        # Nothing feasible across any depth (≥2)
        return None, mask_out

    D_star, F_bid_star, F_ask_star, D_lo, D_hi, key_sorted = choice
    keep_orig = idx_sorted_to_orig[np.array(key_sorted, dtype=int)]
    mask_out[keep_orig] = True

    result = ImpliedFutureResult(
        F=float(0.5 * (F_bid_star + F_ask_star)),
        F_bid=float(F_bid_star),
        F_ask=float(F_ask_star),
        D=float((D_lo + D_hi) / 2),
        D_min=float(D_lo),
        D_max=float(D_hi),
    )

    if plot:
        implied_future_from_option_quotes_plot(
            K, call_bid, call_ask, put_bid, put_ask, result, mask_out, ax=ax
        )
    return result, mask_out


# ======================================================================
# Helpers (private)
# ======================================================================


def _filter_and_sort_finite(
    K: np.ndarray, Cb: np.ndarray, Ca: np.ndarray, Pb: np.ndarray, Pa: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep rows that are finite on all fields; return arrays sorted by K + back-map."""
    K = np.asarray(K, float)
    Cb = np.asarray(Cb, float)
    Ca = np.asarray(Ca, float)
    Pb = np.asarray(Pb, float)
    Pa = np.asarray(Pa, float)

    finite = (
        np.isfinite(K)
        & np.isfinite(Cb)
        & np.isfinite(Ca)
        & np.isfinite(Pb)
        & np.isfinite(Pa)
    )
    idx_finite = np.nonzero(finite)[0]
    if idx_finite.size == 0:
        return K[:0], Cb[:0], Ca[:0], Pb[:0], Pa[:0], idx_finite

    Kf, Cbf, Caf, Pbf, Paf = (
        K[idx_finite],
        Cb[idx_finite],
        Ca[idx_finite],
        Pb[idx_finite],
        Pa[idx_finite],
    )
    order = np.argsort(Kf)
    idx_sorted_to_orig = idx_finite[order]
    return Kf[order], Cbf[order], Caf[order], Pbf[order], Paf[order], idx_sorted_to_orig


def _make_LU(
    Cb: np.ndarray, Ca: np.ndarray, Pb: np.ndarray, Pa: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """L = C_bid - P_ask; U = C_ask - P_bid."""
    L = Cb - Pa
    U = Ca - Pb
    return L, U


def _critical_discounts(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, eps: float
) -> np.ndarray:
    """
    Event discounts where the sorted order of interval endpoints can change.
    Uses equalities among {a_i, b_i} with a_i = D*K_i + L_i, b_i = D*K_i + U_i.
    Only positive D are kept.
    """
    n = len(K)
    cand: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = K[i] - K[j]
            if abs(denom) < eps:
                continue
            # a_i = a_j ; b_i = b_j ; a_i = b_j
            D1 = (L[j] - L[i]) / denom
            D2 = (U[j] - U[i]) / denom
            D3 = (U[j] - L[i]) / denom
            for D in (D1, D2, D3):
                if np.isfinite(D) and (D > 0):
                    cand.append(float(D))
    if not cand:
        return np.array([], dtype=float)
    return np.array(sorted(set(cand)), dtype=float)


def _with_midpoints(Ds: np.ndarray) -> np.ndarray:
    """Add midpoints between consecutive Ds; de-duplicate; keep ascending."""
    if Ds.size == 0:
        return Ds
    mids = (Ds[:-1] + Ds[1:]) * 0.5
    out = np.concatenate([Ds, mids])
    out.sort()
    mask = np.ones_like(out, dtype=bool)
    mask[1:] = np.abs(np.diff(out)) > 1e-15
    return out[mask]


def _intervals_at_D(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, D: float
) -> Tuple[np.ndarray, np.ndarray]:
    a = D * K + L
    b = D * K + U
    return a, b


def _max_overlap_subset_at_D(
    a: np.ndarray, b: np.ndarray, row_valid: np.ndarray, eps: float
) -> Tuple[int, float, np.ndarray]:
    """
    Sweep-line on [a_i, b_i] to find maximum overlap depth and an x* achieving it.
    Returns (depth, x_star, indices_active_at_x_star) while ignoring invalid rows.
    """
    events: List[Tuple[float, int, int]] = []
    for i, ok in enumerate(row_valid):
        if not ok:
            continue
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

    S = np.where(row_valid & (a <= x_star + eps) & (b + eps >= x_star))[0]
    return best_depth, x_star, S


def _collect_subsets_by_depth(
    K: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    row_valid: np.ndarray,
    eval_Ds: Iterable[float],
    eps: float,
) -> Tuple[Dict[int, List[Tuple[int, ...]]], Dict[Tuple[int, ...], List[float]]]:
    """
    For each D in eval_Ds, take the *maximum-overlap* active set at x* and group by depth.
    Returns:
      depths_to_subsets: depth -> list of subset keys (tuples of sorted indices)
      subset_to_Ds     : subset key -> list of D values where it appeared
    """
    depths_to_subsets: DefaultDict[int, List[Tuple[int, ...]]] = defaultdict(list)
    subset_to_Ds: DefaultDict[Tuple[int, ...], List[float]] = defaultdict(list)

    for D in eval_Ds:
        a, b = _intervals_at_D(K, L, U, D)
        depth, _, keep_idx = _max_overlap_subset_at_D(a, b, row_valid, eps=eps)
        if depth <= 0:
            continue
        key = tuple(keep_idx.tolist())
        # Deduplicate per depth
        if not depths_to_subsets[depth] or depths_to_subsets[depth][-1] != key:
            depths_to_subsets[depth].append(key)
        subset_to_Ds[key].append(D)

    return dict(depths_to_subsets), dict(subset_to_Ds)


def _feasible_D_interval_for_subset(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, S: np.ndarray, eps: float
) -> Optional[Tuple[float, float]]:
    """
    Feasible D interval such that ⋂_{i∈S} [D*K_i+L_i, D*K_i+U_i] ≠ ∅ and D>0.
    Reject subsets with <2 distinct strikes.
    """
    Ks, Ls, Us = K[S], L[S], U[S]
    if np.unique(Ks).size < 2:
        return None

    D_lo = 0.0
    D_hi = np.inf
    m = len(S)
    for p in range(m):
        for q in range(m):
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
    eps: float,
) -> np.ndarray:
    """
    Candidate Ds to check the forward band width on a fixed subset S: endpoints of the
    feasible interval plus cross-equalities among a_i/b_i for i∈S within [D_lo, D_hi].
    """
    Ks, Ls, Us = K[S], L[S], U[S]
    cand: List[float] = [D_lo, D_hi]
    m = len(S)
    for p in range(m):
        for q in range(p + 1, m):
            denom = Ks[p] - Ks[q]
            if abs(denom) < eps:
                continue
            D1 = (Ls[q] - Ls[p]) / denom  # a_p = a_q
            D2 = (Us[q] - Us[p]) / denom  # b_p = b_q
            for D in (D1, D2):
                if np.isfinite(D) and (D > 0) and (D_lo - 1e-14 <= D <= D_hi + 1e-14):
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
    """
    For fixed subset S and discount D, return (F_bid, F_ask) s.t.
    F ∈ [F_bid, F_ask] iff there exists X with X ∈ ⋂_{i∈S} [D*K_i+L_i, D*K_i+U_i] and F = X / D.
    """
    Ks, Ls, Us = K[S], L[S], U[S]
    a = D * Ks + Ls
    b = D * Ks + Us
    X_low = float(np.max(a))
    X_high = float(np.min(b))
    return X_low / D, X_high / D


def _choose_subset_with_fallback(
    K: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    depths_to_subsets: Dict[int, List[Tuple[int, ...]]],
    subset_to_Ds: Dict[Tuple[int, ...], List[float]],
    eps: float,
) -> Optional[Tuple[float, float, float, float, float, Tuple[int, ...]]]:
    """
    Iterate depths from high to low (≥2). For each subset at that depth:
      - reject if it has <2 distinct Ks
      - compute feasible D interval; skip if None
      - build width candidate Ds within [D_lo, D_hi]; maximize width = F_ask - F_bid
    Return (D_star, F_bid_star, F_ask_star, D_lo, D_hi, subset_key) or None.
    """
    if not depths_to_subsets:
        return None

    for depth in sorted(depths_to_subsets.keys(), reverse=True):
        if depth < 2:
            continue

        best: Optional[Tuple[float, float, float, float, float, Tuple[int, ...]]] = None
        best_w = -np.inf

        for key in depths_to_subsets[depth]:
            S_sorted = np.array(key, dtype=int)
            # Early reject duplicate-K subsets
            if np.unique(K[S_sorted]).size < 2:
                continue

            D_interval = _feasible_D_interval_for_subset(K, L, U, S_sorted, eps=eps)
            if D_interval is None:
                continue

            D_lo, D_hi = D_interval
            D_cands = _width_candidate_discounts_for_subset(
                K, L, U, S_sorted, D_lo, D_hi, eps=eps
            )
            if D_cands.size == 0:
                continue

            for D in D_cands:
                F_bid, F_ask = _forward_band_for_subset_at_D(K, L, U, S_sorted, D)
                w = F_ask - F_bid
                if (w > best_w + 1e-15) or (
                    abs(w - best_w) <= 1e-15 and (best is None or D < best[0])
                ):
                    best_w = w
                    best = (D, F_bid, F_ask, D_lo, D_hi, key)

        if best is not None:
            return best

    return None


def _narrowest_valid_singleton_index(
    K: np.ndarray, L: np.ndarray, U: np.ndarray, row_valid: np.ndarray, eps: float
) -> Optional[int]:
    """
    Return index (in the *sorted* slice) of the narrowest valid singleton IF there are
    at least two distinct strikes among valid rows. Otherwise return None.
    """
    valid_idx = np.where(row_valid)[0]
    if valid_idx.size == 0:
        return None

    # require ≥ 2 distinct strikes among valid rows; else return None
    if np.unique(K[valid_idx]).size < 2:
        return None

    widths = U[valid_idx] - L[valid_idx]
    arg = np.argmin(widths)  # ties -> first
    return int(valid_idx[arg])
