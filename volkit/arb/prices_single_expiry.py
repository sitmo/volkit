# volkit/arb/detect/detect_arb_prices_single_expiry.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import math
import numpy as np

from .report import ArbReport


# =============================================================================
# Public API
# =============================================================================


def arb_from_option_prices_single_expiry(
    K: Sequence[float],
    C: Optional[Sequence[float]] = None,
    P: Optional[Sequence[float]] = None,
    *,
    F: float,
    D: float,
    # tolerances
    tol_opt: float = 0.0,  # absolute $ tolerance for option inequalities
    tol_opt_rel: float = 0.0,  # relative % of option price
    tol_fut: float = 0.0,  # absolute $ tolerance on the future/forward
    tol_fut_rel: float = 0.0,  # relative % of future
    # enable/disable specific checks
    check_bounds: bool = True,
    check_monotonicity: bool = True,
    check_vertical: bool = True,
    check_convexity: bool = True,
    check_parity: bool = True,
    assume_sorted: bool = True,
    as_report: bool = False,
) -> Dict[str, Any]:
    """
    Detect static no-arbitrage violations for a *single expiration* from mid PRICES,
    and construct explicit *arbitrage trade recipes* (as leg lists) for each violation.
    """
    K, C, P, meta = _prep_inputs(K, C, P, F, D, assume_sorted)
    n = K.size

    # effective tolerances
    fut_pad, opt_tau = _build_tolerances(F, tol_fut, tol_fut_rel, tol_opt, tol_opt_rel)

    # violations & trades accumulators
    violations = _empty_violations_dict()
    trades: List[Dict[str, Any]] = []

    # checks
    if check_bounds:
        _check_bounds(K, C, P, F, D, fut_pad, opt_tau, violations, trades)

    if n >= 2 and (check_monotonicity or check_vertical):
        _check_monotonicity_and_vertical(
            K,
            C,
            P,
            D,
            opt_tau,
            do_mono=check_monotonicity,
            do_vertical=check_vertical,
            violations=violations,
            trades=trades,
        )

    if n >= 3 and check_convexity:
        _check_convexity(K, C, P, opt_tau, violations, trades)

    if check_parity and (C is not None) and (P is not None):
        _check_parity(K, C, P, F, D, fut_pad, opt_tau, violations, trades)

    # wrap up
    n_viol = sum(
        len(v)
        for cat in violations.values()
        for v in cat.values()
        if isinstance(v, list)
    )
    report_dict = {
        "ok": (n_viol == 0),
        "violations": violations,
        "trades": trades,
        "tolerances": {
            "tol_opt": float(tol_opt),
            "tol_opt_rel": float(tol_opt_rel),
            "tol_fut": float(tol_fut),
            "tol_fut_rel": float(tol_fut_rel),
        },
        "enabled_checks": {
            "bounds": bool(check_bounds),
            "monotonicity": bool(check_monotonicity),
            "vertical": bool(check_vertical),
            "convexity": bool(check_convexity),
            "parity": bool(check_parity),
        },
        "meta": meta,
    }
    return ArbReport.from_dict(report_dict) if as_report else report_dict


# =============================================================================
# Input prep & tolerance utilities
# =============================================================================


def _prep_inputs(K, C, P, F, D, assume_sorted):
    K = np.asarray(K, dtype=float)
    if K.ndim != 1:
        raise ValueError("K must be 1-D")
    n = K.size

    # Convert only if provided
    C_arr = None if C is None else np.asarray(C, dtype=float)
    P_arr = None if P is None else np.asarray(P, dtype=float)

    if (C_arr is None) and (P_arr is None):
        raise ValueError("Provide at least one of C or P.")
    if C_arr is not None and C_arr.shape != (n,):
        raise ValueError("C must have same shape as K")
    if P_arr is not None and P_arr.shape != (n,):
        raise ValueError("P must have same shape as K")

    if not assume_sorted:
        perm = np.argsort(K)
        K = K[perm]
        if C_arr is not None:
            C_arr = C_arr[perm]
        if P_arr is not None:
            P_arr = P_arr[perm]
        sorted_flag = True
    else:
        perm = None
        sorted_flag = False

    meta = {"sorted": sorted_flag, "perm": perm, "F": float(F), "D": float(D)}
    return K, C_arr, P_arr, meta


def _build_tolerances(
    F: float,
    tol_fut: float,
    tol_fut_rel: float,
    tol_opt: float,
    tol_opt_rel: float,
):
    """
    Return:
      fut_pad: absolute $ padding in price space contributed by F (multiply by D where needed)
      opt_tau: function(*prices) -> combined absolute/relative option tolerance
    """
    fut_pad = tol_fut + tol_fut_rel * abs(F)

    def opt_tau(*prices: float) -> float:
        base = float(tol_opt)
        if tol_opt_rel > 0:
            mx = 1.0
            for x in prices:
                if x is not None:
                    mx = max(mx, abs(float(x)))
            base += float(tol_opt_rel) * mx
        return float(base)

    return float(fut_pad), opt_tau


def _empty_violations_dict() -> Dict[str, Dict[str, list]]:
    return {
        "bounds": {"call_lb": [], "call_ub": [], "put_lb": [], "put_ub": []},
        "monotonicity": {"call_bad_pairs": [], "put_bad_pairs": []},
        "vertical": {
            "call_lb_pairs": [],
            "call_ub_pairs": [],
            "put_lb_pairs": [],
            "put_ub_pairs": [],
        },
        "convexity": {"call_bad_triples": [], "put_bad_triples": []},
        "parity": {"bad_idx": [], "max_abs_err": np.nan},
    }


# =============================================================================
# Leg builders & net cost
# =============================================================================


def _f(x) -> float:
    return float(x)


def _leg_option(
    asset: str, strike: float, side: str, qty: float, price: float
) -> Dict[str, Any]:
    return {
        "asset": asset,
        "side": side,
        "qty": _f(qty),
        "strike": _f(strike),
        "price": _f(price),
        "notional": None,
    }


def _leg_bond(D: float, side: str, notional: float) -> Dict[str, Any]:
    return {
        "asset": "bond",
        "side": side,
        "qty": 1.0,
        "strike": None,
        "price": _f(D * notional),
        "notional": _f(notional),
    }


def _net_cost_from_legs(legs: List[Dict[str, Any]]) -> float:
    s = 0.0
    for L in legs:
        sign = 1.0 if L["side"] == "sell" else -1.0
        s += sign * _f(L["qty"]) * _f(L["price"])
    return _f(s)


# =============================================================================
# Checks
# =============================================================================


def _check_bounds(
    K: np.ndarray,
    C: Optional[np.ndarray],
    P: Optional[np.ndarray],
    F: float,
    D: float,
    fut_pad: float,
    opt_tau,
    violations: Dict[str, Dict[str, list]],
    trades: List[Dict[str, Any]],
) -> None:
    """Per-strike upper/lower bounds for calls & puts."""
    n = K.size

    if C is not None:
        for i in range(n):
            if not (math.isfinite(K[i]) and math.isfinite(C[i])):  # skip NaNs
                continue

            # Call lower bound: max(0, D*(F-K))
            lb = max(0.0, F - D * K[i])
            tau = opt_tau(C[i]) + D * fut_pad
            if C[i] < lb - tau:
                legs = [_leg_option("call", K[i], "buy", 1.0, C[i])]
                note = "Call priced below its intrinsic value (too cheap)."
                if F > K[i]:
                    legs.append(_leg_bond(D, "sell", F - K[i]))
                    note = "Call priced below intrinsic value; too cheap vs forward."
                violations["bounds"]["call_lb"].append(i)
                trades.append(
                    {
                        "type": "bounds",
                        "notes": note,
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

            # Call upper bound: C <= D*F
            ub = D * F
            tau = opt_tau(C[i]) + D * fut_pad
            if C[i] > ub + tau:
                legs = [
                    _leg_option("call", K[i], "sell", 1.0, C[i]),
                    _leg_bond(D, "buy", F),
                ]
                violations["bounds"]["call_ub"].append(i)
                trades.append(
                    {
                        "type": "bounds",
                        "notes": "Call costs more than the forward (option overpriced).",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

    if P is not None:
        for i in range(n):
            if not (math.isfinite(K[i]) and math.isfinite(P[i])):
                continue

            # Put lower bound: max(0, D*(K-F))
            lb = max(0.0, D * K[i] - F)
            tau = opt_tau(P[i]) + D * fut_pad
            if P[i] < lb - tau:
                legs = [_leg_option("put", K[i], "buy", 1.0, P[i])]
                note = "Put priced below its intrinsic value (too cheap)."
                if K[i] > F:
                    legs.append(_leg_bond(D, "sell", K[i] - F))
                    note = "Put priced below intrinsic value; too cheap vs forward."
                violations["bounds"]["put_lb"].append(i)
                trades.append(
                    {
                        "type": "bounds",
                        "notes": note,
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

            # Put upper bound: P <= D*K
            ub = D * K[i]
            tau = opt_tau(P[i])
            if P[i] > ub + tau:
                legs = [
                    _leg_option("put", K[i], "sell", 1.0, P[i]),
                    _leg_bond(D, "buy", K[i]),
                ]
                violations["bounds"]["put_ub"].append(i)
                trades.append(
                    {
                        "type": "bounds",
                        "notes": "Put costs more than the discounted strike (overpriced).",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )


def _check_monotonicity_and_vertical(
    K: np.ndarray,
    C: Optional[np.ndarray],
    P: Optional[np.ndarray],
    D: float,
    opt_tau,
    *,
    do_mono: bool,
    do_vertical: bool,
    violations: Dict[str, Dict[str, list]],
    trades: List[Dict[str, Any]],
) -> None:
    """Adjacent-strike monotonicity & vertical-spread cap checks."""
    dK = np.diff(K)
    idx = np.where(np.isfinite(dK) & (dK > 0))[0]
    for i in idx:
        j = i + 1
        cap = D * dK[i]

        # Calls
        if C is not None and math.isfinite(C[i]) and math.isfinite(C[j]):
            spread = C[i] - C[j]
            tau_pair = opt_tau(C[i], C[j])

            if do_mono and spread < -tau_pair:
                legs = [
                    _leg_option("call", K[i], "buy", 1.0, C[i]),
                    _leg_option("call", K[j], "sell", 1.0, C[j]),
                ]
                violations["monotonicity"]["call_bad_pairs"].append((i, j))
                trades.append(
                    {
                        "type": "monotonicity",
                        "notes": "Higher-strike call priced above a lower-strike call (should be cheaper).",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

            if do_vertical and spread > cap + tau_pair:
                legs = [
                    _leg_option("call", K[i], "sell", 1.0, C[i]),
                    _leg_option("call", K[j], "buy", 1.0, C[j]),
                    _leg_bond(D, "buy", dK[i]),
                ]
                violations["vertical"]["call_ub_pairs"].append((i, j))
                trades.append(
                    {
                        "type": "vertical",
                        "notes": "Call spread priced above its maximum possible payoff.",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

        # Puts
        if P is not None and math.isfinite(P[i]) and math.isfinite(P[j]):
            spread = P[j] - P[i]
            tau_pair = opt_tau(P[i], P[j])

            if do_mono and spread < -tau_pair:
                legs = [
                    _leg_option("put", K[j], "buy", 1.0, P[j]),
                    _leg_option("put", K[i], "sell", 1.0, P[i]),
                ]
                violations["monotonicity"]["put_bad_pairs"].append((i, j))
                trades.append(
                    {
                        "type": "monotonicity",
                        "notes": "Higher-strike put priced below a lower-strike put (should be pricier).",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

            if do_vertical and spread > cap + tau_pair:
                legs = [
                    _leg_option("put", K[j], "sell", 1.0, P[j]),
                    _leg_option("put", K[i], "buy", 1.0, P[i]),
                    _leg_bond(D, "buy", dK[i]),
                ]
                violations["vertical"]["put_ub_pairs"].append((i, j))
                trades.append(
                    {
                        "type": "vertical",
                        "notes": "Put spread priced above its maximum possible payoff.",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )


def _check_convexity(
    K: np.ndarray,
    C: Optional[np.ndarray],
    P: Optional[np.ndarray],
    opt_tau,
    violations: Dict[str, Dict[str, list]],
    trades: List[Dict[str, Any]],
) -> None:
    """Adjacent-triple convexity (butterfly) checks for calls & puts."""
    n = K.size
    for i in range(n - 2):
        j, k = i + 1, i + 2
        Ki, Kj, Kk = K[i], K[j], K[k]
        if (
            not (math.isfinite(Ki) and math.isfinite(Kj) and math.isfinite(Kk))
            or Kk <= Ki
        ):
            continue
        lam = (Kk - Kj) / (Kk - Ki)

        if (
            C is not None
            and math.isfinite(C[i])
            and math.isfinite(C[j])
            and math.isfinite(C[k])
        ):
            lhs = C[j]
            rhs = lam * C[i] + (1 - lam) * C[k]
            tau = opt_tau(C[i], C[j], C[k])
            if lhs > rhs + tau:
                legs = [
                    _leg_option("call", Ki, "buy", float(lam), C[i]),
                    _leg_option("call", Kk, "buy", float(1 - lam), C[k]),
                    _leg_option("call", Kj, "sell", 1.0, C[j]),
                ]
                violations["convexity"]["call_bad_triples"].append((i, j, k))
                trades.append(
                    {
                        "type": "convexity",
                        "notes": "Middle-strike call overpriced relative to nearby strikes (curve not convex).",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )

        if (
            P is not None
            and math.isfinite(P[i])
            and math.isfinite(P[j])
            and math.isfinite(P[k])
        ):
            lhs = P[j]
            rhs = lam * P[i] + (1 - lam) * P[k]
            tau = opt_tau(P[i], P[j], P[k])
            if lhs > rhs + tau:
                legs = [
                    _leg_option("put", Ki, "buy", float(lam), P[i]),
                    _leg_option("put", Kk, "buy", float(1 - lam), P[k]),
                    _leg_option("put", Kj, "sell", 1.0, P[j]),
                ]
                violations["convexity"]["put_bad_triples"].append((i, j, k))
                trades.append(
                    {
                        "type": "convexity",
                        "notes": "Middle-strike put overpriced relative to nearby strikes (curve not convex).",
                        "legs": legs,
                        "net_cost": _net_cost_from_legs(legs),
                    }
                )


def _check_parity(
    K: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    F: float,
    D: float,
    fut_pad: float,
    opt_tau,
    violations: Dict[str, Dict[str, list]],
    trades: List[Dict[str, Any]],
) -> None:
    """Put–call parity per strike."""
    residuals: List[float] = []
    n = K.size
    for i in range(n):
        if not (math.isfinite(K[i]) and math.isfinite(C[i]) and math.isfinite(P[i])):
            continue
        lhs = C[i] - P[i]
        rhs = D * (F - K[i])
        tau = opt_tau(C[i], P[i]) + D * fut_pad
        m = lhs - rhs
        residuals.append(abs(m))

        if m > tau:  # call-rich
            legs = [
                _leg_option("call", K[i], "sell", 1.0, C[i]),
                _leg_option("put", K[i], "buy", 1.0, P[i]),
                _leg_bond(D, "buy", F - K[i]),
            ]
            violations["parity"]["bad_idx"].append(i)
            trades.append(
                {
                    "type": "parity",
                    "notes": "Call too expensive relative to same-strike put (put–call parity broken).",
                    "legs": legs,
                    "net_cost": _net_cost_from_legs(legs),
                }
            )
        elif m < -tau:  # put-rich
            legs = [
                _leg_option("call", K[i], "buy", 1.0, C[i]),
                _leg_option("put", K[i], "sell", 1.0, P[i]),
                _leg_bond(D, "sell", F - K[i]),
            ]
            violations["parity"]["bad_idx"].append(i)
            trades.append(
                {
                    "type": "parity",
                    "notes": "Put too expensive relative to same-strike call (put–call parity broken).",
                    "legs": legs,
                    "net_cost": _net_cost_from_legs(legs),
                }
            )

    if residuals:
        violations["parity"]["max_abs_err"] = float(np.nanmax(residuals))
