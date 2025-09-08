# volkit/arb/detect/detect_arb_prices_single_expiry.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import math
import numpy as np

from .report import ArbReport


def arb_from_option_prices_single_expiry(
    K: Sequence[float],
    C: Optional[Sequence[float]] = None,
    P: Optional[Sequence[float]] = None,
    *,
    F: float,
    D: float,
    # tolerances
    tol_opt: float = 0.0,       # absolute $ tolerance for option inequalities
    tol_opt_rel: float = 0.0,   # relative % of option price
    tol_fut: float = 0.0,       # absolute $ tolerance on the future/forward
    tol_fut_rel: float = 0.0,   # relative % of future
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

    # ---------- helpers ----------
    def f(x) -> float:
        return float(x)

    def leg_option(asset: str, strike: float, side: str, qty: float, price: float) -> Dict[str, Any]:
        return {
            "asset": asset, "side": side, "qty": f(qty),
            "strike": f(strike), "price": f(price), "notional": None,
        }

    def leg_bond(side: str, notional: float) -> Dict[str, Any]:
        return {
            "asset": "bond", "side": side, "qty": 1.0,
            "strike": None, "price": f(D * notional), "notional": f(notional),
        }

    def net_cost_from_legs(legs: List[Dict[str, Any]]) -> float:
        s = 0.0
        for L in legs:
            sign = 1.0 if L["side"] == "sell" else -1.0
            s += sign * f(L["qty"]) * f(L["price"])
        return f(s)

    # ---------- prep ----------
    K = np.asarray(K, dtype=float)
    n = K.size
    if C is not None:
        C = np.asarray(C, dtype=float)
        if C.shape != K.shape:
            raise ValueError("C must have same shape as K")
    if P is not None:
        P = np.asarray(P, dtype=float)
        if P.shape != K.shape:
            raise ValueError("P must have same shape as K")

    if not assume_sorted:
        perm = np.argsort(K)
        K = K[perm]
        if C is not None:
            C = C[perm]
        if P is not None:
            P = P[perm]
        sorted_flag = True
    else:
        perm = None
        sorted_flag = False

    has_C = C is not None
    has_P = P is not None
    if not (has_C or has_P):
        raise ValueError("Provide at least one of C or P.")

    fut_pad = tol_fut + tol_fut_rel * abs(F)

    def opt_tau(*prices: float) -> float:
        base = tol_opt
        if tol_opt_rel > 0:
            mx = 1.0
            for x in prices:
                if x is not None:
                    mx = max(mx, abs(float(x)))
            base += tol_opt_rel * mx
        return base

    # ---------- checks ----------
    violations: Dict[str, Dict[str, list]] = {
        "bounds": {"call_lb": [], "call_ub": [], "put_lb": [], "put_ub": []},
        "monotonicity": {"call_bad_pairs": [], "put_bad_pairs": []},
        "vertical": {"call_lb_pairs": [], "call_ub_pairs": [], "put_lb_pairs": [], "put_ub_pairs": []},
        "convexity": {"call_bad_triples": [], "put_bad_triples": []},
        "parity": {"bad_idx": [], "max_abs_err": np.nan},
    }
    trades: List[Dict[str, Any]] = []

    # --- BOUNDS ---
    if check_bounds:
        if has_C:
            for i in range(n):
                if not (math.isfinite(K[i]) and math.isfinite(C[i])):
                    continue
                lb = D * max(0.0, F - K[i])
                tau = opt_tau(C[i]) + D * fut_pad
                if C[i] < lb - tau:
                    legs = [leg_option("call", K[i], "buy", 1.0, C[i])]
                    note = "Call priced below its intrinsic value (too cheap)."
                    if F > K[i]:
                        legs.append(leg_bond("sell", F - K[i]))
                        note = "Call priced below intrinsic value; too cheap vs forward."
                    violations["bounds"]["call_lb"].append(i)
                    trades.append({
                        "type": "bounds", "notes": note,
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

                ub = D * F
                tau = opt_tau(C[i]) + D * fut_pad
                if C[i] > ub + tau:
                    legs = [
                        leg_option("call", K[i], "sell", 1.0, C[i]),
                        leg_bond("buy", F),
                    ]
                    violations["bounds"]["call_ub"].append(i)
                    trades.append({
                        "type": "bounds",
                        "notes": "Call costs more than the forward (option overpriced).",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

        if has_P:
            for i in range(n):
                if not (math.isfinite(K[i]) and math.isfinite(P[i])):
                    continue
                lb = D * max(0.0, K[i] - F)
                tau = opt_tau(P[i]) + D * fut_pad
                if P[i] < lb - tau:
                    legs = [leg_option("put", K[i], "buy", 1.0, P[i])]
                    note = "Put priced below its intrinsic value (too cheap)."
                    if K[i] > F:
                        legs.append(leg_bond("sell", K[i] - F))
                        note = "Put priced below intrinsic value; too cheap vs forward."
                    violations["bounds"]["put_lb"].append(i)
                    trades.append({
                        "type": "bounds", "notes": note,
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

                ub = D * K[i]
                tau = opt_tau(P[i])
                if P[i] > ub + tau:
                    legs = [
                        leg_option("put", K[i], "sell", 1.0, P[i]),
                        leg_bond("buy", K[i]),
                    ]
                    violations["bounds"]["put_ub"].append(i)
                    trades.append({
                        "type": "bounds",
                        "notes": "Put costs more than the discounted strike (overpriced).",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

    # --- MONOTONICITY & VERTICAL ---
    if K.size >= 2:
        dK = np.diff(K)
        idx = np.where(np.isfinite(dK) & (dK > 0))[0]
        for i in idx:
            j = i + 1
            cap = D * dK[i]

            if has_C and math.isfinite(C[i]) and math.isfinite(C[j]):
                spread = C[i] - C[j]
                tau_pair = opt_tau(C[i], C[j])

                if check_monotonicity and spread < -tau_pair:
                    legs = [
                        leg_option("call", K[i], "buy", 1.0, C[i]),
                        leg_option("call", K[j], "sell", 1.0, C[j]),
                    ]
                    violations["monotonicity"]["call_bad_pairs"].append((i, j))
                    trades.append({
                        "type": "monotonicity",
                        "notes": "Higher-strike call priced above a lower-strike call (should be cheaper).",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

                if check_vertical and spread > cap + tau_pair:
                    legs = [
                        leg_option("call", K[i], "sell", 1.0, C[i]),
                        leg_option("call", K[j], "buy", 1.0, C[j]),
                        leg_bond("buy", dK[i]),
                    ]
                    violations["vertical"]["call_ub_pairs"].append((i, j))
                    trades.append({
                        "type": "vertical",
                        "notes": "Call spread priced above its maximum possible payoff.",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

            if has_P and math.isfinite(P[i]) and math.isfinite(P[j]):
                spread = P[j] - P[i]
                tau_pair = opt_tau(P[i], P[j])

                if check_monotonicity and spread < -tau_pair:
                    legs = [
                        leg_option("put", K[j], "buy", 1.0, P[j]),
                        leg_option("put", K[i], "sell", 1.0, P[i]),
                    ]
                    violations["monotonicity"]["put_bad_pairs"].append((i, j))
                    trades.append({
                        "type": "monotonicity",
                        "notes": "Higher-strike put priced below a lower-strike put (should be pricier).",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

                if check_vertical and spread > cap + tau_pair:
                    legs = [
                        leg_option("put", K[j], "sell", 1.0, P[j]),
                        leg_option("put", K[i], "buy", 1.0, P[i]),
                        leg_bond("buy", dK[i]),
                    ]
                    violations["vertical"]["put_ub_pairs"].append((i, j))
                    trades.append({
                        "type": "vertical",
                        "notes": "Put spread priced above its maximum possible payoff.",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

    # --- CONVEXITY ---
    if check_convexity and K.size >= 3:
        for i in range(n - 2):
            j, k = i + 1, i + 2
            Ki, Kj, Kk = K[i], K[j], K[k]
            if not (math.isfinite(Ki) and math.isfinite(Kj) and math.isfinite(Kk)) or Kk <= Ki:
                continue
            lam = (Kk - Kj) / (Kk - Ki)

            if has_C and math.isfinite(C[i]) and math.isfinite(C[j]) and math.isfinite(C[k]):
                lhs = C[j]
                rhs = lam * C[i] + (1 - lam) * C[k]
                tau = opt_tau(C[i], C[j], C[k])
                if lhs > rhs + tau:
                    legs = [
                        leg_option("call", Ki, "buy", float(lam), C[i]),
                        leg_option("call", Kk, "buy", float(1 - lam), C[k]),
                        leg_option("call", Kj, "sell", 1.0, C[j]),
                    ]
                    violations["convexity"]["call_bad_triples"].append((i, j, k))
                    trades.append({
                        "type": "convexity",
                        "notes": "Middle-strike call overpriced relative to nearby strikes (curve not convex).",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

            if has_P and math.isfinite(P[i]) and math.isfinite(P[j]) and math.isfinite(P[k]):
                lhs = P[j]
                rhs = lam * P[i] + (1 - lam) * P[k]
                tau = opt_tau(P[i], P[j], P[k])
                if lhs > rhs + tau:
                    legs = [
                        leg_option("put", Ki, "buy", float(lam), P[i]),
                        leg_option("put", Kk, "buy", float(1 - lam), P[k]),
                        leg_option("put", Kj, "sell", 1.0, P[j]),
                    ]
                    violations["convexity"]["put_bad_triples"].append((i, j, k))
                    trades.append({
                        "type": "convexity",
                        "notes": "Middle-strike put overpriced relative to nearby strikes (curve not convex).",
                        "legs": legs, "net_cost": net_cost_from_legs(legs),
                    })

    # --- PARITY ---
    if check_parity and has_C and has_P:
        residuals = []
        for i in range(n):
            if not (math.isfinite(K[i]) and math.isfinite(C[i]) and math.isfinite(P[i])):
                continue
            lhs = C[i] - P[i]
            rhs = D * (F - K[i])
            tau = opt_tau(C[i], P[i]) + D * fut_pad
            m = lhs - rhs
            residuals.append(abs(m))
            if m > tau:   # call-rich
                legs = [
                    leg_option("call", K[i], "sell", 1.0, C[i]),
                    leg_option("put",  K[i], "buy",  1.0, P[i]),
                    leg_bond("buy", F - K[i]),
                ]
                violations["parity"]["bad_idx"].append(i)
                trades.append({
                    "type": "parity",
                    "notes": "Call too expensive relative to same-strike put (put–call parity broken).",
                    "legs": legs, "net_cost": net_cost_from_legs(legs),
                })
            elif m < -tau:  # put-rich
                legs = [
                    leg_option("call", K[i], "buy",  1.0, C[i]),
                    leg_option("put",  K[i], "sell", 1.0, P[i]),
                    leg_bond("sell", F - K[i]),
                ]
                violations["parity"]["bad_idx"].append(i)
                trades.append({
                    "type": "parity",
                    "notes": "Put too expensive relative to same-strike call (put–call parity broken).",
                    "legs": legs, "net_cost": net_cost_from_legs(legs),
                })

        if residuals:
            violations["parity"]["max_abs_err"] = float(np.nanmax(residuals))

    # --- Wrap up ---
    n_viol = sum(len(v) for cat in violations.values() for v in cat.values() if isinstance(v, list))
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
        "meta": {
            "sorted": sorted_flag,
            "perm": perm,
            "F": float(F),
            "D": float(D),
        },
    }

    return ArbReport.from_dict(report_dict) if as_report else report_dict
