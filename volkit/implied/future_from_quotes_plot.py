# volkit/implied/future_plot.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from volkit.implied.future_res import ImpliedFutureResult


def _per_strike_forward_quotes(
    K, C_bid, C_ask, P_bid, P_ask, D_min, D_max
):  # pragma: no cover
    """Per-strike feasible forward *quotes* using the shared D-band.

    For each strike, combine option quote bands with the discount band and
    project via put–call parity: F = K + y / D where y ∈ [Cb-Pa, Ca-Pb] and
    D ∈ [D_min, D_max].

    Returns
    -------
    (F_lo, F_hi) : arrays aligned with K.
    """
    K = np.asarray(K, float)
    y_lo = np.asarray(C_bid, float) - np.asarray(P_ask, float)
    y_hi = np.asarray(C_ask, float) - np.asarray(P_bid, float)

    # Evaluate all 4 corners of [y_lo,y_hi] × [D_min,D_max]
    c1 = y_lo / D_min
    c2 = y_lo / D_max
    c3 = y_hi / D_min
    c4 = y_hi / D_max

    F_lo = K + np.minimum.reduce([c1, c2, c3, c4])
    F_hi = K + np.maximum.reduce([c1, c2, c3, c4])
    return F_lo, F_hi


def implied_future_from_option_quotes_plot(
    K: np.ndarray,
    Cb: np.ndarray,
    Ca: np.ndarray,
    Pb: np.ndarray,
    Pa: np.ndarray,
    res: "ImpliedFutureResult | None",
    mask: np.ndarray,
    ax=None,
    *,
    eps: float = 1e-12,
    uniform_x: bool = True,
    max_xticks: int = 25,
    label_rotation: int = 45,
):  # pragma: no cover
    """Plot per-strike implied forward *quotes* and the global forward band.

    - Per-strike quotes are computed with the *same* D-band (res.D_min/res.D_max),
      ensuring each strike's interval contains the global intersection band.
    - Bars are green for inliers (mask=True), red otherwise.
    - Horizontal dashed lines mark (F_bid, F_ask).
    - X-axis can be uniform spacing with strike labels, or actual strikes.

    Returns matplotlib Figure if it created one; otherwise returns the Axes.
    """
    # Handle infeasible result early
    if res is None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5.2))
            created_fig = True
        else:
            fig, created_fig = None, False
        ax.text(
            0.5,
            0.5,
            "No feasible result (res=None).",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Implied forward F")
        ax.set_title("Implied forward quotes — infeasible")
        return fig if created_fig else ax

    # Validate / coerce inputs
    K = np.asarray(K, float)
    Cb = np.asarray(Cb, float)
    Ca = np.asarray(Ca, float)
    Pb = np.asarray(Pb, float)
    Pa = np.asarray(Pa, float)
    mask = np.asarray(mask, bool)

    # Compute per-strike forward quotes using the shared D-band
    Fi_lo, Fi_hi = _per_strike_forward_quotes(K, Cb, Ca, Pb, Pa, res.D_min, res.D_max)

    finite = (
        np.isfinite(K)
        & np.isfinite(Fi_lo)
        & np.isfinite(Fi_hi)
        & (Fi_lo <= Fi_hi + 1e-12)
    )

    if not np.any(finite):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5.2))
            created_fig = True
        else:
            fig, created_fig = None, False
        ax.text(
            0.5,
            0.5,
            "No finite per-strike forward quotes.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Implied forward F")
        ax.set_title("Implied forward quotes — empty")
        return fig if created_fig else ax

    # Sort by strike among valid points only
    order = np.argsort(K[finite])
    Kp = K[finite][order]
    lo_p = Fi_lo[finite][order]
    hi_p = Fi_hi[finite][order]
    mask_p = mask[finite][order]

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.2))
        created_fig = True

    # Choose x positions
    if uniform_x:
        xpos = np.arange(len(Kp), dtype=float)
        # Ticks: at most max_xticks evenly spread
        if len(xpos) <= max_xticks:
            tick_idx = np.arange(len(xpos))
        else:
            tick_idx = np.unique(np.linspace(0, len(xpos) - 1, max_xticks, dtype=int))
        ax.set_xticks(xpos[tick_idx])
        ax.set_xticklabels(
            [f"{v:g}" for v in Kp[tick_idx]], rotation=label_rotation, ha="right"
        )
        ax.set_xlim(-0.5, len(xpos) - 0.5)
        ax.set_xlabel("Strike K (labels; bars uniformly spaced)")
    else:
        xpos = Kp
        ax.set_xticks(Kp)
        ax.set_xticklabels([f"{v:g}" for v in Kp], rotation=label_rotation, ha="right")
        ax.set_xlabel("Strike K")

    # Draw per-strike bars
    for xi, y0, y1, keep in zip(xpos, lo_p, hi_p, mask_p):
        ax.vlines(
            xi,
            y0,
            y1,
            color=("green" if keep else "red"),
            linewidth=(3.0 if keep else 1.6),
            alpha=(1.0 if keep else 0.65),
        )

    # Global band
    ax.axhline(res.F_bid, linestyle="--", color="black", linewidth=1.1)
    ax.axhline(res.F_ask, linestyle="--", color="black", linewidth=1.1)

    ax.set_ylabel("Implied forward F (per-strike quotes, D-band)")
    ax.set_title(
        f"Global: F_bid={res.F_bid:.6g}, F_ask={res.F_ask:.6g};  D_bid={res.D_min:.6g}, D_ask={res.D_max:.6g}"
    )

    # Y limits with small padding
    y_all = np.concatenate([lo_p, hi_p, [res.F_bid, res.F_ask]])
    ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad = 0.04 * (ymax - ymin + 1e-12)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Legend proxies
    n_keep = int(np.count_nonzero(mask_p))
    n_drop = int(len(mask_p) - n_keep)
    handles = [
        Line2D([], [], color="green", linewidth=3.0, label=f"Kept ({n_keep})"),
        Line2D(
            [], [], color="red", linewidth=1.6, alpha=0.65, label=f"Dropped ({n_drop})"
        ),
        Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            linewidth=1.1,
            label="Global F_bid / F_ask",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)

    return fig if created_fig else ax
