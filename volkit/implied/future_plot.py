# volkit/implied/future_plot.py

import numpy as np
from volkit.implied.future_res import ImpliedFutureResult


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
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter

    if res is None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
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
            ax.set_title("Implied forward intervals — infeasible")
            return fig
        else:
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
            ax.set_title("Implied forward intervals — infeasible")
            return ax

    K = np.asarray(K, float)
    Cb = np.asarray(Cb, float)
    Ca = np.asarray(Ca, float)
    Pb = np.asarray(Pb, float)
    Pa = np.asarray(Pa, float)
    mask = np.asarray(mask, bool)

    D_star = 0.5 * (res.D_min + res.D_max)
    L = Cb - Pa
    U = Ca - Pb

    finite = (
        np.isfinite(K)
        & np.isfinite(L)
        & np.isfinite(U)
        & np.isfinite(D_star)
        & (D_star > eps)
    )
    Fi_lo = np.empty_like(K)
    Fi_hi = np.empty_like(K)
    Fi_lo[finite] = (D_star * K[finite] + L[finite]) / D_star
    Fi_hi[finite] = (D_star * K[finite] + U[finite]) / D_star
    ok = finite & np.isfinite(Fi_lo) & np.isfinite(Fi_hi) & (Fi_lo <= Fi_hi + 1e-12)
    if not ok.any():
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
            ax.text(
                0.5,
                0.5,
                "No finite forward intervals at D*.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Implied forward F")
            ax.set_title("Implied forward intervals — empty at D*")
            return fig
        else:
            ax.text(
                0.5,
                0.5,
                "No finite forward intervals at D*.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Implied forward F")
            ax.set_title("Implied forward intervals — empty at D*")
            return ax

    # ✅ fixed here
    order = np.argsort(K[ok])
    Kp = K[ok][order]
    lo_p = Fi_lo[ok][order]
    hi_p = Fi_hi[ok][order]
    mask_p = mask[ok][order]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        created_fig = True

    if uniform_x:
        xpos = np.arange(len(Kp), dtype=float)
        if len(xpos) <= max_xticks:
            tick_idx = np.arange(len(xpos))
        else:
            tick_idx = np.unique(np.linspace(0, len(xpos) - 1, max_xticks, dtype=int))
        tick_locs = xpos[tick_idx]
        tick_labels = [f"{val:g}" for val in Kp[tick_idx]]
        ax.xaxis.set_major_locator(FixedLocator(tick_locs))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax.set_xlim(-0.5, len(xpos) - 0.5)
        ax.set_xlabel("Strike K (labels; bars uniformly spaced)")
    else:
        xpos = Kp
        ax.set_xlabel("Strike K")

    for xi, y0, y1, keep in zip(xpos, lo_p, hi_p, mask_p):
        ax.vlines(
            xi,
            y0,
            y1,
            color=("green" if keep else "red"),
            linewidth=(3.0 if keep else 1.5),
            alpha=(1.0 if keep else 0.6),
        )

    ax.axhline(res.F_bid, linestyle="--", color="black", linewidth=1.2)
    ax.axhline(res.F_ask, linestyle="--", color="black", linewidth=1.2)

    ax.set_ylabel("Implied forward F at D* (mid of D-band)")
    ax.set_title(
        f"F_bid={res.F_bid:.6g}, F_ask={res.F_ask:.6g}, D_bid={res.D_min:.6g}, D_ask={res.D_max:.6g}"
    )

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(label_rotation)
        lbl.set_horizontalalignment("right")

    y_all = np.concatenate([lo_p, hi_p, [res.F_bid, res.F_ask]])
    ypad = 0.04 * (np.nanmax(y_all) - np.nanmin(y_all) + 1e-12)
    ax.set_ylim(np.nanmin(y_all) - ypad, np.nanmax(y_all) + ypad)

    # Count bars actually drawn (i.e., among `ok`)
    n_keep = int(np.count_nonzero(mask_p))
    n_drop = int(len(mask_p) - n_keep)

    # Legend proxies (so we don't have to capture actual vlines)
    import matplotlib.lines as mlines

    green_proxy = mlines.Line2D(
        [], [], color="green", linewidth=3.0, label=f"Kept ({n_keep})"
    )
    red_proxy = mlines.Line2D(
        [], [], color="red", linewidth=1.5, alpha=0.6, label=f"Dropped ({n_drop})"
    )
    band_proxy = mlines.Line2D(
        [], [], color="black", linestyle="--", linewidth=1.2, label="F_bid*, F_ask*"
    )

    ax.legend(handles=[green_proxy, red_proxy, band_proxy], loc="best", frameon=False)

    return fig if created_fig else ax
