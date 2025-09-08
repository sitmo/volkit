import numpy as np
import matplotlib.pyplot as plt


def implied_future_from_option_prices_plot(
    K: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    inlier: np.ndarray,
    *,
    F_hat: float,
    F_bid: float,
    F_ask: float,
    D_min: float,
    D_max: float,
    D_display: float,
    ax=None,
) -> None:  # pragma: no cover
    """
    Best-effort plot of per-strike implied forwards.
    - Computes y = C - P internally
    - Uses provided D_display (already chosen by caller)
    - Draws inlier/outlier scatter and horizontal F̂ / [F_bid,F_ask] lines
    - Title shows the D band [D_min, D_max]
    - Fully guarded: never raises
    """
    ax_ = ax if ax is not None else plt.gca()

    # Defensive guard for D_display
    D_plot = float(D_display)
    if not np.isfinite(D_plot) or D_plot <= 0.0:
        D_plot = np.finfo(float).eps

    y = np.asarray(C, float) - np.asarray(P, float)
    K = np.asarray(K, float)
    inlier = np.asarray(inlier, bool)

    F_i = K + y / D_plot

    in_K, in_F = K[inlier], F_i[inlier]
    out_K, out_F = K[~inlier], F_i[~inlier]

    if in_K.size:
        ax_.scatter(in_K, in_F, s=18, label="per-strike F (inliers)", alpha=0.9)
    if out_K.size:
        ax_.scatter(
            out_K, out_F, marker="x", label="per-strike F (outliers)", alpha=0.8
        )

    ax_.axhline(F_bid, linestyle="--", color="black", linewidth=1.1, label="F_bid")
    ax_.axhline(F_ask, linestyle="--", color="black", linewidth=1.1, label="F_ask")
    ax_.axhline(F_hat, linestyle=":", color="gray", linewidth=1.0, label="F̂ (point)")

    ax_.set_xlabel("Strike K")
    ax_.set_ylabel(f"Implied future F @ D_display={D_plot:.6g}")
    ax_.set_title(
        f"Implied forwards per strike — F_bid={F_bid:.6g}, F_ask={F_ask:.6g}; "
        f"D∈[{float(D_min):.6g},{float(D_max):.6g}]"
    )

    # Y padding (robust to empty outlier set)
    y_vals = np.array(
        [
            F_bid,
            F_ask,
            F_hat,
            *(in_F.tolist() if in_K.size else []),
            *(out_F.tolist() if out_K.size else []),
        ]
    )
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size:
        pad = 0.04 * (y_vals.max() - y_vals.min() + 1e-12)
        ax_.set_ylim(y_vals.min() - pad, y_vals.max() + pad)

    ax_.legend(loc="best")
