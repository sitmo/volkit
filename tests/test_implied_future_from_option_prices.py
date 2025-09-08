# file: tests/test_estimate_future_from_option_prices.py
import math
import numpy as np
import pytest
import matplotlib.pyplot as plt

from volkit.estimate.future_from_option_prices import (
    estimate_future_from_option_prices,
    ImpliedFutureResult,
    _constrained_ols_line,
    _mad,
    _mad_threshold,
    _feasible_D_band_from_pairwise_slopes,
    _future_band_from_D_band,
    _count_distinct,
    _canon_D_band,
)



@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


def _make_synthetic_prices(K, D, F, noise=None):
    """Build consistent (C, P) such that C - P = D(F - K)."""
    K = np.asarray(K)
    y = D * (F - K)
    # Baseline put, always positive
    P = 0.25 + 0.01 * (K - K.mean()) ** 2
    C = P + y
    if noise is not None:
        C = C + noise
    return C, P


def _make_prices(K, D, F):
    K = np.asarray(K, float)
    y = D * (F - K)
    P = 0.25 + 0.01 * (K - K.mean()) ** 2
    C = P + y
    return C, P


def test_perfect_parity_basic(rng):
    n = 31
    K = np.linspace(80, 120, n)
    D_true = math.exp(-0.02 * 0.5)
    F_true = 101.5
    C, P = _make_synthetic_prices(K, D_true, F_true)

    res, mask = estimate_future_from_option_prices(K, C, P)
    assert isinstance(res, ImpliedFutureResult)
    assert mask.shape == (n,) and mask.all()
    assert abs(res.D - D_true) < 1e-10
    assert abs(res.F - F_true) < 1e-10
    # Bands contain point (robust against tiny float noise)
    assert res.D_min - 1e-15 <= res.D <= res.D_max + 1e-15
    assert res.F_bid - 1e-15 <= res.F <= res.F_ask + 1e-15


def test_outliers_are_removed_and_refit_is_stable(rng):
    n = 60
    K = np.linspace(50, 150, n)
    D_true = 0.93
    F_true = 103.0
    C, P = _make_synthetic_prices(K, D_true, F_true)

    out_idx = rng.choice(n, size=6, replace=False)
    C_noisy = C.copy()
    C_noisy[out_idx[:3]] += 8.0
    C_noisy[out_idx[3:]] -= 5.0

    res, mask = estimate_future_from_option_prices(
        K, C_noisy, P, tol_sigma=3.0, rel_tol=1e-3
    )

    # Outliers should be excluded
    assert mask.sum() <= n - 4

    # Estimates remain close to truth
    assert abs(res.D - D_true) < 5e-3
    assert abs(res.F - F_true) < 0.5


def test_insufficient_distinct_strikes_returns_none(rng):
    n = 10
    K = np.full(n, 100.0)
    D_true = 0.9
    F_true = 100.0
    C, P = _make_synthetic_prices(K, D_true, F_true)

    res, mask = estimate_future_from_option_prices(K, C, P)
    assert res is None
    assert mask.shape == (n,)
    assert not mask.any()


def test_clamping_of_D_bounds(rng):
    # Force very steep negative slope so raw D>1 => clamp
    K = np.array([90.0, 110.0, 130.0, 150.0])
    y = np.array([30.0, 0.0, -30.0, -60.0])
    P = 5.0 + 0.02 * np.maximum(0.0, K - 120)
    C = P + y

    res, mask = estimate_future_from_option_prices(K, C, P)

    assert isinstance(res, ImpliedFutureResult)
    assert 0.0 < res.D <= 1.0
    assert res.D_max <= 1.0 and res.D_min >= 0.0
    assert res.D_min <= res.D <= res.D_max


def test_positive_slope_case_clamps_to_eps(rng):
    # y increases with K -> slope > 0 -> D would be negative; ensure clamp to eps
    K = np.array([90.0, 100.0, 110.0])
    y = np.array([-5.0, 0.0, 5.0])  # slope > 0
    P = np.full_like(K, 2.0)
    C = P + y

    res, mask = estimate_future_from_option_prices(K, C, P, eps=1e-9)
    assert isinstance(res, ImpliedFutureResult)
    assert res.D >= 1e-9
    # bands contain point
    assert res.D_min - 1e-15 <= res.D <= res.D_max + 1e-15


def test_duplicate_strikes_are_collapsed(rng):
    K = np.array([90, 90, 100, 100, 110, 110], dtype=float)
    D_true, F_true = 0.97, 99.0
    C, P = _make_synthetic_prices(K, D_true, F_true)
    C[1] += 4.0

    res, mask = estimate_future_from_option_prices(K, C, P)
    assert abs(res.D - D_true) < 5e-3
    assert abs(res.F - F_true) < 0.2
    assert mask.sum() >= 4


def test_plot_branch_executes(rng):
    K = np.linspace(80, 120, 20)
    D_true, F_true = 0.95, 100.0
    C, P = _make_synthetic_prices(K, D_true, F_true)

    fig, ax = plt.subplots()
    res, mask = estimate_future_from_option_prices(K, C, P, plot=True, ax=ax)
    assert isinstance(res, ImpliedFutureResult)
    plt.close(fig)


def test_band_monotonicity(rng):
    K = np.linspace(60, 140, 40)
    D_true, F_true = 0.92, 103.5
    C, P = _make_synthetic_prices(K, D_true, F_true)

    res, mask = estimate_future_from_option_prices(K, C, P)
    assert res.D_min - 1e-15 <= res.D <= res.D_max + 1e-15
    assert res.F_bid - 1e-15 <= res.F <= res.F_ask + 1e-15


# --------------------- public API error/edge coverage ---------------------


def test_api_rejects_non_1d_and_length_mismatch():
    K = np.array([[90.0, 100.0, 110.0]]).T  # 2-D column
    C = np.array([1.0, 2.0, 3.0])
    P = np.array([0.5, 0.5, 0.5])

    with pytest.raises(ValueError):
        estimate_future_from_option_prices(K, C, P)

    # length mismatch
    K = np.array([90.0, 100.0, 110.0])
    C = np.array([1.0, 2.0])  # shorter
    P = np.array([0.5, 0.6, 0.7])
    with pytest.raises(ValueError):
        estimate_future_from_option_prices(K, C, P)


def test_api_finite_mask_too_small_returns_none():
    # Only one finite row -> early return with all-False mask
    K = np.array([100.0, np.nan, np.nan])
    C = np.array([1.0, 2.0, 3.0])
    P = np.array([0.5, 0.5, 0.5])

    res, mask = estimate_future_from_option_prices(K, C, P)
    assert res is None
    assert mask.shape == (3,) and not mask.any()


def test_trim_loop_breaks_when_less_than_two_inliers():
    # Make residual thresholding ultra-strict so new_inlier.sum() < 2
    K = np.linspace(50, 150, 12)
    C, P = _make_prices(K, D=0.95, F=100.0)
    C = C + np.linspace(-10, 10, K.size)  # inject large non-linear noise

    res, mask = estimate_future_from_option_prices(
        K, C, P, tol_sigma=0.0, abs_tol=0.0, rel_tol=0.0
    )
    # Function should *not* crash and should still return a result or None with a mask
    # We accept either a valid result (with >=2 inliers) or early None.
    assert mask.shape == (K.size,)
    assert (res is None) or isinstance(res, ImpliedFutureResult)


def test_negative_D_hat_returns_none_when_clip_allows():
    # Positive slope in y => D_hat < 0. Allow it via clip_D, then expect early None
    K = np.array([90.0, 100.0, 110.0])
    y = np.array([-5.0, 0.0, 5.0])  # slope > 0
    P = np.full_like(K, 1.0)
    C = P + y

    res, mask = estimate_future_from_option_prices(K, C, P, clip_D=(-np.inf, 1.0))
    assert res is None
    # finite_mask was all True, inliers start as all True, so scatter-back marks all
    assert mask.shape == (K.size,) and mask.all()


# --------------------- helper-level edge coverage ---------------------


def test__constrained_ols_line_value_error_and_flat_case():
    # Value error when not enough points
    with pytest.raises(ValueError):
        _constrained_ols_line(np.array([100.0]), np.array([1.0]), eps=1e-12)

    # varK <= eps triggers flat solution (b = 0, a = mean(y))
    K = np.array([100.0, 100.0])
    y = np.array([1.0, 2.0])
    a, b = _constrained_ols_line(K, y, eps=1.0)  # huge eps to force branch
    assert a == pytest.approx(y.mean())
    assert b == 0.0


def test__mad_empty_and_threshold_minimum():
    eps = 1e-9
    assert _mad(np.array([]), eps) == eps
    # threshold falls back to 5*eps minimum
    resid = np.array([])
    assert _mad_threshold(resid, mult=0.1, eps=eps) == pytest.approx(5 * eps)


def test__feasible_D_band_edge_cases():
    eps = 1e-12

    # n < 2
    Dmn, Dmx = _feasible_D_band_from_pairwise_slopes(
        np.array([100.0]),
        np.array([0.0]),
        q_low=0.25,
        q_high=0.75,
        clip_D=(1e-6, 1.0),
        eps=eps,
    )
    assert (Dmn, Dmx) == (None, None)

    # no slopes because all dK <= eps
    Dmn, Dmx = _feasible_D_band_from_pairwise_slopes(
        np.array([100.0, 100.0 + 1e-13]),
        np.array([0.0, 0.0]),
        q_low=0.25,
        q_high=0.75,
        clip_D=(1e-6, 1.0),
        eps=1e-12,
    )
    assert (Dmn, Dmx) == (None, None)

    # D_cands empty due to non-finite slopes
    Dmn, Dmx = _feasible_D_band_from_pairwise_slopes(
        np.array([90.0, 110.0]),
        np.array([np.nan, np.nan]),
        q_low=0.25,
        q_high=0.75,
        clip_D=(1e-6, 1.0),
        eps=eps,
    )
    assert (Dmn, Dmx) == (None, None)


def test__future_band_from_D_band_empty_inputs():
    F_lo, F_hi = _future_band_from_D_band(np.array([]), np.array([]), 0.9, 0.95)
    assert (F_lo, F_hi) == (None, None)


def test__count_distinct_empty_returns_zero():
    assert _count_distinct(np.array([]), eps=1e-12) == 0


def test__feasible_D_band_inverted_clip_degenerate_point():
    K = np.array([90.0, 110.0, 130.0])
    D_true = 0.95
    C, P = _make_prices(K, D=D_true, F=100.0)
    y = C - P
    clip = (1.0, 1e-6)
    Dmn, Dmx = _feasible_D_band_from_pairwise_slopes(
        K, y, q_low=0.25, q_high=0.75, clip_D=clip, eps=1e-12
    )
    lo, hi = min(clip), max(clip)
    assert lo <= Dmn <= hi
    assert lo <= Dmx <= hi
    assert pytest.approx(Dmn, rel=1e-6) == D_true
    assert pytest.approx(Dmx, rel=1e-6) == D_true


def test_post_trim_distinct_collapse_returns_none():
    # Construct y so initial OLS gives b_hat=0 and a_hat=y_mean=0, leaving only
    # the two K=100 points as inliers under tol_sigma=0. This causes the final
    # distinct(K[inlier]) < 2 check to trigger an early None.
    K = np.array([100.0, 100.0, 101.0, 101.0, 102.0, 102.0])
    P = np.full_like(K, 5.0)
    y = np.array([0.0, 0.0, 10.0, -10.0, 20.0, -20.0])
    C = P + y

    res, mask = estimate_future_from_option_prices(
        K, C, P, tol_sigma=0.0, abs_tol=0.0, rel_tol=0.0
    )
    assert res is None
    assert mask.shape == (K.size,) and not mask.any()


def test_d_band_inverted_clip_produces_degenerate_result():
    K = np.linspace(80.0, 120.0, 10)
    D_true, F_true = 0.97, 100.0
    C, P = _make_prices(K, D=D_true, F=F_true)

    clip = (1.0, 1e-6)  # inverted on purpose
    res, mask = estimate_future_from_option_prices(K, C, P, clip_D=clip)

    assert isinstance(res, ImpliedFutureResult)
    assert mask.shape == (K.size,) and mask.all()

    lo, hi = min(clip), max(clip)
    assert lo <= res.D_min <= res.D_max <= hi
    assert res.D_min - 1e-12 <= res.D <= res.D_max + 1e-12
    assert abs(res.D - D_true) < 1e-6
    assert abs(res.F - F_true) < 1e-6


def test_trim_loop_completes_without_break_max_iter_1():
    # Force a change in inliers on the first (and only) iteration so the for-loop
    # completes by exhausting range rather than via a break condition.
    K = np.linspace(50.0, 150.0, 20)
    C, P = _make_prices(K, D=0.95, F=100.0)
    C = C.copy()
    C[[0, 1, 18, 19]] += 25.0  # large outliers

    res, mask = estimate_future_from_option_prices(
        K, C, P, tol_sigma=2.0, max_trim_iters=1
    )
    assert isinstance(res, ImpliedFutureResult)
    assert mask.sum() < K.size  # outliers removed in the single iteration


def test_fallback_band_when_future_intersection_is_empty():
    """
    With the inlier-quantile band + convex hull, an empty future intersection
    no longer collapses to a tiny band; the final F-band reflects cross-sectional
    dispersion. We only assert containment and a non-trivial width.
    """
    K = np.linspace(80.0, 120.0, 10)
    y = -0.97 * K + 97.0 + 0.2 * (K - 100.0) ** 2
    P = np.zeros_like(K)
    C = y.copy()

    res, mask = estimate_future_from_option_prices(
        K, C, P, trim_mad_mult=1e12  # keep all inliers
    )

    assert isinstance(res, ImpliedFutureResult)
    assert mask.shape == (K.size,) and mask.all()

    # Containment and width > 0 (quantile hull yields a meaningful band)
    assert res.F_bid <= res.F <= res.F_ask
    assert (res.F_ask - res.F_bid) > 0.0

    # D estimate still near the constrained slope (around 0.97)
    assert abs(res.D - 0.97) < 1e-2


# ---------------- Additional helper coverage to hit remaining branches ---------


def test__canon_D_band_seeds_from_hat_and_clamps():
    # Missing band -> seeds from D_hat, clipped inside [c_lo, c_hi], with soft ±tiny containment
    D_min, D_max = _canon_D_band(None, None, D_hat=1.2, c_lo=1e-6, c_hi=1.0, tiny=1e-9)
    assert 1e-6 <= D_min <= D_max <= 1.0
    # Near 1.0 because D_hat is clipped to 1.0
    assert D_min <= 1.0 and D_max >= 1.0 - 1e-9


def test__canon_D_band_soft_contain_and_reorder():
    # Inverted band and outside clip; also check soft containment around D_hat ± tiny
    D_min, D_max = _canon_D_band(1.5, 0.2, D_hat=0.95, c_lo=1e-6, c_hi=1.0, tiny=1e-9)
    assert 1e-6 <= D_min <= D_max <= 1.0
    assert D_min <= 0.95 <= D_max


def test__canon_D_band_collapses_when_band_outside_clip():
    # Raw band inside (0.2, 0.3) but clip is (0.5, 0.6) => entirely outside clip ⇒ collapse to tiny around D_hat
    D_min, D_max = _canon_D_band(0.2, 0.3, D_hat=0.55, c_lo=0.5, c_hi=0.6, tiny=1e-9)
    assert 0.5 <= D_min <= D_max <= 0.6
    assert D_min <= 0.55 <= D_max  # contains D_hat


def test__future_band_from_D_band_empty_intersection_branch():
    # Two strikes create disjoint F-intervals for D∈[0.5,1.0]
    K = np.array([0.0, 1.0])
    y = np.array([+1.0, -1.0])
    Flo, Fhi = _future_band_from_D_band(K, y, D_min=0.5, D_max=1.0)
    assert (Flo, Fhi) == (None, None)


def test_sigma_scale_fallback_branch(rng):
    # Perfect linear data -> residuals zero -> MAD scale = 0 -> fallback scale path covered
    K = np.linspace(50, 150, 25)
    D_true, F_true = 0.94, 102.0
    C, P = _make_prices(K, D=D_true, F=F_true)
    res, mask = estimate_future_from_option_prices(K, C, P, tol_sigma=3.0)
    assert isinstance(res, ImpliedFutureResult)
    assert mask.all()


def test_trim_loop_breaks_on_no_change_inliers(rng):
    # Clean linear data with mild noise below threshold -> inliers don't change -> break on equality
    K = np.linspace(70, 130, 21)
    D_true, F_true = 0.97, 100.0
    C, P = _make_prices(K, D=D_true, F=F_true)
    noise = 1e-8 * (K - K.mean())  # tiny, below MAD threshold
    C = C + noise
    res, mask = estimate_future_from_option_prices(
        K, C, P, trim_mad_mult=3.5, max_trim_iters=5
    )
    assert isinstance(res, ImpliedFutureResult)
    assert mask.all()


def test__future_band_from_D_band_nan_discount_is_rejected():
    # Hits: if not np.all(np.isfinite(d)) -> return (None, None)
    K = np.array([100.0, 101.0])
    y = np.array([1.0, -1.0])
    Flo, Fhi = _future_band_from_D_band(K, y, D_min=np.nan, D_max=0.95)
    assert (Flo, Fhi) == (None, None)


def test__future_band_from_D_band_nonpositive_discount_is_rejected():
    # Hits: if d[0] <= 0.0 -> return (None, None)
    K = np.array([100.0, 101.0])
    y = np.array([1.0, -1.0])
    Flo, Fhi = _future_band_from_D_band(K, y, D_min=-1e-4, D_max=0.5)
    assert (Flo, Fhi) == (None, None)
