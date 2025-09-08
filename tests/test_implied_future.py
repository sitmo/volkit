# tests/test_implied_future.py
import numpy as np

from volkit import estimate_future_from_option_quotes
from volkit.estimate.future_from_option_quotes import (
    _max_overlap_subset_at_D,
    _feasible_D_interval_for_subset,
    _width_candidate_discounts_for_subset,
    _future_band_for_subset_at_D,
    _filter_and_sort_finite,
    _with_midpoints,
    _choose_subset_with_fallback,
)

import volkit.estimate.future_from_option_quotes as vf

EPS = 1e-12


def make_parity_consistent_quotes(K, F, D, s_call=0.4, s_put=0.6):
    """
    Build call/put bid/ask arrays that satisfy (C - P) = D*(F - K) in mid,
    with symmetric bid/ask spreads (but possibly different widths for C and P).
    This yields L = Cb - Pa = D(F-K) - (s_call + s_put)
           and U = Ca - Pb = D(F-K) + (s_call + s_put),
    so at the true discount D all intervals align around D*F with width (s_c+s_p).
    """
    K = np.asarray(K, float)
    CP_mid = D * (F - K)  # C_mid - P_mid

    # choose a neutral P_mid, derive C_mid = CP_mid + P_mid
    # any decomposition works; pick P_mid = 0 for simplicity
    P_mid = np.zeros_like(K, dtype=float)
    C_mid = CP_mid + P_mid

    Ca = C_mid + s_call
    Cb = C_mid - s_call
    Pa = P_mid + s_put
    Pb = P_mid - s_put
    return Cb, Ca, Pb, Pa


def test_known_future_and_band_all_rows_kept_and_nans_filtered(monkeypatch):
    # Known setup
    K = np.array([3900, 3950, 4000, 4050, 4100, 4150], dtype=float)

    F_true = 4025.0
    D_true = 0.97
    s_c, s_p = 0.5, 0.7
    Cb, Ca, Pb, Pa = make_parity_consistent_quotes(K, F_true, D_true, s_c, s_p)

    # Inject a NaN row to check finite filtering (should be dropped)
    K_with_nan = np.concatenate([K, [np.nan]])
    Cb_with_nan = np.concatenate([Cb, [np.nan]])
    Ca_with_nan = np.concatenate([Ca, [np.nan]])
    Pb_with_nan = np.concatenate([Pb, [np.nan]])
    Pa_with_nan = np.concatenate([Pa, [np.nan]])

    # Monkeypatch the plotting function so plot=True path runs without side effects
    import volkit.estimate.future_from_option_quotes as futmod

    monkeypatch.setattr(
        futmod, "estimate_future_from_option_quotes_plot", lambda *a, **k: None
    )

    result, mask = estimate_future_from_option_quotes(
        K_with_nan, Cb_with_nan, Ca_with_nan, Pb_with_nan, Pa_with_nan, plot=True
    )

    # Result should exist
    assert result is not None

    # All finite rows should be kept (True) and the NaN row should be False
    assert mask.shape == K_with_nan.shape
    assert mask[:-1].all() and not mask[-1]

    # The future band should straddle F_true with known width ~ (s_c + s_p)/D
    band_half = (s_c + s_p) / D_true
    assert np.isclose(result.F, F_true, rtol=0, atol=1e-9)
    assert np.isclose(result.F_bid, F_true - band_half, rtol=0, atol=1e-9)
    assert np.isclose(result.F_ask, F_true + band_half, rtol=0, atol=1e-9)

    # The feasible discount interval should contain the true discount
    assert result.D_min <= D_true <= result.D_max


def test_arb_row_is_excluded_and_valid_subset_has_overlap():
    # Three strikes, two clean, one irreparably bad (invalid interval)
    K = np.array([4000.0, 4100.0, 4200.0])

    F_true = 4075.0
    D_true = 0.95
    s_c, s_p = 0.3, 0.4

    Cb, Ca, Pb, Pa = make_parity_consistent_quotes(K, F_true, D_true, s_c, s_p)

    # Make the middle strike invalid: enforce L > U by setting Ca << Pb and Pa >> Cb
    j_bad = 1
    Cb[j_bad] = 0.0
    Ca[j_bad] = -10.0  # ask << bid
    Pb[j_bad] = 10.0
    Pa[j_bad] = 20.0  # ask >> bid
    # Now L = Cb - Pa ≈ -20, U = Ca - Pb ≈ -20; with eps it’s safely L > U

    result, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa)

    # We should still get a result (feasible 2-strike subset)
    assert result is not None

    # Exactly two strikes kept: the good ones (0 and 2); the bad one (1) excluded
    kept_idx = np.nonzero(mask)[0].tolist()
    assert kept_idx == [0, 2]

    # future still close to true (inferred from the two good rows)
    assert np.isclose(result.F, F_true, atol=1e-8)
    assert result.F_bid <= result.F <= result.F_ask


def test_duplicate_strikes_only_returns_none_and_empty_mask():
    # Two rows but same K -> not enough distinct strikes
    K = np.array([4000.0, 4000.0])
    F_true, D_true = 4010.0, 0.94
    Cb, Ca, Pb, Pa = make_parity_consistent_quotes(K, F_true, D_true)

    result, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa)

    assert result is None
    # The function returns an all-False mask in this early exit path
    assert mask.shape == K.shape and not mask.any()


def test_no_overlap_anywhere_returns_none_and_no_selection():
    # Two distinct strikes but every row has L > U (invalid intervals)
    K = np.array([4000.0, 4100.0])
    # Make Ca < Cb and Pa > Pb to force L = Cb - Pa > U = Ca - Pb
    Cb = np.array([10.0, 11.0])
    Ca = np.array([9.0, 10.0])  # ask < bid (pathological but tests robustness)
    Pb = np.array([10.0, 10.0])
    Pa = np.array([11.0, 11.0])  # ask > bid (as usual), but with large spread

    result, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa)

    # No feasible overlap depth >= 2 anywhere -> result None
    assert result is None
    # best subset depth is 0; mask remains all False
    assert mask.shape == K.shape and not mask.any()


def test_tiny_input_less_than_two_finite_rows():
    # One finite row and one NaN -> early exit: finite.sum() < 2
    K = np.array([4000.0, np.nan])
    Cb = np.array([1.0, np.nan])
    Ca = np.array([1.1, np.nan])
    Pb = np.array([0.9, np.nan])
    Pa = np.array([1.2, np.nan])

    result, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa)
    assert result is None
    assert mask.shape == K.shape and not mask.any()


def test__max_overlap_subset_at_D_no_valid_events():
    # All intervals invalid: a_i > b_i
    a = np.array([2.0, 3.0])
    b = np.array([1.0, 2.5])
    row_valid = np.ones_like(a, dtype=bool)
    depth, x_star, S = _max_overlap_subset_at_D(a, b, row_valid, eps=1e-12)
    assert depth == 0
    assert S.size == 0
    assert np.isnan(x_star)


def test__feasible_D_interval_for_subset_duplicate_Ks_returns_none():
    K = np.array([100.0, 100.0, 110.0])
    L = np.array([0.0, 0.0, 0.0])
    U = np.array([0.2, 0.2, 0.3])
    S = np.array([0, 1], dtype=int)  # duplicate strikes only
    assert _feasible_D_interval_for_subset(K, L, U, S, eps=1e-12) is None


def test__feasible_D_interval_for_subset_inconsistent_bounds_returns_none():
    # Two strikes; choose L/U so (L0-U1)/(K1-K0) > (U0-L1)/(K1-K0)
    K = np.array([100.0, 110.0])
    L = np.array([6.0, 6.0])  # large lows
    U = np.array([5.1, 5.0])  # smaller highs
    S = np.array([0, 1], dtype=int)
    out = _feasible_D_interval_for_subset(K, L, U, S, eps=1e-12)
    assert out is None


def test__width_candidate_discounts_for_subset_includes_midpoints_and_endpoints():
    K = np.array([100.0, 110.0])
    L = np.array([0.0, 0.1])
    U = np.array([0.4, 0.7])
    S = np.array([0, 1], dtype=int)
    D_lo, D_hi = 0.5, 2.0
    Ds = _width_candidate_discounts_for_subset(K, L, U, S, D_lo, D_hi, eps=1e-12)
    # Endpoints plus some mids/equalities, sorted unique
    assert np.any(np.isclose(Ds, D_lo))
    assert np.any(np.isclose(Ds, D_hi))
    assert np.all(np.diff(Ds) >= 0)


def test__future_band_for_subset_at_D_basic():
    K = np.array([100.0, 110.0, 120.0])
    # Very wide per-strike intervals so their intersection is non-empty at D=0.95
    L = np.array([-1000.0, -1000.0, -1000.0])
    U = np.array([+1000.0, +1000.0, +1000.0])
    S = np.array([0, 1, 2], dtype=int)
    D = 0.95
    F_bid, F_ask = _future_band_for_subset_at_D(K, L, U, S, D)
    assert np.isfinite(F_bid) and np.isfinite(F_ask)
    assert F_bid <= F_ask


def test_best_depth_lt_2_empty_selection_mask_is_all_false():
    """
    Ensure eval_Ds is non-empty, but every row interval is invalid at all D,
    so the sweep finds no events → best_depth=0<2 → branch at lines ~101–108.
    Expect: res is None, mask is all-False.
    """
    K = np.array([100.0, 200.0])

    # Make per-row intervals invalid (U < L) so no row is ever “valid” at any D.
    # Also choose values so candidate D ratios include a positive number:
    # For i=0, j=1: (U1 - L0)/(K0 - K1) = (-3 - 1)/(-100) = +0.04 > 0
    Cb = np.array([1.0, 0.0])  # L = Cb - Pa
    Ca = np.array([-2.0, -3.0])  # U = Ca - Pb
    Pb = np.array([0.0, 0.0])
    Pa = np.array([0.0, 0.0])

    res, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa, eps=1e-12)
    assert res is None
    assert mask.shape == (2,)
    assert mask.sum() == 0


def test_duplicate_strike_best_subset_triggers_continue_and_returns_none():
    """
    Depth-2 overlap comes only from the pair with duplicate strikes (K=100,100).
    That subset is rejected in step 3 → continue. The third row is always invalid,
    so there's no alternative depth-2 subset → res=None, mask all-False.
    """
    K = np.array([100.0, 100.0, 120.0])

    # Duplicate-K rows (0,1): valid intervals at any D (tight but U>=L)
    Cb01 = np.array([0.0, 0.2])  # L0=0.0, L1=0.2
    Ca01 = np.array([0.1, 0.3])  # U0=0.1, U1=0.3
    Pb01 = np.array([0.0, 0.0])
    Pa01 = np.array([0.0, 0.0])

    # Row 2 (non-duplicate K=120): always invalid (U2 < L2) so it never contributes
    Cb2, Ca2 = 5.0, 4.0  # L2=5.0, U2=4.0 ⇒ invalid at all D
    Pb2, Pa2 = 0.0, 0.0

    Cb = np.array([Cb01[0], Cb01[1], Cb2])
    Ca = np.array([Ca01[0], Ca01[1], Ca2])
    Pb = np.array([Pb01[0], Pb01[1], Pb2])
    Pa = np.array([Pa01[0], Pa01[1], Pa2])

    res, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa, eps=1e-12)
    assert res is None
    assert mask.shape == (3,)
    assert not mask.any()


EPS = 1e-12


def test_no_positive_D_candidates_picks_best_singleton_by_narrowest_band():
    # Arrange two strikes whose cross-equalities yield D <= 0 only.
    # Make both per-strike bands valid (U > L), but different widths.
    # L_i = Cb_i - Pa_i ; U_i = Ca_i - Pb_i
    K = np.array([100.0, 200.0])
    # Row 0: tight band [0.00, 0.05]
    Cb0, Ca0, Pb0, Pa0 = 0.00, 0.05, 0.00, 0.00  # width = 0.05
    # Row 1: wider band [1.00, 2.00]
    Cb1, Ca1, Pb1, Pa1 = 1.00, 2.00, 0.00, 0.00  # width = 1.00

    Cb = np.array([Cb0, Cb1])
    Ca = np.array([Ca0, Ca1])
    Pb = np.array([Pb0, Pb1])
    Pa = np.array([Pa0, Pa1])

    # Act
    res, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa, eps=EPS)

    # Assert: no global overlap for any positive D -> res is None,
    # and mask has exactly the narrowest-band row marked True.
    assert res is None
    assert mask.shape == (2,)
    assert mask.sum() == 1
    # Row 0 (narrower band) should be the chosen singleton
    assert mask[0] and not mask[1]


# -------------------------------------------


def test_no_positive_D_candidates_with_duplicate_only_rows_marks_none():
    K = np.array([100.0, 100.0])
    Cb = np.array([0.00, 0.10])
    Ca = np.array([0.20, 0.30])
    Pb = np.array([0.00, 0.00])
    Pa = np.array([0.00, 0.00])
    res, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa, eps=1e-12)
    assert res is None
    # both rows valid, but only one distinct strike among valid rows → mark none
    assert mask.shape == (2,)
    assert not mask.any()

    # ==================================================================================


def test_public_choice_none_path_returns_none_and_mask_all_false():
    """
    Hit estimate_future_from_option_quotes line 94->96:

    - Have positive D candidates (via a third, INVALID row with very negative L,U
      to generate D > 0 in _critical_discounts).
    - Only depth-2 overlap comes from two rows with DUPLICATE strikes -> rejected
      later by unique-K check, so _choose_subset_with_fallback returns None.
    - Public function must then return (None, all-False mask).
    """
    # Two duplicate-K rows (valid)
    K = np.array([100.0, 100.0, 120.0])
    Cb = np.array([0.00, 0.05, 0.0])
    Ca = np.array([0.10, 0.15, -6.0])  # third row ask very negative
    Pb = np.array([0.00, 0.00, 0.0])
    Pa = np.array([0.00, 0.00, -5.0])  # third row L=-5, U=-6 (invalid: U<L)

    # Ds will be positive due to (i=dup, j=invalid) cross-equalities,
    # but the only overlap subset at depth>=2 is (0,1) with duplicate Ks -> rejected.
    res, mask = estimate_future_from_option_quotes(K, Cb, Ca, Pb, Pa, eps=EPS)

    assert res is None
    assert mask.shape == (3,)
    assert not mask.any()  # all False


def test__filter_and_sort_finite_all_invalid_returns_empty():
    """
    Hit _filter_and_sort_finite early return (idx_finite.size == 0) -> line 134.
    """
    K = np.array([np.nan, np.inf])
    Cb = np.array([np.nan, np.inf])
    Ca = np.array([np.nan, np.inf])
    Pb = np.array([np.nan, np.inf])
    Pa = np.array([np.nan, np.inf])

    K1, Cb1, Ca1, Pb1, Pa1, idx_sorted_to_orig = _filter_and_sort_finite(
        K, Cb, Ca, Pb, Pa
    )
    assert K1.size == 0
    assert idx_sorted_to_orig.size == 0
    # and shapes stay aligned
    assert Cb1.size == Ca1.size == Pb1.size == Pa1.size == 0


def test__with_midpoints_empty_input_returns_empty():
    """
    Hit the short-circuit branch in _with_midpoints for Ds.size == 0 -> line 179.
    """
    out = _with_midpoints(np.array([], dtype=float))
    assert out.size == 0


def test__width_candidate_discounts_skips_duplicate_pairs_and_no_mids():
    """
    Hit the denom≈0 'continue' in _width_candidate_discounts_for_subset (line 302)
    and the 'no mids added' branch (Ds.size < 2).

    Use subset S selecting two entries with identical K, and set D_lo==D_hi so the
    only candidate is the endpoint (no midpoints inserted).
    """
    K = np.array([100.0, 100.0, 120.0])
    L = np.array([0.0, 0.1, 0.0])
    U = np.array([0.2, 0.3, 0.4])

    S = np.array([0, 1], dtype=int)  # duplicate-K subset triggers denom≈0 skip
    D_lo = D_hi = 0.5
    Ds = _width_candidate_discounts_for_subset(K, L, U, S, D_lo, D_hi, eps=EPS)

    # Exactly one candidate (the endpoint), and branch with denom≈0 was executed.
    assert np.allclose(Ds, np.array([0.5]))


def test__choose_subset_with_fallback_empty_mapping_returns_none():
    """
    Hit the guard inside _choose_subset_with_fallback: if not depths_to_subsets -> line 348.
    """
    K = np.array([100.0, 120.0])
    L = np.array([0.0, 0.0])
    U = np.array([1.0, 1.0])
    res = _choose_subset_with_fallback(
        K, L, U, depths_to_subsets={}, subset_to_Ds={}, eps=EPS
    )
    assert res is None


def test__choose_subset_with_fallback_skips_depth1_and_selects_best():
    """
    Cover two paths in _choose_subset_with_fallback:
      - the 'depth < 2: continue' branch (line 352)
      - the 'best is not None -> return best' success path.

    We provide two candidate depth levels: 1 and 2.
    For depth=2 we give a workable subset with a positive feasible D-interval.
    """
    # Three distinct strikes
    K = np.array([100.0, 120.0, 140.0])

    # Choose L,U so that (0,1) has feasible D in (0, hi] and non-empty width
    # For p<q with denom=20:
    #   D_lo = max((L_p - U_q)/20), D_hi = min((U_p - L_q)/20)
    L = np.array([0.0, -1.0, 5.0])  # 2nd row low to allow overlap with 1st at small D
    U = np.array([1.0, 2.0, 6.0])

    depths_to_subsets = {
        1: [(0,), (1,)],  # ensures the 'continue' branch is taken for depth=1
        2: [(0, 1)],  # a valid pair; unique K guaranteed
    }
    # subset_to_Ds is unused by the current implementation but pass an empty dict for completeness
    choice = _choose_subset_with_fallback(
        K, L, U, depths_to_subsets, subset_to_Ds={}, eps=EPS
    )
    assert choice is not None

    D_star, F_bid, F_ask, D_lo, D_hi, subset_key = choice
    assert subset_key == (0, 1)
    # The maximizer can be at the boundary (endpoints are included in candidates).
    assert D_lo - 1e-15 <= D_star <= D_hi + 1e-15

    assert F_bid <= F_ask

    # Sanity: width positive at chosen D
    Fb2, Fa2 = _future_band_for_subset_at_D(K, L, U, np.array(subset_key, int), D_star)
    assert Fa2 - Fb2 >= 0.0


def test__feasible_D_interval_handles_nonfinite_left_bound_skipped():
    """
    Drive the (not isfinite(lo)) path on line 277->279 by injecting np.inf into L.
    This is only to cover the branch; the function should simply treat non-finite 'lo'
    as 'skip' and still return an interval (or None if inconsistent).
    """
    K = np.array([100.0, 120.0])
    L = np.array([np.inf, 0.0])  # non-finite left endpoint for pair (will be skipped)
    U = np.array([1.0, 2.0])

    # Using the full subset
    S = np.array([0, 1], dtype=int)
    D_interval = _feasible_D_interval_for_subset(K, L, U, S, eps=EPS)

    # With lo skipped and hi finite, we still expect some interval or None.
    # Either outcome is acceptable for the branch coverage; assert type/shape.
    assert (D_interval is None) or (
        isinstance(D_interval, tuple) and len(D_interval) == 2
    )


# ---------------------------------------------------------------------------
# Tail coverage tests for volkit/implied/future.py
# These target specific uncovered/partial branches shown in the coverage HTML.
# ---------------------------------------------------------------------------


def test__feasible_D_interval_for_subset_nonfinite_hi_path_returns_none():
    """
    Hit the branch in _feasible_D_interval_for_subset where `hi` is non-finite:
      hi = (U_p - L_q) / (K_q - K_p) = inf  -> `if np.isfinite(hi)` is False.
    That skips updating D_hi and ultimately returns None because D_hi stays inf.
    Targets line ~279 alternate path.
    """
    # Two distinct strikes, subset S = [0,1]
    K = np.array([100.0, 120.0])
    # Choose L,U so that for p=0,q=1:
    #   lo = (L0 - U1) / (120-100) = (0 - 10) / 20 = -0.5 (finite)
    #   hi = (U0 - L1) / (120-100) = (inf - 0) / 20 = inf  (non-finite branch)
    L = np.array([0.0, 0.0])
    U = np.array([np.inf, 10.0])

    S = np.array([0, 1], dtype=int)

    res = _feasible_D_interval_for_subset(K, L, U, S, eps=EPS)
    assert res is None  # D_hi remains inf → rejected


def test__choose_subset_with_fallback_empty_maps_return_none():
    """
    Hit the early 'if not depths_to_subsets: return None' path (line ~348).
    """
    K = np.array([100.0, 120.0])
    L = np.array([0.0, 0.0])
    U = np.array([1.0, 2.0])

    choice = _choose_subset_with_fallback(
        K,
        L,
        U,
        depths_to_subsets={},  # empty -> immediate None
        subset_to_Ds={},
        eps=EPS,
    )
    assert choice is None


def test__choose_subset_with_fallback_skips_depth1_only_and_returns_none():
    """
    Cover the 'depth < 2: continue' branch (lines ~351–352) by giving ONLY
    depth-1 subsets. With no depth >= 2 candidates, function should return None.
    """
    # Any inputs suffice since depth==1 candidates are skipped unconditionally.
    K = np.array([100.0, 120.0, 140.0])
    L = np.array([0.0, 0.0, 0.0])
    U = np.array([1.0, 2.0, 3.0])

    depths_to_subsets = {
        1: [(0,), (1,), (2,)],  # depth 1 only -> 'continue' branch is executed
    }

    choice = _choose_subset_with_fallback(
        K,
        L,
        U,
        depths_to_subsets=depths_to_subsets,
        subset_to_Ds={},  # not used in the depth<2 path
        eps=EPS,
    )
    assert choice is None


def test__with_midpoints_empty_input_fast_path():
    """
    _with_midpoints should return the same empty array when input is empty.
    Hits the Ds.size == 0 early return (line ~178–179).
    """
    Ds = np.array([], dtype=float)
    out = _with_midpoints(Ds)
    assert out.size == 0


def test__filter_and_sort_finite_no_rows_fast_path():
    """
    _filter_and_sort_finite should return empty arrays when no row is finite.
    Hits the idx_finite.size == 0 fast path (lines ~133–134).
    """
    K = np.array([np.nan, np.inf])
    Cb = np.array([np.nan, np.nan])
    Ca = np.array([np.inf, np.nan])
    Pb = np.array([np.nan, np.inf])
    Pa = np.array([np.inf, np.inf])

    Kf, Cbf, Caf, Pbf, Paf, idx = _filter_and_sort_finite(K, Cb, Ca, Pb, Pa)
    assert Kf.size == Cbf.size == Caf.size == Pbf.size == Paf.size == 0
    assert idx.size == 0


# ====================================================9872

EPS = 1e-12


def test__choose_subset_with_fallback_depth2_duplicateK_rejected_then_none():
    """
    Cover the branch that rejects depth-2 candidates with duplicate strikes:
      if np.unique(K[S_sorted]).size < 2: continue
    and then falls through to the final `return None`.

    depths_to_subsets has depth=2 with a single subset (0,1), but K[0]==K[1].
    """
    K = np.array([100.0, 100.0, 140.0])  # duplicate Ks at indices 0 and 1
    L = np.array([0.0, 0.1, 0.0])
    U = np.array([0.2, 0.3, 1.0])

    depths_to_subsets = {
        2: [(0, 1)],  # only candidate has duplicate K -> must be rejected
    }
    choice = _choose_subset_with_fallback(
        K,
        L,
        U,
        depths_to_subsets=depths_to_subsets,
        subset_to_Ds={},  # not used by current implementation
        eps=EPS,
    )
    assert choice is None


def test__choose_subset_with_fallback_depth2_infeasible_interval_then_none():
    """
    Distinct-K depth=2 subset, but _feasible_D_interval_for_subset returns None.
    This must fall through and return None.
    """
    # Two distinct strikes
    K = np.array([100.0, 120.0])

    # Engineer lo > hi so the feasible interval is None.
    # For p=0,q=1 with denom=20:
    #   lo = (L0 - U1)/20
    #   hi = (U0 - L1)/20
    # Choose: lo = (10.0 - 20.05)/20 = -0.5025
    #         hi = (10.1 - 20.0)/20 = -0.495
    # After D_lo = max(D_lo, eps), we get D_lo = 1e-12 > D_hi -> None.
    L = np.array([10.0, 20.0])
    U = np.array([10.1, 20.05])
    S = np.array([0, 1], dtype=int)
    assert _feasible_D_interval_for_subset(K, L, U, S, eps=EPS) is None

    depths_to_subsets = {2: [(0, 1)]}
    choice = _choose_subset_with_fallback(
        K, L, U, depths_to_subsets, subset_to_Ds={}, eps=EPS
    )
    assert choice is None


def test__choose_subset_with_fallback_handles_empty_D_candidates_via_monkeypatch(
    monkeypatch,
):
    """
    Force the `D_cands.size == 0` branch inside _choose_subset_with_fallback by
    monkeypatching _width_candidate_discounts_for_subset to return an empty array.
    This path is otherwise unreachable because the helper always seeds [D_lo, D_hi].
    """
    # Choose inputs so the feasible D-interval definitely exists (distinct K, consistent L,U)
    K = np.array([100.0, 120.0])
    L = np.array([0.0, 0.0])
    U = np.array([1.0, 1.0])
    depths_to_subsets = {2: [(0, 1)]}  # a valid 2-deep subset

    # Sanity: without monkeypatch, an interval exists and D_cands would be non-empty
    S = np.array([0, 1], dtype=int)
    assert _feasible_D_interval_for_subset(K, L, U, S, eps=EPS) is not None

    # Monkeypatch to force the empty-candidates branch
    monkeypatch.setattr(
        vf,
        "_width_candidate_discounts_for_subset",
        lambda *args, **kwargs: np.array([], dtype=float),
    )

    choice = _choose_subset_with_fallback(
        K,
        L,
        U,
        depths_to_subsets=depths_to_subsets,
        subset_to_Ds={},
        eps=EPS,
    )
    # With no D candidates, the subset is skipped; with no other subsets, we fall through to None
    assert choice is None
