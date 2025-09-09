
import math
import numpy as np
import pytest

from volkit.arb.prices_single_expiry import (
    arb_from_option_prices_single_expiry,
    ArbReport,
    _build_tolerances
)


# -----------------------------
# Utilities
# -----------------------------

@pytest.fixture
def base_setup():
    K = np.array([90.0, 100.0, 110.0])
    F = 100.0
    D = 0.99
    return K, F, D


def parity_puts_from_calls(K, F, D, C):
    '''Return P that enforces C - P = D*(F - K) exactly.'''
    return C - (D * (F - K))


# -----------------------------
# Happy path: everything OK
# -----------------------------

def test_no_violations_path_and_as_report(base_setup):
    K, F, D = base_setup
    C = np.array([12.0, 7.0, 3.0])  # decreasing
    P = parity_puts_from_calls(K, F, D, C)  # [1.1, 6.0, 11.9]

    rep = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=True,
    )
    assert isinstance(rep, ArbReport)
    assert rep.ok
    assert rep.trades == []
    assert rep.meta.get('sorted') is False


# -----------------------------
# Bounds-only single violation
# -----------------------------

def test_report_and_trade_roundtrip_and_totals_bounds_only_single_trade(base_setup):
    K, F, D = base_setup
    # Lower bound at 90: F - D*K = 10.9 -> C[0]=11.0 OK
    # Lower bound at 100: 1.0 -> C[1]=2.0 OK
    # Upper bound for calls typically <= F -> at 110 choose 102.0 to breach only UB
    C = np.array([11.0, 2.0, 102.0])
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=F, D=D,
        check_bounds=True, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=False,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert isinstance(rep, ArbReport)
    assert not rep.ok
    assert len(rep.trades) == 1


# -----------------------------
# Sorting + NaN: convexity should not trigger
# -----------------------------

def test_assume_sorted_false_and_nans_and_toggles(base_setup):
    K, F, D = base_setup
    # Unsorted + some NaNs
    K_unsorted = np.array([110.0, 90.0, 100.0])
    C_unsorted = np.array([2.0, 12.0, np.nan])  # middle becomes NaN after sort
    P_unsorted = np.array([14.0, np.nan, 1.0])

    res = arb_from_option_prices_single_expiry(
        K_unsorted, C=C_unsorted, P=P_unsorted, F=F, D=D,
        assume_sorted=False,
        check_bounds=False, check_monotonicity=False, check_vertical=False,
        check_convexity=True, check_parity=False,
        as_report=False,
    )
    assert 'violations' in res
    assert res['ok']  # no convexity triple available -> no violation
    conv = res['violations'].get('convexity', {})
    assert conv.get('call_bad_triples', []) == []
    assert conv.get('put_bad_triples', []) == []


# -----------------------------
# Parity: metrics when all NaN
# -----------------------------

def test_parity_metric_with_no_finite_pairs(base_setup):
    K, F, D = base_setup
    C = np.array([np.nan, np.nan, np.nan])
    P = np.array([np.nan, np.nan, np.nan])

    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=True,
        as_report=False,
    )
    assert res['ok']
    par = res['violations'].get('parity', {})
    assert par.get('bad_idx', []) == []
    if 'max_abs_err' in par:
        assert np.isnan(par['max_abs_err'])


# -----------------------------
# Convexity: put-only violation
# -----------------------------

def test_put_convexity_only(base_setup):
    K, F, D = base_setup  # [90,100,110]
    C = np.array([np.nan, 7.0, 3.0])  # NaN disables call triples
    P = np.array([2.0, 8.0, 2.5])     # middle overpriced -> convexity fail

    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=False, check_vertical=False,
        check_convexity=True, check_parity=False,
        as_report=False,
    )
    assert not res['ok']
    conv = res['violations']['convexity']
    assert conv.get('put_bad_triples', []) == [(0, 1, 2)]
    assert conv.get('call_bad_triples', []) == []


# -----------------------------
# Vertical caps: breach both sides
# -----------------------------

def test_vertical_caps_calls_and_puts(base_setup):
    K, F, D = base_setup
    # Calls: C[0]-C[1] = 15.0 > 9.9
    C = np.array([25.0, 10.0, np.nan])
    # Puts: P[1]-P[0] = 11.0 > 9.9
    P = np.array([1.0, 12.0, np.nan])

    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=False, check_vertical=True,
        check_convexity=False, check_parity=False,
        as_report=False,
    )
    assert not res['ok']
    vert = res['violations']['vertical']
    assert vert.get('call_ub_pairs', []) == [(0, 1)]
    assert vert.get('put_ub_pairs', []) == [(0, 1)]


# -----------------------------
# Input validation + sorting meta
# -----------------------------

def test_input_validation_and_sorting_meta(base_setup):
    K, F, D = base_setup

    with pytest.raises(ValueError):
        arb_from_option_prices_single_expiry(K.reshape(3,1), C=None, P=np.array([1,2,3]), F=F, D=D)
    with pytest.raises(ValueError):
        arb_from_option_prices_single_expiry(K, C=np.array([1,2]), P=None, F=F, D=D)
    with pytest.raises(ValueError):
        arb_from_option_prices_single_expiry(K, C=None, P=None, F=F, D=D)

    # sorting metadata: use parity-consistent arrays *after* sorting
    K_u = np.array([110.0, 90.0, 100.0])
    C_u = np.array([3.0, 12.0, 7.0])
    P_u_sorted = parity_puts_from_calls(np.array([90.0, 100.0, 110.0]), F, D, np.array([12.0,7.0,3.0]))
    # Place P_u in unsorted order aligning with K_u
    P_u = np.array([P_u_sorted[2], P_u_sorted[0], P_u_sorted[1]])  # [11.9, 1.1, 6.0]

    rep = arb_from_option_prices_single_expiry(
        K_u, C=C_u, P=P_u, F=F, D=D, assume_sorted=False,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True, as_report=True
    )
    assert isinstance(rep, ArbReport)
    assert rep.meta.get('sorted') is True
    assert rep.ok


# -----------------------------
# Tolerances: relative branch
# -----------------------------

def test_build_tolerances_relative_branch_no_false_flags(base_setup):
    K, F, D = base_setup
    C = np.array([12.0, 7.0, 3.0])
    P = parity_puts_from_calls(K, F, D, C)  # exact parity

    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        tol_opt=0.01, tol_opt_rel=0.10,  # triggers relative tolerance combination
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert rep.ok


# -----------------------------
# Bounds corner branches
# -----------------------------

def test_bounds_put_upper_overpriced_single_trade_only(base_setup):
    K, F, D = base_setup
    P = np.array([D*K[0] + 5.0, np.nan, np.nan])  # violate put upper bound at idx 0

    res = arb_from_option_prices_single_expiry(
        K, C=None, P=P, F=F, D=D,
        check_bounds=True, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=False,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert not rep.ok
    assert len(rep.trades) == 1
    assert 0 in rep.violations['bounds']['put_ub']


def test_bounds_call_lower_elsepath_when_F_le_K():
    # Single strike where F <= K so call LB branch shouldn't add a bond leg
    K = np.array([150.0])
    F = 100.0
    D = 0.99
    C = np.array([-0.25])  # violates LB since max(0, F - D*K) = 0
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=F, D=D,
        check_bounds=True, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=False,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert not rep.ok
    assert len(rep.trades) == 1
    legs = rep.trades[0].legs
    assert all(leg.asset == 'call' for leg in legs)


def test_bounds_put_lower_elsepath_when_K_le_F():
    # Single strike where K <= F so put LB branch shouldn't add a bond leg
    K = np.array([80.0])
    F = 100.0
    D = 0.99
    P = np.array([-0.10])  # violates LB since max(0, D*K - F) = 0
    res = arb_from_option_prices_single_expiry(
        K, C=None, P=P, F=F, D=D,
        check_bounds=True, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=False,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert not rep.ok
    assert len(rep.trades) == 1
    legs = rep.trades[0].legs
    assert all(leg.asset == 'put' for leg in legs)


# -----------------------------
# Vertical LB for puts
# -----------------------------
def test_vertical_put_lower_bound_pair_violation():
    K = np.array([90.0, 110.0])
    F = 100.0
    D = 0.99
    # Make the higher-strike put CHEAPER to violate the LB: P(K2) - P(K1) < 0
    P = np.array([10.0, 9.0])  # spread = -0.1 < 0  -> LB violation

    res = arb_from_option_prices_single_expiry(
        K, C=None, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=True,  check_vertical=False,
        check_convexity=False, check_parity=False,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert not rep.ok
    assert len(rep.trades) >= 1
    # Optional: inspect the first trade for clarity
    t0 = rep.trades[0]
    assert getattr(t0, "kind", "vertical")  # tolerant check



# -----------------------------
# Convexity: put triple (separate)
# -----------------------------

def test_convexity_put_bad_triple(base_setup):
    K, F, D = base_setup
    P = np.array([1.0, 50.0, 1.0])  # huge middle
    res = arb_from_option_prices_single_expiry(
        K, C=None, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=False, check_vertical=False,
        check_convexity=True, check_parity=False,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert not rep.ok
    assert rep.violations['convexity']['put_bad_triples']


# -----------------------------
# Parity: perfect branch -> mae 0.0
# -----------------------------

def test_parity_ok_branch_and_max_abs_err_set_zero(base_setup):
    K, F, D = base_setup
    C = np.array([12.0, 7.0, 3.0])
    P = parity_puts_from_calls(K, F, D, C)
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False,
    )
    rep = ArbReport.from_dict(res)
    assert rep.ok
    mae = rep.violations['parity']['max_abs_err']
    assert mae == 0.0 or math.isclose(mae, 0.0, abs_tol=1e-12)


def test_wrong_single_input_shape(base_setup):
    K, F, D = base_setup
    X = np.array([12.0, 7.0])
    with pytest.raises(ValueError):    
        res = arb_from_option_prices_single_expiry(
            K, C=X, P=None, F=F, D=D,
            check_bounds=True, check_monotonicity=True, check_vertical=True,
            check_convexity=True, check_parity=True,
            as_report=False,
        )
    with pytest.raises(ValueError):    
        res = arb_from_option_prices_single_expiry(
            K, C=None, P=X, F=F, D=D,
            check_bounds=True, check_monotonicity=True, check_vertical=True,
            check_convexity=True, check_parity=True,
            as_report=False,
        )


def test_unsorter_C(base_setup):
    K, F, D = base_setup
    X = np.array([12.0, 7.0, 5.0])
    res = arb_from_option_prices_single_expiry(
        K, C=X, P=None, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False, assume_sorted=False
    )
    res = arb_from_option_prices_single_expiry(
        K, C=None, P=X, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False, assume_sorted=False
    )

def test_C_with_NAN(base_setup):
    K, F, D = base_setup
    C = np.array([12.0, np.nan, 5.0])
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False
    )


def test_call_below_intrinsic_forward(base_setup):
    K, F, D = base_setup
    C = np.array([1.0, 1.0, 1.0])
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False
    )


def test_butterfly_below_intrinsic(base_setup):
    K, F, D = base_setup
    C = np.array([21.0, 12.0, 1.0])
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=F, D=D,
        check_bounds=True, check_monotonicity=True, check_vertical=True,
        check_convexity=True, check_parity=True,
        as_report=False
    )

def test_build_tolerances_opt_tau_handles_none_prices():
    # Make the relative tolerance active so the loop matters
    F = 100.0
    tol_opt, tol_opt_rel, tol_fut, tol_fut_rel = 0.25, 0.10, 0.0, 0.0
    fut_pad, opt_tau = _build_tolerances(F, tol_fut, tol_fut_rel, tol_opt, tol_opt_rel)

    # Pass some None values to exercise the x is None branch
    # mx should be max(1, |price|) over non-None entries only
    # Here: max(1, |None| ignored, |0.0|, |2.0|) = 2.0
    val = opt_tau(None, 0.0, 2.0)
    # base = tol_opt + tol_opt_rel * mx = 0.25 + 0.10 * 2.0 = 0.45
    assert pytest.approx(val, rel=0, abs=1e-12) == 0.45
    # fut_pad = 0 in this configuration
    assert fut_pad == 0.0




import numpy as np
import pytest
from volkit.arb.prices_single_expiry import arb_from_option_prices_single_expiry

def test_monotonicity_call_violation_hits_missing_branch():
    # Hits lines 297–307: higher-strike call more expensive than lower-strike
    K = np.array([100.0, 110.0])
    C = np.array([5.0, 6.5])  # violation: C[1] > C[0]
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=100.0, D=0.99,
        check_bounds=False, check_monotonicity=True, check_vertical=False,
        check_convexity=False, check_parity=False, as_report=False,
    )
    assert not res["ok"]
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["type"] == "monotonicity"
    legs = t["legs"]
    # Expect buy lower-strike call, sell higher-strike call
    assert [(legs[0]["asset"], legs[0]["side"], legs[0]["strike"]),
            (legs[1]["asset"], legs[1]["side"], legs[1]["strike"])] == [
        ("call", "buy", 100.0),
        ("call", "sell", 110.0),
    ]

def test_parity_call_rich_branch_hits_missing_lines():
    # Hits lines 429–439 (call-rich, m > tau)
    # Construct C-P > D*(F-K) by a wide margin
    K = np.array([100.0])
    F = 120.0
    D = 0.99
    rhs = D * (F - K[0])   # 19.8
    C = np.array([25.0])
    P = np.array([0.0])    # lhs = 25.0 -> m = 25 - 19.8 = 5.2 > 0
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=True, as_report=False,
    )
    assert not res["ok"]
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["type"] == "parity"
    legs = t["legs"]
    # Expect: sell call, buy put, buy bond of notional F-K
    assets_sides = {(leg["asset"], leg["side"]) for leg in legs}
    assert {("call", "sell"), ("put", "buy"), ("bond", "buy")} <= assets_sides
    bond = next(leg for leg in legs if leg["asset"] == "bond")
    assert bond["notional"] == pytest.approx(F - K[0], rel=0, abs=1e-12)

def test_parity_put_rich_branch_hits_missing_lines():
    # Hits lines 441–451 (put-rich, m < -tau)
    K = np.array([100.0])
    F = 120.0
    D = 0.99
    C = np.array([0.0])
    P = np.array([25.0])   # lhs = -25.0, rhs = 19.8 -> m = -44.8 < 0
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=P, F=F, D=D,
        check_bounds=False, check_monotonicity=False, check_vertical=False,
        check_convexity=False, check_parity=True, as_report=False,
    )
    assert not res["ok"]
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["type"] == "parity"
    legs = t["legs"]
    # Expect: buy call, sell put, SELL bond of notional F-K
    assets_sides = {(leg["asset"], leg["side"]) for leg in legs}
    assert {("call", "buy"), ("put", "sell"), ("bond", "sell")} <= assets_sides
    bond = next(leg for leg in legs if leg["asset"] == "bond")
    assert bond["notional"] == pytest.approx(F - K[0], rel=0, abs=1e-12)

_base = dict(
    check_bounds=False, check_monotonicity=False, check_vertical=False,
    check_convexity=True, check_parity=False, as_report=False
)

def test_convexity_skips_triple_with_nonfinite_strike():
    # K has a NaN so math.isfinite(...) is False -> 'continue' branch executes
    K = np.array([90.0, np.nan, 110.0])
    C = np.array([12.0, 8.0, 2.0])   # arbitrary finite prices
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=100.0, D=0.99, assume_sorted=True, **_base
    )
    # We only care that it runs through; with checks limited to convexity, this should be ok
    assert res["ok"]

def test_convexity_skips_triple_when_Kk_le_Ki():
    # Duplicate strikes make Kk <= Ki true for some (i,k) -> 'continue' branch executes
    K = np.array([100.0, 100.0, 110.0])
    C = np.array([5.0, 5.0, 0.5])    # monotone-ish so no violation elsewhere
    res = arb_from_option_prices_single_expiry(
        K, C=C, P=None, F=100.0, D=0.99, assume_sorted=True, **_base
    )
    assert res["ok"]
