import numpy as np
import pytest

from volkit import rate_to_discount, discount_to_rate


def test_with_t_basic():
    r = 0.1
    t = 0.5
    expected = np.exp(-r * t)
    assert np.isclose(rate_to_discount(r, t=t), expected, rtol=1e-12, atol=0)


def test_with_days_default_conversion():
    r = 0.1
    expected = np.exp(-r * 0.5)
    assert np.isclose(rate_to_discount(r, days=365/2), expected, rtol=1e-12, atol=0)


def test_with_days_custom_days_per_year():
    r = 0.1
    expected = np.exp(-r * 0.25)
    assert np.isclose(rate_to_discount(r, days=90, days_per_year=360), expected, rtol=1e-12, atol=0)


def test_negative_rate_supported():
    r = -0.02
    t = 1.0
    # negative rate -> discount > 1
    out = rate_to_discount(r, t=t)
    assert out > 1.0
    assert np.isclose(out, np.exp(-r * t), rtol=1e-12, atol=0)


def test_zero_rate_returns_one():
    assert rate_to_discount(0.0, t=1.0) == 1.0


def test_both_none_raises():
    with pytest.raises(ValueError, match="Exactly one of `t` or `days` must be provided"):
        rate_to_discount(0.1)


def test_both_provided_raises():
    with pytest.raises(ValueError, match="Exactly one of `t` or `days` must be provided"):
        rate_to_discount(0.1, t=1.0, days=365)


def test_t_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        rate_to_discount(0.1, t=0.0)


def test_t_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        rate_to_discount(0.1, t=-1.0)


def test_days_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        rate_to_discount(0.1, days=0)


def test_days_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        rate_to_discount(0.1, days=-5)


def test_rate_nan_raises():
    with pytest.raises(ValueError, match="finite"):
        rate_to_discount(np.nan, t=1.0)


def test_rate_inf_raises():
    with pytest.raises(ValueError, match="finite"):
        rate_to_discount(np.inf, t=1.0)


def test_round_trip_t_path():
    # discount_to_rate -> rate_to_discount should round-trip
    d = 0.937
    t = 0.75
    r = discount_to_rate(d, t=t)
    d_back = rate_to_discount(r, t=t)
    assert np.isclose(d_back, d, rtol=1e-12, atol=0)


def test_round_trip_days_path_default():
    d = 0.88
    days = 63  # 0.25 years if 252/yr
    r = discount_to_rate(d, days=days)
    d_back = rate_to_discount(r, days=days)
    assert np.isclose(d_back, d, rtol=1e-12, atol=0)


def test_round_trip_days_path_custom_dpy():
    d = 0.92
    days = 90
    dpy = 360
    r = discount_to_rate(d, days=days, days_per_year=dpy)
    d_back = rate_to_discount(r, days=days, days_per_year=dpy)
    assert np.isclose(d_back, d, rtol=1e-12, atol=0)
