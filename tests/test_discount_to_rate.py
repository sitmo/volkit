import numpy as np
import pytest

from volkit import discount_to_rate  # adjust import to your package layout


def test_with_t_basic():
    t = 0.5
    d = 0.95
    expected = -np.log(d) / t
    assert np.isclose(discount_to_rate(d, t=t), expected, rtol=1e-12, atol=0)


def test_with_days_default_conversion():
    d = 0.95
    expected = -np.log(d) / 0.5
    assert np.isclose(discount_to_rate(d, days=365/2), expected, rtol=1e-12, atol=0)


def test_with_days_custom_days_per_year():
    d = 0.95
    expected = -np.log(d) / 0.25
    assert np.isclose(discount_to_rate(d, days=90, days_per_year=360), expected, rtol=1e-12, atol=0)


def test_both_none_raises():
    with pytest.raises(ValueError, match="Exactly one of `t` or `days` must be provided"):
        discount_to_rate(0.99)


def test_both_provided_raises():
    with pytest.raises(ValueError, match="Exactly one of `t` or `days` must be provided"):
        discount_to_rate(0.99, t=1.0, days=252)


def test_t_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        discount_to_rate(0.99, t=0.0)


def test_t_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        discount_to_rate(0.99, t=-1.0)


def test_days_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        discount_to_rate(0.99, days=0)


def test_days_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        discount_to_rate(0.99, days=-5)


def test_discount_rate_one_returns_zero_rate():
    assert discount_to_rate(1.0, t=1.0) == 0.0


def test_discount_rate_zero_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        discount_to_rate(0.0, t=1.0)


def test_discount_rate_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        discount_to_rate(-0.1, t=1.0)


def test_discount_rate_nan_raises():
    with pytest.raises(ValueError, match="finite and strictly positive"):
        discount_to_rate(np.nan, t=1.0)


def test_discount_rate_inf_raises():
    with pytest.raises(ValueError, match="finite and strictly positive"):
        discount_to_rate(np.inf, t=1.0)
