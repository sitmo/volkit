# volkit/utils/__init__.py

import numpy as np

__all__ = ["discount_to_rate", "rate_to_discount"]


def discount_to_rate(discount_rate, *, t=None, days=None, days_per_year=365):
    """
    Convert a discount factor into a continuously compounded annualized rate.

    Exactly one of ``t`` or ``days`` must be provided to specify the time horizon.

    Parameters
    ----------
    discount_rate : float
        Discount factor. Must be finite and strictly positive (> 0).
    t : float, optional
        Time horizon in years. Mutually exclusive with ``days``.
    days : int or float, optional
        Time horizon in days. Mutually exclusive with ``t``.
    days_per_year : int or float, default=365
        Number of days per year used to convert ``days`` into years.

    Returns
    -------
    float
        Continuously compounded annualized rate corresponding to the discount factor.

    Raises
    ------
    ValueError
        If neither or both of ``t`` and ``days`` are provided.
        If ``discount_rate`` is not finite or not strictly positive.
        If the resulting time horizon (years) is not strictly positive.

    Examples
    --------
    >>> discount_to_rate(0.95, t=0.5)  # roughly half a year
    0.102586...
    >>> discount_to_rate(0.95, days=90)  # 90 trading days with 365/yr
    0.20802...
    >>> discount_to_rate(1.0, t=1.0)
    0.0
    """
    # Exactly one of t or days must be provided
    if not ((t is None) ^ (days is None)):
        raise ValueError("Exactly one of `t` or `days` must be provided.")

    # Validate discount_rate
    if not np.isfinite(discount_rate) or discount_rate <= 0:
        raise ValueError("`discount_rate` must be finite and strictly positive (> 0).")

    # Determine t in years
    if days is not None:
        t_years = float(days) / float(days_per_year)
    else:
        t_years = float(t)

    if t_years <= 0:
        raise ValueError("Time horizon must be strictly positive (t > 0 or days > 0).")

    return float(-np.log(discount_rate) / t_years)


def rate_to_discount(rate, *, t=None, days=None, days_per_year=365):
    """
    Convert a continuously compounded annualized rate into a discount factor.

    Exactly one of ``t`` or ``days`` must be provided to specify the time horizon.

    Parameters
    ----------
    rate : float
        Continuously compounded annualized rate. May be negative or positive,
        but must be finite (no NaN/Inf).
    t : float, optional
        Time horizon in years. Mutually exclusive with ``days``.
    days : int or float, optional
        Time horizon in days. Mutually exclusive with ``t``.
    days_per_year : int or float, default=365
        Number of days per year used to convert ``days`` into years.

    Returns
    -------
    float
        Discount factor in (0, +inf) given by ``exp(-rate * t_years)``.

    Raises
    ------
    ValueError
        If neither or both of ``t`` and ``days`` are provided.
        If ``rate`` is not finite.
        If the resulting time horizon (years) is not strictly positive.

    Examples
    --------
    >>> rate_to_discount(0.1, t=0.5)
    0.9512...
    >>> rate_to_discount(0.1, days=90)
    0.9756...
    >>> rate_to_discount(0.0, t=1.0)
    1.0
    """
    # Exactly one of t or days must be provided
    if not ((t is None) ^ (days is None)):
        raise ValueError("Exactly one of `t` or `days` must be provided.")

    # Validate rate
    if not np.isfinite(rate):
        raise ValueError("`rate` must be finite (no NaN/Inf).")

    # Determine t in years
    if days is not None:
        t_years = float(days) / float(days_per_year)
    else:
        t_years = float(t)

    if t_years <= 0:
        raise ValueError("Time horizon must be strictly positive (t > 0 or days > 0).")

    return float(np.exp(-rate * t_years))
