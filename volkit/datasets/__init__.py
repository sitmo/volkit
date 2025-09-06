# volkit/dataset/__init__.py
import os
import pandas as pd


def spxw(min_vol: int = 0, D: int | None = None) -> pd.DataFrame:
    """
    Load a sample SPXW options slice and apply simple filters.

    This helper reads ``spxw.csv`` packaged with :mod:`volkit.dataset`, computes
    calendar days-to-expiry ``D`` and trading-year time-to-expiry ``T`` (using
    252 trading days per year), renames core option columns, and optionally
    filters by minimum *both-sides* volume and/or a specific day-to-expiry.

    Parameters
    ----------
    min_vol : int, default 0
        Minimum per-option volume required **for both** the call and the put
        at a given strike. Rows are kept only if
        ``C_vol >= min_vol`` **and** ``P_vol >= min_vol``.
        Use ``0`` to disable the volume filter.
    D : int or None, default None
        If provided, keep only rows whose computed calendar
        days-to-expiry equals this value. For example, ``D=7`` keeps
        the 7-calendar-day slice only. Use ``None`` to keep all expiries
        in the file.

    Returns
    -------
    pandas.DataFrame
        A tidy DataFrame with one row per strike and the following columns:

        - ``T`` : float, time to expiry in trading years (``D / 252``)
        - ``D`` : int, calendar days to expiry (``ExpDate - Date``)
        - ``K`` : float, strike
        - ``C_bid``, ``C_ask``, ``C_vol`` : call bid/ask/volume
        - ``P_bid``, ``P_ask``, ``P_vol`` : put bid/ask/volume

    Notes
    -----
    - ``T`` uses **252** trading days/year to align with options practice.
      If you prefer ACT/365 or ACT/ACT, convert after loading.
    - The function expects a CSV named ``spxw.csv`` to be present in the
      same directory as this module. If it is missing, ``pandas.read_csv``
      will raise ``FileNotFoundError``.
    - Date columns (``Date``, ``ExpDate``) are parsed as datetimes.

    Examples
    --------
    Load everything:

    >>> from volkit.dataset import spxw
    >>> df = spxw()
    >>> df.head()

    Keep only options with at least 100 contracts traded on both sides:

    >>> df = spxw(min_vol=100)

    Work with a single tenor (e.g., 7 days to expiry):

    >>> df_7d = spxw(min_vol=50, D=7)

    Group by time-to-expiry and inspect strikes:

    >>> df.groupby('D')['K'].agg(['min','max','count'])
    """
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "spxw.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date", "ExpDate"])
    df["D"] = (df["ExpDate"] - df["Date"]).dt.days
    df["T"] = df["D"] / 252
    df = df.rename(
        columns={
            "Strike": "K",
            "CallBid": "C_bid",
            "CallAsk": "C_ask",
            "CallVolume": "C_vol",
            "PutBid": "P_bid",
            "PutAsk": "P_ask",
            "PutVolume": "P_vol",
        }
    )

    # Volume filter (both sides must meet the threshold)
    if min_vol > 0:
        df = df[df["C_vol"] >= min_vol]
        df = df[df["P_vol"] >= min_vol]

    # Specific day filter
    if D is not None:
        df = df[df["D"] == D]

    return df[["T", "D", "K", "C_bid", "C_ask", "C_vol", "P_bid", "P_ask", "P_vol"]]
