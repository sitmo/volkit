# volkit/datasets/loader_spxw.py
from typing import Optional
import os
import pandas as pd

def spxw(min_vol: int = 0, D: Optional[int] = None) -> pd.DataFrame:
    """
    Load a sample SPXW options slice and apply simple filters.

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
        A  DataFrame with one row per strike and the following columns:
        - ``quote_date`` : Quote snapshot data (at 12:45 during live sesion)
        - ``expiration_date`` : Option Expiration date
        - ``D`` : int, calendar days to expiry (``ExpDate - Date``)
        - ``T`` : float, time to expiry in trading years (``D / 252``)
        - ``K`` : float, strike
        - ``F_bid``, ``F_ask`` : underlying future bid/ask
        - ``C_bid``, ``C_ask``, ``C_vol``, ``C_oi`` : call bid/ask/volume/open interest
        - ``P_bid``, ``P_ask``, ``P_vol``, ``P_oi`` : put bid/ask/volume/open interest

    Notes
    -----
    - ``T`` uses **252** trading days/year to align with options practice.
      If you prefer ACT/365 or ACT/ACT, convert after loading.
    - Date columns (``quote_date``, ``expiration_date``) are parsed as datetimes.

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
    # columns to pivot and how they should be renamed
    rename_map = {
        "bid_size_1545": "bid_size",
        "bid_1545": "bid",
        "ask_size_1545": "ask_size",
        "ask_1545": "ask",
        "trade_volume": "vol",
        "open_interest": "oi"
    }

    idx = ["quote_date", "expiration", "strike"]

    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "data/spxw20190626.csv")

    df = pd.read_csv(
        csv_path,
        parse_dates=['quote_date', 'expiration']
    )

    # Wide pivot: values × option_type -> single level columns like C_bid, P_ask_size
    pt = (
        df.pivot_table(
            index=idx,
            columns="option_type",
            values=list(rename_map.keys()),
            aggfunc="first",  # or 'last' / np.nanmean etc., if duplicates exist
        )
        .sort_index(axis=1)  # optional
    )

    # flatten MultiIndex columns: ('bid_1545','C') -> 'C_bid'
    pt.columns = [
        f"{opt}_{rename_map[val]}"
        for (val, opt) in pt.columns.to_flat_index()
    ]

    pt = pt.reset_index()

    # Keep the non-pivot columns (take first if duplicated across C/P rows)
    nonpivot = [
        "quote_date", "expiration", "strike",
        "underlying_bid_1545", "underlying_ask_1545",
    ]
    base = (
        df[nonpivot]
        .drop_duplicates(subset=idx)
        .groupby(idx, as_index=False)
        .first()
    )

    # Merge and (optionally) order columns
    out = base.merge(pt, on=idx, how="left")


    out = out.rename(
        columns={
            "strike": "K",
            "underlying_bid_1545": "F_bid",
            "underlying_ask_1545": "F_ask",
            "trade_volume": "vol",
            "PutBid": "P_bid",
            "PutAsk": "P_ask",
            "PutVolume": "P_vol",
            "expiration": "expiration_date"
        }
    )
    out["D"] = (out["expiration_date"] - out["quote_date"]).dt.days
    out["T"] = out["D"] / 252

    if min_vol > 0:
        out = out[out["C_vol"] >= min_vol]
        out = out[out["P_vol"] >= min_vol]

    # Specific day filter
    if D is not None:
        out = out[out["D"] == D]    
    return out

