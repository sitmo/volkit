# volkit/datasets/loader_spxw.py
from typing import Optional
import os
import pandas as pd


def spxw(
    min_volume: int = 0,
    D: Optional[int] = None,
    *,
    full: bool = False,
) -> pd.DataFrame:
    """
    Load a sample SPXW options slice and apply simple filters.

    Parameters
    ----------
    min_volume : int, default 0
        Minimum per-option trade volume required **for both** the call and the put
        at a given strike. Rows are kept only if
        ``C_vol >= min_volume`` **and** ``P_vol >= min_volume``.
        Use ``0`` to disable the volume filter.
    D : int or None, default None
        If provided, keep only rows whose computed calendar days-to-expiry equals this value.
        For example, ``D=7`` keeps only the 7-calendar-day slice. Use ``None`` to keep all expiries.
    full : bool, default False
        Select between a compact, most-relevant schema (False) and a full schema (True).

        When ``full=False`` (default), return:
          - ``["K", "D", "T"]`` — strike and days/time to expiry
          - ``["F_bid", "F_ask"]`` — underlying future quotes
          - ``["C_bid", "C_ask", "P_bid", "P_ask"]`` — call/put quotes
          - ``["C_vol", "P_vol"]`` — call/put traded volume (day)

        When ``full=True``, additionally include:
          - ``["quote_date", "expiration_date"]`` — actual datetimes
          - ``["C_bid_size", "C_ask_size", "P_bid_size", "P_ask_size"]`` — order book sizes
          - ``["C_oi", "P_oi"]`` — open interest

    Returns
    -------
    pandas.DataFrame
        One row per strike with at least: ``K``, ``D``, ``T`` and the selected extra columns.

    Notes
    -----
    - ``T`` uses **252** trading days/year to align with options practice.
      If you prefer ACT/365 or ACT/ACT, convert after loading.
    - Date columns (``quote_date``, ``expiration_date``) are parsed as datetimes.

    Examples
    --------
    Load defaults (compact schema):
    >>> df = spxw()

    Keep only options with at least 100 contracts traded on both sides:
    >>> df = spxw(min_volume=100)

    Work with a single tenor (e.g., 7 days to expiry):
    >>> df_7 = spxw(D=7)

    Get the full column set:
    >>> df_full = spxw(full=True)
    """
    # columns to pivot and how they should be renamed
    rename_map = {
        "bid_size_1545": "bid_size",
        "bid_1545": "bid",
        "ask_size_1545": "ask_size",
        "ask_1545": "ask",
        "trade_volume": "vol",
        "open_interest": "oi",
    }
    idx = ["quote_date", "expiration", "strike"]

    # Locate CSV
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "data", "spxw20190626.csv")

    df = pd.read_csv(data_path, parse_dates=["quote_date", "expiration"])

    # Wide pivot: values × option_type -> single level columns like C_bid, P_ask_size
    pt = df.pivot_table(
        index=idx,
        columns="option_type",
        values=list(rename_map.keys()),
        aggfunc="first",  # if duplicates exist, take first
    ).sort_index(axis=1)

    # flatten MultiIndex columns: ('bid_1545','C') -> 'C_bid'
    pt.columns = [
        f"{opt}_{rename_map[val]}" for (val, opt) in pt.columns.to_flat_index()
    ]
    pt = pt.reset_index()

    # Keep the non-pivot columns (take the first if duplicated across C/P rows)
    nonpivot = [
        "quote_date",
        "expiration",
        "strike",
        "underlying_bid_1545",
        "underlying_ask_1545",
    ]
    base = df[nonpivot].drop_duplicates(subset=idx).groupby(idx, as_index=False).first()

    # Merge
    out = base.merge(pt, on=idx, how="left")

    # Rename to final schema
    out = out.rename(
        columns={
            "strike": "K",
            "underlying_bid_1545": "F_bid",
            "underlying_ask_1545": "F_ask",
            "expiration": "expiration_date",
        }
    )

    # Days and time to expiry
    out["D"] = (out["expiration_date"] - out["quote_date"]).dt.days
    out["T"] = out["D"] / 252.0

    # Row filters
    if min_volume > 0:
        out = out[(out["C_vol"] >= min_volume) & (out["P_vol"] >= min_volume)]

    if D is not None:
        out = out[out["D"] == D]

    # Column selection ---------------------------------------------------------
    cols = [
        "K",
        "D",
        "T",
        "F_bid",
        "F_ask",
        "C_bid",
        "C_ask",
        "P_bid",
        "P_ask",
        "C_vol",
        "P_vol",
    ]
    if full:
        cols += [
            "quote_date",
            "expiration_date",
            "C_bid_size",
            "C_ask_size",
            "P_bid_size",
            "P_ask_size",
            "C_oi",
            "P_oi",
        ]

    # Be forgiving if a column is missing in a custom CSV
    cols = [c for c in cols if c in out.columns]

    # Stable ordering for reproducibility (these columns always exist in `out`)
    out = out.sort_values(["quote_date", "expiration_date", "K"]).reset_index(drop=True)

    return out[cols]
