# tests/test_loader_spxw.py
import os
import pandas as pd

from volkit.datasets import spxw
from volkit.datasets import loader_spxw as _mod


def _get_sample_csv_path() -> str:
    here = os.path.dirname(_mod.__file__)
    return os.path.join(here, "data", "spxw20190626.csv")


def test_spxw_defaults_compact_schema():
    df = spxw()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Compact (default) schema
    expected = {
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
    }
    assert expected.issubset(df.columns)
    # Dates/sizes/OI are NOT in compact by default
    assert "quote_date" not in df.columns
    assert "C_bid_size" not in df.columns
    assert "C_oi" not in df.columns


def test_spxw_full_schema():
    df = spxw(full=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Compact + full extras
    expected = {
        # compact
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
        # extras
        "quote_date",
        "expiration_date",
        "C_bid_size",
        "C_ask_size",
        "P_bid_size",
        "P_ask_size",
        "C_oi",
        "P_oi",
    }
    assert expected.issubset(df.columns)


def test_spxw_min_volume_branch_off_and_on():
    # Branch: min_volume == 0 (no filtering)
    df0 = spxw(min_volume=0)
    assert len(df0) > 0

    # Branch: min_volume > 0 (filter executes)
    df1 = spxw(min_volume=1)
    assert isinstance(df1, pd.DataFrame)

    # Same schema (compact)
    assert set(df0.columns) == set(df1.columns)


def test_spxw_D_filter_valid_and_invalid():
    # Use an existing D to ensure non-empty result
    df_all = spxw()
    assert len(df_all) > 0
    some_D = int(df_all["D"].iloc[0])

    df_D = spxw(D=some_D)
    assert len(df_D) > 0
    assert df_D["D"].nunique() == 1
    assert int(df_D["D"].iloc[0]) == some_D

    # Invalid D → empty DataFrame (still with compact columns)
    df_empty = spxw(D=-10_000)
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0
    assert list(df_empty.columns) == [
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
