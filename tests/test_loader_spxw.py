# tests/test_loader_spxw.py
import os
import pandas as pd

# Import the function + module to locate the packaged CSV
from volkit.datasets import spxw
from volkit.datasets import loader_spxw as _mod


def _get_sample_csv_path() -> str:
    here = os.path.dirname(_mod.__file__)
    return os.path.join(here, "data", "spxw20190626.csv")


def test_spxw_defaults_loads_dataframe():
    df = spxw()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Defaults include dates, underlying, quotes, volumes (sizes & OI off)
    expected_subset = {"K", "D", "T", "quote_date", "expiration_date",
                       "F_bid", "F_ask", "C_bid", "C_ask", "P_bid", "P_ask",
                       "C_vol", "P_vol"}
    assert expected_subset.issubset(df.columns)


def test_spxw_min_volume_branch_off_and_on():
    # Branch: min_volume == 0 (no filter)
    df0 = spxw(min_volume=0)
    assert isinstance(df0, pd.DataFrame)
    assert len(df0) > 0

    # Branch: min_volume > 0 (filter executes even if it doesn't reduce rows)
    df1 = spxw(min_volume=1)  # execute the filtering branch
    assert isinstance(df1, pd.DataFrame)
    # Same schema regardless of reduction
    assert set(df0.columns) == set(df1.columns)


def test_spxw_D_filter_present_and_absent():
    # Pick a D that actually exists to avoid brittle tests
    df_all = spxw()
    assert len(df_all) > 0
    some_D = int(df_all["D"].iloc[0])

    # Branch: D filter keeps only that tenor
    df_D = spxw(D=some_D)
    assert len(df_D) > 0
    assert df_D["D"].nunique() == 1
    assert int(df_D["D"].iloc[0]) == some_D

    # Branch: D filter that matches nothing -> empty DataFrame,
    # still with the selected columns (here: minimal schema)
    df_empty = spxw(
        D=-123456,  # unrealistic D to ensure empty
        include_dates=False,
        include_underlying=False,
        include_quotes=False,
        include_sizes=False,
        include_open_interest=False,
        include_volumes=False,
    )
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0
    assert list(df_empty.columns) == ["K", "D", "T"]


def test_column_flags_minimal_and_combinations():
    # Minimal: only K, D, T
    df_min = spxw(
        include_dates=False,
        include_underlying=False,
        include_quotes=False,
        include_sizes=False,
        include_open_interest=False,
        include_volumes=False,
    )
    assert list(df_min.columns) == ["K", "D", "T"]

    # Include only dates
    df_dates = spxw(
        include_dates=True,
        include_underlying=False,
        include_quotes=False,
        include_sizes=False,
        include_open_interest=False,
        include_volumes=False,
    )
    assert list(df_dates.columns) == ["K", "D", "T", "quote_date", "expiration_date"]

    # Include only underlying
    df_under = spxw(
        include_dates=False,
        include_underlying=True,
        include_quotes=False,
        include_sizes=False,
        include_open_interest=False,
        include_volumes=False,
    )
    assert list(df_under.columns) == ["K", "D", "T", "F_bid", "F_ask"]

    # Include only quotes
    df_quotes = spxw(
        include_dates=False,
        include_underlying=False,
        include_quotes=True,
        include_sizes=False,
        include_open_interest=False,
        include_volumes=False,
    )
    assert list(df_quotes.columns) == ["K", "D", "T", "C_bid", "C_ask", "P_bid", "P_ask"]

    # Include sizes (bid & ask sizes for both C and P)
    df_sizes = spxw(
        include_dates=False,
        include_underlying=False,
        include_quotes=False,
        include_sizes=True,
        include_open_interest=False,
        include_volumes=False,
    )
    for col in ["C_bid_size", "C_ask_size", "P_bid_size", "P_ask_size"]:
        assert col in df_sizes.columns

    # Include open interest
    df_oi = spxw(
        include_dates=False,
        include_underlying=False,
        include_quotes=False,
        include_sizes=False,
        include_open_interest=True,
        include_volumes=False,
    )
    for col in ["C_oi", "P_oi"]:
        assert col in df_oi.columns

    # Include only volumes
    df_vols = spxw(
        include_dates=False,
        include_underlying=False,
        include_quotes=False,
        include_sizes=False,
        include_open_interest=False,
        include_volumes=True,
    )
    assert list(df_vols.columns) == ["K", "D", "T", "C_vol", "P_vol"]


def test_data_path_override_works():
    csv_path = _get_sample_csv_path()
    assert os.path.exists(csv_path), "Packaged sample CSV not found"
    df = spxw(data_path=csv_path)  # explicit override
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
