# tests/test_loader_spxw.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the module under test
from volkit.datasets import loader_spxw


def _make_csv(tmp_path: Path) -> Path:
    """
    Create a temporary data/spxw.csv next to a fake module file and
    monkeypatch loader_spxw.__file__ to point at that fake module.
    Returns the path to the fake module file's folder (so tests can patch __file__).
    """
    # Directory that will impersonate the package folder containing loader_spxw.py
    pkg_dir = tmp_path / "volkit" / "dataset"
    data_dir = pkg_dir / "data"
    data_dir.mkdir(parents=True)

    # Create a CSV with multiple cases:
    # A: D=7, both vols >=100  -> included when min_vol>=100 and D=7
    # B: D=7, P_vol<100        -> excluded by volume filter but present with min_vol=0
    # C: D=10, both vols >=100 -> included when min_vol>=100 and D=None (or D=10)
    # D: D=7, C_vol<100        -> excluded by volume filter
    rows = [
        # Date,       ExpDate,    Strike, CallBid, CallAsk, CallVolume, PutBid, PutAsk, PutVolume
        ["2025-01-01", "2025-01-08", 4700, 10.0, 10.5, 100, 9.5, 10.0, 100],  # A
        ["2025-01-01", "2025-01-08", 4725, 7.0, 7.5, 150, 6.0, 6.5, 90],  # B
        ["2025-01-01", "2025-01-11", 4750, 5.0, 5.5, 200, 5.0, 5.5, 200],  # C
        ["2025-01-01", "2025-01-08", 4775, 3.0, 3.5, 20, 3.0, 3.5, 500],  # D
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "ExpDate",
            "Strike",
            "CallBid",
            "CallAsk",
            "CallVolume",
            "PutBid",
            "PutAsk",
            "PutVolume",
        ],
    )
    df.to_csv(data_dir / "spxw.csv", index=False)

    # Put a fake file so pkg_dir exists on disk; __file__ will be patched to this.
    fake_module_file = pkg_dir / "loader_spxw.py"
    fake_module_file.write_text(
        "# fake placeholder; real module is imported from site-packages\n"
    )

    return fake_module_file


@pytest.fixture
def patch_module_file(tmp_path, monkeypatch):
    """
    Fixture that creates a temp CSV next to a fake module file and points
    loader_spxw.__file__ to that location so the function under test reads our CSV.
    """
    fake_module_file = _make_csv(tmp_path)
    # Patch the module's __file__ so: here = os.path.dirname(__file__)
    monkeypatch.setattr(loader_spxw, "__file__", str(fake_module_file))
    return fake_module_file.parent  # the directory acting as volkit/dataset/


def test_load_no_filters_returns_expected_columns_and_values(patch_module_file):
    # No filters -> all rows present
    df = loader_spxw.spxw()

    # Exact column order
    assert list(df.columns) == [
        "T",
        "D",
        "K",
        "C_bid",
        "C_ask",
        "C_vol",
        "P_bid",
        "P_ask",
        "P_vol",
    ]

    # Shape: 4 rows, 9 columns
    assert df.shape == (4, 9)

    # Renaming worked
    assert set(df["K"].tolist()) == {4700, 4725, 4750, 4775}

    # D computed as calendar days; T = D / 252
    # Expect D values {7, 7, 10, 7}
    assert df["D"].tolist().count(7) == 3
    assert df["D"].tolist().count(10) == 1

    # Check T numerically for both 7 and 10 day rows
    t_for_7 = df.loc[df["D"] == 7, "T"].iloc[0]
    t_for_10 = df.loc[df["D"] == 10, "T"].iloc[0]
    assert np.isclose(t_for_7, 7 / 252)
    assert np.isclose(t_for_10, 10 / 252)

    # Basic sanity on a known row (strike 4700)
    row_4700 = df.loc[df["K"] == 4700].iloc[0]
    assert np.isclose(row_4700["C_bid"], 10.0)
    assert np.isclose(row_4700["P_ask"], 10.0)
    assert row_4700["C_vol"] == 100 and row_4700["P_vol"] == 100


def test_volume_filter_requires_both_sides(patch_module_file):
    # min_vol=100 -> only rows where C_vol >= 100 AND P_vol >= 100 stay
    df = loader_spxw.spxw(min_vol=100)

    # Only rows A (4700, D=7) and C (4750, D=10) remain
    assert set(df["K"].tolist()) == {4700, 4750}
    # Ensure B (P_vol=90) and D (C_vol=20) are filtered out
    assert 4725 not in df["K"].tolist()
    assert 4775 not in df["K"].tolist()

    # D and T still correct after filtering
    assert set(df["D"].tolist()) == {7, 10}
    assert np.isclose(df.loc[df["K"] == 4700, "T"].item(), 7 / 252)
    assert np.isclose(df.loc[df["K"] == 4750, "T"].item(), 10 / 252)


def test_day_filter_keeps_only_requested_tenor(patch_module_file):
    # D=7 -> keep only rows with 7 calendar days to expiry
    df = loader_spxw.spxw(D=7)

    assert df["D"].nunique() == 1
    assert df["D"].iloc[0] == 7
    # Should have 3 strikes: 4700, 4725, 4775
    assert set(df["K"].tolist()) == {4700, 4725, 4775}
    # T matches 7/252
    assert np.allclose(df["T"].values, np.full(len(df), 7 / 252))


def test_combined_filters_can_yield_empty(patch_module_file):
    # Ask for D=7 but also min_vol=200; among D=7 rows none has both vols >=200
    df = loader_spxw.spxw(min_vol=200, D=7)
    assert df.empty


def test_return_column_order_is_stable(patch_module_file):
    # Using any path should preserve the exact output column order
    df = loader_spxw.spxw(min_vol=0, D=None)
    expected = ["T", "D", "K", "C_bid", "C_ask", "C_vol", "P_bid", "P_ask", "P_vol"]
    assert list(df.columns) == expected
