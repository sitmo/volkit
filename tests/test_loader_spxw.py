# tests/test_loader_spxw.py

import pandas as pd

# Import the module under test
from volkit.datasets import spxw


def test_spxw_loading():
    df = spxw()
    assert len(df) > 0
    assert isinstance(df, pd.DataFrame)


def test_spxw_loading_vol_filter():
    df = spxw(min_vol=100)
    assert len(df) > 0
    assert isinstance(df, pd.DataFrame)


def test_spxw_loading_d_filter():
    df = spxw(D=2)
    assert len(df) > 0
    assert isinstance(df, pd.DataFrame)
