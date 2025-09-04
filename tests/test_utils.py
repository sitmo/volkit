# tests/test_iv_and_utils.py
import numpy as np

from volkit.future import _parse_cp, price_euro_future, iv_euro_future


def test_parse_cp_values_and_broadcast():
    cp = _parse_cp(["c", "put", +1, -1, "CALL", "P"])
    np.testing.assert_array_equal(cp, np.array([1, -1, 1, -1, 1, -1]))
    b = _parse_cp("call", target_shape=(2, 3))
    assert b.shape == (2, 3)
    assert np.all(b == 1)

