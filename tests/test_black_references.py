# tests/test_black_references.py
import numpy as np
import pytest

from volkit import price_euro_future

# Each case: (label, F, K, T, r, sigma, cp, expected_price, atol)
# Sources are cited per-case in comments.

CASES = [
    # 1) Hull Example 16.7 (call) — F=620, K=600, r=5%, T=0.5, sigma=20% -> ~44.19
    # Source: Hull example reproduced in university slides.
    # https://math.uni.lu/haas/teaching/17-18_FinMath/Handout16-2.pdf (Example showing 44.19)
    ("Hull_16_7_call", 620.0, 600.0, 0.5, 0.05, 0.20, +1, 44.19, 1e-2),

    # 2) MATLAB blkprice example (call) — F=20, K=20, r=9%, T=4/12, sigma=25% -> 1.1166
    # 3) MATLAB blkprice example (put)  — same params -> 1.1166
    # Source: https://www.mathworks.com/help/finance/blkprice.html
    ("MATLAB_call", 20.0, 20.0, 4.0/12.0, 0.09, 0.25, +1, 1.1166, 1e-4),
    ("MATLAB_put",  20.0, 20.0, 4.0/12.0, 0.09, 0.25, -1, 1.1166, 1e-4),

    # 4) py_vollib doctest (discounted Black) — F=100, K=100, r=2%, T=0.5, sigma=20% -> 5.5811067246
    # Source: https://vollib.org/documentation/1.0.3/autoapi/py_vollib/ref_python/black/index.html
    ("py_vollib_call_par", 100.0, 100.0, 0.5, 0.02, 0.20, +1, 5.581106724604812, 1e-10),

    # 5) blackscholes library docs (call) — F=55, K=50, r=0.25%, T=1, sigma=15% -> 6.2345
    # 6) blackscholes library docs (put)  — same params -> 1.2470
    # Source: https://blackscholes.github.io/binomial-pricing/2_price_calculation/
    ("bs_lib_call", 55.0, 50.0, 1.0, 0.0025, 0.15, +1, 6.2345, 5e-4),
    ("bs_lib_put",  55.0, 50.0, 1.0, 0.0025, 0.15, -1, 1.2470, 5e-4),

]

@pytest.mark.parametrize(
    "label,F,K,T,r,sigma,cp,expected,atol",
    CASES,
)
def test_black_price_matches_references(label, F, K, T, r, sigma, cp, expected, atol):
    """
    Sanity check: price_euro_future reproduces well-known Black-76 prices from
    textbooks and widely used libraries.
    """
    out = price_euro_future(F, K, T, r, sigma, cp)
    # 1) Correct shape (scalar in, scalar out)
    assert np.asarray(out).shape == ()
    # 2) Value against reference
    np.testing.assert_allclose(float(out), float(expected), rtol=0.0, atol=atol)
