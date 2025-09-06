# volkit/__init__.py

from .pricing.future import (
    price_euro_future,
    delta_euro_future,
    gamma_euro_future,
    vega_euro_future,
    theta_euro_future,
    rho_euro_future,
    dual_delta_euro_future,
    vanna_euro_future,
    vomma_euro_future,
    lambda_euro_future,
)

from .implied.future import implied_future_from_option_quotes

from .implied.vol_future import implied_vol_euro_future

from .utils import discount_to_rate, rate_to_discount

__all__ = [
    "parse_cp",
    "price_euro_future",
    "delta_euro_future",
    "gamma_euro_future",
    "vega_euro_future",
    "theta_euro_future",
    "rho_euro_future",
    "dual_delta_euro_future",
    "vanna_euro_future",
    "vomma_euro_future",
    "lambda_euro_future",
    "implied_vol_euro_future",
    "implied_future_from_option_quotes",
    "discount_to_rate",
    "rate_to_discount",
]

__version__ = "0.1.4"
