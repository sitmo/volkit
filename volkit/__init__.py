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

from .estimate.future_from_option_quotes import estimate_future_from_option_quotes
from .estimate.future_from_option_prices import estimate_future_from_option_prices
from .estimate.future_res import ImpliedFutureResult

from .estimate.vol_from_option_prices import estimate_vol_from_option_prices

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
    "estimate_vol_from_option_prices",
    "estimate_future_from_option_quotes",
    "estimate_future_from_option_prices",
    "ImpliedFutureResult",
    "discount_to_rate",
    "rate_to_discount",
]

__version__ = "0.2.1"