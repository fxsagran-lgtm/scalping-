"""Hard safety switch. This module exists so that nothing can place real orders.

There is intentionally no exchange credential handling anywhere in this package and
all network access is read-only public market data (HTTP GET).
"""

LIVE_TRADING_ENABLED = False


class LiveTradingDisabledError(RuntimeError):
    pass


def place_real_order(*_args, **_kwargs):
    """Always raises. Kept so any future caller fails loudly instead of trading."""
    raise LiveTradingDisabledError(
        "Real exchange orders are disabled in this project (analysis and paper trading only)."
    )
