from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((ad == 0) & (au > 0), 100.0)
    out = out.mask((ad == 0) & (au == 0), 50.0)
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at 00:00 UTC. Cumulative, so it only uses past data."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    day = df.index.normalize()
    cum_pv = (tp * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0.0, np.nan)


def confirmed_swings(high: pd.Series, low: pd.Series, n: int) -> pd.DataFrame:
    """Swing highs/lows that are only *known* n bars after they occur (no look-ahead).

    A pivot at bar i needs n lower highs on each side, so it is confirmed at bar i+n.
    Returns the last and previous confirmed pivot prices as of each bar.
    """
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    N = len(h)
    cols = {k: np.full(N, np.nan) for k in ("sh_last", "sh_prev", "sl_last", "sl_prev")}
    shl = shp = sll = slp = np.nan
    for j in range(N):
        i = j - n
        if i >= n:
            if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:j + 1].max():
                shp, shl = shl, h[i]
            if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:j + 1].min():
                slp, sll = sll, l[i]
        cols["sh_last"][j], cols["sh_prev"][j] = shl, shp
        cols["sl_last"][j], cols["sl_prev"][j] = sll, slp
    return pd.DataFrame(cols, index=high.index)
