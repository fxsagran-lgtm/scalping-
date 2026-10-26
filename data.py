"""Real public market data only. No synthetic data, no silent fallback between sources.

All requests are read-only HTTP GETs to public market-data endpoints.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

OHLCV = ["open", "high", "low", "close", "volume"]
INTERVAL_SECONDS = {"5m": 300, "15m": 900}

SOURCES = {
    # Public market-data mirror of Binance spot (works in many regions where api.binance.com returns 451).
    "binance_vision": {"label": "Binance spot (data-api.binance.vision)", "quote": "USDT"},
    "binance": {"label": "Binance spot (api.binance.com)", "quote": "USDT"},
    # Coinbase quotes BTC-USD / ETH-USD (USD, NOT USDT). Only used if you pick it explicitly.
    "coinbase": {"label": "Coinbase Exchange (USD pairs, not USDT)", "quote": "USD"},
}
_BINANCE_HOST = {
    "binance_vision": "https://data-api.binance.vision",
    "binance": "https://api.binance.com",
}
_COINBASE_HOST = "https://api.exchange.coinbase.com"


class DataUnavailableError(RuntimeError):
    """Raised when real data cannot be fetched. The app shows this; it never substitutes fake data."""


class NoNewCandlesError(DataUnavailableError):
    """The request succeeded but the exchange returned no (new) closed candles."""


@dataclass
class CandleData:
    df: pd.DataFrame
    source: str
    symbol: str
    timeframe: str
    fetched_at_utc: Optional[pd.Timestamp]
    from_cache: bool = False
    note: str = ""
    gaps: List[dict] = field(default_factory=list)

    @property
    def latest_open_utc(self) -> pd.Timestamp:
        return self.df.index[-1]

    @property
    def latest_close_utc(self) -> pd.Timestamp:
        return self.df.index[-1] + pd.Timedelta(seconds=INTERVAL_SECONDS[self.timeframe])

    def age_minutes(self, now: Optional[pd.Timestamp] = None) -> float:
        now = now or pd.Timestamp.now(tz="UTC")
        return (now - self.latest_close_utc).total_seconds() / 60.0

    def banner(self, now: Optional[pd.Timestamp] = None) -> str:
        s = SOURCES.get(self.source, {}).get("label", self.source)
        fetched = self.fetched_at_utc.strftime("%Y-%m-%d %H:%M UTC") if self.fetched_at_utc is not None else "n/a"
        txt = (
            f"{self.symbol} {self.timeframe} | source: {s} | candles: {len(self.df):,} | "
            f"latest closed candle opened {self.latest_open_utc:%Y-%m-%d %H:%M} UTC "
            f"({self.age_minutes(now):.1f} min old) | last fetched {fetched}"
        )
        if self.from_cache:
            txt += " | CACHED (no network refresh)"
        if self.gaps:
            txt += f" | {sum(g['missing'] for g in self.gaps)} missing candles in {len(self.gaps)} gap(s)"
        return txt + (f" | {self.note}" if self.note else "")


def utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.as_unit("ns")


def find_gaps(df: pd.DataFrame, timeframe: str) -> List[dict]:
    step = pd.Timedelta(seconds=INTERVAL_SECONDS[timeframe])
    d = df.index.to_series().diff()
    bad = d[d > step]
    return [
        {"after": str(ts - dd), "resumes": str(ts), "missing": int(dd / step) - 1}
        for ts, dd in bad.items()
    ]


def _clean(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=OHLCV)
    return df


def _get_json(session, url, params, retries=3, backoff=1.5):
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=20)
        except requests.RequestException as e:  # network failure
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:160]}"
            if r.status_code in (400, 401, 403, 404, 451):
                break  # retrying will not help
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    hint = " (HTTP 451/403 usually means this region is blocked; pick another source explicitly.)" if last and ("451" in last or "403" in last) else ""
    raise DataUnavailableError(f"Could not fetch {url}: {last}{hint}")


def _binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _coinbase_product(symbol: str) -> str:
    base = symbol.split("/")[0].upper()
    return f"{base}-USD"


def fetch_candles(symbol: str, timeframe: str, start: pd.Timestamp, end: Optional[pd.Timestamp] = None,
                  source: str = "binance_vision", session=None, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Fetch closed candles in [start, end). The still-forming candle is always dropped."""
    if timeframe not in INTERVAL_SECONDS:
        raise ValueError(f"unsupported timeframe {timeframe}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source}")
    session = session or requests.Session()
    step = INTERVAL_SECONDS[timeframe]
    now = now or pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    end = end or now
    rows = []
    if source in _BINANCE_HOST:
        url = _BINANCE_HOST[source] + "/api/v3/klines"
        cur = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        while cur < end_ms:
            data = _get_json(session, url, {"symbol": _binance_symbol(symbol), "interval": timeframe,
                                            "startTime": cur, "endTime": end_ms, "limit": 1000})
            if not data:
                break
            rows.extend([[int(k[0]), *map(float, k[1:6])] for k in data])
            nxt = int(data[-1][0]) + step * 1000
            if nxt <= cur:
                break
            cur = nxt
            if len(data) < 1000:
                break
    else:  # coinbase: max 300 candles per request, newest first
        url = f"{_COINBASE_HOST}/products/{_coinbase_product(symbol)}/candles"
        cur = start
        while cur < end:
            win_end = min(cur + pd.Timedelta(seconds=step * 300), end)
            data = _get_json(session, url, {"granularity": step, "start": cur.isoformat(), "end": win_end.isoformat()})
            for t, lo, hi, op, cl, vol in data:
                rows.append([int(t) * 1000, float(op), float(hi), float(lo), float(cl), float(vol)])
            cur = win_end
            time.sleep(0.12)
    if not rows:
        raise NoNewCandlesError(f"No candles returned for {symbol} {timeframe} from {source}.")
    df = pd.DataFrame(rows, columns=["t"] + OHLCV)
    df.index = utc_index(pd.to_datetime(df.pop("t"), unit="ms", utc=True))
    df.index.name = "open_time"
    df = _clean(df, timeframe)
    # drop the candle that has not closed yet
    closed = df.index + pd.Timedelta(seconds=step) <= now
    return df[closed]


def _cache_paths(cache_dir, source, symbol, timeframe):
    stem = f"{source}__{symbol.replace('/', '')}__{timeframe}"
    d = Path(cache_dir)
    return d / f"{stem}.csv", d / f"{stem}.meta.json"


def load_cached(symbol, timeframe, source, cache_dir="data_cache") -> Optional[CandleData]:
    csv, meta = _cache_paths(cache_dir, source, symbol, timeframe)
    if not csv.exists():
        return None
    df = pd.read_csv(csv, index_col=0)
    df.index = utc_index(pd.to_datetime(df.index, utc=True))
    df.index.name = "open_time"
    fetched = None
    if meta.exists():
        fetched = pd.Timestamp(json.loads(meta.read_text()).get("fetched_at_utc"))
    return CandleData(df, source, symbol, timeframe, fetched, from_cache=True, gaps=find_gaps(df, timeframe))


def load_or_fetch(symbol: str, timeframe: str, start: pd.Timestamp, source: str = "binance_vision",
                  cache_dir: str = "data_cache", offline: bool = False, session=None,
                  now: Optional[pd.Timestamp] = None) -> CandleData:
    """Incrementally refresh a CSV cache with real candles.

    If the network fails this RAISES, unless offline=True was chosen explicitly
    (then the cached real candles are returned and clearly labelled as cached).
    """
    start = pd.Timestamp(start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    cached = load_cached(symbol, timeframe, source, cache_dir)
    if offline:
        if cached is None:
            raise DataUnavailableError(f"Offline mode but no cached {symbol} {timeframe} data from {source}.")
        cached.df = cached.df[cached.df.index >= start]
        if cached.df.empty:
            raise DataUnavailableError("Cached data does not cover the requested range.")
        return cached
    step = pd.Timedelta(seconds=INTERVAL_SECONDS[timeframe])
    if cached is not None and cached.df.index[0] <= start + pd.Timedelta(days=1):
        base = cached.df
        fetch_from = base.index[-1] + step
    else:
        base = None
        fetch_from = start
    try:
        new = fetch_candles(symbol, timeframe, fetch_from, source=source, session=session, now=now)
    except NoNewCandlesError:
        if base is None:
            raise
        new = base.iloc[0:0]  # request succeeded; cache is simply already up to date
    df = pd.concat([base, new]) if base is not None else new
    df = _clean(df, timeframe)
    csv, meta = _cache_paths(cache_dir, source, symbol, timeframe)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv)
    fetched_at = now or pd.Timestamp.now(tz="UTC")
    meta.write_text(json.dumps({"fetched_at_utc": fetched_at.isoformat(), "source": source,
                                "symbol": symbol, "timeframe": timeframe}))
    df = df[df.index >= start]
    return CandleData(df, source, symbol, timeframe, fetched_at, from_cache=False, gaps=find_gaps(df, timeframe))
