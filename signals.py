"""Signal engine. Everything is computed from CLOSED candles only.

A signal is decided at the close of a 5m candle; the entry is the NEXT candle's open.
15m trend data is merged by candle *close* time, so a 15m candle is never used before it has closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CostConfig, StrategyConfig
from .data import utc_index
from .indicators import atr, confirmed_swings, ema, rsi, vwap_daily

FIVE = pd.Timedelta(minutes=5)
FIFTEEN = pd.Timedelta(minutes=15)


def trend_frame(df15: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    d = df15[["open", "high", "low", "close", "volume"]].astype(float).copy()
    d.index = utc_index(d.index)
    t = pd.DataFrame(index=d.index)
    t["ema20"] = ema(d["close"], cfg.ema_fast)
    t["ema50"] = ema(d["close"], cfg.ema_slow)
    t["atr"] = atr(d, cfg.atr_len)
    sw = confirmed_swings(d["high"], d["low"], cfg.swing_n_15m)
    hh_hl = (sw.sh_last > sw.sh_prev) & (sw.sl_last > sw.sl_prev)
    lh_ll = (sw.sh_last < sw.sh_prev) & (sw.sl_last < sw.sl_prev)
    gap = (t["ema20"] - t["ema50"]) / t["atr"]
    slope = t["ema50"] - t["ema50"].shift(cfg.trend_slope_bars)
    bull = (t["ema20"] > t["ema50"]) & (d["close"] > t["ema50"]) & (gap >= cfg.min_trend_gap_atr)
    bear = (t["ema20"] < t["ema50"]) & (d["close"] < t["ema50"]) & (-gap >= cfg.min_trend_gap_atr)
    if cfg.require_15m_slope:
        bull &= slope > 0
        bear &= slope < 0
    if cfg.require_15m_structure:
        bull &= hh_hl
        bear &= lh_ll
    t["bull"], t["bear"] = bull.fillna(False), bear.fillna(False)
    t["gap_atr"] = gap
    t["hh_hl"], t["lh_ll"] = hh_hl, lh_ll
    t["available_at"] = t.index + FIFTEEN  # known only once the 15m candle has CLOSED
    return t


def build_features(df5: pd.DataFrame, df15: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    f = df5[["open", "high", "low", "close", "volume"]].astype(float).copy()
    f.index = utc_index(f.index)
    f["ema20"] = ema(f["close"], cfg.ema_fast)
    f["ema50"] = ema(f["close"], cfg.ema_slow)
    f["rsi"] = rsi(f["close"], cfg.rsi_len)
    f["atr"] = atr(f, cfg.atr_len)
    f["atr_pct"] = f["atr"] / f["close"] * 100.0
    f["vwap"] = vwap_daily(f)
    sw = confirmed_swings(f["high"], f["low"], cfg.swing_n_5m)
    f["sh5"], f["sl5"] = sw["sh_last"], sw["sl_last"]

    t = trend_frame(df15, cfg).reset_index(drop=True).sort_values("available_at")
    left = pd.DataFrame({"close_time": (f.index + FIVE).as_unit("ns")})
    m = pd.merge_asof(
        left, t, left_on="close_time", right_on="available_at", direction="backward",
        tolerance=pd.Timedelta(minutes=cfg.trend_max_age_min),
    )
    f["t_bull"] = m["bull"].eq(True).to_numpy()
    f["t_bear"] = m["bear"].eq(True).to_numpy()
    f["t_gap_atr"] = m["gap_atr"].to_numpy()
    f["t_hh_hl"] = m["hh_hl"].eq(True).to_numpy()
    f["t_lh_ll"] = m["lh_ll"].eq(True).to_numpy()
    f["t_ema20"], f["t_ema50"] = m["ema20"].to_numpy(), m["ema50"].to_numpy()
    return f


def evaluate_conditions(f: pd.DataFrame, cfg: StrategyConfig, costs: CostConfig) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    c, o, h, l, a = f["close"], f["open"], f["high"], f["low"], f["atr"]
    yes = pd.Series(True, index=f.index)
    session = pd.Series(f.index.hour, index=f.index).isin(cfg.allowed_hours_utc) if cfg.allowed_hours_utc else yes
    valid = f[["ema20", "ema50", "rsi", "atr", "vwap"]].notna().all(axis=1) & (f["volume"] > 0)
    vol_ok = f["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    rt = costs.round_trip_frac
    out = {}

    # ---------------- BUY: bullish pullback ----------------
    stop = l.rolling(cfg.stop_lookback).min() - cfg.stop_buffer_atr * a
    risk = c - stop
    touched = (l <= f["ema20"] + cfg.touch_atr * a).astype(float).rolling(cfg.pullback_lookback).max() > 0
    conds = pd.DataFrame({
        "data_valid": valid,
        "trend_15m": f["t_bull"],
        "pullback_touch": touched,
        "reclaim_ema20": c > f["ema20"],
        "confirm_candle": (c > o) if cfg.require_confirm_candle else yes,
        "break_prev_extreme": (c > h.shift(1)) if cfg.require_break_prev_extreme else yes,
        "vwap_side": (c > f["vwap"]) if cfg.require_vwap_side else yes,
        "ema_order_5m": (f["ema20"] > f["ema50"]) if cfg.require_5m_ema_order else yes,
        "rsi_band": f["rsi"].between(cfg.rsi_pullback_low, cfg.rsi_pullback_high),
        "rsi_turning": (f["rsi"] > f["rsi"].shift(1)) if cfg.require_rsi_turn else yes,
        "not_extended": (c - f["ema20"]) <= cfg.max_extension_atr * a,
        "volatility_ok": vol_ok,
        "stop_distance_ok": (risk > 0) & (risk / a).between(cfg.min_stop_atr, cfg.max_stop_atr),
        "costs_ok": (risk > 0) & (rt * c <= cfg.max_cost_r * risk),
        "room_to_target": f["sh5"].isna() | (f["sh5"] <= c) | ((f["sh5"] - c) >= cfg.min_room_r * risk),
        "session_ok": session,
    }).fillna(False).astype(bool)
    out["BUY"] = (conds, stop, risk)

    # ---------------- SELL: bearish pullback (mirror) ----------------
    stop_s = h.rolling(cfg.stop_lookback).max() + cfg.stop_buffer_atr * a
    risk_s = stop_s - c
    touched_s = (h >= f["ema20"] - cfg.touch_atr * a).astype(float).rolling(cfg.pullback_lookback).max() > 0
    lo, hi = 100.0 - cfg.rsi_pullback_high, 100.0 - cfg.rsi_pullback_low
    conds_s = pd.DataFrame({
        "data_valid": valid,
        "trend_15m": f["t_bear"],
        "pullback_touch": touched_s,
        "reclaim_ema20": c < f["ema20"],
        "confirm_candle": (c < o) if cfg.require_confirm_candle else yes,
        "break_prev_extreme": (c < l.shift(1)) if cfg.require_break_prev_extreme else yes,
        "vwap_side": (c < f["vwap"]) if cfg.require_vwap_side else yes,
        "ema_order_5m": (f["ema20"] < f["ema50"]) if cfg.require_5m_ema_order else yes,
        "rsi_band": f["rsi"].between(lo, hi),
        "rsi_turning": (f["rsi"] < f["rsi"].shift(1)) if cfg.require_rsi_turn else yes,
        "not_extended": (f["ema20"] - c) <= cfg.max_extension_atr * a,
        "volatility_ok": vol_ok,
        "stop_distance_ok": (risk_s > 0) & (risk_s / a).between(cfg.min_stop_atr, cfg.max_stop_atr),
        "costs_ok": (risk_s > 0) & (rt * c <= cfg.max_cost_r * risk_s),
        "room_to_target": f["sl5"].isna() | (f["sl5"] >= c) | ((c - f["sl5"]) >= cfg.min_room_r * risk_s),
        "session_ok": session,
    }).fillna(False).astype(bool)
    out["SELL"] = (conds_s, stop_s, risk_s)
    return out


PASS_TEXT = {
    "trend_15m": "15m trend aligned (EMA20/EMA50 order, price vs EMA50, slope, market structure)",
    "pullback_touch": "5m price pulled back to the EMA20 zone recently",
    "reclaim_ema20": "5m candle closed back on the trend side of EMA20",
    "confirm_candle": "confirmation candle in trend direction",
    "break_prev_extreme": "close broke the previous candle's high/low",
    "vwap_side": "price on the trend side of VWAP",
    "ema_order_5m": "5m EMA20 vs EMA50 agrees with trend",
    "rsi_band": "RSI(14) in pullback band",
    "rsi_turning": "RSI turning back in trend direction",
    "not_extended": "entry not stretched away from EMA20",
    "volatility_ok": "ATR % within allowed volatility range",
    "stop_distance_ok": "stop distance within configured ATR bounds",
    "costs_ok": "fees + slippage small relative to risk",
    "room_to_target": "room before the nearest 5m swing level",
    "session_ok": "inside allowed trading hours",
    "data_valid": "indicators warmed up",
}


def _reasons(side: str, row: pd.Series, conds_row: pd.Series, risk: float, cfg: StrategyConfig, costs: CostConfig) -> List[str]:
    out = []
    for k, ok in conds_row.items():
        if ok:
            out.append(PASS_TEXT.get(k, k))
    out.append(f"RSI {row['rsi']:.1f}, ATR {row['atr_pct']:.3f}% of price, 15m EMA gap {row['t_gap_atr']:.2f} ATR")
    out.append(f"round-trip costs ≈ {costs.round_trip_frac * row['close'] / risk:.2f}R")
    return out


def generate_signals(f: pd.DataFrame, cfg: StrategyConfig, costs: CostConfig, symbol: str) -> pd.DataFrame:
    """All rule-valid candidates, with cooldown and daily cap applied (causally)."""
    ev = evaluate_conditions(f, cfg, costs)
    idx = f.index
    cand = []
    for side, (conds, stop, risk) in ev.items():
        ok = conds.all(axis=1)
        ok.iloc[: cfg.warmup_bars] = False
        for pos in np.flatnonzero(ok.to_numpy()):
            cand.append((int(pos), side))
    cand.sort()
    rows, last_pos, per_day = [], -10**9, {}
    for pos, side in cand:
        ts = idx[pos]
        day = (ts + FIVE).date()
        reason = ""
        if pos - last_pos < cfg.cooldown_bars:
            reason = "cooldown"
        elif per_day.get((symbol, day), 0) >= cfg.max_signals_per_symbol_day:
            reason = "daily_cap"
        accepted = reason == ""
        if accepted:
            last_pos = pos
            per_day[(symbol, day)] = per_day.get((symbol, day), 0) + 1
        conds, stop, risk = ev[side]
        r = float(risk.iloc[pos])
        entry = float(f["close"].iloc[pos])
        sgn = 1 if side == "BUY" else -1
        rows.append({
            "symbol": symbol, "bar_open": ts, "decision_time": ts + FIVE, "side": side,
            "entry_ref": entry, "stop": float(stop.iloc[pos]), "risk": r, "atr": float(f["atr"].iloc[pos]),
            **{f"tp{i + 1}": entry + sgn * m * r for i, m in enumerate(cfg.tp_r)},
            "accepted": accepted, "suppressed_reason": reason,
            "reasons": " | ".join(_reasons(side, f.iloc[pos], conds.iloc[pos], r, cfg, costs)),
        })
    cols = ["symbol", "bar_open", "decision_time", "side", "entry_ref", "stop", "risk", "atr", "tp1", "tp2", "tp3",
            "accepted", "suppressed_reason", "reasons"]
    out = pd.DataFrame(rows, columns=cols)
    out["accepted"] = out["accepted"].astype(bool)
    return out


def filter_funnel(f: pd.DataFrame, cfg: StrategyConfig, costs: CostConfig) -> pd.DataFrame:
    """Which filters bind? Per-condition pass rate and cumulative survivors (post-warmup bars only)."""
    ev = evaluate_conditions(f, cfg, costs)
    rows = []
    for side, (conds, _, _) in ev.items():
        c = conds.iloc[cfg.warmup_bars:]
        alive = pd.Series(True, index=c.index)
        for k in c.columns:
            alive &= c[k]
            rows.append({"side": side, "filter": k, "pass_rate_%": 100 * c[k].mean(), "cumulative_survivors": int(alive.sum())})
    return pd.DataFrame(rows)


@dataclass
class Signal:
    symbol: str
    action: str                      # BUY / SELL / WAIT
    candle_open_utc: Optional[str]
    decision_time_utc: Optional[str]
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: Dict[str, float] = field(default_factory=dict)
    risk_reward: str = ""
    cost_in_r: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    atr: Optional[float] = None
    data_source: str = ""
    latest_candle_open_utc: str = ""
    data_age_minutes: float = float("nan")
    note: str = "Entry = next 5m candle open; entry shown is the last close and will differ. Analysis only."

    def to_dict(self):
        return asdict(self)


def latest_signal(symbol: str, df5: pd.DataFrame, df15: pd.DataFrame, cfg: StrategyConfig, costs: CostConfig,
                  now: Optional[pd.Timestamp] = None, data_source: str = "") -> Signal:
    now = now or pd.Timestamp.now(tz="UTC")
    f = build_features(df5, df15, cfg)
    last = f.index[-1]
    close_time = last + FIVE
    age = (now - close_time).total_seconds() / 60.0
    base = dict(symbol=symbol, candle_open_utc=str(last), decision_time_utc=str(close_time),
                data_source=data_source, latest_candle_open_utc=str(last), data_age_minutes=round(age, 1))
    if age > cfg.max_data_age_minutes:
        return Signal(action="WAIT", reasons=[f"Data is stale: latest closed 5m candle is {age:.0f} min old "
                                              f"(limit {cfg.max_data_age_minutes}). No signal while data is stale."], **base)
    if len(f) < cfg.warmup_bars:
        return Signal(action="WAIT", reasons=["Not enough history to warm up indicators."], **base)
    cands = generate_signals(f, cfg, costs, symbol)
    hit = cands[cands["bar_open"] == last]
    if not hit.empty:
        r = hit.iloc[0]
        if r["accepted"]:
            rr = " / ".join(f"1:{m:g}" for m in cfg.tp_r)
            return Signal(
                action=r["side"], entry=r["entry_ref"], stop=r["stop"],
                targets={f"{m:g}R": r[f"tp{i + 1}"] for i, m in enumerate(cfg.tp_r)},
                risk_reward=rr, cost_in_r=round(costs.round_trip_frac * r["entry_ref"] / r["risk"], 3),
                reasons=r["reasons"].split(" | "), atr=r["atr"], **base)
        return Signal(action="WAIT", reasons=[f"Setup present but suppressed ({r['suppressed_reason']}) to avoid over-trading."], **base)
    # WAIT: explain what is missing for the side the 15m trend points to
    ev = evaluate_conditions(f, cfg, costs)
    row = f.iloc[-1]
    if row["t_bull"]:
        side = "BUY"
    elif row["t_bear"]:
        side = "SELL"
    else:
        return Signal(action="WAIT", reasons=["15m trend is not clean (EMA order / slope / structure disagree) - no pullback trade."], **base)
    conds = ev[side][0].iloc[-1]
    missing = [PASS_TEXT.get(k, k) for k, ok in conds.items() if not ok]
    return Signal(action="WAIT", reasons=[f"15m trend favours {side}, but waiting for: " + "; ".join(missing)], **base)
