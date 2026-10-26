"""Event-driven backtest on closed candles.

Conventions (all conservative):
* Signal decided at candle close; entry = NEXT candle open (+ slippage against you).
* If a candle touches both the stop and a target, the STOP is assumed to hit first.
* After TP1 the stop moves to break-even for the NEXT candle (optional).
* Stops / time-exits pay slippage; limit targets do not. Fees are charged on every fill.
* One open position per symbol. Trades still open when data ends are excluded and counted.
"""
from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CostConfig, StrategyConfig, PRESETS
from .signals import FIVE, build_features, generate_signals


def simulate_trade(o, h, l, c, e: int, side: str, stop: float, atr_sig: float,
                   cfg: StrategyConfig, costs: CostConfig) -> dict:
    s = 1 if side == "BUY" else -1
    N = len(o)
    slip, fee = costs.slippage_bps_per_side / 1e4, costs.fee_bps_per_side / 1e4
    ideal_in = float(o[e])
    fill = ideal_in * (1 + s * slip)
    risk = s * (fill - stop)
    if not (risk > 0 and cfg.min_stop_atr <= risk / atr_sig <= cfg.max_stop_atr * 1.25
            and costs.round_trip_frac * fill <= cfg.max_cost_r * 1.25 * risk):
        return {"status": "skipped", "reason": "entry_invalidated"}
    tps = [fill + s * m * risk for m in cfg.tp_r]
    fr = cfg.tp_fractions
    stop_lvl, remaining, be_next = stop, 1.0, False
    legs: List[tuple] = []          # (fraction, ideal_price, fill_price, kind)
    done = [False] * 3
    last_k = min(N - 1, e + cfg.max_hold_bars - 1)
    exit_k, reason = None, None
    for k in range(e, last_k + 1):
        ok_, hk, lk, ck = o[k], h[k], l[k], c[k]
        if k > e and s * (ok_ - stop_lvl) <= 0:                       # gapped through the stop
            legs.append((remaining, ok_, ok_ * (1 - s * slip), "stop_gap"))
            remaining, exit_k, reason = 0.0, k, "stop_gap"
            break
        if (lk <= stop_lvl) if s == 1 else (hk >= stop_lvl):          # stop checked BEFORE targets
            kind = "be_stop" if done[0] else "stop"
            legs.append((remaining, stop_lvl, stop_lvl * (1 - s * slip), kind))
            remaining, exit_k, reason = 0.0, k, kind
            break
        for j in range(3):
            if done[j]:
                continue
            if (hk >= tps[j]) if s == 1 else (lk <= tps[j]):
                legs.append((fr[j], tps[j], tps[j], f"tp{j + 1}"))
                done[j] = True
                remaining -= fr[j]
                if j == 0 and cfg.move_to_be_after_tp1:
                    be_next = True
            else:
                break
        if remaining <= 1e-9:
            remaining, exit_k, reason = 0.0, k, "tp3"
            break
        if be_next:
            stop_lvl, be_next = fill, False
        if k == last_k and (k - e + 1) >= cfg.max_hold_bars:
            legs.append((remaining, ck, ck * (1 - s * slip), "time"))
            remaining, exit_k, reason = 0.0, k, "time"
            break
    if exit_k is None:
        return {"status": "open", "entry_idx": e, "fill": fill, "stop": stop, "stop_now": stop_lvl,
                "risk_per_unit": risk, "ideal_entry": ideal_in, "tps": tps, "tp_hit": list(done),
                "remaining": remaining, "side": side, "legs": legs, "last_idx": last_k}
    ideal_move = s * sum(f_ * (p - ideal_in) for f_, p, _, _ in legs)
    real_move = s * sum(f_ * (q - fill) for f_, _, q, _ in legs)
    fees = fee * (fill + sum(f_ * q for f_, _, q, _ in legs))
    net = real_move - fees
    return {
        "status": "closed", "entry_idx": e, "exit_idx": exit_k, "exit_reason": reason,
        "tp_hits": int(sum(done)), "ideal_entry": ideal_in, "fill": fill, "stop": stop,
        "risk_per_unit": risk, "gross_unit": ideal_move, "slip_unit": ideal_move - real_move,
        "fees_unit": fees, "net_unit": net, "r_gross": ideal_move / risk, "r_net": net / risk,
        "hold_bars": exit_k - e + 1, "legs": legs,
    }


def backtest_symbol(symbol: str, df5: pd.DataFrame, df15: pd.DataFrame, cfg: StrategyConfig,
                    costs: CostConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Counter]:
    """Returns (accepted signals, unsized trades, skip counters) for the whole period."""
    f = build_features(df5, df15, cfg)
    cand = generate_signals(f, cfg, costs, symbol)
    sig = cand[cand["accepted"]].reset_index(drop=True)
    o, h, l, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pos = {ts: i for i, ts in enumerate(f.index)}
    trades, skipped, busy_until = [], Counter(), -1
    for r in sig.itertuples():
        e = pos[r.bar_open] + 1
        if e >= len(f):
            skipped["no_next_candle"] += 1
            continue
        if e <= busy_until:
            skipped["position_already_open"] += 1
            continue
        res = simulate_trade(o, h, l, c, e, r.side, r.stop, r.atr, cfg, costs)
        if res["status"] == "skipped":
            skipped[res["reason"]] += 1
            continue
        if res["status"] == "open":
            skipped["unfinished_at_data_end"] += 1
            continue
        busy_until = res["exit_idx"]
        trades.append({
            "symbol": symbol, "side": r.side, "signal_bar": r.bar_open,
            "entry_time": f.index[e], "exit_time": f.index[res["exit_idx"]] + FIVE,
            **{k: res[k] for k in ("ideal_entry", "fill", "stop", "risk_per_unit", "exit_reason", "tp_hits",
                                   "gross_unit", "slip_unit", "fees_unit", "net_unit", "r_gross", "r_net", "hold_bars")},
            "reasons": r.reasons,
        })
    return sig, pd.DataFrame(trades), skipped


def apply_sizing(trades: pd.DataFrame, initial_equity: float, risk_pct: float, max_leverage: float) -> pd.DataFrame:
    """Chronological sizing: risk risk_pct of realised equity at entry, capped at max_leverage x equity."""
    if trades.empty:
        return trades.assign(qty=[], notional=[], pnl=[], fees=[], slippage=[], gross_pnl=[])
    t = trades.sort_values("entry_time").reset_index(drop=True)
    pending, realised = [], 0.0
    q, pnl, fees, slip, gross, notional = [], [], [], [], [], []
    for i, r in t.iterrows():
        while pending and pending[0][0] <= r.entry_time:
            realised += heapq.heappop(pending)[2]
        eq = max(initial_equity + realised, 0.0)
        qty = min(eq * risk_pct / 100.0 / r.risk_per_unit, eq * max_leverage / r.fill)
        p = qty * r.net_unit
        q.append(qty); notional.append(qty * r.fill); pnl.append(p)
        fees.append(qty * r.fees_unit); slip.append(qty * r.slip_unit); gross.append(qty * r.gross_unit)
        heapq.heappush(pending, (r.exit_time, i, p))
    t["qty"], t["notional"], t["pnl"], t["fees"], t["slippage"], t["gross_pnl"] = q, notional, pnl, fees, slip, gross
    return t


def compute_metrics(trades: pd.DataFrame, n_signals: int, days: float, initial_equity: float) -> dict:
    days = max(days, 1e-9)
    m = {"days": round(days, 2), "setups": int(n_signals), "setups_per_day": n_signals / days,
         "total_trades": int(len(trades)), "trades_per_day": len(trades) / days}
    if trades.empty:
        m.update(win_rate=np.nan, net_pnl=0.0, net_return_pct=0.0, profit_factor=np.nan, max_drawdown_pct=0.0,
                 max_drawdown_usd=0.0, avg_r=np.nan, avg_r_gross=np.nan, fees=0.0, slippage=0.0, gross_pnl=0.0,
                 longs=0, shorts=0, avg_hold_min=np.nan)
        return m
    pnl = trades["pnl"]
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = initial_equity + trades.sort_values("exit_time")["pnl"].cumsum()
    eq = pd.concat([pd.Series([initial_equity]), eq.reset_index(drop=True)])
    peak = eq.cummax()
    dd = (eq - peak)
    m.update(
        win_rate=float((pnl > 0).mean()), net_pnl=float(pnl.sum()), net_return_pct=float(pnl.sum() / initial_equity * 100),
        profit_factor=float(gw / gl) if gl > 0 else (float("inf") if gw > 0 else float("nan")),
        max_drawdown_pct=float((dd / peak).min() * 100), max_drawdown_usd=float(dd.min()),
        avg_r=float(trades["r_net"].mean()), avg_r_gross=float(trades["r_gross"].mean()),
        fees=float(trades["fees"].sum()), slippage=float(trades["slippage"].sum()), gross_pnl=float(trades["gross_pnl"].sum()),
        longs=int((trades["side"] == "BUY").sum()), shorts=int((trades["side"] == "SELL").sum()),
        avg_hold_min=float(trades["hold_bars"].mean() * 5),
    )
    return m


def equity_curve(trades: pd.DataFrame, initial_equity: float) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.sort_values("exit_time")
    return pd.Series((initial_equity + t["pnl"].cumsum()).to_numpy(), index=t["exit_time"].to_numpy(), name="equity")


def verdict(m: dict, min_trades: int = 30) -> str:
    if m["total_trades"] < min_trades:
        return f"INCONCLUSIVE: only {m['total_trades']} trades (need >= {min_trades}) - no conclusion about profitability."
    if m["net_pnl"] <= 0:
        return (f"LOSES MONEY out-of-sample after fees and slippage: net {m['net_pnl']:+,.2f} "
                f"({m['net_return_pct']:+.2f}%), profit factor {m['profit_factor']:.2f}, avg {m['avg_r']:+.3f}R/trade.")
    if m["profit_factor"] < 1.15:
        return "MARGINAL: slightly positive after costs, but too thin to distinguish from noise or cost-model error."
    return ("POSITIVE in this sample after costs. This is a historical result on limited data, not evidence of future profit; "
            "confirm with paper trading before trusting it.")


def run_experiment(data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], cfg: StrategyConfig, costs: CostConfig,
                   train_frac: float = 0.6) -> dict:
    """data: {symbol: (df5, df15)}. Parameters are fixed BEFORE looking at the test segment."""
    per = {}
    for sym, (d5, d15) in data.items():
        sig, tr, sk = backtest_symbol(sym, d5, d15, cfg, costs)
        per[sym] = (sig, tr, sk, d5.index[cfg.warmup_bars], d5.index[-1] + FIVE)
    start = max(v[3] for v in per.values())
    end = min(v[4] for v in per.values())
    cutoff = start + (end - start) * train_frac
    segs = {"train": (start, cutoff), "test": (cutoff, end), "full": (start, end)}
    sets = {s: [s] for s in data}
    if len(data) > 1:
        sets["COMBINED"] = list(data)
    out = {"config": asdict(cfg), "costs": asdict(costs), "cutoff": str(cutoff), "start": str(start), "end": str(end),
           "train_frac": train_frac, "results": {}, "skipped": {s: dict(v[2]) for s, v in per.items()}}
    for name, syms in sets.items():
        out["results"][name] = {}
        for seg, (a, b) in segs.items():
            sigs = [per[s][0][(per[s][0]["bar_open"] >= a) & (per[s][0]["bar_open"] < b)] for s in syms]
            trs = [per[s][1][(per[s][1]["signal_bar"] >= a) & (per[s][1]["signal_bar"] < b)] for s in syms if not per[s][1].empty]
            tr = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame()
            tr = apply_sizing(tr, cfg.initial_equity, cfg.risk_pct, cfg.max_leverage) if not tr.empty else tr
            m = compute_metrics(tr, sum(len(x) for x in sigs), (b - a).total_seconds() / 86400, cfg.initial_equity)
            out["results"][name][seg] = {"metrics": m, "trades": tr, "signals": pd.concat(sigs, ignore_index=True)}
    return out


def compare_presets(data, costs: CostConfig, presets: Optional[dict] = None, train_frac: float = 0.6,
                    min_train_trades: int = 30) -> dict:
    """Run every preset; pick the winner using the TRAIN segment only; report its TEST result."""
    presets = presets or PRESETS
    runs = {n: run_experiment(data, p, costs, train_frac) for n, p in presets.items()}
    key = "COMBINED" if "COMBINED" in next(iter(runs.values()))["results"] else next(iter(next(iter(runs.values()))["results"]))
    rows = []
    for n, r in runs.items():
        tr, te = r["results"][key]["train"]["metrics"], r["results"][key]["test"]["metrics"]
        rows.append({"preset": n, "train_setups_day": tr["setups_per_day"], "train_trades": tr["total_trades"],
                     "train_net": tr["net_pnl"], "train_pf": tr["profit_factor"], "train_maxdd_pct": tr["max_drawdown_pct"],
                     "test_setups_day": te["setups_per_day"], "test_trades": te["total_trades"], "test_net": te["net_pnl"],
                     "test_pf": te["profit_factor"], "test_maxdd_pct": te["max_drawdown_pct"], "test_avg_r": te["avg_r"],
                     "test_win_rate": te["win_rate"]})
    table = pd.DataFrame(rows)
    elig = table[(table["train_trades"] >= min_train_trades) & (table["train_net"] > 0)]
    selected = None if elig.empty else elig.sort_values("train_pf", ascending=False).iloc[0]["preset"]
    return {"runs": runs, "table": table, "selected_on_train": selected, "key": key}
