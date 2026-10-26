"""Paper trading: simulated fills only. Nothing here talks to an exchange account."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .backtest import simulate_trade
from .config import CostConfig, StrategyConfig
from .safety import LIVE_TRADING_ENABLED
from .signals import FIVE, Signal


class PaperBroker:
    def __init__(self, path: str, cfg: StrategyConfig, costs: CostConfig):
        assert LIVE_TRADING_ENABLED is False
        self.path, self.cfg, self.costs = Path(path), cfg, costs
        self.state = {"initial_equity": cfg.initial_equity, "positions": []}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())

    # ---------- persistence ----------
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, default=str))

    def reset(self, initial_equity: Optional[float] = None):
        self.state = {"initial_equity": initial_equity or self.cfg.initial_equity, "positions": []}
        self.save()

    # ---------- account ----------
    @property
    def positions(self) -> List[dict]:
        return self.state["positions"]

    def closed(self) -> List[dict]:
        return [p for p in self.positions if p["status"] == "closed"]

    def equity(self) -> float:
        return self.state["initial_equity"] + sum(p["pnl"] for p in self.closed())

    def _equity_at(self, ts: pd.Timestamp) -> float:
        return self.state["initial_equity"] + sum(p["pnl"] for p in self.closed() if pd.Timestamp(p["exit_time"]) <= ts)

    # ---------- actions ----------
    def open_from_signal(self, sig: Signal) -> bool:
        """Register a paper position for a fresh BUY/SELL signal. Fills at the next candle open."""
        if sig.action not in ("BUY", "SELL") or sig.stop is None or sig.atr is None:
            return False
        pid = f"{sig.symbol}|{sig.candle_open_utc}|{sig.action}"
        if any(p["id"] == pid for p in self.positions):
            return False
        if any(p["symbol"] == sig.symbol and p["status"] in ("pending", "open") for p in self.positions):
            return False  # one position per symbol, same as the backtest
        self.positions.append({"id": pid, "symbol": sig.symbol, "side": sig.action, "signal_bar": sig.candle_open_utc,
                               "stop": sig.stop, "atr": sig.atr, "status": "pending", "reasons": sig.reasons})
        self.save()
        return True

    def update(self, symbol: str, df5: pd.DataFrame) -> int:
        """Advance this symbol's paper positions using closed 5m candles. Returns number that changed."""
        changed = 0
        idx = df5.index
        arr = [df5[k].to_numpy(float) for k in ("open", "high", "low", "close")]
        for p in self.positions:
            if p["symbol"] != symbol or p["status"] == "closed":
                continue
            entry_open = pd.Timestamp(p["signal_bar"]) + FIVE
            if entry_open not in idx:
                continue  # next candle has not closed yet
            e = idx.get_loc(entry_open)
            res = simulate_trade(*arr, e, p["side"], p["stop"], p["atr"], self.cfg, self.costs)
            before = p["status"]
            if res["status"] == "skipped":
                p.update(status="closed", exit_reason="entry_invalidated", pnl=0.0, exit_time=str(entry_open), r_net=0.0)
            else:
                if "qty" not in p:
                    eq = max(self._equity_at(entry_open), 0.0)
                    p["qty"] = min(eq * self.cfg.risk_pct / 100 / res["risk_per_unit"], eq * self.cfg.max_leverage / res["fill"])
                    p["entry_time"], p["fill"] = str(entry_open), res["fill"]
                if res["status"] == "closed":
                    p.update(status="closed", exit_reason=res["exit_reason"], exit_time=str(idx[res["exit_idx"]] + FIVE),
                             pnl=p["qty"] * res["net_unit"], fees=p["qty"] * res["fees_unit"], r_net=res["r_net"])
                else:
                    p.update(status="open", stop_now=res["stop_now"], tp_hit=res["tp_hit"], targets=res["tps"])
            changed += before != p["status"]
        if changed:
            self.save()
        return changed
