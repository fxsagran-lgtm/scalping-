"""Command line runner.

  python -m scalper.cli fetch    --months 6
  python -m scalper.cli backtest --months 6 --preset all
  python -m scalper.cli signal
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtest import compare_presets, run_experiment, verdict
from .config import PRESETS, CostConfig, with_overrides
from .data import DataUnavailableError, load_or_fetch
from .report import metrics_table, write_report
from .signals import latest_signal

SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _start(months: float) -> pd.Timestamp:
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30.4 * months)).normalize()


def _load(args):
    data, banners = {}, []
    for s in args.symbols:
        cd = {}
        for tf in ("5m", "15m"):
            c = load_or_fetch(s, tf, _start(args.months), source=args.source, cache_dir=args.cache, offline=args.offline)
            banners.append(c.banner())
            print(c.banner())
            cd[tf] = c.df
        data[s] = (cd["5m"], cd["15m"])
    return data, banners


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scalper")
    ap.add_argument("cmd", choices=["fetch", "backtest", "signal"])
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--months", type=float, default=6)
    ap.add_argument("--source", default="binance_vision", choices=["binance_vision", "binance", "coinbase"])
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--offline", action="store_true", help="use cached real candles only (labelled as cached)")
    ap.add_argument("--preset", default="all", help="strict | balanced | active | all")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    costs = CostConfig(a.fee_bps, a.slip_bps)
    try:
        data, banners = _load(a)
    except DataUnavailableError as e:
        print(f"DATA ERROR (no synthetic data was substituted): {e}", file=sys.stderr)
        return 2
    if a.cmd == "fetch":
        return 0
    if a.cmd == "signal":
        cfg = PRESETS["balanced" if a.preset == "all" else a.preset]
        for s, (d5, d15) in data.items():
            sig = latest_signal(s, d5, d15, cfg, costs, data_source=a.source)
            print(f"\n{s}: {sig.action}  entry={sig.entry} stop={sig.stop} targets={sig.targets}")
            for r in sig.reasons:
                print("   -", r)
        return 0
    prov = {"banners": banners, "source": a.source, "months_requested": a.months, "symbols": a.symbols}
    if a.preset == "all":
        comp = compare_presets(data, costs, train_frac=a.train_frac)
        print("\nPreset comparison (COMBINED, parameters chosen on TRAIN only):")
        print(comp["table"].round(3).to_string(index=False))
        chosen = comp["selected_on_train"]
        print(f"\nSelected on train: {chosen or 'NONE - no preset was profitable in the train segment'}")
        exp = comp["runs"][chosen or "balanced"]
        path = write_report(a.out, exp, prov, comp)
    else:
        exp = run_experiment(data, PRESETS[a.preset], costs, a.train_frac)
        path = write_report(a.out, exp, prov)
    print("\n" + metrics_table(exp).round(3).to_string(index=False))
    key = "COMBINED" if "COMBINED" in exp["results"] else a.symbols[0]
    print("\nVERDICT:", verdict(exp["results"][key]["test"]["metrics"]))
    print("Report written to", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
