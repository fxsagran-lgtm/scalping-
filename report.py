from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import verdict


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o)) else ("inf" if np.isinf(o) else float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


def metrics_table(exp: dict) -> pd.DataFrame:
    rows = []
    for name, segs in exp["results"].items():
        for seg, r in segs.items():
            m = r["metrics"]
            rows.append({"set": name, "segment": seg, "setups/day": m["setups_per_day"], "trades": m["total_trades"],
                         "win %": m["win_rate"] * 100 if m["win_rate"] == m["win_rate"] else np.nan,
                         "net P/L": m["net_pnl"], "PF": m["profit_factor"], "max DD %": m["max_drawdown_pct"],
                         "avg R": m["avg_r"], "fees": m["fees"], "slippage": m["slippage"]})
    return pd.DataFrame(rows)


def write_report(outdir: str, exp: dict, provenance: dict, comparison: dict = None) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {"generated_utc": stamp, "data_provenance": provenance,
               "config": exp["config"], "costs": exp["costs"], "period": [exp["start"], exp["end"]],
               "train_test_cutoff": exp["cutoff"], "skipped_signals": exp["skipped"],
               "metrics": {n: {s: r["metrics"] for s, r in segs.items()} for n, segs in exp["results"].items()}}
    if comparison:
        payload["preset_comparison"] = comparison["table"].to_dict("records")
        payload["selected_on_train"] = comparison["selected_on_train"]
    (out / f"backtest_{stamp}.json").write_text(json.dumps(_clean(payload), indent=2))
    for n, segs in exp["results"].items():
        for s, r in segs.items():
            if s == "full" and not r["trades"].empty:
                r["trades"].to_csv(out / f"trades_{n}_{stamp}.csv", index=False)
    t = metrics_table(exp)
    lines = [f"# Backtest {stamp}", "", "## Data", *[f"- {v}" for v in provenance.get("banners", [])], "",
             f"Train/test cutoff: {exp['cutoff']} (parameters fixed before the test segment).", "",
             "## Results", t.round(3).to_markdown(index=False) if hasattr(t, "to_markdown") else t.round(3).to_string(index=False), "",
             "## Verdict (out-of-sample, combined)" if "COMBINED" in exp["results"] else "## Verdict (out-of-sample)",
             verdict(exp["results"][next(reversed(exp["results"]))]["test"]["metrics"])]
    (out / f"backtest_{stamp}.md").write_text("\n".join(lines))
    return out / f"backtest_{stamp}.md"
