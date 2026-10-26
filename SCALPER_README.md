# Scalping system add-on (analysis + paper trading only)

Additive: new `scalper/` package, `pages/2_Scalping_System.py` (Streamlit multipage), `tests/`. Your existing dashboard files are not modified.

## Install into your repo (Codespaces terminal, repo root)
1. Copy `scalper/`, `pages/2_Scalping_System.py`, `tests/test_scalper.py`, `tests/__init__.py` (skip if you already have one), `requirements-scalper.txt`, this file.
   - `pages/` must sit NEXT TO your main Streamlit script (the one you pass to `streamlit run`). If you already have a `pages/2_*.py`, rename mine to an unused number.
2. `pip install -r requirements-scalper.txt`
3. `python -m unittest discover -s tests -v`   (all tests must pass)

## Run
- Prefetch real candles (public GET only): `python -m scalper.cli fetch --months 6`
- Backtest (BTC, ETH, combined; train/test split; presets compared on TRAIN only): `python -m scalper.cli backtest --months 6 --preset all`
- Latest signal: `python -m scalper.cli signal`
- Dashboard: `streamlit run <your_main_script>.py` (e.g. `streamlit run app.py`), open the forwarded port 8501; the new page appears in the sidebar as "Scalping System". Your original dashboard remains the main page.
- Region-blocked source (HTTP 451)? Pick another `--source` explicitly (`binance`, `coinbase` = USD pairs, not USDT). It never switches by itself.

## Rules enforced in code
- Signal at 5m close -> entry at NEXT 5m open. 15m data is used only after that 15m candle closes. Swing pivots are confirmed n bars late. A test proves signals don't change when future candles are removed.
- Stop assumed hit before targets if one candle touches both. Fees on every fill; slippage on entries, stops, time exits.
- Exits: 1/3 at 1R, 1/3 at 1.5R, 1/3 at 2R; stop to break-even after 1R; time stop after 4h. 0.5% equity risk/trade, max 1x notional.
- Safety filters (cost <= 0.25R of risk, stop 0.6-2.5 ATR, ATR% band, cooldown, daily cap, data freshness) are identical in every preset and bounded by `SAFETY_LIMITS`; presets only change signal-quality filters.
- `LIVE_TRADING_ENABLED = False`, no credentials, no order endpoints (a test scans the package).
- Reports go to `reports/` with data source, fetch time and latest candle time per series.
