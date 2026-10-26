"""Unit tests. Synthetic candles below are TEST FIXTURES ONLY - the app and CLI never use them."""
import sys, tempfile, unittest, pathlib
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scalper import indicators as ind
from scalper.backtest import (apply_sizing, compute_metrics, run_experiment, simulate_trade, verdict)
from scalper.config import PRESETS, TUNABLE_FIELDS, CostConfig, StrategyConfig, validate, with_overrides
from scalper.data import DataUnavailableError, fetch_candles, load_or_fetch, CandleData
from scalper.paper import PaperBroker
from scalper.safety import LiveTradingDisabledError, place_real_order, LIVE_TRADING_ENABLED
from scalper.signals import FIVE, build_features, evaluate_conditions, generate_signals, latest_signal


def synth(n5=9000, seed=1, start="2026-01-01", drift_cycle=1200):
    rng = np.random.default_rng(seed)
    idx5 = pd.date_range(start, periods=n5, freq="5min", tz="UTC").as_unit("ns")
    regime = np.sin(np.arange(n5) / drift_cycle * 2 * np.pi)
    ret = rng.normal(0, 0.0012, n5) + regime * 0.00025
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0007, n5)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0007, n5)) * close
    d5 = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(1, 10, n5)}, index=idx5)
    d15 = d5.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return d5, d15


D5, D15 = synth()
COSTS = CostConfig(1.0, 0.5)   # LOW-cost fixture so synthetic candles can produce signals; defaults tested separately


class Indicators(unittest.TestCase):
    def test_ema_matches_manual(self):
        s = pd.Series([1.0, 2, 3, 4, 5, 6])
        e = ind.ema(s, 3)
        a, ref = 2 / 4, [1.0]
        for x in s[1:]:
            ref.append(a * x + (1 - a) * ref[-1])
        self.assertAlmostEqual(e.iloc[-1], ref[-1])

    def test_rsi_bounds_and_extremes(self):
        up = pd.Series(np.arange(1, 60, dtype=float))
        self.assertEqual(ind.rsi(up, 14).iloc[-1], 100.0)
        r = ind.rsi(D5["close"], 14).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_atr_constant_range(self):
        df = pd.DataFrame({"high": [11.0] * 40, "low": [9.0] * 40, "close": [10.0] * 40})
        self.assertAlmostEqual(ind.atr(df, 14).iloc[-1], 2.0)

    def test_vwap_resets_daily(self):
        v = ind.vwap_daily(D5)
        firsts = D5.groupby(D5.index.normalize()).head(1).index
        tp = (D5.high + D5.low + D5.close) / 3
        for ts in firsts[1:4]:
            self.assertAlmostEqual(v[ts], tp[ts])        # first bar of a UTC day == its own typical price

    def test_swing_confirmed_only_after_n_bars(self):
        h = pd.Series([1, 2, 3, 9, 3, 2, 1, 2, 3, 2, 1.0]); l = h - 0.5
        sw = ind.confirmed_swings(h, l, 3)
        self.assertTrue(np.isnan(sw["sh_last"].iloc[5]))   # pivot at bar 3 not known until bar 6
        self.assertEqual(sw["sh_last"].iloc[6], 9)


class NoLookAhead(unittest.TestCase):
    def test_features_and_signals_identical_on_truncated_history(self):
        cfg = PRESETS["active"]
        cut = 6000
        d5c = D5.iloc[:cut]
        d15c = D15[D15.index + pd.Timedelta(minutes=15) <= d5c.index[-1] + FIVE]
        full, part = build_features(D5, D15, cfg), build_features(d5c, d15c, cfg)
        cols = ["ema20", "ema50", "rsi", "atr", "vwap", "sh5", "sl5", "t_bull", "t_bear", "t_gap_atr"]
        pd.testing.assert_frame_equal(full.iloc[:cut][cols], part[cols], check_exact=False, rtol=1e-9)
        sf, sp = generate_signals(full, cfg, COSTS, "X"), generate_signals(part, cfg, COSTS, "X")
        a = set(zip(sf[sf.bar_open < part.index[-1]].bar_open, sf[sf.bar_open < part.index[-1]].side))
        b = set(zip(sp[sp.bar_open < part.index[-1]].bar_open, sp[sp.bar_open < part.index[-1]].side))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)


class Exits(unittest.TestCase):
    cfg = StrategyConfig(min_stop_atr=0.5)

    def _run(self, bars, side="BUY", stop=98.0, costs=CostConfig(0, 0), cfg=None):
        a = np.array(bars, float)
        return simulate_trade(a[:, 0], a[:, 1], a[:, 2], a[:, 3], 0, side, stop, 2.0, cfg or self.cfg, costs)

    def test_entry_at_next_open_and_r_multiples(self):
        # entry 100, stop 98 -> risk 2: tp1 102, tp2 103, tp3 104
        r = self._run([[100, 100.5, 99.5, 100], [100, 104.5, 100, 104], ])
        self.assertEqual(r["status"], "closed"); self.assertEqual(r["exit_reason"], "tp3")
        self.assertAlmostEqual(r["r_net"], (1 + 1.5 + 2) / 3)

    def test_stop_first_when_bar_hits_both(self):
        r = self._run([[100, 100.5, 99.5, 100], [100, 105, 97, 100]])
        self.assertEqual(r["exit_reason"], "stop"); self.assertAlmostEqual(r["r_net"], -1.0)

    def test_breakeven_after_tp1(self):
        r = self._run([[100, 102.5, 99.5, 102], [102, 102.2, 99.9, 100]])
        self.assertEqual(r["exit_reason"], "be_stop")
        self.assertAlmostEqual(r["r_net"], 1 / 3)

    def test_short_mirror(self):
        r = self._run([[100, 100.5, 99.5, 100], [100, 100, 95.5, 96]], side="SELL", stop=102.0)
        self.assertEqual(r["exit_reason"], "tp3"); self.assertAlmostEqual(r["r_net"], 1.5)

    def test_cost_math(self):
        cfg = StrategyConfig(max_hold_bars=2, max_cost_r=0.35, min_stop_atr=0.5)
        costs = CostConfig(fee_bps_per_side=10, slippage_bps_per_side=5)
        r = self._run([[100, 100.2, 99.8, 100], [100, 100.2, 99.8, 100]], stop=98.0, costs=costs, cfg=cfg)
        self.assertEqual(r["exit_reason"], "time")
        fill, exit_ = 100 * 1.0005, 100 * 0.9995
        self.assertAlmostEqual(r["net_unit"], (exit_ - fill) - 0.001 * (fill + exit_), places=9)
        self.assertGreater(r["fees_unit"], 0); self.assertGreater(r["slip_unit"], 0)

    def test_gap_invalidates_entry(self):
        r = self._run([[97.5, 100, 97, 99]])
        self.assertEqual(r["status"], "skipped")


class Sizing(unittest.TestCase):
    def test_leverage_cap_and_chronology(self):
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        tr = pd.DataFrame({"entry_time": [t0, t0 + pd.Timedelta(hours=2)], "exit_time": [t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=3)],
                           "fill": [100.0, 100.0], "risk_per_unit": [0.1, 2.0], "net_unit": [-0.1, 1.0], "fees_unit": [0.0] * 2,
                           "slip_unit": [0.0] * 2, "gross_unit": [-0.1, 1.0]})
        s = apply_sizing(tr, 10000, 0.5, 1.0)
        self.assertAlmostEqual(s.notional.iloc[0], 10000.0)            # capped at 1x
        eq2 = 10000 + s.pnl.iloc[0]
        self.assertAlmostEqual(s.qty.iloc[1], eq2 * 0.005 / 2.0)       # uses realised equity at entry

    def test_metrics(self):
        t = pd.DataFrame({"pnl": [100.0, -50.0, -100.0, 30.0], "r_net": [1, -.5, -1, .3], "r_gross": [1.1, -.4, -.9, .4],
                          "exit_time": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"), "side": ["BUY", "SELL", "BUY", "SELL"],
                          "fees": [1.0] * 4, "slippage": [.5] * 4, "gross_pnl": [101.5, -48.5, -98.5, 31.5], "hold_bars": [3] * 4})
        m = compute_metrics(t, 6, 2.0, 10000)
        self.assertAlmostEqual(m["profit_factor"], 130 / 150); self.assertAlmostEqual(m["win_rate"], .5)
        self.assertAlmostEqual(m["max_drawdown_usd"], -150.0); self.assertAlmostEqual(m["setups_per_day"], 3.0)
        self.assertIn("LOSES MONEY", verdict({**m, "total_trades": 50}))


class Experiment(unittest.TestCase):
    def test_chronological_split_and_no_forced_trades(self):
        data = {"A": (D5, D15), "B": synth(seed=7)}
        exp = run_experiment(data, PRESETS["active"], COSTS, 0.6)
        cut = pd.Timestamp(exp["cutoff"])
        for name, segs in exp["results"].items():
            tr, te = segs["train"]["trades"], segs["test"]["trades"]
            if not tr.empty: self.assertTrue((tr.signal_bar < cut).all())
            if not te.empty: self.assertTrue((te.signal_bar >= cut).all())
        self.assertIn("COMBINED", exp["results"])
        flat = pd.DataFrame({"open": 100.0, "high": 100.01, "low": 99.99, "close": 100.0, "volume": 1.0}, index=D5.index[:3000])
        f15 = flat.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        z = run_experiment({"FLAT": (flat, f15)}, PRESETS["active"], COSTS)
        self.assertEqual(z["results"]["FLAT"]["full"]["metrics"]["total_trades"], 0)  # poor conditions -> no trades


class CostFilter(unittest.TestCase):
    def test_default_fees_block_tight_scalps(self):
        """With 0.10%+0.02% per side, round-trip cost is 0.24% of price; a ~0.2%-ATR stop can't pass the 0.25R cost cap."""
        from scalper.signals import filter_funnel
        f = build_features(D5, D15, PRESETS["active"])
        fun = filter_funnel(f, PRESETS["active"], CostConfig())
        buy = fun[(fun.side == "BUY") & (fun["filter"] == "costs_ok")].iloc[0]
        self.assertLess(buy["pass_rate_%"], 5.0)
        self.assertEqual(len(generate_signals(f, PRESETS["active"], CostConfig(), "X").query("accepted")), 0)


class Config(unittest.TestCase):
    def test_safety_limits_enforced(self):
        for bad in (dict(max_cost_r=0.9), dict(min_stop_atr=0.1), dict(max_leverage=3.0), dict(risk_pct=5.0), dict(cooldown_bars=1)):
            with self.assertRaises(ValueError): with_overrides(StrategyConfig(), **bad)

    def test_presets_only_change_tunable_filters(self):
        base = PRESETS["balanced"]
        for p in PRESETS.values():
            for k, v in p.__dict__.items():
                if k not in TUNABLE_FIELDS and k != "name":
                    self.assertEqual(v, getattr(base, k), k)


class Safety(unittest.TestCase):
    def test_live_orders_disabled(self):
        self.assertFalse(LIVE_TRADING_ENABLED)
        with self.assertRaises(LiveTradingDisabledError): place_real_order("BTCUSDT", "BUY", 1)

    def test_no_credentials_or_order_endpoints_in_package(self):
        banned = ["api_key", "apikey", "api_secret", "secret_key", "hmac", "x-mbx-apikey", "/order", "requests.post", ".post("]
        for p in (pathlib.Path(__file__).resolve().parents[1] / "scalper").glob("*.py"):
            txt = p.read_text().lower()
            for b in banned: self.assertNotIn(b, txt, f"{b} found in {p.name}")


def _binance_rows(start_ms, n, step=300000):
    return [[start_ms + i * step, "1", "2", "0.5", "1.5", "10", 0, 0, 0, 0, 0, 0] for i in range(n)]


class DataHonesty(unittest.TestCase):
    def test_network_failure_raises_not_synthetic(self):
        import requests
        sess = mock.Mock(); sess.get.side_effect = requests.ConnectionError("down")
        with mock.patch("scalper.data.time.sleep"):
            with self.assertRaises(DataUnavailableError):
                fetch_candles("BTC/USDT", "5m", pd.Timestamp("2026-01-01", tz="UTC"), session=sess)

    def test_http_451_gives_clear_error(self):
        sess = mock.Mock(); sess.get.return_value = mock.Mock(status_code=451, text="blocked")
        with self.assertRaises(DataUnavailableError) as cm:
            fetch_candles("BTC/USDT", "5m", pd.Timestamp("2026-01-01", tz="UTC"), session=sess)
        self.assertIn("region", str(cm.exception))

    def test_paging_and_drops_forming_candle_and_banner(self):
        start = pd.Timestamp("2026-01-01", tz="UTC"); s_ms = int(start.timestamp() * 1000)
        pages = [mock.Mock(status_code=200, json=lambda: _binance_rows(s_ms, 1000)),
                 mock.Mock(status_code=200, json=lambda: _binance_rows(s_ms + 1000 * 300000, 10))]
        sess = mock.Mock(); sess.get.side_effect = pages
        now = start + pd.Timedelta(minutes=5 * 1009 + 2)   # last candle (index 1009) is still forming
        df = fetch_candles("BTC/USDT", "5m", start, session=sess, now=now)
        self.assertEqual(len(df), 1009); self.assertEqual(sess.get.call_count, 2)
        with tempfile.TemporaryDirectory() as d:
            sess.get.side_effect = pages  # exhausted mocks are fine: new pages
            sess.get.side_effect = [mock.Mock(status_code=200, json=lambda: _binance_rows(s_ms, 10))]
            cd = load_or_fetch("BTC/USDT", "5m", start, cache_dir=d, session=sess, now=start + pd.Timedelta(hours=2))
            b = cd.banner(now=start + pd.Timedelta(hours=2))
            self.assertIn("Binance", b); self.assertIn("2026-01-01 00:45", b)
            with self.assertRaises(DataUnavailableError):      # cache exists, network down -> still raises unless offline=True
                import requests
                sess.get.side_effect = requests.ConnectionError("down")
                with mock.patch("scalper.data.time.sleep"):
                    load_or_fetch("BTC/USDT", "5m", start, cache_dir=d, session=sess, now=start + pd.Timedelta(hours=5))
            off = load_or_fetch("BTC/USDT", "5m", start, cache_dir=d, offline=True)
            self.assertTrue(off.from_cache); self.assertIn("CACHED", off.banner())


class LiveAndPaper(unittest.TestCase):
    def test_stale_data_forces_wait(self):
        s = latest_signal("X", D5, D15, PRESETS["active"], COSTS, now=D5.index[-1] + pd.Timedelta(hours=3))
        self.assertEqual(s.action, "WAIT"); self.assertIn("stale", s.reasons[0].lower())

    def test_signal_fields_present_when_fresh(self):
        cfg = PRESETS["active"]; f = build_features(D5, D15, cfg)
        sg = generate_signals(f, cfg, COSTS, "X"); sg = sg[sg.accepted]
        self.assertGreater(len(sg), 5)
        r = sg.iloc[len(sg) // 2]; k = D5.index.get_loc(r.bar_open)
        s = latest_signal("X", D5.iloc[:k + 1], D15[D15.index + pd.Timedelta(minutes=15) <= D5.index[k] + FIVE], cfg, COSTS,
                          now=D5.index[k] + FIVE + pd.Timedelta(minutes=1))
        self.assertEqual(s.action, r.side)
        self.assertEqual(set(s.targets), {"1R", "1.5R", "2R"}); self.assertGreater(len(s.reasons), 3)
        risk = abs(s.entry - s.stop)
        self.assertAlmostEqual(abs(s.targets["2R"] - s.entry), 2 * risk, places=6)

    def test_paper_roundtrip_persists(self):
        cfg = PRESETS["active"]; f = build_features(D5, D15, cfg)
        sg = generate_signals(f, cfg, COSTS, "X"); sg = sg[sg.accepted]
        r = sg.iloc[len(sg) // 2]; k = D5.index.get_loc(r.bar_open)
        sig = latest_signal("X", D5.iloc[:k + 1], D15[D15.index + pd.Timedelta(minutes=15) <= D5.index[k] + FIVE], cfg, COSTS,
                            now=D5.index[k] + FIVE + pd.Timedelta(minutes=1))
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "paper.json"; b = PaperBroker(str(p), cfg, COSTS)
            self.assertTrue(b.open_from_signal(sig)); self.assertFalse(b.open_from_signal(sig))   # no duplicates
            b.update("X", D5.iloc[:k + 1]); self.assertEqual(b.positions[0]["status"], "pending")  # next candle not closed yet
            b.update("X", D5)
            self.assertIn(b.positions[0]["status"], ("open", "closed"))
            b2 = PaperBroker(str(p), cfg, COSTS); self.assertEqual(len(b2.positions), 1)


if __name__ == "__main__":
    unittest.main()
