"""Scalping System page (Streamlit multipage). The existing dashboard is untouched.

ANALYSIS + PAPER TRADING ONLY. No exchange credentials, no real orders.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from scalper.backtest import compare_presets, run_experiment, verdict, equity_curve
from scalper.config import PRESETS, CostConfig
from scalper.data import DataUnavailableError, SOURCES, load_or_fetch
from scalper.paper import PaperBroker
from scalper.report import metrics_table
from scalper.safety import LIVE_TRADING_ENABLED
from scalper.signals import build_features, filter_funnel, latest_signal

st.set_page_config(page_title="Scalping System", layout="centered")
st.title("Scalping System (BTC & ETH)")
st.error("Analysis and paper trading only. Real orders are disabled."
         if not LIVE_TRADING_ENABLED else "SAFETY SWITCH INCONSISTENT - do not use.")
st.caption("Setups/day of 4-5 is a research target, not a promise. The strategy may simply not meet it, or may lose money after fees.")

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Data source", list(SOURCES), format_func=lambda k: SOURCES[k]["label"])
    preset = st.selectbox("Preset", list(PRESETS), index=1)
    fee = st.number_input("Fee per side (bps)", 0.0, 50.0, 10.0, 0.5, help="Use your real fee tier.")
    slip = st.number_input("Slippage per side (bps)", 0.0, 50.0, 2.0, 0.5)
    offline = st.checkbox("Use cached candles only (no network)", value=False)
costs, cfg = CostConfig(fee, slip), PRESETS[preset]


@st.cache_data(ttl=60, show_spinner=False)
def get_live(symbol, source, offline, _bucket):
    start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=5)
    c5 = load_or_fetch(symbol, "5m", start, source=source, offline=offline)
    c15 = load_or_fetch(symbol, "15m", start, source=source, offline=offline)
    return c5, c15


tab_sig, tab_bt, tab_paper = st.tabs(["Signals", "Backtest", "Paper trading"])

with tab_sig:
    if st.button("Refresh now"):
        st.cache_data.clear()
    broker = PaperBroker("paper_state/paper_account.json", cfg, costs)
    for sym in SYMBOLS:
        st.subheader(sym)
        try:
            c5, c15 = get_live(sym, source, offline, pd.Timestamp.now(tz="UTC").floor("min"))
        except DataUnavailableError as e:
            st.error(f"Real data unavailable - no synthetic data is used. {e}")
            continue
        st.caption(c5.banner())
        # live view needs enough history to warm indicators: warmup_bars of 5m (~2 days) -> fetch 5 days above
        sig = latest_signal(sym, c5.df, c15.df, cfg, costs, data_source=SOURCES[source]["label"])
        (st.success if sig.action == "BUY" else st.error if sig.action == "SELL" else st.info)(f"**{sig.action}**")
        if sig.action in ("BUY", "SELL"):
            st.table(pd.DataFrame({"Level": ["Entry (last close; fills next open)", "Stop loss", *[f"Target {k}" for k in sig.targets]],
                                   "Price": [sig.entry, sig.stop, *sig.targets.values()]}).set_index("Level").round(4))
            st.write(f"Risk/reward: {sig.risk_reward} | round-trip costs ≈ {sig.cost_in_r}R")
        for r in sig.reasons:
            st.write("• " + r)
        st.caption(f"Latest closed 5m candle opened {sig.latest_candle_open_utc} UTC | data age {sig.data_age_minutes} min")
        if sig.action in ("BUY", "SELL") and st.button(f"Open PAPER trade ({sym})", key=f"p_{sym}"):
            st.write("Paper position created." if broker.open_from_signal(sig) else "Already tracked / position open.")
        with st.expander("Which filters block signals? (pass rates)"):
            st.dataframe(filter_funnel(build_features(c5.df, c15.df, cfg), cfg, costs).round(1), hide_index=True)

with tab_bt:
    months = st.slider("History (months)", 1, 12, 6)
    train = st.slider("Train fraction (chronological)", 0.4, 0.8, 0.6, 0.05)
    mode = st.radio("Run", ["Compare all presets", "Selected preset only"], horizontal=True)
    if st.button("Fetch real candles & run backtest"):
        start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30.4 * months)).normalize()
        try:
            data, banners = {}, []
            prog = st.progress(0.0)
            for i, s in enumerate(SYMBOLS):
                cd = {}
                for tf in ("5m", "15m"):
                    c = load_or_fetch(s, tf, start, source=source, offline=offline)
                    banners.append(c.banner()); cd[tf] = c.df
                data[s] = (cd["5m"], cd["15m"]); prog.progress((i + 1) / len(SYMBOLS))
            st.session_state["bt"] = dict(
                banners=banners,
                comp=compare_presets(data, costs, train_frac=train) if mode.startswith("Compare") else None,
                exp=None if mode.startswith("Compare") else run_experiment(data, cfg, costs, train))
        except DataUnavailableError as e:
            st.error(f"Real data unavailable - nothing was simulated. {e}")
    bt = st.session_state.get("bt")
    if bt:
        for b in bt["banners"]:
            st.caption(b)
        exp = bt["exp"]
        if bt["comp"]:
            comp = bt["comp"]
            st.subheader("Preset comparison (combined BTC+ETH)")
            st.dataframe(comp["table"].round(3), hide_index=True)
            sel = comp["selected_on_train"]
            st.write(f"Selected using TRAIN data only: **{sel or 'none - no preset was profitable in train'}**")
            exp = comp["runs"][sel or "balanced"]
        st.subheader("Metrics")
        st.dataframe(metrics_table(exp).round(3), hide_index=True)
        key = "COMBINED" if "COMBINED" in exp["results"] else SYMBOLS[0]
        v = verdict(exp["results"][key]["test"]["metrics"])
        (st.error if v.startswith(("LOSES", "INCONCLUSIVE")) else st.warning if v.startswith("MARGINAL") else st.info)(v)
        eq = equity_curve(exp["results"][key]["test"]["trades"], exp["config"]["initial_equity"])
        if len(eq):
            st.line_chart(eq)
        st.caption(f"Out-of-sample segment starts {exp['cutoff']}. Skipped signals: {exp['skipped']}")

with tab_paper:
    broker = PaperBroker("paper_state/paper_account.json", cfg, costs)
    for sym in SYMBOLS:
        try:
            c5, _ = get_live(sym, source, offline, pd.Timestamp.now(tz="UTC").floor("min"))
            broker.update(sym, c5.df)
        except DataUnavailableError as e:
            st.warning(f"{sym}: {e}")
    st.metric("Paper equity", f"{broker.equity():,.2f}")
    if broker.positions:
        st.dataframe(pd.DataFrame(broker.positions).drop(columns=["reasons"], errors="ignore"), hide_index=True)
    else:
        st.write("No paper trades yet. Open one from the Signals tab when a BUY/SELL appears.")
    if st.checkbox("I understand this deletes all paper history") and st.button("Reset paper account"):
        broker.reset(); st.rerun()
