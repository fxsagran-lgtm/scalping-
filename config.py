from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

# Hard limits: no preset or UI control may go beyond these. They exist so that
# chasing signal frequency can never silently weaken risk protections.
SAFETY_LIMITS = dict(
    max_cost_r_ceiling=0.35,      # round-trip cost may never exceed 0.35R of the trade's risk
    min_stop_atr_floor=0.5,       # stops tighter than 0.5 ATR are noise-level
    max_stop_atr_ceiling=3.0,
    max_leverage_ceiling=1.0,     # spot-equivalent notional only
    risk_pct_ceiling=1.0,         # % of equity risked per trade
    max_signals_per_symbol_day_ceiling=6,
    min_cooldown_bars_floor=3,
    max_data_age_minutes_ceiling=15,
)


@dataclass(frozen=True)
class CostConfig:
    fee_bps_per_side: float = 10.0       # 0.10% taker. Set to YOUR real fee tier.
    slippage_bps_per_side: float = 2.0   # applied to entries, stops and time exits (not limit targets)

    @property
    def round_trip_frac(self) -> float:
        return 2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side) / 1e4


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "balanced"

    # ---- fixed indicator lengths (per spec) ----
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    atr_len: int = 14
    swing_n_15m: int = 3
    swing_n_5m: int = 3
    warmup_bars: int = 600               # 5m bars skipped at the start so indicators are warm

    # ---- TUNABLE filters (presets may vary these) ----
    require_15m_structure: bool = True   # HH/HL for longs, LH/LL for shorts
    require_15m_slope: bool = True
    trend_slope_bars: int = 3
    min_trend_gap_atr: float = 0.3       # (EMA20-EMA50)/ATR15 must be at least this
    pullback_lookback: int = 6           # bars in which price must have touched the EMA20 zone
    touch_atr: float = 0.25
    require_confirm_candle: bool = True  # bullish (bearish) candle closing back through EMA20
    require_break_prev_extreme: bool = False
    require_vwap_side: bool = True
    require_5m_ema_order: bool = True
    rsi_pullback_low: float = 40.0       # long band; short band is mirrored (100 - x)
    rsi_pullback_high: float = 62.0
    require_rsi_turn: bool = True
    max_extension_atr: float = 1.0
    min_room_r: float = 1.0              # nearest 5m swing must leave this much room (0 disables)
    allowed_hours_utc: Optional[Tuple[int, ...]] = None

    # ---- SAFETY filters (identical in every preset; bounded by SAFETY_LIMITS) ----
    min_atr_pct: float = 0.04            # 5m ATR as % of price: skip dead markets
    max_atr_pct: float = 0.60            # ... and chaotic ones
    stop_lookback: int = 6
    stop_buffer_atr: float = 0.10
    min_stop_atr: float = 0.6
    max_stop_atr: float = 2.5
    max_cost_r: float = 0.25
    cooldown_bars: int = 6
    max_signals_per_symbol_day: int = 4
    max_data_age_minutes: int = 12
    trend_max_age_min: int = 30

    # ---- exits ----
    tp_r: Tuple[float, float, float] = (1.0, 1.5, 2.0)
    tp_fractions: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    move_to_be_after_tp1: bool = True
    max_hold_bars: int = 48              # 4 hours of 5m bars

    # ---- sizing ----
    initial_equity: float = 10_000.0
    risk_pct: float = 0.5
    max_leverage: float = 1.0


TUNABLE_FIELDS = (
    "require_15m_structure", "require_15m_slope", "trend_slope_bars", "min_trend_gap_atr",
    "pullback_lookback", "touch_atr", "require_confirm_candle", "require_break_prev_extreme",
    "require_vwap_side", "require_5m_ema_order", "rsi_pullback_low", "rsi_pullback_high",
    "require_rsi_turn", "max_extension_atr", "min_room_r", "allowed_hours_utc",
)


def validate(cfg: StrategyConfig) -> StrategyConfig:
    L = SAFETY_LIMITS
    problems = []
    if cfg.max_cost_r > L["max_cost_r_ceiling"]:
        problems.append(f"max_cost_r {cfg.max_cost_r} > {L['max_cost_r_ceiling']}")
    if cfg.min_stop_atr < L["min_stop_atr_floor"]:
        problems.append(f"min_stop_atr {cfg.min_stop_atr} < {L['min_stop_atr_floor']}")
    if cfg.max_stop_atr > L["max_stop_atr_ceiling"]:
        problems.append(f"max_stop_atr {cfg.max_stop_atr} > {L['max_stop_atr_ceiling']}")
    if cfg.max_leverage > L["max_leverage_ceiling"]:
        problems.append(f"max_leverage {cfg.max_leverage} > {L['max_leverage_ceiling']}")
    if cfg.risk_pct > L["risk_pct_ceiling"]:
        problems.append(f"risk_pct {cfg.risk_pct} > {L['risk_pct_ceiling']}")
    if cfg.max_signals_per_symbol_day > L["max_signals_per_symbol_day_ceiling"]:
        problems.append("max_signals_per_symbol_day too high")
    if cfg.cooldown_bars < L["min_cooldown_bars_floor"]:
        problems.append("cooldown_bars too low")
    if cfg.max_data_age_minutes > L["max_data_age_minutes_ceiling"]:
        problems.append("max_data_age_minutes too high")
    if abs(sum(cfg.tp_fractions) - 1.0) > 1e-9:
        problems.append("tp_fractions must sum to 1")
    if list(cfg.tp_r) != sorted(cfg.tp_r):
        problems.append("tp_r must be ascending")
    if problems:
        raise ValueError("Config violates safety limits: " + "; ".join(problems))
    return cfg


# Presets differ ONLY in tunable (signal-quality) filters. Safety filters never change.
PRESETS = {
    "strict": StrategyConfig(
        name="strict", min_trend_gap_atr=0.6, touch_atr=0.15, require_break_prev_extreme=True,
        rsi_pullback_low=42.0, rsi_pullback_high=58.0, max_extension_atr=0.7, min_room_r=1.5,
    ),
    "balanced": StrategyConfig(name="balanced"),
    "active": StrategyConfig(
        name="active", min_trend_gap_atr=0.1, touch_atr=0.4, pullback_lookback=10,
        require_5m_ema_order=False, require_rsi_turn=False,
        rsi_pullback_low=35.0, rsi_pullback_high=68.0, max_extension_atr=1.3, min_room_r=0.7,
    ),
}
for _p in PRESETS.values():
    validate(_p)


def with_overrides(cfg: StrategyConfig, **kw) -> StrategyConfig:
    return validate(replace(cfg, **kw))
