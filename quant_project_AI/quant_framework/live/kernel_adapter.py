"""Adapter that wraps strategy signal logic for live trading.

Delegates signal computation to the Numba backtest kernels — the kernel
is the single source of truth for all 18 strategies.  This eliminates
signal-logic duplication between backtest and live paths.

MultiTFAdapter adds multi-timeframe signal fusion on top, combining
positions from multiple KernelAdapter instances (one per interval) via
TrendFilter, Consensus, or Primary fusion modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest.config import BacktestConfig
from ..backtest.kernels import (
    KERNEL_NAMES,
    DEFAULT_PARAM_GRIDS,
    config_to_kernel_costs,
    eval_kernel_position,
)

logger = logging.getLogger(__name__)

_INTERVAL_RANK = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5, "1w": 6}

_PARAM_DEFAULTS: Dict[str, tuple] = {
    name: DEFAULT_PARAM_GRIDS[name][0] for name in KERNEL_NAMES
}


class KernelAdapter:
    """Generates buy/sell signals by running the actual Numba kernel on
    the rolling OHLCV window.

    The kernel determines the current position (+1 long, -1 short, 0 flat)
    using exactly the same logic as the backtester.  When the position
    changes relative to the previous bar, a buy/sell signal is emitted.
    """

    def __init__(
        self,
        strategy_name: str,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[BacktestConfig] = None,
    ):
        if strategy_name not in KERNEL_NAMES:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {KERNEL_NAMES}")
        self._name = strategy_name
        self._params_dict = params or {}
        self._kernel_params = self._resolve_kernel_params(strategy_name, params)
        self._config = config or BacktestConfig.crypto()
        self._costs = config_to_kernel_costs(self._config)
        self._prev_position: int = 0

    @property
    def name(self) -> str:
        return self._name

    _DICT_KEY_ORDER: Dict[str, tuple] = {
        "MA": ("ma_short", "ma_long"),
        "RSI": ("rsi_period", "os", "ob"),
        "MACD": ("fast", "slow", "signal"),
        "Drift": ("lookback", "drift_threshold", "hold_period"),
        "RAMOM": ("mom_period", "vol_period", "entry_z", "exit_z"),
        "Turtle": ("entry_period", "exit_period", "atr_period", "atr_mult"),
        "Bollinger": ("bb_period", "bb_std"),
        "Keltner": ("ema_period", "atr_period", "atr_mult"),
        "MultiFactor": ("rsi_period", "mom_period", "vol_period", "long_thresh", "short_thresh"),
        "VolRegime": ("atr_period", "vol_threshold", "ma_short", "ma_long", "rsi_os", "rsi_ob"),
        "MESA": ("fast_limit", "slow_limit"),
        "KAMA": ("period", "fast_sc", "slow_sc"),
        "Donchian": ("entry_period", "exit_period"),
        "ZScore": ("lookback", "entry_z", "exit_z", "stop_z"),
        "MomBreak": ("high_period", "proximity_pct", "atr_period", "atr_trail"),
        "RegimeEMA": ("vol_period", "ema_fast", "ema_slow", "vol_threshold"),
        "DualMom": ("abs_period", "rel_period", "entry_z", "exit_z"),
        "Consensus": ("ma_short", "ma_long", "rsi_period", "rsi_os", "rsi_ob",
                       "bb_period", "bb_std", "min_votes"),
    }

    @staticmethod
    def _resolve_kernel_params(name: str, params: Optional[Dict[str, Any]]) -> tuple:
        """Convert user-friendly param dict/list/tuple to kernel tuple, or use defaults.

        Supports all 18 strategies via explicit key-to-position mapping.
        """
        if params is None:
            return _PARAM_DEFAULTS[name]
        if isinstance(params, tuple):
            return params
        if isinstance(params, list):
            return tuple(params)

        base = list(_PARAM_DEFAULTS[name])
        keys = KernelAdapter._DICT_KEY_ORDER.get(name, ())
        for idx, key in enumerate(keys):
            if key in params and idx < len(base):
                base[idx] = type(base[idx])(params[key])
        return tuple(base)

    def generate_signal(
        self,
        window_df: pd.DataFrame,
        symbol: str,
        *,
        arrays: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the Numba kernel on the full window and emit a signal on
        position change.

        ``arrays`` is an optional pre-built dict of contiguous float64 arrays
        (keys: open, high, low, close) from ``_RollingWindow.to_arrays()``.
        When provided, no ``np.ascontiguousarray`` copies are made — the
        arrays are used directly.

        Returns a signal dict with action='buy'|'sell' or None.
        """
        if arrays is not None:
            c = arrays.get("close")
            if c is None or len(c) < 30:
                return None
            o = arrays["open"]
            h = arrays["high"]
            l = arrays["low"]
        else:
            if len(window_df) < 30:
                return None
            for col in ("open", "high", "low", "close"):
                if col not in window_df.columns:
                    return None
            vals = window_df[["close", "open", "high", "low"]].values
            if vals.dtype != np.float64 or not vals.flags.c_contiguous:
                vals = np.ascontiguousarray(vals, dtype=np.float64)
            c, o, h, l = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

        try:
            co = self._costs
            cur_pos = eval_kernel_position(
                self._name, self._kernel_params, c, o, h, l,
                co["sb"], co["ss"], co["cm"], co["lev"], co["dc"],
                co["sl"], co["pfrac"], co["sl_slip"],
            )
        except Exception as e:
            logger.debug("Kernel eval failed for %s/%s: %s", self._name, symbol, e)
            return None

        action: Optional[str] = None
        if cur_pos > 0 and self._prev_position <= 0:
            action = "buy"
        elif cur_pos < 0 and self._prev_position >= 0:
            action = "sell"

        if action is None:
            self._prev_position = cur_pos
            return None

        self._prev_position = cur_pos
        return {
            "action": action,
            "symbol": symbol,
            "price": float(c[-1]),
            "strategy": self._name,
        }

    def get_position(self) -> int:
        """Current kernel position: +1 long, -1 short, 0 flat."""
        return self._prev_position

    def set_position(self, position: int) -> None:
        """Force the runtime position state to a recovered value."""
        self._prev_position = 1 if position > 0 else (-1 if position < 0 else 0)


# ── Multi-Timeframe Adapter ──────────────────────────────────────────

FUSION_MODES = ("trend_filter", "consensus", "primary")


class MultiTFAdapter:
    """Combines signals from multiple KernelAdapters (one per interval).

    Fusion modes
    -------------
    trend_filter  Highest-TF position sets allowed direction; only lower-TF
                  signals that agree with the trend are forwarded.
    consensus     Majority vote across all TF positions; signal emitted when
                  the fused position changes.
    primary       Only the designated primary-TF adapter generates signals;
                  other TFs are tracked for dashboard display only.
    """

    def __init__(
        self,
        adapters: Dict[str, KernelAdapter],
        mode: str = "trend_filter",
        primary_interval: Optional[str] = None,
    ):
        if mode not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode '{mode}'. Choose from {FUSION_MODES}")
        if not adapters:
            raise ValueError("At least one (interval, KernelAdapter) pair is required")

        self._adapters = adapters
        self._mode = mode
        self._sorted_intervals = sorted(
            adapters.keys(), key=lambda x: _INTERVAL_RANK.get(x, 99),
        )
        self._primary = primary_interval or self._sorted_intervals[-1]
        self._prev_fused_position: int = 0
        self._tf_positions: Dict[str, int] = {iv: 0 for iv in self._sorted_intervals}

    @property
    def name(self) -> str:
        parts = [f"{iv}:{self._adapters[iv].name}" for iv in self._sorted_intervals]
        return f"MultiTF[{self._mode}]({' | '.join(parts)})"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def intervals(self) -> List[str]:
        return list(self._sorted_intervals)

    @property
    def tf_positions(self) -> Dict[str, int]:
        """Per-interval position states for dashboard display."""
        return dict(self._tf_positions)

    @property
    def fused_position(self) -> int:
        """Current fused (consensus / trend_filter / primary) position."""
        return self._prev_fused_position

    def set_fused_position(self, position: int) -> None:
        """Force the fused runtime position to a recovered value."""
        self._prev_fused_position = 1 if position > 0 else (-1 if position < 0 else 0)

    @property
    def tf_strategies(self) -> Dict[str, str]:
        """Per-interval strategy names."""
        return {iv: a.name for iv, a in self._adapters.items()}

    def warmup(
        self,
        windows: Dict[str, pd.DataFrame],
        symbol: str,
        *,
        arrays_map: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> None:
        """Run each kernel adapter on its historical window to establish
        initial positions *without* emitting any signals.  Must be called
        after historical data has been loaded into the feed windows.

        ``arrays_map`` is an optional per-interval dict of pre-built arrays
        from ``_RollingWindow.to_arrays()`` to skip DataFrame→array conversion.
        """
        for iv in self._sorted_intervals:
            adapter = self._adapters.get(iv)
            df = windows.get(iv)
            if adapter is None or df is None or df.empty:
                continue
            arrs = arrays_map.get(iv) if arrays_map else None
            adapter.generate_signal(df, symbol, arrays=arrs)
            self._tf_positions[iv] = adapter.get_position()

        pos_str = " | ".join(
            f"{iv}={self._tf_positions[iv]:+d}" for iv in self._sorted_intervals
        )
        logger.info("MultiTF warmup [%s] %s → tf_positions: %s", symbol, self._mode, pos_str)

        positions = list(self._tf_positions.values())
        n = len(positions)
        longs = sum(1 for p in positions if p > 0)
        shorts = sum(1 for p in positions if p < 0)
        threshold = n / 2.0
        if longs > threshold:
            self._prev_fused_position = 1
        elif shorts > threshold:
            self._prev_fused_position = -1
        else:
            self._prev_fused_position = 0
        logger.info(
            "MultiTF warmup fused_position=%+d (longs=%d shorts=%d threshold=%.1f)",
            self._prev_fused_position, longs, shorts, threshold,
        )

    def on_bar(
        self,
        window_df: pd.DataFrame,
        symbol: str,
        interval: str,
        *,
        arrays: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process a bar close for the given interval. Returns a trade signal
        only when the fused position changes.

        ``arrays`` — pre-built contiguous float64 arrays to bypass
        DataFrame → array conversion.
        """
        adapter = self._adapters.get(interval)
        if adapter is None:
            return None

        sig = adapter.generate_signal(window_df, symbol, arrays=arrays)
        self._tf_positions[interval] = adapter.get_position()

        pos_str = " | ".join(
            f"{iv}={self._tf_positions[iv]:+d}" for iv in self._sorted_intervals
        )
        logger.info(
            "MultiTF on_bar [%s@%s] updated %s=%+d → all: %s  fused_prev=%+d",
            symbol, interval, interval, self._tf_positions[interval],
            pos_str, self._prev_fused_position,
        )

        if self._mode == "trend_filter":
            return self._fuse_trend_filter(sig, symbol, interval)
        elif self._mode == "consensus":
            return self._fuse_consensus(symbol)
        else:
            return self._fuse_primary(sig, interval)

    # ── Fusion implementations ───────────────────────────────────────

    def _fuse_trend_filter(
        self, lower_sig: Optional[Dict[str, Any]], symbol: str, interval: str,
    ) -> Optional[Dict[str, Any]]:
        """Higher TF sets trend; lower TF provides entry timing."""
        highest_iv = self._sorted_intervals[-1]
        trend = self._tf_positions.get(highest_iv, 0)

        if interval == highest_iv:
            return self._emit_on_position_change(trend, symbol)

        if lower_sig is None:
            return None

        action = lower_sig["action"]
        if trend > 0 and action == "buy":
            return self._emit_on_position_change(1, symbol)
        elif trend < 0 and action == "sell":
            return self._emit_on_position_change(-1, symbol)
        elif trend == 0:
            return self._emit_on_position_change(0, symbol)

        return None

    def _fuse_consensus(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Majority vote across all TF positions."""
        positions = list(self._tf_positions.values())
        n = len(positions)
        longs = sum(1 for p in positions if p > 0)
        shorts = sum(1 for p in positions if p < 0)
        threshold = n / 2.0

        if longs > threshold:
            fused = 1
        elif shorts > threshold:
            fused = -1
        else:
            fused = 0

        if fused != self._prev_fused_position:
            logger.info(
                "MultiTF CONSENSUS [%s] fused %+d→%+d (longs=%d shorts=%d/%d)",
                symbol, self._prev_fused_position, fused, longs, shorts, n,
            )

        return self._emit_on_position_change(fused, symbol)

    def _fuse_primary(
        self, sig: Optional[Dict[str, Any]], interval: str,
    ) -> Optional[Dict[str, Any]]:
        """Only signals from the primary TF are forwarded."""
        if interval != self._primary:
            return None
        return sig

    def _emit_on_position_change(
        self, new_pos: int, symbol: str,
    ) -> Optional[Dict[str, Any]]:
        if new_pos == self._prev_fused_position:
            return None

        action: Optional[str] = None
        if new_pos > 0 and self._prev_fused_position <= 0:
            action = "buy"
        elif new_pos < 0 and self._prev_fused_position >= 0:
            action = "sell"
        elif new_pos == 0:
            action = "sell" if self._prev_fused_position > 0 else "buy"

        self._prev_fused_position = new_pos
        if action is None:
            return None

        strat_names = " | ".join(
            f"{iv}:{self._adapters[iv].name}" for iv in self._sorted_intervals
        )
        return {
            "action": action,
            "symbol": symbol,
            "price": 0.0,
            "strategy": f"MultiTF[{self._mode}]({strat_names})",
        }
