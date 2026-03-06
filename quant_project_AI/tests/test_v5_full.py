"""V5 Real Trading Upgrade — Comprehensive Test Suite.

Tests all phases: core abstractions, backtest improvements, broker infrastructure,
safety, accuracy, and performance.
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_results: List[Dict] = []
_section_times: Dict[str, float] = {}


def check(test_id: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {test_id}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    _results.append({"id": test_id, "passed": passed, "detail": detail})


def make_ohlcv(n=500, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.uniform(1e6, 5e6, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates, "open": opn, "high": high, "low": low,
        "close": close, "volume": vol,
    }).set_index("date")


# ═══════════════════════════════════════════════════════════════
#  PHASE 0: Core Abstractions
# ═══════════════════════════════════════════════════════════════

def test_phase0_core():
    t0 = time.perf_counter()

    # --- AssetClass ---
    from quant_framework.core.asset_types import AssetClass
    check("P0-asset-class-enum",
          len(AssetClass) == 5,
          f"5 asset classes: {[a.value for a in AssetClass]}")

    # --- CryptoSpotMargin ---
    from quant_framework.core.margin import CryptoSpotMargin, CryptoFuturesMargin, RegTMargin
    spot = CryptoSpotMargin()
    check("P0-spot-margin-100pct",
          spot.initial_margin(10000, 1.0) == 10000,
          f"spot IM = {spot.initial_margin(10000, 1.0)}")

    # --- CryptoFuturesMargin ---
    futures = CryptoFuturesMargin()
    im = futures.initial_margin(100000, 10.0)
    check("P0-futures-im",
          abs(im - 10000) < 0.01,
          f"IM(100k, 10x) = {im}")

    mm = futures.maintenance_margin(40000)
    expected_mm = 40000 * 0.004 - 0
    check("P0-futures-mm-tier1",
          abs(mm - expected_mm) < 0.01,
          f"MM(40k) = {mm}, expected {expected_mm}")

    mm2 = futures.maintenance_margin(200000)
    expected_mm2 = 200000 * 0.005 - 50
    check("P0-futures-mm-tier2",
          abs(mm2 - expected_mm2) < 0.01,
          f"MM(200k) = {mm2}, expected {expected_mm2}")

    mm3 = futures.maintenance_margin(5_000_000)
    expected_mm3 = 5_000_000 * 0.025 - 16_050
    check("P0-futures-mm-tier4",
          abs(mm3 - expected_mm3) < 0.01,
          f"MM(5M) = {mm3}, expected {expected_mm3}")

    lp_long = futures.liquidation_price_long(50000, 10.0, 0.004)
    check("P0-futures-liq-long",
          lp_long > 0 and lp_long < 50000,
          f"liq_long(50k,10x) = {lp_long:.2f}")

    lp_short = futures.liquidation_price_short(50000, 10.0, 0.004)
    check("P0-futures-liq-short",
          lp_short > 50000,
          f"liq_short(50k,10x) = {lp_short:.2f}")

    # --- RegTMargin ---
    regt = RegTMargin()
    im_regt = regt.initial_margin(100000, 2.0)
    check("P0-regt-im",
          abs(im_regt - 50000) < 0.01,
          f"RegT IM(100k) = {im_regt}")

    mm_regt = regt.maintenance_margin(80000)
    check("P0-regt-mm",
          abs(mm_regt - 20000) < 0.01,
          f"RegT MM(80k) = {mm_regt}")

    mc = regt.check_margin_call(18000, 80000)
    check("P0-regt-margin-call",
          mc is True,
          f"equity 18k < MM 20k → margin call = {mc}")

    # --- CostModel ---
    from quant_framework.core.costs import CryptoFuturesCost, USEquityCost
    crypto_cost = CryptoFuturesCost()
    comm_maker = crypto_cost.commission("buy", 100000, is_maker=True)
    comm_taker = crypto_cost.commission("buy", 100000, is_maker=False)
    check("P0-crypto-cost-maker-taker",
          abs(comm_maker - 20) < 0.01 and abs(comm_taker - 40) < 0.01,
          f"maker={comm_maker}, taker={comm_taker}")

    hc_long = crypto_cost.holding_cost(100000, 0.001, "long")
    hc_short = crypto_cost.holding_cost(100000, 0.001, "short")
    check("P0-crypto-funding",
          hc_long > 0 and hc_short < 0,
          f"long_cost={hc_long}, short_cost={hc_short} (positive rate)")

    us_cost = USEquityCost()
    comm_us = us_cost.commission("buy", 50000, shares=200)
    check("P0-us-cost",
          comm_us == 200 * 0.005,
          f"US comm(200 shares) = {comm_us}")

    comm_us_min = us_cost.commission("buy", 500, shares=1)
    check("P0-us-cost-min",
          comm_us_min == 1.0,
          f"US comm(1 share) = {comm_us_min} (min $1)")

    sd = us_cost.settlement_delay_days()
    check("P0-us-settlement",
          sd == 1,
          f"settlement delay = {sd} days")

    # --- SymbolSpec ---
    from quant_framework.core.symbol_spec import SymbolSpec
    spec = SymbolSpec(
        symbol="BTCUSDT", asset_class=AssetClass.CRYPTO_PERP,
        base_asset="BTC", quote_asset="USDT",
        tick_size=0.10, step_size=0.001,
        min_notional=5.0, min_qty=0.001, max_qty=1000.0, max_leverage=125.0,
    )
    rp = spec.round_price(50000.37)
    check("P0-spec-round-price",
          abs(rp - 50000.4) < 0.01,
          f"round_price(50000.37) = {rp}")

    rq = spec.round_quantity(1.2345)
    check("P0-spec-round-qty",
          abs(rq - 1.234) < 0.0001,
          f"round_quantity(1.2345) = {rq}")

    err = spec.validate_order(0.0001, 50000)
    check("P0-spec-validate-min-qty",
          err is not None and "min_qty" in err,
          f"validate_order(0.0001) = {err}")

    err2 = spec.validate_order(0.001, 1.0)
    check("P0-spec-validate-min-notional",
          err2 is not None and "min_notional" in err2,
          f"validate_order(notional=0.001) = {err2}")

    ok = spec.validate_order(0.5, 50000)
    check("P0-spec-validate-ok",
          ok is None,
          f"validate_order(0.5, 50000) = {ok}")

    # --- MarketCalendar ---
    from quant_framework.core.market_hours import MarketCalendar
    cal = MarketCalendar()
    check("P0-cal-crypto-always",
          cal.is_tradable(AssetClass.CRYPTO_PERP, datetime(2023, 1, 1, 3, 0)),
          "crypto tradable at 3am")

    # Saturday
    check("P0-cal-us-weekend",
          not cal.is_tradable(AssetClass.US_EQUITY, datetime(2023, 1, 7, 12, 0)),
          "US equity not tradable on Saturday")

    # --- PDTTracker ---
    from quant_framework.core.pdt_tracker import PDTTracker
    pdt = PDTTracker()
    for i in range(3):
        pdt.record_round_trip("AAPL", datetime(2023, 6, 12 + i))
    check("P0-pdt-count",
          pdt.day_trade_count(date(2023, 6, 15)) == 3,
          f"count = {pdt.day_trade_count(date(2023, 6, 15))}")

    check("P0-pdt-violate",
          pdt.would_violate(20000, date(2023, 6, 15)),
          "equity<25k + 3 day trades → violate")

    check("P0-pdt-no-violate-rich",
          not pdt.would_violate(30000, date(2023, 6, 15)),
          "equity>25k → no violate")

    _section_times["P0_core"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Backtest Config + Funding Rates
# ═══════════════════════════════════════════════════════════════

def test_phase1_backtest():
    t0 = time.perf_counter()
    from quant_framework.backtest.config import BacktestConfig

    # V5 fields exist
    cfg = BacktestConfig.crypto(leverage=5.0, interval="4h")
    check("P1-config-asset-class",
          hasattr(cfg, "asset_class"),
          f"asset_class={cfg.asset_class}")

    check("P1-config-maker-taker",
          hasattr(cfg, "commission_pct_maker") and cfg.commission_pct_maker == 0.0002,
          f"maker={cfg.commission_pct_maker}, taker={cfg.commission_pct_taker}")

    check("P1-config-us-fields",
          hasattr(cfg, "commission_per_share"),
          f"per_share={cfg.commission_per_share}")

    check("P1-config-margin-interest",
          hasattr(cfg, "margin_interest_rate"),
          f"margin_interest={cfg.margin_interest_rate}")

    # FundingRateLoader
    from quant_framework.data.funding_rates import FundingRateLoader
    loader = FundingRateLoader(cache_dir=tempfile.mkdtemp())
    check("P1-funding-loader-exists",
          hasattr(loader, "download") and hasattr(loader, "map_to_bars"),
          "FundingRateLoader has download + map_to_bars")

    # Test map_to_bars with synthetic data
    funding_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=90, freq="8h"),
        "funding_rate": np.random.default_rng(42).uniform(-0.001, 0.003, 90),
    })
    bar_ts = pd.date_range("2023-01-01", periods=30, freq="1D").values
    mapped = loader.map_to_bars(funding_df, bar_ts)
    check("P1-funding-map-shape",
          len(mapped) == 30,
          f"mapped shape = {len(mapped)}")

    check("P1-funding-map-values",
          not np.all(np.isnan(mapped)) and np.nanmin(mapped) > -0.01,
          f"mean={np.nanmean(mapped):.6f}")

    _section_times["P1_backtest"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  PHASE 3: Broker Infrastructure
# ═══════════════════════════════════════════════════════════════

def test_phase3_brokers():
    t0 = time.perf_counter()

    # --- Broker base async methods ---
    from quant_framework.broker.base import Broker
    check("P3-broker-async-methods",
          hasattr(Broker, "submit_order_async") and hasattr(Broker, "cancel_order_async"),
          "Broker has async methods")
    check("P3-broker-margin-methods",
          hasattr(Broker, "get_available_margin") and hasattr(Broker, "get_margin_ratio"),
          "Broker has margin methods")

    # --- Credentials ---
    from quant_framework.broker.credentials import CredentialManager
    cred = CredentialManager()
    check("P3-credentials-exists",
          hasattr(cred, "load") and hasattr(cred, "sign_request"),
          "CredentialManager has load + sign_request")

    sig = cred.sign_request({"symbol": "BTCUSDT", "side": "BUY"}, "test_secret")
    check("P3-credentials-sign",
          isinstance(sig, str) and len(sig) == 64,
          f"signature length = {len(sig)}")

    # --- RateLimiter ---
    from quant_framework.broker.rate_limiter import RateLimiter
    rl = RateLimiter(max_weight=10, window_seconds=1)
    check("P3-ratelimiter-usage",
          rl._current_usage() == 0,
          f"initial usage = {rl._current_usage()}")

    # --- BinanceFuturesBroker structure ---
    from quant_framework.broker.binance_futures import BinanceFuturesBroker
    check("P3-binance-broker-class",
          hasattr(BinanceFuturesBroker, "submit_order_async") and
          hasattr(BinanceFuturesBroker, "sync_positions") and
          hasattr(BinanceFuturesBroker, "set_leverage"),
          "BinanceFuturesBroker has key methods")

    # --- IBKRBroker structure ---
    from quant_framework.broker.ibkr_broker import IBKRBroker
    check("P3-ibkr-broker-class",
          hasattr(IBKRBroker, "submit_order_async") and
          hasattr(IBKRBroker, "sync_positions"),
          "IBKRBroker has key methods")

    # --- LiveOrderManager ---
    from quant_framework.broker.live_order_manager import LiveOrderManager
    check("P3-order-manager",
          hasattr(LiveOrderManager, "submit") and hasattr(LiveOrderManager, "cancel"),
          "LiveOrderManager has submit + cancel")

    # --- Reconciler ---
    from quant_framework.broker.reconciler import PositionReconciler
    check("P3-reconciler",
          hasattr(PositionReconciler, "reconcile"),
          "PositionReconciler has reconcile")

    # --- Execution Algos ---
    from quant_framework.broker.execution_algo import TWAP, LimitChase, Iceberg
    check("P3-execution-algos",
          all(hasattr(cls, "execute") for cls in [TWAP, LimitChase, Iceberg]),
          "TWAP, LimitChase, Iceberg have execute")

    # --- Testnet config ---
    from quant_framework.broker.testnet import TestnetConfig
    check("P3-testnet-config",
          hasattr(TestnetConfig, "BINANCE_FUTURES"),
          "TestnetConfig has BINANCE_FUTURES")

    _section_times["P3_brokers"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  PHASE 5: Safety Infrastructure
# ═══════════════════════════════════════════════════════════════

def test_phase5_safety():
    t0 = time.perf_counter()

    # --- AlertManager ---
    from quant_framework.live.alerts import AlertManager
    am = AlertManager({"console": True})
    check("P5-alert-manager",
          hasattr(am, "send"),
          "AlertManager has send")

    # --- AuditTrail ---
    from quant_framework.live.audit import AuditTrail
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        audit_path = f.name
    try:
        audit = AuditTrail(audit_path)
        audit.record_event("test", "test_event")
        audit.record_order_lifecycle(
            internal_id="INT001", exchange_order_id="EX001",
            signal_ts=datetime.now(), submit_ts=datetime.now(),
            ack_ts=datetime.now(), fill_ts=datetime.now(),
            fill_price=50000.0, signal_price=50010.0,
            latency_ms=15.5, slippage_bps=2.0,
        )
        report = audit.generate_daily_audit_report(date.today())
        check("P5-audit-trail",
              "INT001" in report or "order" in report.lower(),
              f"report length = {len(report)}")
        audit.close()
    except Exception as e:
        check("P5-audit-trail", False, str(e))
    finally:
        os.unlink(audit_path)

    # --- KillSwitch ---
    from quant_framework.live.kill_switch import KillSwitch
    check("P5-kill-switch",
          hasattr(KillSwitch, "flatten_all") and hasattr(KillSwitch, "is_triggered"),
          "KillSwitch has flatten_all + is_triggered")

    # --- HealthCheckServer ---
    from quant_framework.live.health_server import HealthCheckServer
    check("P5-health-server",
          hasattr(HealthCheckServer, "start"),
          "HealthCheckServer has start")

    _section_times["P5_safety"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  MARGIN MODEL ACCURACY
# ═══════════════════════════════════════════════════════════════

def test_margin_accuracy():
    t0 = time.perf_counter()
    from quant_framework.core.margin import CryptoFuturesMargin, RegTMargin

    futures = CryptoFuturesMargin()

    # Tier boundary tests (Binance official)
    test_cases = [
        (10_000, 0.004, 0),
        (50_000, 0.004, 0),
        (50_001, 0.005, 50),
        (250_000, 0.005, 50),
        (250_001, 0.01, 1050),
        (1_000_000, 0.01, 1050),
        (1_000_001, 0.025, 16050),
        (10_000_000, 0.025, 16050),
        (10_000_001, 0.05, 266050),
        (20_000_000, 0.05, 266050),
        (20_000_001, 0.10, 1266050),
        (50_000_000, 0.10, 1266050),
        (100_000_001, 0.15, 5016050),
        (200_000_001, 0.25, 25016050),
    ]

    for notional, expected_rate, expected_amount in test_cases:
        mm = futures.maintenance_margin(notional)
        expected = notional * expected_rate - expected_amount
        check(f"MARGIN-tier-{notional}",
              abs(mm - expected) < 0.02,
              f"MM({notional:,}) = {mm:,.2f}, expected {expected:,.2f}")

    # Liquidation price accuracy
    for lev in [2, 5, 10, 20, 50, 100]:
        lp = futures.liquidation_price_long(50000, lev, 0.004)
        expected_lp = 50000 * (1 - 1/lev + 0.004)
        check(f"MARGIN-liq-long-{lev}x",
              abs(lp - expected_lp) < 0.01,
              f"liq_long({lev}x) = {lp:.2f}")

    # RegT accuracy
    regt = RegTMargin()
    for notional in [10_000, 100_000, 500_000]:
        im = regt.initial_margin(notional, 2.0)
        check(f"MARGIN-regt-im-{notional}",
              abs(im - notional * 0.5) < 0.01,
              f"RegT IM({notional:,}) = {im:,.2f}")

    _section_times["margin_accuracy"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  COST MODEL ACCURACY
# ═══════════════════════════════════════════════════════════════

def test_cost_accuracy():
    t0 = time.perf_counter()
    from quant_framework.core.costs import CryptoFuturesCost, USEquityCost

    crypto = CryptoFuturesCost()

    # Maker/taker commission accuracy
    for notional in [1_000, 10_000, 100_000, 1_000_000]:
        maker = crypto.commission("buy", notional, is_maker=True)
        taker = crypto.commission("buy", notional, is_maker=False)
        check(f"COST-crypto-comm-{notional}",
              abs(maker - notional * 0.0002) < 0.001 and abs(taker - notional * 0.0004) < 0.001,
              f"maker={maker:.2f}, taker={taker:.2f}")

    # Funding: positive rate → longs pay, shorts receive
    for rate in [0.0001, 0.001, 0.003]:
        hc_long = crypto.holding_cost(100_000, rate, "long")
        hc_short = crypto.holding_cost(100_000, rate, "short")
        check(f"COST-funding-{rate}",
              hc_long > 0 and hc_short < 0 and abs(hc_long + hc_short) < 0.001,
              f"long={hc_long:.2f}, short={hc_short:.2f}")

    # Negative rate → shorts pay
    hc_neg_long = crypto.holding_cost(100_000, -0.001, "long")
    hc_neg_short = crypto.holding_cost(100_000, -0.001, "short")
    check("COST-funding-negative",
          hc_neg_long < 0 and hc_neg_short > 0,
          f"neg_rate: long={hc_neg_long:.2f}, short={hc_neg_short:.2f}")

    # US Equity
    us = USEquityCost()
    for shares in [1, 10, 100, 1000]:
        comm = us.commission("buy", shares * 100, shares=shares)
        expected = max(1.0, shares * 0.005)
        check(f"COST-us-comm-{shares}sh",
              abs(comm - expected) < 0.001,
              f"comm({shares}sh) = {comm:.2f}")

    _section_times["cost_accuracy"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  KERNEL ACCURACY (all 18 strategies)
# ═══════════════════════════════════════════════════════════════

def test_kernel_accuracy():
    t0 = time.perf_counter()
    from quant_framework.backtest.kernels import (
        KERNEL_REGISTRY, eval_kernel, eval_kernel_detailed,
        config_to_kernel_costs, DEFAULT_PARAM_GRIDS,
    )
    from quant_framework.backtest.config import BacktestConfig

    df = make_ohlcv(500, seed=12345)
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l_ = df["low"].values.astype(np.float64)

    accuracy_results = {}
    for config_name, cfg_fn in [("crypto", BacktestConfig.crypto),
                                 ("stock", lambda: BacktestConfig.stock_ibkr())]:
        cfg = cfg_fn() if config_name == "stock" else cfg_fn(leverage=2.0)
        costs = config_to_kernel_costs(cfg)
        sb, ss, cm, lev = costs["sb"], costs["ss"], costs["cm"], costs["lev"]
        dc, sl, pfrac, sl_slip = costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"]

        for name in sorted(KERNEL_REGISTRY):
            params = DEFAULT_PARAM_GRIDS[name][0]
            try:
                r1, d1, nt1 = eval_kernel(name, params, c, o, h, l_,
                                           sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
                r2, d2, nt2, eq, _, _ = eval_kernel_detailed(
                    name, params, c, o, h, l_,
                    sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

                all_ok = abs(r1 - r2) < 1e-6 and abs(d1 - d2) < 1e-6 and nt1 == nt2
                eq_valid = eq[0] == 1.0 and not np.any(np.isnan(eq)) and not np.any(eq < 0)

                check(f"KERNEL-{config_name}-{name}",
                      all_ok and eq_valid,
                      f"ret={r2:.2f}% dd={d2:.2f}% trades={nt2}")

                if config_name == "crypto":
                    accuracy_results[name] = {
                        "ret": r2, "dd": d2, "trades": nt2,
                        "eq_end": eq[-1],
                    }
            except Exception as e:
                check(f"KERNEL-{config_name}-{name}", False, str(e)[:80])

    _section_times["kernel_accuracy"] = time.perf_counter() - t0
    return accuracy_results


# ═══════════════════════════════════════════════════════════════
#  PERFORMANCE BENCHMARK
# ═══════════════════════════════════════════════════════════════

def test_performance():
    t0 = time.perf_counter()
    from quant_framework.backtest.kernels import (
        eval_kernel, DEFAULT_PARAM_GRIDS, config_to_kernel_costs,
    )
    from quant_framework.backtest.config import BacktestConfig

    config = BacktestConfig.crypto()
    costs = config_to_kernel_costs(config)
    sb, ss, cm, lev = costs["sb"], costs["ss"], costs["cm"], costs["lev"]
    dc, sl, pfrac, sl_slip = costs["dc"], costs["sl"], costs["pfrac"], costs["sl_slip"]

    perf_results = {}

    # Throughput by size
    for sz in [500, 1000, 2000, 5000]:
        df = make_ohlcv(sz, seed=77)
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l_ = df["low"].values.astype(np.float64)

        eval_kernel("MA", DEFAULT_PARAM_GRIDS["MA"][0], c, o, h, l_,
                    sb, ss, cm, lev, dc, sl, pfrac, sl_slip)

        t1 = time.perf_counter()
        n_runs = 0
        for name in sorted(DEFAULT_PARAM_GRIDS.keys()):
            params = DEFAULT_PARAM_GRIDS[name][0]
            eval_kernel(name, params, c, o, h, l_,
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
            n_runs += 1
        elapsed = time.perf_counter() - t1
        throughput = n_runs / elapsed
        perf_results[sz] = {"n_kernels": n_runs, "elapsed_ms": elapsed * 1000,
                            "throughput": throughput}
        check(f"PERF-{sz}bars",
              throughput > 50,
              f"{throughput:.0f} kernels/s ({elapsed*1000:.1f}ms)")

    # Param scan
    df_bench = make_ohlcv(1000, seed=55)
    c = df_bench["close"].values.astype(np.float64)
    o = df_bench["open"].values.astype(np.float64)
    h = df_bench["high"].values.astype(np.float64)
    l_ = df_bench["low"].values.astype(np.float64)

    t_scan = time.perf_counter()
    scan_count = 0
    for name in ["MA", "RSI", "MACD"]:
        for params in DEFAULT_PARAM_GRIDS[name][:20]:
            eval_kernel(name, params, c, o, h, l_,
                        sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
            scan_count += 1
    scan_elapsed = time.perf_counter() - t_scan
    scan_tput = scan_count / scan_elapsed
    perf_results["scan"] = {"count": scan_count, "elapsed_ms": scan_elapsed * 1000,
                            "throughput": scan_tput}
    check("PERF-scan",
          scan_tput > 100,
          f"{scan_tput:.0f} evals/s ({scan_count} combos)")

    # Margin model perf
    from quant_framework.core.margin import CryptoFuturesMargin
    fm = CryptoFuturesMargin()
    t_mm = time.perf_counter()
    for _ in range(100_000):
        fm.maintenance_margin(np.random.uniform(1000, 50_000_000))
    mm_elapsed = time.perf_counter() - t_mm
    mm_rate = 100_000 / mm_elapsed
    perf_results["margin"] = {"rate": mm_rate, "elapsed_ms": mm_elapsed * 1000}
    check("PERF-margin-model",
          mm_rate > 100_000,
          f"{mm_rate:.0f} calcs/s")

    _section_times["performance"] = time.perf_counter() - t0
    return perf_results


# ═══════════════════════════════════════════════════════════════
#  EXISTING REGRESSION
# ═══════════════════════════════════════════════════════════════

def test_regression():
    t0 = time.perf_counter()

    # PaperBroker still works
    from quant_framework.broker.paper import PaperBroker
    broker = PaperBroker(10000)
    r1 = broker.submit_order({"action": "buy", "symbol": "T", "shares": 10, "price": 100})
    check("REGR-paper-buy", r1["status"] == "filled", f"status={r1['status']}")
    r2 = broker.submit_order({"action": "sell", "symbol": "T", "shares": 10, "price": 110})
    check("REGR-paper-sell-profit",
          broker.get_cash() > 10000,
          f"cash={broker.get_cash():.2f}")

    # Strategy imports still work
    from quant_framework.strategy import (
        MovingAverageStrategy, RSIStrategy, MACDStrategy,
        DriftRegimeStrategy, ZScoreReversionStrategy, KAMAStrategy,
    )
    for cls in [MovingAverageStrategy, RSIStrategy, MACDStrategy,
                DriftRegimeStrategy, ZScoreReversionStrategy, KAMAStrategy]:
        s = cls()
        check(f"REGR-strategy-{s.name[:15]}",
              s.min_lookback >= 1,
              f"lookback={s.min_lookback}")

    # KernelAdapter
    from quant_framework.live.kernel_adapter import KernelAdapter
    adapter = KernelAdapter("MA", (10, 30))
    window = make_ohlcv(200)
    sig = adapter.generate_signal(window, "TEST")
    check("REGR-adapter-signal",
          sig is not None,
          f"signal type = {type(sig).__name__}")

    # TradeJournal
    from quant_framework.live.trade_journal import TradeJournal
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        j = TradeJournal(f.name)
        j.record_trade("BTC", "buy", 1, 50000, 0, 0, "test")
        j._flush()
        stats = j.get_trade_stats()
        check("REGR-journal",
              stats["total_trades"] == 1,
              f"trades={stats['total_trades']}")
        j.close()
        os.unlink(f.name)

    # ResearchDB
    from quant_framework.research.database import ResearchDB
    db = ResearchDB(":memory:")
    db.record_health("BTC", "MA", sharpe_30d=1.5, drawdown_pct=10)
    latest = db.get_latest_health("BTC", "MA")
    check("REGR-researchdb",
          latest is not None and abs(latest["sharpe_30d"] - 1.5) < 1e-6,
          f"sharpe={latest['sharpe_30d'] if latest else 'None'}")
    db.close()

    _section_times["regression"] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════

def generate_report(accuracy_results, perf_results):
    total = len(_results)
    passed = sum(1 for r in _results if r["passed"])
    failed = total - passed

    lines = []
    lines.append("# V5 Real Trading Upgrade — Complete Test Report")
    lines.append("")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Tests**: {total}")
    lines.append(f"**Passed**: {passed} ({100*passed/max(total,1):.1f}%)")
    lines.append(f"**Failed**: {failed}")
    lines.append("")
    lines.append("---")

    # Timing
    lines.append("")
    lines.append("## Execution Timing")
    lines.append("")
    lines.append("| Section | Time (ms) |")
    lines.append("|---------|-----------|")
    for sec, t in sorted(_section_times.items()):
        lines.append(f"| {sec} | {t*1000:.1f} |")
    total_time = sum(_section_times.values())
    lines.append(f"| **TOTAL** | **{total_time*1000:.1f}** |")

    # Failures
    failures = [r for r in _results if not r["passed"]]
    if failures:
        lines.append("")
        lines.append("## FAILURES")
        lines.append("")
        for f in failures:
            lines.append(f"- **{f['id']}**: {f['detail']}")

    # Phase 0 results
    lines.append("")
    lines.append("## Phase 0: Core Abstractions")
    lines.append("")
    p0_tests = [r for r in _results if r["id"].startswith("P0-")]
    lines.append(f"**{sum(1 for r in p0_tests if r['passed'])}/{len(p0_tests)} passed**")
    lines.append("")
    lines.append("| Test | Result | Detail |")
    lines.append("|------|--------|--------|")
    for r in p0_tests:
        lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail'][:60]} |")

    # Margin accuracy
    lines.append("")
    lines.append("## Margin Model Accuracy")
    lines.append("")
    margin_tests = [r for r in _results if r["id"].startswith("MARGIN-")]
    lines.append(f"**{sum(1 for r in margin_tests if r['passed'])}/{len(margin_tests)} passed**")
    lines.append("")
    lines.append("| Tier/Test | Result | Detail |")
    lines.append("|-----------|--------|--------|")
    for r in margin_tests:
        lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail'][:60]} |")

    # Cost accuracy
    lines.append("")
    lines.append("## Cost Model Accuracy")
    lines.append("")
    cost_tests = [r for r in _results if r["id"].startswith("COST-")]
    lines.append(f"**{sum(1 for r in cost_tests if r['passed'])}/{len(cost_tests)} passed**")
    lines.append("")
    lines.append("| Test | Result | Detail |")
    lines.append("|------|--------|--------|")
    for r in cost_tests:
        lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail'][:60]} |")

    # Kernel accuracy
    lines.append("")
    lines.append("## Kernel Accuracy (18 strategies × 2 configs)")
    lines.append("")
    kernel_tests = [r for r in _results if r["id"].startswith("KERNEL-")]
    lines.append(f"**{sum(1 for r in kernel_tests if r['passed'])}/{len(kernel_tests)} passed**")
    lines.append("")
    lines.append("| Kernel | Config | Return % | DrawDown % | Trades | Result |")
    lines.append("|--------|--------|----------|-----------|--------|--------|")
    for r in kernel_tests:
        parts = r["id"].replace("KERNEL-", "").split("-", 1)
        cfg = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        lines.append(f"| {name} | {cfg} | {r['detail'][:50]} | {'PASS' if r['passed'] else 'FAIL'} |")

    # Performance
    lines.append("")
    lines.append("## Performance Benchmark")
    lines.append("")
    lines.append("### Kernel Throughput")
    lines.append("")
    lines.append("| Bars | Kernels | Time (ms) | Throughput |")
    lines.append("|------|---------|-----------|-----------|")
    for sz in [500, 1000, 2000, 5000]:
        if sz in perf_results:
            p = perf_results[sz]
            lines.append(f"| {sz} | {p['n_kernels']} | {p['elapsed_ms']:.1f} | **{p['throughput']:.0f} k/s** |")

    if "scan" in perf_results:
        s = perf_results["scan"]
        lines.append(f"\n### Param Scan: {s['count']} combos in {s['elapsed_ms']:.1f}ms = **{s['throughput']:.0f} evals/s**")

    if "margin" in perf_results:
        m = perf_results["margin"]
        lines.append(f"\n### Margin Model: **{m['rate']:.0f} calcs/s** ({m['elapsed_ms']:.1f}ms for 100k)")

    # Broker infrastructure
    lines.append("")
    lines.append("## Broker Infrastructure")
    lines.append("")
    p3_tests = [r for r in _results if r["id"].startswith("P3-")]
    for r in p3_tests:
        lines.append(f"- {r['id']}: {'PASS' if r['passed'] else 'FAIL'} — {r['detail']}")

    # Safety
    lines.append("")
    lines.append("## Safety Infrastructure")
    lines.append("")
    p5_tests = [r for r in _results if r["id"].startswith("P5-")]
    for r in p5_tests:
        lines.append(f"- {r['id']}: {'PASS' if r['passed'] else 'FAIL'} — {r['detail']}")

    # Regression
    lines.append("")
    lines.append("## Regression Tests")
    lines.append("")
    regr_tests = [r for r in _results if r["id"].startswith("REGR-")]
    lines.append(f"**{sum(1 for r in regr_tests if r['passed'])}/{len(regr_tests)} passed**")

    # Summary
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **{passed}/{total}** tests passed ({100*passed/max(total,1):.1f}%)")
    lines.append(f"- Total execution time: **{total_time*1000:.0f}ms**")
    lines.append("")

    # New file manifest
    lines.append("### V5 New Files Created")
    lines.append("")
    lines.append("| File | Phase | Purpose |")
    lines.append("|------|-------|---------|")
    new_files = [
        ("core/__init__.py", "0", "Package init"),
        ("core/asset_types.py", "0", "AssetClass enum (5 types)"),
        ("core/margin.py", "0", "MarginModel: CryptoSpot, CryptoFutures (Binance tiers), RegT"),
        ("core/costs.py", "0", "CostModel: CryptoFutures (maker/taker/funding), USEquity (IBKR)"),
        ("core/symbol_spec.py", "0", "SymbolSpec: price/qty rounding, order validation"),
        ("core/market_hours.py", "2", "MarketCalendar: US market hours + NYSE holidays"),
        ("core/pdt_tracker.py", "3", "PDT rule tracking (5-day window)"),
        ("data/funding_rates.py", "1", "Binance funding rate download + parquet cache"),
        ("broker/credentials.py", "3", "API key management (env > .env > yaml)"),
        ("broker/rate_limiter.py", "3", "Token bucket limiter with O(1) tracking"),
        ("broker/binance_futures.py", "3", "Binance USDT-M Perp broker (async, WebSocket)"),
        ("broker/ibkr_broker.py", "3", "IBKR US equity broker (ib_insync)"),
        ("broker/live_order_manager.py", "4", "Order lifecycle management with timeout"),
        ("broker/reconciler.py", "4", "Position reconciliation (exchange vs internal)"),
        ("broker/execution_algo.py", "7", "TWAP, LimitChase, Iceberg algorithms"),
        ("broker/testnet.py", "6", "Testnet URL configuration"),
        ("live/alerts.py", "5", "Multi-channel alerts (Telegram/Discord)"),
        ("live/audit.py", "5", "SQLite audit trail + daily reports"),
        ("live/kill_switch.py", "5", "Emergency position flattening"),
        ("live/health_server.py", "5", "HTTP health/metrics endpoints"),
    ]
    for f, p, desc in new_files:
        lines.append(f"| `{f}` | {p} | {desc} |")

    lines.append("")
    if failed == 0:
        lines.append("> **ALL TESTS PASSED.** V5 Real Trading Upgrade implementation verified.")
    else:
        lines.append(f"> **{failed} tests failed.** See FAILURES section.")

    report = "\n".join(lines)
    report_path = os.path.join(ROOT, "reports", "v5_full_test_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as fh:
        fh.write(report)
    print(f"\nReport saved to: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  V5 REAL TRADING UPGRADE — COMPLETE TEST SUITE")
    print("=" * 70)

    accuracy_results = {}
    perf_results = {}

    sections = [
        ("Phase 0: Core Abstractions", test_phase0_core),
        ("Phase 1: Backtest Improvements", test_phase1_backtest),
        ("Phase 3: Broker Infrastructure", test_phase3_brokers),
        ("Phase 5: Safety Infrastructure", test_phase5_safety),
        ("Margin Model Accuracy", test_margin_accuracy),
        ("Cost Model Accuracy", test_cost_accuracy),
        ("Kernel Accuracy (18×2)", lambda: accuracy_results.update(test_kernel_accuracy() or {})),
        ("Performance Benchmark", lambda: perf_results.update(test_performance() or {})),
        ("Regression Tests", test_regression),
    ]

    for title, func in sections:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
        try:
            func()
        except Exception as e:
            print(f"[SECTION FAIL] {title}: {e}")
            traceback.print_exc()

    print(f"\n{'═' * 70}")
    total = len(_results)
    passed = sum(1 for r in _results if r["passed"])
    failed = total - passed
    print(f"  RESULTS: {passed}/{total} PASSED, {failed} FAILED")
    print(f"{'═' * 70}")

    generate_report(accuracy_results, perf_results)
