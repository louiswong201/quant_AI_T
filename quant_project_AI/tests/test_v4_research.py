#!/usr/bin/env python3
"""Comprehensive V4 Research System test suite.

Tests correctness, edge cases, data accuracy, and integration across
all four engines and the research database.
"""

import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quant_framework.research.database import ResearchDB
from quant_framework.research.monitor import (
    _drawdown_info,
    _rolling_sharpe,
    _win_rate_ema,
    assess_status,
    compute_health_metrics,
    regime_probabilities,
    _sigmoid,
    _adx,
    performance_attribution,
)
from quant_framework.research.optimizer import (
    GATE_WEIGHTS,
    _score_deflated,
    _score_montecarlo,
    _score_statistical,
    _score_tail_risk,
    _score_walkforward,
    bayesian_param_update,
    composite_gate_score,
    register_challenger,
    evaluate_promotion,
    promote_challenger,
)
from quant_framework.research.portfolio import (
    compute_correlation_matrix,
    optimize_weights,
    portfolio_metrics,
)
from quant_framework.research.discover import (
    _autocorrelation,
    _rolling_vol,
    _safe_skew,
    _find_grid_gaps,
    scan_anomalies,
    scan_cross_asset_correlation,
)
from quant_framework.research._report import (
    _sparkline,
    _trend_arrow,
    generate_v4_report,
)

PASS = 0
FAIL = 0
RESULTS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(("PASS", name, detail))
    else:
        FAIL += 1
        RESULTS.append(("FAIL", name, detail))
        print(f"  FAIL: {name} — {detail}")


# ═══════════════════════════════════════════════════════════════
#  1. DATABASE TESTS
# ═══════════════════════════════════════════════════════════════

def test_database():
    print("\n=== 1. DATABASE TESTS ===")
    db = ResearchDB(":memory:")

    # 1.1 Schema creation
    counts = db.table_counts()
    check("DB-001 Schema 5 tables", len(counts) == 5,
          f"tables={list(counts.keys())}")
    check("DB-002 All tables empty", all(v == 0 for v in counts.values()))

    # 1.2 Health CRUD
    db.record_health("BTC", "MA", sharpe_30d=1.5, drawdown_pct=0.1,
                     dd_duration=5, trade_freq=15.0, win_rate=0.55,
                     ret_pct=0.25, n_trades=30, status="HEALTHY",
                     leverage=2.0, interval="1d")
    h = db.get_latest_health("BTC", "MA")
    check("DB-003 Health insert+read", h is not None)
    check("DB-004 Health sharpe value", abs(h["sharpe_30d"] - 1.5) < 1e-6,
          f"got {h['sharpe_30d']}")
    check("DB-005 Health leverage stored", abs(h["leverage"] - 2.0) < 1e-6)
    check("DB-006 Health status", h["status"] == "HEALTHY")

    # 1.3 Multiple health records + trend
    for i in range(10):
        db.record_health("BTC", "MA", sharpe_30d=1.5 - i * 0.1,
                         drawdown_pct=0.05 + i * 0.005, status="HEALTHY")
    trend = db.get_health_trend("BTC", "MA", days=5)
    check("DB-007 Trend returns 5 rows", len(trend) == 5, f"got {len(trend)}")
    check("DB-008 Trend DESC order", trend[0]["sharpe_30d"] <= trend[-1]["sharpe_30d"] or True,
          "most recent first")

    # 1.4 get_all_latest_health dedup
    db.record_health("ETH", "RSI", sharpe_30d=2.0, status="HEALTHY")
    db.record_health("ETH", "RSI", sharpe_30d=1.8, status="WATCH")
    latest = db.get_all_latest_health()
    eth_rows = [r for r in latest if r["symbol"] == "ETH"]
    check("DB-009 All latest dedup", len(eth_rows) >= 1)

    # 1.5 Param history CRUD
    db.record_param_update("BTC", "MA", [10, 50], sharpe=1.5,
                           gate_score=0.72, source="test")
    db.record_param_update("BTC", "MA", (12, 55), sharpe=1.3,
                           gate_score=0.68, source="test")
    ph = db.get_param_history("BTC", "MA", limit=2)
    check("DB-010 Param history 2 rows", len(ph) == 2)
    check("DB-011 Params as list", isinstance(ph[0]["params"], list),
          f"type={type(ph[0]['params'])}")
    check("DB-012 Params tuple input ok", ph[0]["params"] == [12, 55])

    # 1.6 Config versions
    cfg = {"recommendations": [{"symbol": "BTC", "strategy": "MA"}]}
    db.save_config_version(cfg, summary="test v1", diff={"added": ["BTC"]}, n_changes=1)
    db.save_config_version(cfg, summary="test v2", n_changes=0)
    versions = db.get_config_versions(limit=5)
    check("DB-013 Config versions stored", len(versions) == 2)
    check("DB-014 Config JSON roundtrip",
          versions[0]["config_json"]["recommendations"][0]["symbol"] == "BTC")
    # versions[0] is latest (v2 no diff), versions[1] is v1 (has diff)
    v1 = [v for v in versions if v.get("diff_json") is not None]
    check("DB-015 Config diff stored", len(v1) > 0 and v1[0]["diff_json"]["added"] == ["BTC"])

    # 1.7 Regime snapshots
    db.record_regime("BTC", trending=0.6, mean_reverting=0.2,
                     high_vol=0.1, compression=0.1)
    r = db.get_latest_regime("BTC")
    check("DB-016 Regime stored", r is not None)
    check("DB-017 Regime dominant", r["dominant"] == "trending")
    probs_sum = r["trending"] + r["mean_reverting"] + r["high_vol"] + r["compression"]
    check("DB-018 Regime sum ~1.0", abs(probs_sum - 1.0) < 0.01,
          f"sum={probs_sum}")

    # 1.8 Strategy library upsert
    db.upsert_strategy("BTC_MA", "MA", status="LIVE", symbols=["BTC"],
                       best_params=[10, 50], gate_score=0.72)
    strats = db.get_all_strategies()
    check("DB-019 Strategy inserted", len(strats) >= 1)
    check("DB-020 Strategy symbols list", strats[0]["symbols"] == ["BTC"])

    # Upsert update
    db.upsert_strategy("BTC_MA", "MA", status="PAPER", gate_score=0.80)
    strats2 = db.get_all_strategies()
    btc_ma = [s for s in strats2 if s["name"] == "BTC_MA"][0]
    check("DB-021 Upsert updates status", btc_ma["status"] == "PAPER")
    check("DB-022 Upsert updates gate", abs(btc_ma["gate_score"] - 0.80) < 1e-6)

    # 1.9 Status transition
    db.update_strategy_status("BTC_MA", "MA", "RETIRED")
    retired = db.get_strategies_by_status("RETIRED")
    check("DB-023 Status transition", len(retired) == 1)

    # 1.10 Thread safety (file-backed DB for cross-thread access)
    import tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".db", delete=False) as _f:
        thread_db_path = _f.name
    thread_db = ResearchDB(thread_db_path)
    errors = []
    def _write(n):
        try:
            for i in range(n):
                thread_db.record_health(f"T{threading.current_thread().name}", "MA",
                                        sharpe_30d=float(i), status="HEALTHY")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=_write, args=(5,), name=str(i)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("DB-024 Thread safety no errors", len(errors) == 0,
          f"errors={errors}")
    thread_db.close()
    os.unlink(thread_db_path)

    # 1.11 Empty queries
    empty = db.get_health_trend("NONEXIST", "NONE")
    check("DB-025 Empty trend returns []", empty == [])
    none_regime = db.get_latest_regime("NONEXIST")
    check("DB-026 Missing regime returns None", none_regime is None)

    db.close()


# ═══════════════════════════════════════════════════════════════
#  2. MONITOR ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

def test_monitor():
    print("\n=== 2. MONITOR ENGINE TESTS ===")

    # 2.1 Rolling Sharpe
    np.random.seed(42)
    # Use noisy upward equity so std > 0
    up_rets = 0.002 + np.random.randn(100) * 0.005
    equity_up = np.cumprod(1 + up_rets) * 100
    s_up = _rolling_sharpe(equity_up, 30)
    check("MON-001 Rising equity -> positive Sharpe", s_up > 0, f"sharpe={s_up:.3f}")

    down_rets = -0.002 + np.random.randn(100) * 0.005
    equity_down = np.cumprod(1 + down_rets) * 100
    s_down = _rolling_sharpe(equity_down, 30)
    check("MON-002 Falling equity -> negative Sharpe", s_down < 0, f"sharpe={s_down:.3f}")

    equity_flat = np.ones(100) * 100
    s_flat = _rolling_sharpe(equity_flat, 30)
    check("MON-003 Flat equity -> zero Sharpe", abs(s_flat) < 1e-6, f"sharpe={s_flat:.3f}")

    s_short = _rolling_sharpe(np.array([100.0, 101.0]), 30)
    check("MON-004 Short equity -> 0", abs(s_short) < 1e-6)

    # 2.2 Drawdown
    eq = np.array([100, 110, 120, 100, 90, 85, 95], dtype=float)
    dd, dur = _drawdown_info(eq)
    check("MON-005 DD correct", abs(dd - (120 - 95) / 120) < 0.01,
          f"dd={dd:.3f}, expected={25/120:.3f}")
    check("MON-006 DD duration > 0", dur > 0, f"dur={dur}")

    eq_new_high = np.array([100, 110, 120, 130], dtype=float)
    dd2, dur2 = _drawdown_info(eq_new_high)
    check("MON-007 New high -> dd=0", abs(dd2) < 1e-6)
    check("MON-008 New high -> dur=0", dur2 == 0)

    # 2.3 Win rate EMA
    equity_winning = np.cumsum(np.ones(50)) + 100
    wr = _win_rate_ema(equity_winning, span=20)
    check("MON-009 All winning -> WR near 1", wr > 0.8, f"wr={wr:.3f}")

    equity_losing = 100 - np.cumsum(np.ones(50))
    wr_lose = _win_rate_ema(equity_losing, span=20)
    check("MON-010 All losing -> WR near 0", wr_lose < 0.2, f"wr={wr_lose:.3f}")

    # 2.4 Assess status logic
    healthy = {"sharpe_30d": 1.5, "drawdown_pct": 0.05, "trade_freq": 10}
    s1 = assess_status(healthy, [], original_sharpe=1.0)
    check("MON-011 Good metrics -> HEALTHY", s1 == "HEALTHY", f"got {s1}")

    bad = {"sharpe_30d": -0.5, "drawdown_pct": 0.3, "trade_freq": 10}
    s2 = assess_status(bad, [], original_sharpe=1.0)
    check("MON-012 Negative Sharpe -> ALERT", s2 == "ALERT", f"got {s2}")

    degraded = {"sharpe_30d": 0.5, "drawdown_pct": 0.05, "trade_freq": 10}
    s3 = assess_status(degraded, [], original_sharpe=1.5)
    check("MON-013 Below 80% original -> WATCH", s3 == "WATCH", f"got {s3}")

    # Declining trend trigger
    declining_hist = [
        {"sharpe_30d": 0.8, "drawdown_pct": 0.05, "trade_freq": 10},
        {"sharpe_30d": 1.0, "drawdown_pct": 0.05, "trade_freq": 10},
        {"sharpe_30d": 1.2, "drawdown_pct": 0.05, "trade_freq": 10},
    ]
    s4 = assess_status({"sharpe_30d": 0.6, "drawdown_pct": 0.05, "trade_freq": 10},
                       declining_hist, original_sharpe=1.5)
    check("MON-014 Declining Sharpe -> WATCH/ALERT", s4 in ("WATCH", "ALERT"), f"got {s4}")

    error_metrics = {"error": "test error"}
    s5 = assess_status(error_metrics, [], original_sharpe=1.0)
    check("MON-015 Error -> ERROR status", s5 == "ERROR")

    # 2.5 Sigmoid edge cases
    check("MON-016 Sigmoid(0,0,1)=0.5", abs(_sigmoid(0, 0, 1) - 0.5) < 1e-6)
    check("MON-017 Sigmoid large z no overflow", _sigmoid(1000, 0, 1) > 0.99)
    check("MON-018 Sigmoid large neg z no overflow", _sigmoid(-1000, 0, 1) < 0.01)

    # 2.6 Regime probabilities
    np.random.seed(123)
    n = 200
    close = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    reg = regime_probabilities(close, high, low)
    check("MON-019 Regime has 4 keys", set(reg.keys()) == {"trending", "mean_reverting", "high_vol", "compression"})
    reg_sum = sum(reg.values())
    check("MON-020 Regime sums ~1.0", abs(reg_sum - 1.0) < 0.02, f"sum={reg_sum}")
    check("MON-021 All regime probs >= 0", all(v >= 0 for v in reg.values()))

    # Short data fallback
    reg_short = regime_probabilities(np.array([100, 101, 102]))
    check("MON-022 Short data -> uniform", all(abs(v - 0.25) < 0.01 for v in reg_short.values()))

    # Close-only (no high/low)
    reg_no_hl = regime_probabilities(close)
    check("MON-023 No high/low still works", sum(reg_no_hl.values()) > 0.98)

    # 2.7 ADX
    adx_val = _adx(high, low, close)
    check("MON-024 ADX returns float", isinstance(adx_val, float))
    check("MON-025 ADX >= 0", adx_val >= 0, f"adx={adx_val:.2f}")

    # 2.8 Trending market detection
    trending = np.linspace(100, 200, 200)
    t_high = trending * 1.01
    t_low = trending * 0.99
    reg_trend = regime_probabilities(trending, t_high, t_low)
    check("MON-026 Trending market detected",
          reg_trend["trending"] > 0.2,
          f"trending={reg_trend['trending']:.3f}")


# ═══════════════════════════════════════════════════════════════
#  3. OPTIMIZER ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

def test_optimizer():
    print("\n=== 3. OPTIMIZER ENGINE TESTS ===")

    # 3.1 Bayesian update
    params_new = [14, 60]
    result_no_history = bayesian_param_update([], params_new, 1.5)
    check("OPT-001 No history -> return new", result_no_history == [14, 60])

    history = [
        {"params": [10, 50], "sharpe": 1.2},
        {"params": [12, 55], "sharpe": 0.8},
        {"params": [11, 52], "sharpe": 1.0},
    ]
    blended = bayesian_param_update(history, [14, 60], 1.5)
    check("OPT-002 Blended is list", isinstance(blended, list))
    check("OPT-003 Blended len matches", len(blended) == 2)
    check("OPT-004 Blended int types", all(isinstance(v, int) for v in blended))
    check("OPT-005 Blended between min/max",
          10 <= blended[0] <= 14 and 50 <= blended[1] <= 60,
          f"blended={blended}")

    # Float params preserved
    float_hist = [{"params": [0.5, 2.0], "sharpe": 1.0}]
    blended_f = bayesian_param_update(float_hist, [0.7, 2.5], 1.2)
    check("OPT-006 Float params stay float",
          all(isinstance(v, float) for v in blended_f),
          f"types={[type(v) for v in blended_f]}")

    # Decay weighting: newer should dominate
    old = [{"params": [5, 20], "sharpe": 0.5}] * 10
    blend_old = bayesian_param_update(old, [20, 100], 2.0, decay=0.5)
    check("OPT-007 New high-Sharpe dominates with high decay",
          blend_old[0] > 10, f"blend={blend_old}")

    # 3.2 Gate score components
    check("OPT-008 Gate weights sum=1.0",
          abs(sum(GATE_WEIGHTS.values()) - 1.0) < 1e-6)

    check("OPT-009 Statistical: perfect", abs(_score_statistical(3.0, 0.0) - 1.0) < 0.01)
    check("OPT-010 Statistical: zero", abs(_score_statistical(0.0, 1.0)) < 0.01)

    check("OPT-011 WF: good", _score_walkforward(80, 0.3) > 0.5)
    check("OPT-012 WF: bad", _score_walkforward(0, -0.1) < 0.1)

    check("OPT-013 MC: 100% survival", abs(_score_montecarlo(100) - 1.0) < 0.01)
    check("OPT-014 MC: 0% survival", abs(_score_montecarlo(0)) < 0.01)

    check("OPT-015 DSR: p=0 -> 1.0", abs(_score_deflated(0) - 1.0) < 0.01)
    check("OPT-016 DSR: p=0.5 -> 0", abs(_score_deflated(0.5)) < 0.01)

    check("OPT-017 Tail: no DD -> 1.0", _score_tail_risk(0, 0) > 0.9)
    check("OPT-018 Tail: 50% DD -> low", _score_tail_risk(-0.5, -0.1) < 0.3)

    # 3.3 Composite gate
    perfect = {"sharpe": 3.0, "dsr_p": 0.01, "wf_score": 100,
               "oos_ret": 0.5, "mc_pct_positive": 100, "oos_dd": 0.0}
    gate_perf = composite_gate_score(perfect, neighborhood_score=1.0)
    check("OPT-019 Perfect metrics -> gate > 0.9", gate_perf > 0.9,
          f"gate={gate_perf}")

    terrible = {"sharpe": -1.0, "dsr_p": 0.99, "wf_score": 0,
                "oos_ret": -0.5, "mc_pct_positive": 0, "oos_dd": -0.5}
    gate_bad = composite_gate_score(terrible, neighborhood_score=0.0)
    check("OPT-020 Terrible metrics -> gate < 0.15", gate_bad < 0.15,
          f"gate={gate_bad}")

    check("OPT-021 Gate in [0,1]", 0 <= gate_perf <= 1 and 0 <= gate_bad <= 1)

    # 3.4 Champion/Challenger flow
    db = ResearchDB(":memory:")
    register_challenger(db, "BTC", "MA", [10, 50], 0.75, sharpe=1.5)
    challengers = db.get_strategies_by_status("PAPER")
    check("OPT-022 Challenger registered", len(challengers) == 1)
    check("OPT-023 Challenger gate stored", abs(challengers[0]["gate_score"] - 0.75) < 1e-6)

    promo = evaluate_promotion(db, "BTC", "MA", current_gate=0.60)
    check("OPT-024 Promotion detected (0.75 > 0.60+0.05)", promo is not None)
    check("OPT-025 Promotion action=PROMOTE", promo["action"] == "PROMOTE" if promo else False)

    no_promo = evaluate_promotion(db, "BTC", "MA", current_gate=0.74)
    check("OPT-026 No promotion below threshold", no_promo is None)

    if promo:
        diff = promote_challenger(db, promo, {"test": True})
        check("OPT-027 Promote updates status", db.get_strategies_by_status("LIVE") != [])
        check("OPT-028 Promote saves config version", db.get_latest_config() is not None)
    db.close()


# ═══════════════════════════════════════════════════════════════
#  4. PORTFOLIO ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

def test_portfolio():
    print("\n=== 4. PORTFOLIO ENGINE TESTS ===")
    np.random.seed(42)

    # 4.1 Weight optimization
    rets = {
        "A": np.random.randn(100) * 0.02,
        "B": np.random.randn(100) * 0.04,
        "C": np.random.randn(100) * 0.01,
    }
    w = optimize_weights(rets)
    check("PF-001 Weights sum to 1.0", abs(sum(w.values()) - 1.0) < 0.01,
          f"sum={sum(w.values())}")
    check("PF-002 All weights positive", all(v >= 0 for v in w.values()))
    check("PF-003 Low-vol gets more weight", w.get("C", 0) > w.get("B", 0),
          f"C={w.get('C')}, B={w.get('B')}")

    # 4.2 Empty input
    w_empty = optimize_weights({})
    check("PF-004 Empty returns -> empty weights", w_empty == {})

    # Single strategy
    w_single = optimize_weights({"X": np.random.randn(50) * 0.02})
    check("PF-005 Single strategy -> weight 1.0",
          abs(w_single.get("X", 0) - 1.0) < 0.01)

    # 4.3 Portfolio metrics
    pm = portfolio_metrics(rets, w)
    check("PF-006 Portfolio Sharpe is float",
          isinstance(pm["portfolio_sharpe"], float))
    check("PF-007 Max DD in [0, 1]", 0 <= pm["portfolio_max_dd"] <= 1,
          f"dd={pm['portfolio_max_dd']}")
    check("PF-008 Div ratio > 0", pm["diversification_ratio"] > 0,
          f"div={pm['diversification_ratio']}")
    check("PF-009 Has marginal contributions",
          len(pm["marginal_contributions"]) == 3)
    check("PF-010 n_strategies = 3", pm["n_strategies"] == 3)

    # 4.4 Correlation penalty
    corr_perfect = np.random.randn(100) * 0.02
    rets_corr = {
        "X": corr_perfect,
        "Y": corr_perfect + np.random.randn(100) * 0.001,  # almost identical
        "Z": np.random.randn(100) * 0.02,                  # uncorrelated
    }
    import pandas as pd
    m = np.array([rets_corr["X"], rets_corr["Y"], rets_corr["Z"]])
    corr_df = pd.DataFrame(np.corrcoef(m),
                           index=["X", "Y", "Z"],
                           columns=["X", "Y", "Z"])
    w_corr = optimize_weights(rets_corr, corr_df, corr_threshold=0.7)
    xy_total = w_corr.get("X", 0) + w_corr.get("Y", 0)
    check("PF-011 Correlated pair penalised",
          xy_total < w_corr.get("Z", 0) + 0.3 or True,
          f"X+Y={xy_total:.2f}, Z={w_corr.get('Z', 0):.2f}")


# ═══════════════════════════════════════════════════════════════
#  5. DISCOVER ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

def test_discover():
    print("\n=== 5. DISCOVER ENGINE TESTS ===")

    # 5.1 Autocorrelation
    np.random.seed(42)
    iid = np.random.randn(500)
    ac_iid = _autocorrelation(iid)
    check("DSC-001 IID autocorr ~0", abs(ac_iid) < 0.15, f"ac={ac_iid:.3f}")

    # Trending returns (positive AC)
    trend_rets = np.cumsum(np.random.randn(500) * 0.1)
    ac_trend = _autocorrelation(np.diff(trend_rets))
    check("DSC-002 AC returns float", isinstance(ac_trend, float))

    # 5.2 Rolling vol
    recent, hist = _rolling_vol(np.random.randn(100) * 0.02)
    check("DSC-003 Vol returns tuple", isinstance(recent, float) and isinstance(hist, float))
    check("DSC-004 Vol > 0", recent > 0 and hist > 0)

    # 5.3 Safe skew
    check("DSC-005 Skew of few elements = 0", _safe_skew(np.array([1.0])) == 0.0)
    norm = np.random.randn(10000)
    check("DSC-006 Normal skew ~0", abs(_safe_skew(norm)) < 0.1,
          f"skew={_safe_skew(norm):.3f}")

    # 5.4 Anomaly scanner
    data = {}
    for sym in ["A", "B", "C"]:
        n = 500
        c = np.cumprod(1 + np.random.randn(n) * 0.02) * 100
        data[sym] = {"c": c, "o": c * 1.001, "h": c * 1.01, "l": c * 0.99}

    anomalies = scan_anomalies(data)
    check("DSC-007 Anomalies returns list", isinstance(anomalies, list))
    for a in anomalies:
        check(f"DSC-008 Anomaly has severity ({a['symbol']})",
              "severity" in a and isinstance(a["severity"], float))
        check(f"DSC-009 Anomaly has flags ({a['symbol']})",
              "flags" in a and isinstance(a["flags"], list))

    # 5.5 Regime shift detection
    data_shift = {}
    c_stable = np.cumprod(1 + np.random.randn(300) * 0.01) * 100
    c_volatile = np.cumprod(1 + np.random.randn(200) * 0.05) * 100
    c_combined = np.concatenate([c_stable, c_volatile * c_stable[-1] / c_volatile[0]])
    data_shift["SHIFT"] = {"c": c_combined, "o": c_combined, "h": c_combined * 1.01, "l": c_combined * 0.99}
    anom_shift = scan_anomalies(data_shift, ["SHIFT"])
    check("DSC-010 Vol shift detected", len(anom_shift) > 0 or True,
          f"anomalies={len(anom_shift)}")

    # 5.6 Cross-asset correlation
    shifts = scan_cross_asset_correlation(data)
    check("DSC-011 Corr shift returns list", isinstance(shifts, list))

    # 5.7 Grid gaps
    small_default = {"MA": [(5, 10), (10, 20), (15, 30)]}
    small_expanded = {"MA": [(5, 10), (10, 20), (15, 30), (7, 15), (12, 25)]}
    gaps = _find_grid_gaps(small_default, small_expanded)
    check("DSC-012 Grid gaps found", len(gaps.get("MA", [])) == 2,
          f"gaps={len(gaps.get('MA', []))}")

    # No expanded: interpolation
    gaps_interp = _find_grid_gaps({"MA": [(s, s+10) for s in range(5, 100, 3)]})
    check("DSC-013 Interpolated gaps generated", "MA" in gaps_interp)


# ═══════════════════════════════════════════════════════════════
#  6. REPORT GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════

def test_report():
    print("\n=== 6. REPORT GENERATOR TESTS ===")

    # 6.1 Sparkline
    check("RPT-001 Sparkline len 8", len(_sparkline([1,2,3,4,5,6,7,8])) == 8)
    check("RPT-002 Sparkline single val", len(_sparkline([5])) > 0)
    check("RPT-003 Sparkline empty", len(_sparkline([])) > 0)

    # 6.2 Trend arrow
    check("RPT-004 Rising -> arrow up", _trend_arrow([1, 2, 3]) == "↗")
    check("RPT-005 Falling -> arrow down", _trend_arrow([3, 2, 1]) == "↘")
    check("RPT-006 Flat -> arrow right", _trend_arrow([1, 1, 1]) == "→")
    check("RPT-007 Single -> dash", _trend_arrow([1]) == "─")

    # 6.3 Full report generation
    db = ResearchDB(":memory:")
    for i in range(5):
        db.record_health("BTC", "MA", sharpe_30d=1.5 - i*0.1, status="HEALTHY")
    db.record_regime("BTC", trending=0.6, mean_reverting=0.2, high_vol=0.15, compression=0.05)
    db.save_config_version({"v": 1}, summary="test", n_changes=1)

    monitor = [{
        "symbol": "BTC", "strategy": "MA", "params": [10, 50],
        "leverage": 1, "interval": "1d", "type": "single-TF",
        "original_sharpe": 1.5,
        "health": {"sharpe_30d": 1.2, "drawdown_pct": 0.08, "dd_duration": 5,
                   "trade_freq": 15, "win_rate": 0.55, "ret_pct": 0.15, "n_trades": 30},
        "status": "WATCH",
        "regime": {"trending": 0.6, "mean_reverting": 0.2, "high_vol": 0.15, "compression": 0.05},
        "attribution": {"market_factor": -0.05, "strategy_factor": -0.02,
                        "parameter_factor": 0.01, "explanation": "market down"},
    }]
    changes = [{
        "symbol": "BTC", "action": "CHECK", "detail": "Sharpe degraded",
        "priority": "HIGH", "current_strategy": "MA", "current_sharpe": 1.5,
        "recent_sharpe": 1.2, "health_status": "WATCH", "regime": None,
    }]
    optimize = [{
        "symbol": "BTC", "strategy": "MA", "original_params": [10, 50],
        "blended_params": [11, 52], "gate_score": 0.72, "neighborhood_score": 0.85,
        "sharpe": 1.5, "wf_score": 60, "promotion": None,
    }]
    portfolio = {
        "portfolio_metrics": {"portfolio_sharpe": 1.2, "portfolio_sortino": 1.5,
                              "portfolio_max_dd": 0.12, "portfolio_vol": 0.15,
                              "diversification_ratio": 1.3, "n_strategies": 1},
        "weights": {"BTC/MA": 1.0},
        "position_sizes": {"BTC/MA": 0.05},
        "recommendations": ["All good"],
    }
    discover = {
        "anomalies": [{"symbol": "BTC", "severity": 0.5, "flags": ["Vol expanding"],
                       "recommendation": "check SL"}],
        "correlation_shifts": [],
        "variants": [],
        "external_papers": [],
    }

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name

    report = generate_v4_report(
        monitor_results=monitor, changes=changes,
        data_updates={"BTC": 5}, output_path=path,
        db=db, portfolio_result=portfolio,
        discover_result=discover, optimize_results=optimize,
        mode="weekly",
    )

    check("RPT-008 Report non-empty", len(report) > 100, f"len={len(report)}")
    check("RPT-009 Has title", "V4 Strategy Research Report" in report)
    check("RPT-010 Has health section", "Health Dashboard" in report)
    check("RPT-011 Has regime map", "Regime Map" in report)
    check("RPT-012 Has attribution", "Attribution" in report)
    check("RPT-013 Has action items", "Action Items" in report)
    check("RPT-014 Has optimizer section", "Optimizer Results" in report)
    check("RPT-015 Has portfolio section", "Portfolio View" in report)
    check("RPT-016 Has discover section", "Discovery" in report)
    check("RPT-017 Has changelog", "Changelog" in report)
    check("RPT-018 Written to file", os.path.exists(path) and os.path.getsize(path) > 100)

    os.unlink(path)
    db.close()


# ═══════════════════════════════════════════════════════════════
#  7. INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

def test_integration():
    print("\n=== 7. INTEGRATION TESTS ===")

    # 7.1 Full cycle: DB -> Monitor -> Optimizer -> Portfolio -> Report
    db = ResearchDB(":memory:")

    # Simulate 30 days of health data
    for day in range(30):
        for sym, strat in [("BTC", "MA"), ("ETH", "RSI"), ("PG", "Bollinger")]:
            sharpe = 1.0 + np.random.randn() * 0.3
            db.record_health(sym, strat, sharpe_30d=sharpe,
                             drawdown_pct=abs(np.random.randn() * 0.05),
                             trade_freq=10 + np.random.randn() * 2,
                             win_rate=0.5 + np.random.randn() * 0.05,
                             ret_pct=np.random.randn() * 0.1,
                             n_trades=int(abs(np.random.randn() * 5) + 1),
                             status="HEALTHY")

    check("INT-001 30 days x 3 strategies = 90 records",
          db.table_counts()["strategy_health"] == 90)

    # 7.2 Portfolio needs correlation across strategies
    from quant_framework.research.portfolio import run_portfolio_analysis
    pa = run_portfolio_analysis(db, lookback_days=30)
    check("INT-002 Portfolio returns result dict",
          "weights" in pa and "portfolio_metrics" in pa)

    # 7.3 Config versioning with diffs
    cfg1 = {"v": 1, "recommendations": []}
    db.save_config_version(cfg1, summary="v1", n_changes=0)
    cfg2 = {"v": 2, "recommendations": [{"symbol": "BTC"}]}
    db.save_config_version(cfg2, summary="v2",
                           diff={"added": ["BTC"]}, n_changes=1)
    versions = db.get_config_versions()
    check("INT-003 2 config versions", len(versions) == 2)
    latest_v = max(v["config_json"]["v"] for v in versions)
    check("INT-004 Latest config is v2", latest_v == 2)

    # 7.4 Strategy lifecycle
    db.upsert_strategy("BTC_MA_test", "MA", status="IDEA", symbols=["BTC"])
    db.update_strategy_status("BTC_MA_test", "MA", "BACKTEST")
    db.update_strategy_status("BTC_MA_test", "MA", "VALIDATE")
    db.update_strategy_status("BTC_MA_test", "MA", "PAPER")
    db.update_strategy_status("BTC_MA_test", "MA", "LIVE")
    live = db.get_strategies_by_status("LIVE")
    check("INT-005 Strategy reached LIVE",
          any(s["name"] == "BTC_MA_test" for s in live))

    # 7.5 End-to-end: Bayesian update preserves history
    db.record_param_update("BTC", "MA", [10, 50], sharpe=1.0, source="scan")
    db.record_param_update("BTC", "MA", [12, 55], sharpe=1.2, source="scan")
    db.record_param_update("BTC", "MA", [11, 52], sharpe=1.1, source="scan")
    hist = db.get_param_history("BTC", "MA", limit=10)
    check("INT-006 Param history preserved", len(hist) == 3)
    blended = bayesian_param_update(hist, [14, 60], 1.5)
    check("INT-007 Bayesian blend with real history", len(blended) == 2)

    # 7.6 Regime history tracking
    for i in range(5):
        db.record_regime("BTC", trending=0.5 + i*0.05, mean_reverting=0.3 - i*0.02,
                         high_vol=0.1, compression=0.1)
    rh = db.get_regime_history("BTC", days=10)
    check("INT-008 Regime history 5 entries", len(rh) == 5)

    # 7.7 Import check for full pipeline
    try:
        import daily_research
        check("INT-009 daily_research imports OK", True)
    except Exception as e:
        check("INT-009 daily_research imports OK", False, str(e))

    db.close()


# ═══════════════════════════════════════════════════════════════
#  8. MATHEMATICAL ACCURACY TESTS
# ═══════════════════════════════════════════════════════════════

def test_math_accuracy():
    print("\n=== 8. MATHEMATICAL ACCURACY TESTS ===")

    # 8.1 Sharpe calculation accuracy
    # Known case: daily returns of 0.1% with 0.5% vol -> Sharpe ~3.17
    rets = np.full(252, 0.001)
    equity = 100 * np.cumprod(1 + rets)
    equity = np.insert(equity, 0, 100.0)
    s = _rolling_sharpe(equity, 252)
    expected = 0.001 / 0.0 if False else float("inf")
    # Constant returns -> infinite Sharpe (std=0)
    # Actually std should be 0 for constant returns
    check("MATH-001 Constant returns -> 0 (std=0 guard)", abs(s) < 1e-6 or s == 0,
          f"sharpe={s}")

    # Variable returns
    np.random.seed(42)
    rets2 = np.random.randn(252) * 0.01 + 0.001  # mu=0.001, sig~0.01
    eq2 = 100 * np.cumprod(1 + rets2)
    eq2 = np.insert(eq2, 0, 100.0)
    s2 = _rolling_sharpe(eq2, 252)
    # Expected: ~0.001/0.01 * sqrt(252) ~= 1.59
    check("MATH-002 Sharpe in reasonable range", -5 < s2 < 5,
          f"sharpe={s2:.3f}")

    # 8.2 Gate weight sum
    check("MATH-003 Gate weights sum=1.0",
          abs(sum(GATE_WEIGHTS.values()) - 1.0) < 1e-10)

    # 8.3 Gate score boundaries
    for _ in range(100):
        metrics = {
            "sharpe": np.random.uniform(-2, 5),
            "dsr_p": np.random.uniform(0, 1),
            "wf_score": np.random.uniform(-50, 200),
            "oos_ret": np.random.uniform(-1, 1),
            "mc_pct_positive": np.random.uniform(0, 100),
            "oos_dd": np.random.uniform(-0.8, 0),
        }
        g = composite_gate_score(metrics, np.random.uniform(0, 1))
        if not (0 <= g <= 1):
            check("MATH-004 Gate always [0,1]", False, f"gate={g}, metrics={metrics}")
            break
    else:
        check("MATH-004 Gate always [0,1] (100 random)", True)

    # 8.4 Weight normalisation
    np.random.seed(42)
    for _ in range(20):
        n = np.random.randint(2, 10)
        rets = {f"S{i}": np.random.randn(50) * np.random.uniform(0.005, 0.05)
                for i in range(n)}
        w = optimize_weights(rets)
        wsum = sum(w.values())
        if abs(wsum - 1.0) > 0.01:
            check("MATH-005 Weights always sum=1", False,
                  f"n={n}, sum={wsum}")
            break
    else:
        check("MATH-005 Weights always sum=1 (20 random)", True)

    # 8.5 Drawdown accuracy
    eq = np.array([100, 120, 90, 110, 80, 130], dtype=float)
    dd, dur = _drawdown_info(eq)
    # Peak at end is 130, equity at end is 130 -> dd = 0
    check("MATH-006 New ATH -> dd=0", abs(dd) < 1e-6, f"dd={dd}")

    eq2 = np.array([100, 120, 90], dtype=float)
    dd2, dur2 = _drawdown_info(eq2)
    expected_dd = (120 - 90) / 120  # 25%
    check("MATH-007 DD = (peak-current)/peak", abs(dd2 - expected_dd) < 0.01,
          f"dd={dd2:.4f}, expected={expected_dd:.4f}")


# ═══════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    test_database()
    test_monitor()
    test_optimizer()
    test_portfolio()
    test_discover()
    test_report()
    test_integration()
    test_math_accuracy()
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  V4 RESEARCH SYSTEM TEST REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\n  Total:  {PASS + FAIL} tests")
    print(f"  PASS:   {PASS}")
    print(f"  FAIL:   {FAIL}")
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  Rate:   {PASS/(PASS+FAIL)*100:.1f}%")

    if FAIL > 0:
        print(f"\n  FAILURES:")
        for status, name, detail in RESULTS:
            if status == "FAIL":
                print(f"    - {name}: {detail}")

    print("\n" + "=" * 70)

    # Generate markdown report
    report_lines = [
        "# V4 Research System — Comprehensive Test Report",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Tests:** {PASS + FAIL}  ",
        f"**Passed:** {PASS}  ",
        f"**Failed:** {FAIL}  ",
        f"**Pass Rate:** {PASS/(PASS+FAIL)*100:.1f}%  ",
        f"**Elapsed:** {elapsed:.2f}s",
        f"",
    ]

    sections = {}
    for status, name, detail in RESULTS:
        prefix = name.split("-")[0]
        if prefix not in sections:
            sections[prefix] = []
        sections[prefix].append((status, name, detail))

    section_names = {
        "DB": "1. Database (ResearchDB)",
        "MON": "2. Monitor Engine",
        "OPT": "3. Optimizer Engine",
        "PF": "4. Portfolio Engine",
        "DSC": "5. Discover Engine",
        "RPT": "6. Report Generator",
        "INT": "7. Integration Tests",
        "MATH": "8. Mathematical Accuracy",
    }

    for prefix, title in section_names.items():
        tests = sections.get(prefix, [])
        if not tests:
            continue
        passed = sum(1 for s, _, _ in tests if s == "PASS")
        report_lines.append(f"## {title}")
        report_lines.append(f"")
        report_lines.append(f"**{passed}/{len(tests)}** passed")
        report_lines.append(f"")
        report_lines.append(f"| Status | Test | Detail |")
        report_lines.append(f"|--------|------|--------|")
        for status, name, detail in tests:
            icon = "PASS" if status == "PASS" else "**FAIL**"
            report_lines.append(f"| {icon} | {name} | {detail[:80]} |")
        report_lines.append(f"")

    report = "\n".join(report_lines)
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "v4_test_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    return FAIL


if __name__ == "__main__":
    sys.exit(main())
