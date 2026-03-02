# Parameter Decay Study Report

> **160** optimization origins | **8** symbols | train=500 bars (~2 years) | step=63 bars (~3 months)
> Strategies analyzed: MA, MACD, MomBreak (Top-3 from 10-Layer V3 report)
> Elapsed: 33s | All Numba @njit compiled

---

## Core Question

**How often should you re-optimize strategy parameters to maintain robust performance?**

This study measures parameter decay by:

1. Optimizing on a rolling training window of 500 bars (~2 years)
2. Freezing those params and measuring forward performance at 1-12 months
3. Repeating across 160 optimization origins across 8 symbols
4. Tracking cumulative return, win rate, and marginal monthly contribution

---

## 1. Cumulative Forward Return by Month Since Optimization

| Strategy | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 | Half-Life |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| MA | +0.0% | -0.5% | +0.5% | +2.0% | +1.9% | +5.2% | +5.8% | +8.4% | +10.3% | +11.1% | +13.3% | +15.7% | 2 mo |
| MACD | +1.2% | +2.4% | +5.2% | +6.4% | +7.4% | +8.5% | +9.0% | +9.6% | +10.7% | +11.0% | +12.7% | +14.9% | >12 mo |
| MomBreak | +0.3% | +1.6% | +3.7% | +5.7% | +5.4% | +7.8% | +7.4% | +8.2% | +9.2% | +9.5% | +10.1% | +11.4% | >12 mo |

**解读**:
- **MA**: 前2个月几乎没有正向收益，随后逐步回升。说明 MA 策略的最优参数对短期市场变化敏感，但长期趋势性资产仍可获利。
- **MACD**: 从 M1 起即有 +1.2% 正收益，且持续增长至 M12 (+14.9%)，半衰期超 12 个月。**参数最为稳定。**
- **MomBreak**: 与 MACD 类似，从 M1 (+0.3%) 起正向，但 M5/M7 出现轻微回撤。整体仍正向增长。

---

## 2. Win Rate Decay (% of origins with positive return)

| Strategy | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MA | 9% | 21% | 29% | 39% | 41% | 50% | 52% | 53% | 56% | 53% | 54% | 56% |
| MACD | 46% | 49% | 54% | 54% | 53% | 55% | 57% | 56% | 56% | 56% | 56% | 54% |
| MomBreak | 8% | 32% | 45% | 46% | 52% | 54% | 56% | 54% | 56% | 58% | 59% | 59% |

**解读**:
- **MA 和 MomBreak 的 M1 胜率极低 (8-9%)**：说明刚优化完的参数在第一个月的胜率不高，这是因为优化参数需要趋势延续才能获利，短期 (21 天) 信号不足。
- **MACD M1 = 46%**：接近随机但有轻微正向偏差，说明 MACD 信号在短期内仍有一定效力。
- **所有策略在 M6 达到 50%+ 胜率**：说明 6 个月是参数产生稳定正向收益的最小时间窗口。
- **M12 胜率: MA 56%, MACD 54%, MomBreak 59%**：都略高于 50%，说明参数在 12 个月后仍有微弱优势。

---

## 3. Marginal Monthly Return (每月增量收益)

The incremental return contributed by the Nth month. When this turns consistently negative, the optimized params are actively hurting performance.

| Strategy | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 | Suggestion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| MA | +0.0% | -0.5% | +0.9% | +1.5% | -0.1% | +3.3% | +0.6% | +2.6% | +1.8% | +0.9% | +2.2% | +2.4% | Re-opt <= 6 mo |
| MACD | +1.2% | +1.2% | +2.8% | +1.1% | +1.0% | +1.1% | +0.4% | +0.6% | +1.1% | +0.3% | +1.7% | +2.2% | Re-opt <= 6 mo |
| MomBreak | +0.3% | +1.2% | +2.1% | +2.0% | -0.2% | +2.4% | -0.4% | +0.7% | +1.0% | +0.4% | +0.5% | +1.3% | Re-opt <= 6 mo |

**解读**:
- **没有任何策略出现连续负增量**，说明这三个策略的参数没有"彻底失效"的现象。
- **MA 在 M2 出现 -0.5%，MomBreak 在 M5/M7 出现轻微负增量**：说明这些时间点市场可能经历了与优化期不同的行情。
- **整体趋势是每月正增量递减**（MACD: M3=+2.8% → M10=+0.3%），说明参数的"alpha"在逐渐衰减。

---

## 4. Per-Symbol Performance (M1/M3/M6/M12 回报及波动率)

| Strategy | Symbol | M1 | M3 | M6 | M12 | Volatility |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **MA** | AAPL | +0.4% | +1.7% | +2.6% | +2.9% | 9.0% |
| | AMZN | +0.0% | -2.1% | -1.7% | +0.9% | 7.6% |
| | BTC | -0.3% | +0.7% | +5.2% | +11.6% | 16.8% |
| | ETH | +0.1% | -0.5% | -0.3% | -0.4% | 16.6% |
| | GOOGL | -0.1% | -0.1% | +1.2% | +9.7% | 8.8% |
| | SOL | +0.8% | -0.5% | +19.3% | +62.1% | 61.7% |
| | SPY | +0.0% | +0.6% | +2.0% | +6.6% | 5.7% |
| | TSLA | -1.3% | +5.3% | +8.3% | +16.8% | 22.5% |
| **MACD** | AAPL | +0.7% | +0.7% | +0.5% | +2.0% | 8.7% |
| | AMZN | -0.9% | +0.9% | -0.5% | -2.7% | 13.3% |
| | BTC | +2.6% | +4.7% | +2.2% | +4.7% | 24.1% |
| | ETH | +1.6% | +7.0% | +7.0% | +8.0% | 23.8% |
| | GOOGL | +1.1% | +1.0% | +2.2% | +6.3% | 16.9% |
| | SOL | +1.5% | +10.4% | +28.0% | +50.1% | 48.3% |
| | SPY | -0.5% | +0.8% | +0.8% | +2.6% | 5.9% |
| | TSLA | +2.0% | +12.1% | +21.2% | +37.8% | 28.5% |
| **MomBreak** | AAPL | +0.0% | +1.1% | +1.7% | +6.5% | 7.1% |
| | AMZN | +0.0% | +1.3% | +2.4% | +5.9% | 7.5% |
| | BTC | +0.5% | +3.3% | +2.5% | +5.8% | 14.7% |
| | ETH | +0.9% | +4.4% | +3.2% | +0.2% | 17.8% |
| | GOOGL | +0.0% | +0.5% | +3.5% | +13.6% | 11.7% |
| | SOL | +0.6% | +9.2% | +30.8% | +32.2% | 42.7% |
| | SPY | +0.0% | +0.2% | -0.3% | +1.0% | 6.1% |
| | TSLA | +0.0% | +5.7% | +11.4% | +24.4% | 12.3% |

**解读 — 关键发现**:

### 按资产类型

| 资产类型 | 代表 | 特征 | 参数衰减速度 |
|:---|:---|:---|:---|
| **高波动加密货币** | SOL | M12 高达 +62%, 但波动率 61.7% | 快 — 受单次大行情驱动 |
| **主流加密货币** | BTC, ETH | M6 +2~7%, 波动率 15-24% | 中等偏快 |
| **高波动股票** | TSLA | M12 +17~38%, 波动率 12-28% | 中等 |
| **大盘科技股** | AAPL, GOOGL | M12 +3~14%, 波动率 7-17% | 慢 |
| **指数 ETF** | SPY | M12 +1~7%, 波动率 5-6% | 最慢 |

### 策略间比较

- **MACD 在加密货币上表现最好**: ETH M12 +8.0%, SOL +50.1%, BTC +4.7%
- **MomBreak 在股票上更稳定**: AAPL +6.5%, GOOGL +13.6%, 波动率更低
- **MA 在 ETH 和 AMZN 上失效**: M12 分别为 -0.4% 和 +0.9%，说明这些资产的趋势特征不够明显
- **SPY 所有策略收益最低但波动最低**: 最适合保守参数 + 长周期调整

---

## 5. Re-Optimization Frequency Recommendations

### Decision Framework

| Signal | What It Means | Action |
|:---|:---|:---|
| 半衰期 < N 月 | 累积收益在 N 月内减半 | 在 N 月前重新优化 |
| 胜率 < 50% at month M | 参数不优于随机 | 在 M 月前重新优化 |
| 连续 2+ 月负增量 | 参数正在伤害表现 | 立即重新优化 |
| 三个信号对齐 | 高置信度判断 | 取最短信号 |

### Strategy-Specific Recommendations

**MA (Moving Average)**
- Half-life: 2 months (最短)
- M1 胜率仅 9% → 需要至少 6 个月才能累积足够交易次数产生正向收益
- 累积收益在 M2 一度转负 → 短期内参数不可靠
- **Recommendation: 每 3 个月重新优化，但不要期望短期内获利**
- **适用于趋势性强的资产 (BTC, TSLA, SOL)，不适合震荡市 (ETH, AMZN)**

**MACD**
- Half-life: >12 months (最长、最稳定)
- M1 胜率 46% → 优化后立即有接近 50% 的方向正确率
- 每月增量始终为正 → 参数没有失效迹象
- **Recommendation: 每 3-6 个月重新优化即可**
- **所有资产类型通用，参数鲁棒性最强**

**MomBreak (Momentum Breakout)**
- Half-life: >12 months
- M1 胜率仅 8% → 与 MA 类似，需要时间积累
- M5 和 M7 出现轻微负增量 → 在市场回调期间可能有回撤
- **Recommendation: 每 3 个月重新优化**
- **最适合趋势延续期，在震荡市中应降低仓位**

### Consolidated Schedule

| Asset Type | Symbols | Re-Opt Frequency | Reason |
|:---|:---|:---|:---|
| **Crypto** | BTC, ETH, SOL | **Every 2-3 months** | High regime turnover, 波动率 15-62% |
| **Growth Stocks** | TSLA | **Every 3 months** | Moderate volatility shifts |
| **Large-cap Tech** | AAPL, GOOGL, AMZN | **Every 3-6 months** | Slower regime changes |
| **Index ETF** | SPY | **Every 6 months** | Most stable regime, 波动率 ~6% |

---

## 6. Key Findings

1. **MACD 参数最稳定**: 半衰期 >12 个月，M1 即有正收益，是最值得信赖的策略
2. **MA 和 MomBreak 需要时间积累**: M1 胜率 8-9%，但 M6 后胜率 >50%
3. **所有策略的边际收益逐月递减**: 说明参数"alpha"确实在衰减，但不会完全消失
4. **高波动资产 (SOL, TSLA) 的绝对收益高但风险大**: 需要更频繁调参
5. **保守方案: 每季度 (3 个月) 全面重新优化** — 对所有资产通用
6. **激进方案: 加密货币每 2 个月，股票每 3-6 个月** — 区分资产类型
7. **SPY 最稳定**: 参数衰减最慢，适合长周期持仓 + 半年调参

---

## 7. Integration with Robust Backtest System

This study is integrated with the 10-Layer Anti-Overfitting System via `robust_backtest_api.py`.

### Quick Start — 3 Lines of Code

```python
from robust_backtest_api import run_robust_pipeline

results = run_robust_pipeline(
    symbols=["AAPL", "BTC", "SPY"],
    strategies=["MA", "MACD", "MomBreak"],
)
# results["ranking"] → sorted list of (rank, strategy, score, verdict)
# results["best_params"] → {symbol: {strategy: params_tuple}}
# results["report"] → markdown report string
```

### Full API Usage

```python
from robust_backtest_api import run_robust_pipeline, RobustConfig, quick_scan, scan_top_strategies

# 1. Full 10-layer scan with custom config
config = RobustConfig(
    mc_paths=50,           # More MC paths for higher confidence
    shuffle_paths=30,      # More OHLC shuffle paths
    bootstrap_paths=30,    # More bootstrap paths
    embargo=5,             # Embargo between train/val
)
results = run_robust_pipeline(
    symbols=["AAPL", "GOOGL", "TSLA", "BTC", "ETH", "SOL", "SPY", "AMZN"],
    strategies=["MA", "MACD", "MomBreak"],
    config=config,
)

# 2. Quick single-strategy scan
result = quick_scan("MomBreak", symbols=["BTC", "ETH"])

# 3. Scan top-3 strategies across all 8 assets
result = scan_top_strategies()

# 4. CLI usage
# python robust_backtest_api.py --symbols AAPL BTC SPY --top-only
# python robust_backtest_api.py --strategies MA MACD --mc-paths 50
```

### Re-Optimization Schedule

```
┌────────────────────────────────────────────────────────────────┐
│              Quarterly Re-Optimization Cycle                   │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│  Q1 Jan  │  Q2 Apr  │  Q3 Jul  │  Q4 Oct  │  Q1 Jan (repeat) │
│          │          │          │          │                   │
│  Step 1: Download latest data                                 │
│  Step 2: run_robust_pipeline(...)                             │
│  Step 3: Select ROBUST/STRONG strategies                      │
│  Step 4: Deploy best_params to live system                    │
│  Step 5: Trade for 3 months                                   │
│          │          │          │          │                   │
│  Crypto: Can run monthly mini-scan for BTC/ETH/SOL            │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
```

### File Structure

```
examples/
  walk_forward_robust_scan.py   ← 10-Layer core (21 Numba kernels)
  robust_backtest_api.py        ← Reusable API wrapper
  param_decay_study.py          ← This decay study
docs/
  PARAM_DECAY_STUDY.md          ← This report
  ROBUST_ANTI_OVERFIT_REPORT.md ← V3 10-Layer report
  COMPREHENSIVE_BACKTEST_ANALYSIS_V3.md ← Detailed analysis
results/
  decay_study.csv               ← Raw decay data
  robust_scan/summary.csv       ← Latest scan results
```
