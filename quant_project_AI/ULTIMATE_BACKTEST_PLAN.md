# Ultimate Comprehensive Backtest — 执行计划

## 概述

本计划对 **18 个策略 × 132,442 参数组合 × 30 symbols × 5 杠杆 × 3 时间周期** 进行全方位回测，
通过 **11 层防过拟合**机制选出最优参数组合，直接用于 live trading。

---

## 参数网格对比

| 等级 | 总组合数 | vs DEFAULT | 用途 |
|------|----------|-----------|------|
| DEFAULT | 8,440 | 1.0x | 快速测试 |
| EXPANDED | 94,042 | 11.1x | 标准全面扫描 |
| **ULTRA** | **132,442** | **15.7x** | **极致全面扫描** |

### 各策略参数量

| 策略 | DEFAULT | EXPANDED | ULTRA | 提升倍数 |
|------|---------|----------|-------|----------|
| MA | 938 | 2,413 | 8,118 | 8.7x |
| RSI | 600 | 4,800 | 8,600 | 14.3x |
| MACD | 3,256 | 5,265 | 5,727 | 1.8x |
| Drift | 264 | 4,060 | 7,344 | 27.8x |
| RAMOM | 750 | 7,182 | 13,824 | 18.4x |
| Turtle | 420 | 3,114 | 11,040 | 26.3x |
| Bollinger | 110 | 588 | 1,287 | 11.7x |
| Keltner | 135 | 690 | 1,332 | 9.9x |
| MultiFactor | 720 | 29,160 | 15,120 | 21.0x |
| VolRegime | 288 | 16,128 | 25,200 | 87.5x |
| MESA | 9 | 63 | 77 | 8.6x |
| KAMA | 144 | 2,592 | 3,024 | 21.0x |
| Donchian | 84 | 399 | 936 | 11.1x |
| ZScore | 108 | 3,960 | 9,324 | 86.3x |
| MomBreak | 240 | 1,386 | 3,072 | 12.8x |
| RegimeEMA | 162 | 4,473 | 4,473 | 27.6x |
| DualMom | 20 | 89 | 120 | 6.0x |
| Consensus | 192 | 7,680 | 13,824 | 72.0x |

---

## 执行步骤

### Step 0: 环境准备

```bash
cd quant_project_AI

# 确认 Python 环境
python3 --version     # 需要 3.10+

# 确认依赖已安装
pip install yfinance ccxt numba numpy pandas scipy
```

### Step 1: 下载全量数据 (约5-10分钟)

下载 **10 crypto + 20 stock = 30 symbols** 的 **daily / 4h / 1h** 三个时间周期数据。

```bash
python3 download_data.py
```

预期输出:
```
  Crypto (10 symbols):
    BTC (BTC-USD) | 1d:3000+ | 1h:7000+ | 4h:1800+
    ETH (ETH-USD) | 1d:2500+ | 1h:7000+ | 4h:1800+
    ...
  Stocks (20 symbols):
    AAPL | 1d:10000+ | 1h:4000+ | 4h:1000+
    MSFT | 1d:10000+ | 1h:4000+ | 4h:1000+
    ...
  Files: 30 daily, 25+ 4h, 25+ 1h
```

**注意**: yfinance 的 1h 数据最多 730 天，daily 可以获取全历史。4h 由 1h 重采样得到。

#### 如果需要更多数据（可选）:

```bash
# 下载多时间框架数据（含 15m，通过 Binance API）
python3 examples/download_multi_tf_data.py
```

### Step 2: 验证数据完整性

```bash
python3 -c "
import os
data_dir = 'data'
for sub in ['.', 'daily', '4h', '1h']:
    d = os.path.join(data_dir, sub) if sub != '.' else data_dir
    if os.path.isdir(d):
        csvs = [f for f in os.listdir(d) if f.endswith('.csv')]
        print(f'{sub:>6}: {len(csvs)} files')
        for f in sorted(csvs)[:5]:
            import pandas as pd
            df = pd.read_csv(os.path.join(d, f))
            print(f'        {f}: {len(df)} bars, cols: {list(df.columns)[:5]}')
"
```

### Step 3: Numba 预热（首次运行建议）

```bash
python3 warmup.py
```

这会预编译所有 Numba JIT 函数，避免首次运行时编译延迟。

### Step 4: 运行 Ultimate 回测

#### 选项 A: 完整 ULTRA 扫描（推荐，约 30-120 分钟）

```bash
python3 run_ultimate_scan.py
```

完整 4 阶段管道:
- Phase 1: 单时间框架 × 5 杠杆 × 18 策略 × 132K 参数
- Phase 2: 多时间框架融合（top Phase 1 survivors × TF combos）
- Phase 3: 质量过滤 + CPCV 交叉验证
- Phase 4: 导出 `live_trading_config.json` + 详细报告

#### 选项 B: 快速模式（约 10-30 分钟）

```bash
python3 run_ultimate_scan.py --fast
```

使用 EXPANDED 网格 (94K) + 更少的 MC 路径。

#### 选项 C: 自定义杠杆（按需）

```bash
# 只测试 1x 和 2x 杠杆
python3 run_ultimate_scan.py --leverage 1,2

# 只测试高杠杆（crypto）
python3 run_ultimate_scan.py --leverage 3,5,10
```

#### 选项 D: 分阶段执行

```bash
# 先跑 Phase 1（最耗时）
python3 run_ultimate_scan.py --phase 1

# 检查 Phase 1 结果后再跑后续
python3 run_ultimate_scan.py --phase 2
python3 run_ultimate_scan.py --phase 3
python3 run_ultimate_scan.py --phase 4
```

#### 选项 E: 导出更多推荐

```bash
# 导出 top 100 而非默认 50
python3 run_ultimate_scan.py --top 100
```

### Step 5: 查看结果

回测完成后自动生成:

| 文件 | 路径 | 内容 |
|------|------|------|
| live trading 配置 | `reports/live_trading_config.json` | 最优参数集合 |
| 带时间戳的配置 | `reports/live_trading_config_YYYYMMDD_HHMMSS.json` | 存档 |
| 详细报告 | `reports/ultimate_backtest_report_YYYYMMDD_HHMMSS.md` | 完整分析 |

```bash
# 查看最优推荐
cat reports/live_trading_config.json | python3 -m json.tool | head -100
```

### Step 6: 使用最优参数启动 Live Trading

```bash
# 加载最优参数进行 live trading
python3 run_live_trading.py --config reports/live_trading_config.json
```

---

## 11 层防过拟合机制详解

| 层次 | 机制 | 作用 |
|------|------|------|
| 1 | Purged Walk-Forward | 6 个滚动窗口，训练/验证/测试分离，带 embargo gap |
| 2 | Multi-Metric Scoring | return / drawdown × trade_count_factor 综合评分 |
| 3 | Minimum Trade Filter | ≥ 20 trades 最低交易次数 |
| 4 | Validation Gate | 验证集不能比训练集大幅度恶化 |
| 5 | Cross-Window Consistency | 参数必须在多个时间段都有效 |
| 6 | Monte Carlo Price Perturbation | 50 次价格扰动测试（OOS 数据）|
| 7 | OHLC Shuffle Perturbation | 30 次 OHLC 重排测试 |
| 8 | Block Bootstrap Resampling | 30 次分块自举测试 |
| 9 | Deflated Sharpe Ratio | 修正多重假设检验 |
| 10 | Composite Ranking | 综合排名 |
| 11 | CPCV Cross-Validation | 组合交叉验证 |

## 质量过滤阈值

Phase 3 的过滤条件:

| 指标 | 阈值 | 含义 |
|------|------|------|
| Sharpe Ratio | > 0.3 | 风险调整回报 |
| DSR p-value | < 0.15 | 统计显著性 |
| MC Survival | > 50% | Monte Carlo 正收益比例 |
| OOS Return | > 0% | 样本外正收益 |
| Max Drawdown (multi-TF) | < 60% | 最大回撤限制 |

---

## 预估运行时间

| 场景 | Mac M-series | Windows i7-7700 | 说明 |
|------|-------------|-----------------|------|
| --fast (4 symbols) | ~2-5 min | ~5-15 min | 仅有 4 个 daily symbols |
| --fast (30 symbols) | ~15-30 min | ~30-60 min | 下载数据后 |
| ULTRA (4 symbols) | ~5-10 min | ~10-30 min | 仅有 4 个 daily symbols |
| ULTRA (30 symbols) | ~30-90 min | ~60-180 min | 完整 30 symbols × 3 TF |
| ULTRA (30 sym, 5 lev) | ~60-120 min | ~120-360 min | 最全面扫描 |

**提示**: 首次运行会有 Numba JIT 编译延迟（~30-60秒），后续运行走缓存。

---

## 回测覆盖的维度

### 18 个策略类型

| 类别 | 策略名 | 参数数 | 描述 |
|------|--------|--------|------|
| 趋势跟踪 | MA | 2 | 双均线交叉 |
| 趋势跟踪 | MACD | 3 | MACD 信号线交叉 |
| 趋势跟踪 | DualMom | 2 | 双动量（快/慢） |
| 趋势跟踪 | Turtle | 4 | 海龟通道突破 |
| 趋势跟踪 | Donchian | 3 | 唐奇安通道 |
| 均值回归 | RSI | 3 | RSI 超买超卖 |
| 均值回归 | Bollinger | 2 | 布林带均值回归 |
| 均值回归 | ZScore | 4 | Z-Score 均值回归 |
| 自适应 | MESA | 2 | MESA 自适应 |
| 自适应 | KAMA | 5 | 考夫曼自适应均线 |
| 自适应 | Keltner | 3 | 肯特纳通道 |
| 动量 | RAMOM | 4 | 风险调整动量 |
| 动量 | MomBreak | 4 | 动量突破 |
| 多因子 | MultiFactor | 5 | RSI+动量+波动率 |
| 多因子 | Consensus | 7 | 多信号投票 |
| 体制 | Drift | 3 | 漂移体制检测 |
| 体制 | VolRegime | 6 | 波动率体制 |
| 体制 | RegimeEMA | 5 | 体制 EMA 三重 |

### 5 个杠杆级别

| 杠杆 | 适用 | Stop Loss |
|------|------|-----------|
| 1x | 所有资产 | 40% |
| 2x | 所有资产 | 40% |
| 3x | 所有资产 | 26.7% |
| 5x | 仅 Crypto | 16% |
| 10x | 仅 Crypto | 8% |

### 3 个时间周期

| 周期 | Bars/Year (Crypto) | Bars/Year (Stock) |
|------|-------|-------|
| 1d | 365 | 252 |
| 4h | 2,190 | 441 |
| 1h | 8,760 | 1,764 |

### 2 种成本模型

| 模型 | Commission | Slippage | Funding | 适用 |
|------|-----------|----------|---------|------|
| crypto | 0.04% | 3 bps | 0.03%/day | BTC, ETH, SOL, ... |
| stock_ibkr | 0.05% | 5 bps | margin interest | AAPL, MSFT, NVDA, ... |

---

## 产出物

1. **`reports/live_trading_config.json`** — 结构化的最优参数配置
   ```json
   {
     "recommendations": [
       {
         "rank": 1,
         "symbol": "BTC",
         "strategy": "RAMOM",
         "params": [15, 10, 2.0, 0.5],
         "leverage": 2,
         "interval": "1d",
         "cpcv_validated": true,
         "backtest_metrics": {
           "oos_ret": 45.2,
           "sharpe": 1.85,
           "dsr_p": 0.032,
           "mc_pct_positive": 0.87
         }
       }
     ]
   }
   ```

2. **`reports/ultimate_backtest_report_*.md`** — 详细回测报告

3. **终端输出** — 实时 top 15 排名 + 最终推荐列表

---

## 常见问题

### Q: 运行太慢怎么办？

```bash
# 用 --fast 模式
python3 run_ultimate_scan.py --fast

# 或减少杠杆维度
python3 run_ultimate_scan.py --leverage 1,2

# 或只跑 Phase 1 看初步结果
python3 run_ultimate_scan.py --phase 1
```

### Q: 没有多时间框架数据？

Phase 2（多 TF 融合）会自动跳过。Phase 1 只用 daily 数据也完全可以选出最优参数。

### Q: 如何只对 crypto 或 stock 做回测？

目前脚本会自动按 symbol 类型区分 crypto/stock 的成本模型。如需限制 symbol，在 `download_data.py` 中注释掉不需要的 symbol。

### Q: 结果的 wf_score 代表什么？

`wf_score = sharpe × trade_factor × dsr_bonus`
- `sharpe`: 样本外 Sharpe ratio
- `trade_factor`: 交易次数因子 (min(1, sqrt(trades/30)))
- `dsr_bonus`: DSR p-value 奖励

### Q: 如何确定参数不是过拟合？

11 层防过拟合 + CPCV 验证确保:
- 参数在 6 个不同时间窗口均有效
- 加入价格噪声后仍然盈利 (MC > 50%)
- 打乱 OHLC 后策略仍有信号 (shuffle test)
- 统计上显著 (DSR p < 0.15)
