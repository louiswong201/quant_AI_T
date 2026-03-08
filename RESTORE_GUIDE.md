# 2026-03-07 改动文件还原指南

## 概要

本次共 **47 个文件**（35 个修改 + 12 个新增），分装在 6 个 zip 包中。

| Zip 包 | 文件数 | 内容 |
|--------|--------|------|
| `pack1_backtest_core.zip` | 10 | 回测引擎 + 核心 + 仪表盘 + daily_research |
| `pack2_broker_live.zip` | 10 | 经纪商 + 实盘交易 |
| `pack3_research_strategy.zip` | 10 | 研究引擎 + 策略（部分） |
| `pack4_tests_runners.zip` | 6 | 策略（续）+ 测试 + 运行器 |
| `pack5_reports_docs.zip` | 10 | 代码审核报告文档 |
| `pack6_plan.zip` | 1 | Ultimate 回测计划 |

---

## 还原方法

### 方法一：一键还原（推荐）

进入项目根目录 `quant_AI_T/`，逐个解压即可：

```bash
cd /path/to/quant_AI_T

unzip -o zip_20260307/pack1_backtest_core.zip
unzip -o zip_20260307/pack2_broker_live.zip
unzip -o zip_20260307/pack3_research_strategy.zip
unzip -o zip_20260307/pack4_tests_runners.zip
unzip -o zip_20260307/pack5_reports_docs.zip
unzip -o zip_20260307/pack6_plan.zip
```

> `-o` 参数表示覆盖已有文件（不提示确认）。

### 方法二：Windows PowerShell

```powershell
cd C:\path\to\quant_AI_T

Expand-Archive -Path zip_20260307\pack1_backtest_core.zip -DestinationPath . -Force
Expand-Archive -Path zip_20260307\pack2_broker_live.zip -DestinationPath . -Force
Expand-Archive -Path zip_20260307\pack3_research_strategy.zip -DestinationPath . -Force
Expand-Archive -Path zip_20260307\pack4_tests_runners.zip -DestinationPath . -Force
Expand-Archive -Path zip_20260307\pack5_reports_docs.zip -DestinationPath . -Force
Expand-Archive -Path zip_20260307\pack6_plan.zip -DestinationPath . -Force
```

### 方法三：手动解压

将每个 zip 解压后，内部的文件结构与项目一致，直接复制覆盖到项目根目录即可。

---

## 文件清单与目标路径

### Pack 1: `pack1_backtest_core.zip`（回测 + 核心）

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/backtest/__init__.py` | 修改 |
| 2 | `quant_project_AI/quant_framework/backtest/config.py` | 修改 |
| 3 | `quant_project_AI/quant_framework/backtest/kernels.py` | 修改 |
| 4 | `quant_project_AI/quant_framework/backtest/robust_scan.py` | 修改 |
| 5 | `quant_project_AI/quant_framework/core/margin.py` | 修改 |
| 6 | `quant_project_AI/quant_framework/core/pdt_tracker.py` | 修改 |
| 7 | `quant_project_AI/quant_framework/data/funding_rates.py` | 修改 |
| 8 | `quant_project_AI/quant_framework/dashboard/app.py` | 修改 |
| 9 | `quant_project_AI/quant_framework/dashboard/charts.py` | 修改 |
| 10 | `quant_project_AI/daily_research.py` | 修改 |

### Pack 2: `pack2_broker_live.zip`（经纪商 + 实盘）

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/broker/binance_futures.py` | 修改 |
| 2 | `quant_project_AI/quant_framework/broker/execution_algo.py` | 修改 |
| 3 | `quant_project_AI/quant_framework/broker/ibkr_broker.py` | 修改 |
| 4 | `quant_project_AI/quant_framework/broker/rate_limiter.py` | 修改 |
| 5 | `quant_project_AI/quant_framework/live/alerts.py` | 修改 |
| 6 | `quant_project_AI/quant_framework/live/kill_switch.py` | 修改 |
| 7 | `quant_project_AI/quant_framework/live/price_feed.py` | 修改 |
| 8 | `quant_project_AI/quant_framework/live/trade_journal.py` | 修改 |
| 9 | `quant_project_AI/quant_framework/live/trading_runner.py` | 修改 |
| 10 | `quant_project_AI/run_live_trading.py` | 修改 |

### Pack 3: `pack3_research_strategy.zip`（研究 + 策略）

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/research/_report.py` | 修改 |
| 2 | `quant_project_AI/quant_framework/research/database.py` | 修改 |
| 3 | `quant_project_AI/quant_framework/research/discover.py` | 修改 |
| 4 | `quant_project_AI/quant_framework/research/monitor.py` | 修改 |
| 5 | `quant_project_AI/quant_framework/research/optimizer.py` | 修改 |
| 6 | `quant_project_AI/quant_framework/research/portfolio.py` | 修改 |
| 7 | `quant_project_AI/quant_framework/strategy/base_strategy.py` | 修改 |
| 8 | `quant_project_AI/quant_framework/strategy/drift_regime_strategy.py` | 修改 |
| 9 | `quant_project_AI/quant_framework/strategy/momentum_breakout_strategy.py` | 修改 |
| 10 | `quant_project_AI/quant_framework/strategy/rsi_strategy.py` | 修改 |

### Pack 4: `pack4_tests_runners.zip`（测试 + 运行器）

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/strategy/zscore_reversion_strategy.py` | 修改 |
| 2 | `quant_project_AI/run_production_scan.py` | 修改 |
| 3 | `quant_project_AI/run_ultimate_scan.py` | **新增** |
| 4 | `quant_project_AI/tests/test_v4_production.py` | 修改 |
| 5 | `quant_project_AI/tests/test_v4_research.py` | 修改 |
| 6 | `quant_project_AI/tests/test_v52_fixes.py` | 修改 |

### Pack 5: `pack5_reports_docs.zip`（审核报告）

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `CODE_REVIEW_REPORT.md` | **新增** (根目录) |
| 2 | `LIVE_TRADING_CODE_REVIEW.md` | **新增** (根目录) |
| 3 | `quant_project_AI/CODE_REVIEW_REPORT.md` | **新增** |
| 4 | `quant_project_AI/ENTRY_POINT_CODE_REVIEW.md` | **新增** |
| 5 | `quant_project_AI/ENTRY_POINT_DEEP_REVIEW.md` | **新增** |
| 6 | `quant_project_AI/INTEGRATION_DATA_FLOW_REVIEW.md` | **新增** |
| 7 | `quant_project_AI/KERNELS_CODE_REVIEW.md` | **新增** |
| 8 | `quant_project_AI/ROUND3_CODE_REVIEW_REPORT.md` | **新增** |
| 9 | `quant_project_AI/STRATEGY_CODE_REVIEW_REPORT.md` | **新增** |
| 10 | `quant_project_AI/TEST_REVIEW_REPORT.md` | **新增** |

### Pack 6: `pack6_plan.zip`（回测计划）

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/ULTIMATE_BACKTEST_PLAN.md` | **新增** |

---

## 改动分类汇总

### 代码修改（35 个文件）

| 模块 | 文件数 | 主要改动 |
|------|--------|----------|
| backtest (回测) | 4 | config/kernels/robust_scan/init 优化 |
| core (核心) | 2 | margin模型 + PDT追踪器(7天窗口) |
| broker (经纪商) | 4 | Binance/IBKR/执行算法/限流器 |
| live (实盘) | 5 | 交易运行器/价格源/告警/止损开关/交易日志 |
| dashboard (仪表盘) | 2 | app + charts图表修复 |
| data (数据) | 1 | funding_rates费率数据 |
| research (研究) | 6 | 报告/数据库/发现/监控/优化器/组合 |
| strategy (策略) | 5 | base/drift/momentum/rsi/zscore逻辑修正 |
| runners (运行器) | 2 | run_live_trading + run_production_scan |
| tests (测试) | 3 | v4 production/research + v52 fixes |
| other | 1 | daily_research.py |

### 新增文件（12 个）

| 文件 | 说明 |
|------|------|
| `run_ultimate_scan.py` | ULTRA 参数网格回测脚本 (132K combos) |
| `ULTIMATE_BACKTEST_PLAN.md` | 详细回测执行计划 |
| `CODE_REVIEW_REPORT.md` (×2) | 全面代码审核报告 |
| `ENTRY_POINT_CODE_REVIEW.md` | 入口点深度审核 |
| `ENTRY_POINT_DEEP_REVIEW.md` | 入口点二次审核 |
| `INTEGRATION_DATA_FLOW_REVIEW.md` | 集成数据流审核 |
| `KERNELS_CODE_REVIEW.md` | Numba 内核审核 |
| `ROUND3_CODE_REVIEW_REPORT.md` | 第三轮审核报告 |
| `STRATEGY_CODE_REVIEW_REPORT.md` | 策略代码审核 |
| `TEST_REVIEW_REPORT.md` | 测试代码审核 |
| `LIVE_TRADING_CODE_REVIEW.md` | 实盘交易审核 |

---

## 注意事项

1. **解压目标**：所有 zip 都应解压到项目根目录 `quant_AI_T/`
2. **覆盖顺序**：任意顺序解压均可，文件之间无冲突
3. **备份建议**：解压前建议先 `git stash` 或备份当前代码
4. **Numba 缓存**：代码更新后首次运行需重新编译 Numba 缓存，建议先执行 `python3 warmup.py`
