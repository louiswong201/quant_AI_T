# 文件还原指南 — 2026-03-06 更新包

> 共 58 个文件（23 个修改 + 35 个新增），分装在 6 个 ZIP 中。

---

## ZIP 总览

| ZIP 文件 | 文件数 | 内容 | 大小 |
|----------|--------|------|------|
| `zip1_broker.zip` | 10 | Broker 模块（交易所对接） | 13K |
| `zip2_core_data_config.zip` | 10 | Core 模块 + Data 模块 + 平台配置 | 10K |
| `zip3_live_warmup.zip` | 10 | Live Trading 模块 + Numba 预热 | 22K |
| `zip4_strategy_research.zip` | 10 | Strategy 策略 + Research 研究 + Backtest 入口 | 39K |
| `zip5_reports.zip` | 10 | 报告文件（测试报告、基准报告、配置） | 32K |
| `zip6_docs_backtest_tests.zip` | 8 | 文档 + 回测引擎 + 测试文件 | 80K |

---

## 还原步骤

### 方法一：命令行一键还原（推荐）

进入项目根目录（包含 `quant_project_AI/` 的那一层），然后执行：

**macOS / Linux:**

```bash
cd /path/to/quant_AI_T

for z in packaged_updates_20260306/zip*.zip; do
  unzip -o "$z"
done
```

**Windows (PowerShell):**

```powershell
cd C:\path\to\quant_AI_T

Get-ChildItem packaged_updates_20260306\zip*.zip | ForEach-Object {
  Expand-Archive -Path $_.FullName -DestinationPath . -Force
}
```

**Windows (CMD):**

```cmd
cd C:\path\to\quant_AI_T

for %z in (packaged_updates_20260306\zip*.zip) do (
  tar -xf "%z"
)
```

> `-o` (unzip) 和 `-Force` (PowerShell) 表示覆盖已有文件，不会提示确认。

### 方法二：手动解压

1. 将 6 个 ZIP 文件复制到项目根目录（与 `quant_project_AI/` 同级）
2. 逐个解压，每个 ZIP 内已包含完整的 `quant_project_AI/...` 路径
3. 解压后文件会自动落入正确位置

---

## 每个 ZIP 的详细文件清单

### zip1_broker.zip — Broker 交易所对接模块

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/broker/__init__.py` | 修改 |
| 2 | `quant_project_AI/quant_framework/broker/base.py` | 修改 |
| 3 | `quant_project_AI/quant_framework/broker/binance_futures.py` | **新增** |
| 4 | `quant_project_AI/quant_framework/broker/credentials.py` | **新增** |
| 5 | `quant_project_AI/quant_framework/broker/execution_algo.py` | **新增** |
| 6 | `quant_project_AI/quant_framework/broker/ibkr_broker.py` | **新增** |
| 7 | `quant_project_AI/quant_framework/broker/live_order_manager.py` | **新增** |
| 8 | `quant_project_AI/quant_framework/broker/rate_limiter.py` | **新增** |
| 9 | `quant_project_AI/quant_framework/broker/reconciler.py` | **新增** |
| 10 | `quant_project_AI/quant_framework/broker/testnet.py` | **新增** |

### zip2_core_data_config.zip — Core 核心模块 + Data 数据 + 平台配置

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/core/__init__.py` | **新增** |
| 2 | `quant_project_AI/quant_framework/core/asset_types.py` | **新增** |
| 3 | `quant_project_AI/quant_framework/core/costs.py` | **新增** |
| 4 | `quant_project_AI/quant_framework/core/margin.py` | **新增** |
| 5 | `quant_project_AI/quant_framework/core/market_hours.py` | **新增** |
| 6 | `quant_project_AI/quant_framework/core/pdt_tracker.py` | **新增** |
| 7 | `quant_project_AI/quant_framework/core/symbol_spec.py` | **新增** |
| 8 | `quant_project_AI/quant_framework/data/__init__.py` | 修改 |
| 9 | `quant_project_AI/quant_framework/data/funding_rates.py` | **新增** |
| 10 | `quant_project_AI/quant_framework/platform_config.py` | 修改 |

### zip3_live_warmup.zip — Live Trading 实时交易 + Numba 预热

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/live/__init__.py` | 修改 |
| 2 | `quant_project_AI/quant_framework/live/alerts.py` | **新增** |
| 3 | `quant_project_AI/quant_framework/live/audit.py` | **新增** |
| 4 | `quant_project_AI/quant_framework/live/audit_trail.py` | **新增** |
| 5 | `quant_project_AI/quant_framework/live/health_server.py` | **新增** |
| 6 | `quant_project_AI/quant_framework/live/kernel_adapter.py` | 修改 |
| 7 | `quant_project_AI/quant_framework/live/kill_switch.py` | **新增** |
| 8 | `quant_project_AI/quant_framework/live/risk.py` | 修改 |
| 9 | `quant_project_AI/quant_framework/live/trade_journal.py` | 修改 |
| 10 | `quant_project_AI/quant_framework/warmup.py` | 修改 |

### zip4_strategy_research.zip — 策略 + 研究引擎 + 回测入口

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/quant_framework/strategy/base_strategy.py` | 修改 |
| 2 | `quant_project_AI/quant_framework/strategy/drift_regime_strategy.py` | 修改 |
| 3 | `quant_project_AI/quant_framework/strategy/lorentzian_strategy.py` | 修改 |
| 4 | `quant_project_AI/quant_framework/strategy/macd_strategy.py` | 修改 |
| 5 | `quant_project_AI/quant_framework/strategy/momentum_breakout_strategy.py` | 修改 |
| 6 | `quant_project_AI/quant_framework/strategy/zscore_reversion_strategy.py` | 修改 |
| 7 | `quant_project_AI/quant_framework/research/discover.py` | 修改 |
| 8 | `quant_project_AI/quant_framework/research/monitor.py` | 修改 |
| 9 | `quant_project_AI/quant_framework/research/portfolio.py` | 修改 |
| 10 | `quant_project_AI/quant_framework/backtest/__init__.py` | 修改 |

### zip5_reports.zip — 报告与配置

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/reports/full_integration_test_report.md` | **新增** |
| 2 | `quant_project_AI/reports/live_trading_config.json` | 修改 |
| 3 | `quant_project_AI/reports/live_trading_test_report.md` | **新增** |
| 4 | `quant_project_AI/reports/technical_benchmark_report.md` | 修改 |
| 5 | `quant_project_AI/reports/technical_benchmark_report_20260305_233904.md` | **新增** |
| 6 | `quant_project_AI/reports/technical_benchmark_report_20260305_234738.md` | **新增** |
| 7 | `quant_project_AI/reports/technical_benchmark_report_20260306_001130.md` | **新增** |
| 8 | `quant_project_AI/reports/technical_benchmark_report_20260306_001450.md` | **新增** |
| 9 | `quant_project_AI/reports/v52_test_report.md` | **新增** |
| 10 | `quant_project_AI/reports/v5_full_test_report.md` | **新增** |

### zip6_docs_backtest_tests.zip — 文档 + 回测引擎 + 测试

| # | 文件路径 | 类型 |
|---|---------|------|
| 1 | `quant_project_AI/docs/STEP_BY_STEP_GUIDE.md` | **新增** |
| 2 | `quant_project_AI/docs/V5_REAL_TRADING_UPGRADE_PLAN.md` | **新增** |
| 3 | `quant_project_AI/quant_framework/backtest/backtest_engine.py` | 修改 |
| 4 | `quant_project_AI/quant_framework/backtest/config.py` | 修改 |
| 5 | `quant_project_AI/tests/test_full_backtest_live.py` | **新增** |
| 6 | `quant_project_AI/tests/test_live_trading_async.py` | **新增** |
| 7 | `quant_project_AI/tests/test_v52_fixes.py` | **新增** |
| 8 | `quant_project_AI/tests/test_v5_full.py` | **新增** |

---

## 目录结构预览

解压后，这些文件会分布在以下目录中：

```
quant_project_AI/
├── docs/
│   ├── STEP_BY_STEP_GUIDE.md          ← 新增
│   └── V5_REAL_TRADING_UPGRADE_PLAN.md ← 新增
├── quant_framework/
│   ├── backtest/
│   │   ├── __init__.py                ← 修改
│   │   ├── backtest_engine.py         ← 修改
│   │   └── config.py                  ← 修改
│   ├── broker/                        ← 新增目录（8 个新文件 + 2 个修改）
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── binance_futures.py
│   │   ├── credentials.py
│   │   ├── execution_algo.py
│   │   ├── ibkr_broker.py
│   │   ├── live_order_manager.py
│   │   ├── rate_limiter.py
│   │   ├── reconciler.py
│   │   └── testnet.py
│   ├── core/                          ← 新增目录（7 个新文件）
│   │   ├── __init__.py
│   │   ├── asset_types.py
│   │   ├── costs.py
│   │   ├── margin.py
│   │   ├── market_hours.py
│   │   ├── pdt_tracker.py
│   │   └── symbol_spec.py
│   ├── data/
│   │   ├── __init__.py                ← 修改
│   │   └── funding_rates.py           ← 新增
│   ├── live/
│   │   ├── __init__.py                ← 修改
│   │   ├── alerts.py                  ← 新增
│   │   ├── audit.py                   ← 新增
│   │   ├── audit_trail.py             ← 新增
│   │   ├── health_server.py           ← 新增
│   │   ├── kernel_adapter.py          ← 修改
│   │   ├── kill_switch.py             ← 新增
│   │   ├── risk.py                    ← 修改
│   │   └── trade_journal.py           ← 修改
│   ├── platform_config.py             ← 修改
│   ├── research/
│   │   ├── discover.py                ← 修改
│   │   ├── monitor.py                 ← 修改
│   │   └── portfolio.py               ← 修改
│   ├── strategy/
│   │   ├── base_strategy.py           ← 修改
│   │   ├── drift_regime_strategy.py   ← 修改
│   │   ├── lorentzian_strategy.py     ← 修改
│   │   ├── macd_strategy.py           ← 修改
│   │   ├── momentum_breakout_strategy.py ← 修改
│   │   └── zscore_reversion_strategy.py  ← 修改
│   └── warmup.py                      ← 修改
├── reports/
│   ├── full_integration_test_report.md ← 新增
│   ├── live_trading_config.json       ← 修改
│   ├── live_trading_test_report.md    ← 新增
│   ├── technical_benchmark_report.md  ← 修改
│   ├── technical_benchmark_report_20260305_*.md ← 新增 (×2)
│   ├── technical_benchmark_report_20260306_*.md ← 新增 (×2)
│   ├── v52_test_report.md             ← 新增
│   └── v5_full_test_report.md         ← 新增
└── tests/
    ├── test_full_backtest_live.py      ← 新增
    ├── test_live_trading_async.py      ← 新增
    ├── test_v52_fixes.py              ← 新增
    └── test_v5_full.py                ← 新增
```

---

## 注意事项

1. **解压位置**：必须在项目根目录（包含 `quant_project_AI/` 文件夹的那一层）解压，ZIP 内已包含 `quant_project_AI/` 前缀路径
2. **新增目录**：`core/` 和部分 `broker/` 文件是全新目录，解压工具会自动创建
3. **覆盖确认**：23 个修改文件会覆盖原有版本，建议先备份或确认 git 状态
4. **解压顺序**：6 个 ZIP 之间无依赖关系，可任意顺序解压
5. **Numba 缓存**：还原后建议重新预热 Numba 缓存
   - macOS: `python3 -m quant_framework.warmup`
   - Windows: `python -m quant_framework.warmup`

---

## 统计

- 修改文件：23 个
- 新增文件：35 个
- 总计：58 个文件
- ZIP 包数：6 个（每包 ≤ 10 个文件）
