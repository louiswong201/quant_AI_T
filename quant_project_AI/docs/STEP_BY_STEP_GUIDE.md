# Quant Framework — Step-by-Step 使用指南

> 从安装到实盘，完整操作流程。同时支持 macOS 和 Windows。

---

## macOS vs Windows 关键差异速览

在进入详细步骤之前，先了解两个平台的核心差异：

| 项目 | macOS | Windows |
|------|-------|---------|
| **Python 命令** | `python3` | `python` |
| **pip 命令** | `pip3` 或 `pip` | `pip` |
| **终端** | Terminal / iTerm2 | PowerShell / CMD |
| **路径分隔符** | `/` | `\`（代码中均用 `/` 即可，Python 自动处理） |
| **查看文件** | `cat file.json` | `type file.json`（PowerShell 也支持 `cat`） |
| **Numba 预热耗时** | ~40s (M4 Pro) | ~2-5 min (i7-7700) |
| **Numba 线程层** | workqueue（默认） | 推荐安装 TBB (`pip install tbb`) |
| **Numba 缓存** | `__pycache__/` 目录 | 自动隔离到 `%LOCALAPPDATA%` 避免杀毒干扰 |
| **全量回测耗时** | ~15-25 min | ~1-2 小时 |
| **Live Trading 信号处理** | Unix 信号 (SIGTERM) | atexit 回调 + KeyboardInterrupt |
| **asyncio 事件循环** | 默认 | 自动切换 SelectorEventLoop |
| **定时调度** | cron / launchd | Task Scheduler (schtasks) |
| **Shell 脚本** | `.sh` (bash) | `.ps1` (PowerShell) 或 `.bat` |

> 下面所有命令示例中，**Mac 用 `python3`，Windows 用 `python`**。如果你在 Mac 上安装了 Anaconda/Miniconda，`python` 也可能直接可用。

---

## 目录

1. [环境安装](#1-环境安装)
2. [Numba 预热（首次必做）](#2-numba-预热首次必做)
3. [下载历史数据](#3-下载历史数据)
4. [运行全量回测（参数搜索）](#4-运行全量回测参数搜索)
5. [理解回测结果](#5-理解回测结果)
6. [启动 Live Trading（模拟盘）](#6-启动-live-trading模拟盘)
7. [每日策略研究](#7-每日策略研究)
8. [定时调度自动化](#8-定时调度自动化)
9. [添加自定义策略](#9-添加自定义策略)
10. [性能基准测试](#10-性能基准测试)
11. [Windows 专项注意事项](#11-windows-专项注意事项)
12. [常用命令速查表](#12-常用命令速查表)

---

## 1. 环境安装

### 1.1 Python 版本

推荐 **Python 3.9 – 3.11**。Numba 对 3.12+ 的支持可能不完整。

### 1.2 安装依赖

**macOS:**

```bash
cd quant_project_AI
pip3 install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
cd quant_project_AI
pip install -r requirements.txt
pip install tbb    # 推荐：Numba 线程层，Windows 上显著提升稳定性和性能
```

核心依赖：

| 包 | 用途 |
|---|---|
| `numba` | JIT 编译回测内核（18 策略 × 96 函数） |
| `numpy`, `pandas` | 数据处理 |
| `yfinance` | 下载历史行情 |
| `scipy`, `scikit-learn` | 统计检验、抗过拟合 |
| `dash`, `plotly` | Live Trading 仪表盘 |
| `websockets`, `aiohttp` | Binance 实时数据流 |
| `tbb` | (Windows 推荐) Intel 线程构建块，替代 workqueue 线程层 |

### 1.3 验证安装

**macOS:**

```bash
python3 -c "from quant_framework.backtest import BacktestConfig; print('OK')"
```

**Windows:**

```powershell
python -c "from quant_framework.backtest import BacktestConfig; print('OK')"
```

如果看到 `OK`，说明安装成功。首次运行会看到 Numba 缓存相关的警告，这是正常的。

---

## 2. Numba 预热（首次必做）

框架使用 Numba JIT 将 96 个回测函数编译为机器码。**首次运行前必须预热缓存**，否则第一次回测会额外花费 5–10 分钟。

**macOS:**

```bash
python3 -m quant_framework.warmup
```

**Windows:**

```powershell
python -m quant_framework.warmup
```

输出示例：

```
============================================================
  Quant Framework — Numba JIT Cache Warmup
============================================================
  [1/18] Compiling MA             ...
  [2/18] Compiling RSI            ...
  ...
  [18/18] Compiling scan_all_kernels ...
  Compiling robustness helpers (perturb/shuffle/bootstrap) ...
  Compiling Walk-Forward robustness pipeline ...

  Numba cache warmed: 18 strategies + robustness in 38.7s
============================================================
```

| 平台 | 预热耗时 | 说明 |
|------|---------|------|
| macOS M4 Pro | ~40s | Apple Silicon 原生编译 |
| macOS Intel | ~60-90s | x86 编译 |
| Windows i7-7700 | ~2-5 min | JIT 编译较慢；首次需关闭杀毒软件实时扫描以避免缓存写入干扰 |
| Windows + TBB | ~1.5-3 min | 安装 tbb 后略有改善 |

预热完成后，缓存文件存储在 `quant_framework/backtest/__pycache__/` 中。只要不修改 `kernels.py` 或 `robust_scan.py` 的源代码，缓存永久有效。

> **自动预热**：如果忘记手动预热，框架在 `import quant_framework.backtest` 时会自动检测缓存并在后台启动预热子进程。但建议首次手动运行以确认一切正常。

> **Windows 注意**：如果预热后缓存不生效（每次运行仍然很慢），检查杀毒软件是否删除了 `__pycache__/` 中的 `.nbi`/`.nbc` 文件。建议将项目目录加入杀毒软件的排除列表。

---

## 3. 下载历史数据

### 3.1 一键下载全量数据

**macOS:** `python3 download_data.py`　　**Windows:** `python download_data.py`

默认下载：
- **加密货币 Top 10**：BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, MATIC
- **美股 Top 20**：AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, ...
- **时间框架**：1d（日线，最长历史）、4h（4 小时）、1h（1 小时）

### 3.2 仅下载特定类别

**macOS:**

```bash
python3 download_data.py --crypto    # 仅加密货币
python3 download_data.py --stock     # 仅美股
```

**Windows:**

```powershell
python download_data.py --crypto    # 仅加密货币
python download_data.py --stock     # 仅美股
```

### 3.3 数据存储结构

下载完成后，数据保存在 `data/` 目录：

```
data/
├── daily/
│   ├── BTC.csv          # 日线 OHLCV
│   ├── ETH.csv
│   ├── AAPL.csv
│   └── ...
├── 4h/
│   ├── BTC_4h.csv       # 4 小时线
│   └── ...
└── 1h/
    ├── BTC_1h.csv       # 1 小时线
    └── ...
```

每个 CSV 包含列：`date, open, high, low, close, volume`。

### 3.4 数据要求

- 每个 symbol 至少需要 **500 根 bar** 才能进行有效的 Walk-Forward 验证
- 日线数据可获取 10+ 年历史
- 1h 数据受 yfinance 限制，最多 730 天

---

## 4. 运行全量回测（参数搜索）

这是框架的核心能力：在 18 个策略 × 数千参数组合 × 多杠杆 × 多时间框架中，用 11 层抗过拟合验证找到最优交易参数。

### 4.1 完整生产级扫描

**macOS:** `python3 run_production_scan.py`　　**Windows:** `python run_production_scan.py`

**4 个阶段自动执行：**

| 阶段 | 内容 | macOS M4 Pro | Windows i7-7700 |
|------|------|-------------|-----------------|
| Phase 0 | 加载数据 | ~5s | ~10s |
| Phase 1 | 单时间框架 × 4 杠杆 × 18 策略 × 全参数网格，11 层抗过拟合 | ~3-5 min | ~20-40 min |
| Phase 2 | 多时间框架融合（1h+4h, 4h+1d, 1h+4h+1d） | ~5-10 min | ~30-60 min |
| Phase 3 | CPCV 交叉验证最终筛选 | ~2-3 min | ~10-20 min |
| Phase 4 | 导出 `reports/live_trading_config.json` | <1s | <1s |

> **Windows 建议**：首次运行建议先用 `--fast` 模式验证环境没问题，再跑完整版。

### 4.2 快速模式

如果只想快速验证，使用默认参数网格（~8K 组合 vs 50K+）：

**macOS:** `python3 run_production_scan.py --fast`　　**Windows:** `python run_production_scan.py --fast`

### 4.3 只运行某个阶段

**macOS:**

```bash
python3 run_production_scan.py --phase 1    # 仅 Phase 1
python3 run_production_scan.py --phase 3    # 仅 Phase 3（需先完成 Phase 1-2）
```

**Windows:**

```powershell
python run_production_scan.py --phase 1
python run_production_scan.py --phase 3
```

### 4.4 11 层抗过拟合验证说明

框架不是简单的参数搜索。每个策略参数组合必须通过 11 层验证：

```
1. Purged Walk-Forward     — 6 窗口时间序列 CV，训练/验证/测试严格分离
2. Multi-Metric Scoring    — 综合收益、回撤、交易次数的复合评分
3. Minimum Trade Filter    — 最低交易次数 ≥ 20，排除低频噪声
4. Validation Gate         — 验证期表现不能偏离训练期太远
5. Cross-Window Consistency — 参数须在多个窗口都有效
6. Monte Carlo Perturbation — 加随机噪声重复测试 (30 条路径)
7. Shuffle Test            — 打乱 OHLC 角色测试 (20 条路径)
8. Block Bootstrap         — 块状自举重采样 (20 条路径)
9. Deflated Sharpe Ratio   — 修正多重比较偏误
10. Out-of-Sample Gate     — OOS 收益必须为正
11. CPCV                   — 组合清洗交叉验证
```

---

## 5. 理解回测结果

### 5.1 输出文件

回测完成后生成 `reports/live_trading_config.json`：

```json
{
  "generated_at": "2026-03-05T23:25:40",
  "n_symbols": 4,
  "recommendations": [
    {
      "rank": 1,
      "symbol": "BTC",
      "type": "single-TF",
      "strategy": "RSI",
      "params": [5, 15, 75],
      "interval": "1d",
      "leverage": 2,
      "backtest_metrics": {
        "sharpe": 2.41,
        "return_pct": 145.53,
        "max_dd_pct": 9.96,
        "n_trades": 17
      }
    }
  ]
}
```

### 5.2 关键指标解读

| 指标 | 含义 | 优秀标准 |
|------|------|---------|
| `sharpe` | 风险调整后收益（年化收益 / 年化波动率） | > 1.0 |
| `return_pct` | OOS（样本外）总收益率 % | > 0 |
| `max_dd_pct` | 最大回撤 % | < 30% |
| `n_trades` | 总交易次数 | ≥ 20 |
| `dsr_p` | Deflated Sharpe Ratio p 值（修正后显著性） | < 0.10 |
| `mc_pct_positive` | Monte Carlo 正收益路径比例 | > 60% |

### 5.3 策略名称与参数对照

| 策略名 | 参数含义 |
|--------|---------|
| MA | (短期均线, 长期均线) |
| RSI | (周期, 超卖阈值, 超买阈值) |
| MACD | (快线, 慢线, 信号线) |
| Drift | (回望期, 漂移阈值, 持仓期) |
| RAMOM | (动量期, 波动期, 入场z值, 出场z值) |
| Bollinger | (周期, 标准差倍数) |
| Turtle | (入场周期, 出场周期, ATR周期, ATR倍数) |
| Keltner | (EMA周期, ATR周期, ATR倍数) |
| ZScore | (回望期, 入场z值, 出场z值, 止损z值) |
| DualMom | (快速均线, 慢速均线) |
| Consensus | (短MA, 长MA, RSI周期, 动量周期, 超卖, 超买, 投票阈值) |

---

## 6. 启动 Live Trading（模拟盘）

### 6.1 运行所有推荐策略

**macOS:** `python3 run_live_trading.py`　　**Windows:** `python run_live_trading.py`

这会：
1. 读取 `reports/live_trading_config.json`
2. 为每个 symbol 选择排名最高的策略
3. 启动 Paper Trading（模拟盘）
4. 打开 Dashboard 仪表盘（默认 http://127.0.0.1:8050）

### 6.2 常用选项

**macOS:**

```bash
python3 run_live_trading.py --top-n 5                 # Top 5 symbol
python3 run_live_trading.py --symbols BTC,ETH,AAPL    # 指定 symbol
python3 run_live_trading.py --cpcv-only               # 仅 CPCV 验证的策略
python3 run_live_trading.py --initial-cash 100000 --position-size 0.05
python3 run_live_trading.py --no-dashboard
```

**Windows:**

```powershell
python run_live_trading.py --top-n 5
python run_live_trading.py --symbols BTC,ETH,AAPL
python run_live_trading.py --cpcv-only
python run_live_trading.py --initial-cash 100000 --position-size 0.05
python run_live_trading.py --no-dashboard
```

### 6.3 数据源

- **加密货币**：Binance WebSocket 实时数据流
- **美股**：yfinance 定时轮询

### 6.4 交易信号流程

```
价格数据 → KernelAdapter（Numba 内核生成信号）
         → RiskGate（风控检查）
         → PaperBroker（模拟成交）
         → TradeJournal（记录交易）
         → Dashboard（实时显示）
```

### 6.5 停止 Live Trading

- **两个平台都通用**：在终端按 `Ctrl+C`，或通过 Dashboard 的 Kill Switch 按钮
- **macOS 额外方式**：可发送 `kill -SIGTERM <pid>` 信号
- **Windows 特殊处理**：框架自动使用 `atexit` 回调实现优雅关闭，无需额外操作

---

## 7. 每日策略研究

框架内置了 V4 智能研究系统，由 4 个引擎驱动：

| 引擎 | 功能 | 运行频率 |
|------|------|---------|
| Monitor | 健康度趋势、市场 regime 检测、绩效归因 | 每日 |
| Optimize | Bayesian 参数更新、Champion/Challenger 对比 | 每周 |
| Portfolio | 策略间相关性、权重优化、组合指标 | 每周 |
| Discover | 变体挖掘、市场异常扫描、新策略探索 | 每月 |

### 7.1 日常健康检查（~15 秒）

**macOS:** `python3 daily_research.py`　　**Windows:** `python daily_research.py`

输出示例：

```
[Monitor] BTC RSI: Sharpe 2.41 → 2.35 → 2.28 [declining]
[Monitor] ETH Drift: Sharpe 2.27 ✓ stable
[Monitor] AAPL Donchian: regime shift detected (trending → compression)
ACTION: Re-optimize AAPL recommended
```

### 7.2 每周深度分析

**macOS:** `python3 daily_research.py --mode weekly`　　**Windows:** `python daily_research.py --mode weekly`

额外执行：参数再优化、组合相关性更新、市场异常扫描。

### 7.3 每月全面扫描

**macOS:** `python3 daily_research.py --mode monthly`　　**Windows:** `python daily_research.py --mode monthly`

额外执行：扩展参数网格搜索、CPCV 重验证、策略变体挖掘。

### 7.4 针对特定 symbol

**macOS:** `python3 daily_research.py --symbols BTC,ETH --mode weekly`

**Windows:** `python daily_research.py --symbols BTC,ETH --mode weekly`

### 7.5 自动更新 Live Config

**macOS:** `python3 daily_research.py --mode weekly --apply`

**Windows:** `python daily_research.py --mode weekly --apply`

加上 `--apply` 会自动更新 `live_trading_config.json`（会先备份旧版本）。

---

## 8. 定时调度自动化

### 8.1 macOS / Linux (cron)

```bash
crontab -e
```

添加以下行：

```cron
# 每天 08:00 运行日常健康检查
0 8 * * * cd /path/to/quant_project_AI && /path/to/python daily_research.py >> logs/daily.log 2>&1

# 每周一 07:00 运行周度深度分析
0 7 * * 1 cd /path/to/quant_project_AI && /path/to/python daily_research.py --mode weekly >> logs/weekly.log 2>&1

# 每月 1 号 06:00 运行全面扫描
0 6 1 * * cd /path/to/quant_project_AI && /path/to/python daily_research.py --mode monthly >> logs/monthly.log 2>&1
```

或者直接使用内置脚本：

```bash
# 编辑 crontab 使用脚本
0 8 * * * /path/to/quant_project_AI/scripts/scheduled_research.sh daily
0 7 * * 1 /path/to/quant_project_AI/scripts/scheduled_research.sh weekly
0 6 1 * * /path/to/quant_project_AI/scripts/scheduled_research.sh monthly
```

### 8.2 Windows (Task Scheduler)

通过 PowerShell 创建：

```powershell
# 每天 08:00 运行
schtasks /create /tn "Quant_Daily" /tr "python C:\path\to\daily_research.py" /sc daily /st 08:00

# 每周一 07:00 运行
schtasks /create /tn "Quant_Weekly" /tr "python C:\path\to\daily_research.py --mode weekly" /sc weekly /d MON /st 07:00
```

或通过 GUI：
1. 打开「任务计划程序」
2. 创建基本任务
3. 设置触发器（每日/每周）
4. 操作 → 启动程序 → `python` → 参数 `daily_research.py`

### 8.3 跨平台方案 (APScheduler)

如果需要在应用内调度：

```python
from apscheduler.schedulers.blocking import BlockingScheduler
# pip install apscheduler

scheduler = BlockingScheduler()
scheduler.add_job(run_daily, 'cron', hour=8)
scheduler.add_job(run_weekly, 'cron', day_of_week='mon', hour=7)
scheduler.start()
```

---

## 9. 添加自定义策略

详细指南见 `docs/how_to_add_new_strategy.md`。简要流程如下：

### 9.1 定义 Numba 内核

在 `quant_framework/backtest/kernels.py` 中添加 3 个函数：

```python
@njit(cache=True)
def bt_mystrategy_ls(c, o, h, l, my_param1, my_param2, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    """核心策略逻辑：返回 (total_return, max_drawdown, n_trades)"""
    # ... 交易逻辑
    return ret, dd, cnt

@njit(cache=True)
def _eq_mystrategy(c, o, h, l, my_param1, my_param2, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    """带权益曲线的版本：返回 (ret, dd, cnt, equity_array)"""
    # ... 同上 + 记录权益
    return ret, dd, cnt, equity

@njit(cache=True, parallel=True)
def _scan_mystrategy_njit(ga, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    """并行参数扫描版本"""
    # ... prange 并行
```

### 9.2 注册到系统

在 `kernels.py` 中更新：

```python
KERNEL_REGISTRY["MyStrategy"] = KernelSpec(...)

DEFAULT_PARAM_GRIDS["MyStrategy"] = [
    (p1, p2) for p1 in range(5, 50, 5) for p2 in [0.5, 1.0, 1.5, 2.0]
]

INDICATOR_DEPS["MyStrategy"] = {"mas", "atr"}  # 声明依赖的预计算指标
```

### 9.3 预热新策略

**macOS:** `python3 -m quant_framework.warmup`　　**Windows:** `python -m quant_framework.warmup`

### 9.4 测试

```python
from quant_framework.backtest import backtest, BacktestConfig
result = backtest("MyStrategy", (10, 1.5), data, BacktestConfig.crypto())
print(f"Sharpe: {result['sharpe']:.2f}")
```

---

## 10. 性能基准测试

**macOS:** `python3 examples/technical_benchmark.py`　　**Windows:** `python examples/technical_benchmark.py`

输出报告在 `reports/technical_benchmark_report_YYYYMMDD_HHMMSS.md`。

**典型结果对比：**

| 指标 | macOS M4 Pro | Windows i7-7700 |
|------|-------------|-----------------|
| 单次回测延迟 | ~28μs | ~100μs |
| 单次回测吞吐 | 36,000/sec | ~10,000/sec |
| 全策略扫描 (8,440 combos) | 16ms | ~50ms |
| 扫描吞吐 | 520,000 combos/sec | ~170,000 combos/sec |
| WF 优化 (18 策略) | ~7s | ~30s |

---

## 11. Windows 专项注意事项

### 11.1 Numba 配置

框架自动检测 Windows 并优化 Numba 配置：
- 线程数设为物理核心数（非逻辑核心数，避免超线程争抢）
- 推荐安装 TBB（`pip install tbb`）以使用更稳定的线程层
- 缓存目录隔离于 `%LOCALAPPDATA%\quant_framework\numba_cache`，避免 Windows Defender 干扰

### 11.2 asyncio 事件循环

Live Trading 在 Windows 上自动使用 `SelectorEventLoop` 以确保兼容性。

### 11.3 性能预期

Windows i7-7700（4C/8T）相比 M4 Pro（14C）的预期倍数：

| 操作 | 预期耗时 |
|------|---------|
| 单次回测 | ~100μs |
| 全策略扫描 | ~50ms |
| WF 优化 (18 策略) | ~30s |
| 全量生产扫描 | ~1-2 小时 |

### 11.4 Windows 信号处理

Windows 不支持 Unix 信号（SIGTERM）。框架自动切换为 `atexit` 回调和键盘中断处理，确保优雅关闭。

---

## 12. 常用命令速查表

> 下表中 `PY` = `python3`(macOS) 或 `python`(Windows)

### 环境与预热

| 操作 | macOS | Windows |
|------|-------|---------|
| 安装依赖 | `pip3 install -r requirements.txt` | `pip install -r requirements.txt` |
| 安装 TBB | — (可选) | `pip install tbb` (推荐) |
| Numba 缓存预热 | `python3 -m quant_framework.warmup` | `python -m quant_framework.warmup` |
| 静默预热 | `python3 -m quant_framework.warmup -q` | `python -m quant_framework.warmup -q` |

### 数据

| 操作 | macOS | Windows |
|------|-------|---------|
| 下载全部数据 | `python3 download_data.py` | `python download_data.py` |
| 仅加密货币 | `python3 download_data.py --crypto` | `python download_data.py --crypto` |
| 仅美股 | `python3 download_data.py --stock` | `python download_data.py --stock` |

### 回测与参数搜索

| 操作 | macOS | Windows |
|------|-------|---------|
| 完整扫描 | `python3 run_production_scan.py` | `python run_production_scan.py` |
| 快速模式 | `python3 run_production_scan.py --fast` | `python run_production_scan.py --fast` |
| 仅 Phase 1 | `python3 run_production_scan.py --phase 1` | `python run_production_scan.py --phase 1` |

### Live Trading

| 操作 | macOS | Windows |
|------|-------|---------|
| 启动全部 | `python3 run_live_trading.py` | `python run_live_trading.py` |
| Top 5 | `python3 run_live_trading.py --top-n 5` | `python run_live_trading.py --top-n 5` |
| 指定 symbol | `python3 run_live_trading.py --symbols BTC,ETH` | `python run_live_trading.py --symbols BTC,ETH` |
| 仅 CPCV | `python3 run_live_trading.py --cpcv-only` | `python run_live_trading.py --cpcv-only` |

### 策略研究

| 操作 | macOS | Windows |
|------|-------|---------|
| 日常检查 | `python3 daily_research.py` | `python daily_research.py` |
| 周度分析 | `python3 daily_research.py --mode weekly` | `python daily_research.py --mode weekly` |
| 月度全扫描 | `python3 daily_research.py --mode monthly` | `python daily_research.py --mode monthly` |

### 性能测试

| macOS | Windows |
|-------|---------|
| `python3 examples/technical_benchmark.py` | `python examples/technical_benchmark.py` |

### 研究报告

所有输出报告位于 `reports/` 目录：

```
reports/
├── live_trading_config.json             # Live Trading 策略配置
├── technical_benchmark_report*.md       # 性能基准报告
├── v4_production_validation_report.md   # V4 研究系统验证报告
└── multi_timeframe_analysis_report.md   # 多时间框架分析报告
```

查看报告：**macOS** `cat reports/live_trading_config.json`　**Windows** `type reports\live_trading_config.json`

---

## 完整工作流示例

以下是从零开始到启动 Live Trading 的完整流程。

**macOS:**

```bash
# 1. 安装依赖
pip3 install -r requirements.txt

# 2. 预热 Numba 缓存（首次必做，约 40 秒）
python3 -m quant_framework.warmup

# 3. 下载历史数据
python3 download_data.py

# 4. 运行全量回测（找到最优参数）
python3 run_production_scan.py

# 5. 查看回测结果
cat reports/live_trading_config.json

# 6. 启动 Live Trading
python3 run_live_trading.py

# 7. 次日运行策略健康检查
python3 daily_research.py
```

**Windows (PowerShell):**

```powershell
# 1. 安装依赖
pip install -r requirements.txt
pip install tbb    # 推荐

# 2. 预热 Numba 缓存（首次必做，约 2-5 分钟）
python -m quant_framework.warmup

# 3. 下载历史数据
python download_data.py

# 4. 运行全量回测（找到最优参数，约 1-2 小时）
python run_production_scan.py

# 5. 查看回测结果
type reports\live_trading_config.json

# 6. 启动 Live Trading
python run_live_trading.py

# 7. 次日运行策略健康检查
python daily_research.py
```

---

## 更多文档

| 文档 | 内容 |
|------|------|
| `docs/BEGINNER_GUIDE.md` | 初学者完全指南（代码架构级） |
| `docs/STRATEGY_ARSENAL.md` | 18 策略详解 |
| `docs/how_to_add_new_strategy.md` | 添加新策略指南 |
| `docs/BACKTEST_VS_LIVE.md` | 回测与实盘差异 |
| `docs/ARCHITECTURE.md` | 系统架构 |
| `docs/INDEX.md` | 文档总索引 |
