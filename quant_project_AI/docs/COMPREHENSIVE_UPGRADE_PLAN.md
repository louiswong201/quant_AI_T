# Quant Framework 综合升级计划：深度分析与极致优化

> 分析日期：2026-03-02  
> 总代码量：49,474 行 Python（含测试和示例）  
> 核心模块：backtest / strategy / data / rag / live / alpha / broker / dashboard  
> 对标框架：Qlib (微软)、VectorBT PRO、ArcticDB、Zipline、机构级自研系统  
> 总计发现：75 个基础问题 + 73 项极致优化，分 8 阶段 20 周实施

---

## 目录

- **Part I — 项目总览与现状**
  - [1. 项目架构评估](#1-项目架构评估)
  - [2. 与顶级框架差距分析](#2-与顶级框架差距分析)
- **Part II — 五轮深度分析**
  - [3. 第一轮：Critical Bugs（10 个严重缺陷）](#3-第一轮critical-bugs10-个严重缺陷)
  - [4. 第二轮：性能瓶颈（8 个问题）](#4-第二轮性能瓶颈8-个问题)
  - [5. 第三轮：架构设计（8 个问题）](#5-第三轮架构设计8-个问题)
  - [6. 第四轮：代码质量与可维护性（8 个问题）](#6-第四轮代码质量与可维护性8-个问题)
  - [7. 第五轮：安全性与测试覆盖](#7-第五轮安全性与测试覆盖)
- **Part III — 十维极致优化方案**
  - [8. 回测引擎极致性能](#8-回测引擎极致性能)
  - [9. 数据层极致 I/O](#9-数据层极致-io)
  - [10. 策略执行模型升级](#10-策略执行模型升级)
  - [11. 实盘交易 sub-ms 延迟](#11-实盘交易-sub-ms-延迟)
  - [12. 机构级风控体系](#12-机构级风控体系)
  - [13. RAG 金融专属增强](#13-rag-金融专属增强)
  - [14. Alpha 研究工作流](#14-alpha-研究工作流)
  - [15. 生产级可观测性](#15-生产级可观测性)
  - [16. Dashboard 实时化](#16-dashboard-实时化)
  - [17. 缺失的顶级特性](#17-缺失的顶级特性)
- **Part IV — 统一实施计划**
  - [18. 八阶段实施路线图](#18-八阶段实施路线图)
  - [19. 预期效果对比矩阵](#19-预期效果对比矩阵)
- **Part V — 附录**
  - [A. 文件级问题清单](#附录-a文件级问题清单)
  - [B. 依赖分析与 pyproject.toml 建议](#附录-b依赖分析与-pyprojecttoml-建议)
  - [C. 代码规模与复杂度热点](#附录-c代码规模与复杂度热点)

---

# Part I — 项目总览与现状

## 1. 项目架构评估

### 1.1 项目概述

以**极速数据处理 + AI 融合**为核心目标的量化交易框架：

| 模块 | 行数 | 职责 |
|------|------|------|
| `backtest/` | ~8,800 | 回测引擎、18 个 Numba 内核、鲁棒性扫描、TCA |
| `strategy/` | ~2,500 | 13 个策略（MA/RSI/MACD/Lorentzian/MESA 等） |
| `data/` | ~2,200 | 数据管理、缓存、多后端存储（Binary/Arrow/Parquet）、适配器 |
| `rag/` | ~1,800 | RAG 管道（嵌入/检索/重排/向量库） |
| `live/` | ~2,500 | 实盘适配、价格馈送（YFinance/Binance）、风控、交易日志 |
| `alpha/` | ~800 | Alpha 因子（订单流/波动率/跨资产） |
| `tests/` | ~4,500 | 测试套件（含伪测试脚本） |
| `examples/` | ~14,000 | 示例和基准脚本 |

### 1.2 架构优点

- 分层清晰：数据层 → 策略层 → 执行层 → 分析层
- Numba JIT 加速的内核设计（18 个策略内核）
- 多后端存储（Binary Mmap / Arrow IPC / Parquet）
- RAG 与量化融合的创新设计（`as_of_date` 防止未来偷看）
- 11 层反过拟合扫描（walk-forward / Monte Carlo / CPCV / 膨胀 Sharpe）
- 多时间框架融合回测（trend_filter / consensus / primary）

### 1.3 架构缺陷概览

- `kernels.py` 单文件 3,215 行，维护困难
- 策略间大量重复代码（数据解包、指标计算、止损逻辑）
- 策略实例状态在多标的回测中共享，导致严重逻辑错误
- 协议（Protocol）与实现签名不一致
- 数据层未充分利用零拷贝能力（mmap 后仍 copy，Arrow 先加载再过滤）
- 测试覆盖不均，7 个模块零覆盖

---

## 2. 与顶级框架差距分析

| 能力维度 | 当前水平 | 顶级水平 | 差距 |
|----------|---------|---------|------|
| 回测速度 | 中（Numba 但有冗余） | 极快（零拷贝 + prange 并行 + 单 pass 指标） | **大** |
| 数据 I/O | 中（多后端但未充分利用） | 极快（mmap 零拷贝 + 分层缓存 L1/L2/L3） | **大** |
| 实盘延迟 | ~1-3ms（DataFrame 开销） | <0.5ms（纯数组路径 + 环形缓冲区） | **中** |
| 风控体系 | 基础（单标的限制） | 机构级（VaR + 压力测试 + 组合级集中度） | **大** |
| RAG 检索 | 暴力搜索 O(n) | ANN 索引 O(log n)（FAISS HNSW） | **大** |
| Alpha 研究 | 基础 IC/ICIR | 完整（IC 衰减 + 换手 + 拥挤度 + 因子 DSL） | **中** |
| 可观测性 | 基础 logging | 结构化日志 + Prometheus + OpenTelemetry | **大** |
| 状态管理 | 无 | 事件溯源 + 检查点 + 确定性回放 | **大** |
| 执行算法 | 无 | TWAP/VWAP/IS + 智能路由 | **大** |

### 与具体框架对比

| 特性 | Qlib | VectorBT | ArcticDB | 本框架 |
|------|:----:|:--------:|:--------:|:------:|
| 表达式引擎 | ✓ | ✗ | ✗ | ✗ |
| 矩阵布局 (symbol×time) | ✗ | ✓ | ✗ | ✗ |
| 交易日历 | ✓ | ✓ | ✗ | ✗ |
| 数据版本管理 | ✗ | ✗ | ✓ | ✗ |
| Numba prange 多标的 | ✗ | ✓ | ✗ | ✗ |
| RAG 融合 | ✗ | ✗ | ✗ | **✓** |
| 反过拟合 11 层扫描 | ✗ | ✗ | ✗ | **✓** |
| 多 TF 融合回测 | ✗ | ✗ | ✗ | **✓** |

---

# Part II — 五轮深度分析

## 3. 第一轮：Critical Bugs（10 个严重缺陷）

### BUG-01：多标的策略状态共享（P0 — 影响 5 个策略）

**受影响文件：** `momentum_breakout_strategy.py`, `drift_regime_strategy.py`, `adaptive_regime_ensemble.py`, `microstructure_momentum.py`, `zscore_reversion_strategy.py`

策略中的 `_trailing_stop`、`_hold_count`、`_entry_bar` 等状态变量定义为实例属性（单值），而非按标的维护的字典。多标的回测中一个标的的状态会覆盖另一个。

```python
# 当前（错误）                              # 修复后
self._trailing_stop = 0.0                   self._trailing_stop: Dict[str, float] = {}
```

### BUG-02：robust.py 权益曲线使用线性插值（P0）

**文件：** `backtest/robust.py` L131-132

```python
pv = np.linspace(ic, final_value, n)  # 线性插值！不是真实权益
```

Sharpe/最大回撤等全部失真。**修复：** 使用 `eval_kernel_detailed` 获取真实权益。

### BUG-03：bars_per_year 从未传入 kernel costs（P0）

**文件：** `backtest/robust_scan.py`

`config_to_kernel_costs()` 不含 `bars_per_year`，非日线频率（5m/1h/4h）的年化指标全部错误，默认使用 252。

### BUG-04：ZScore 策略退出逻辑错误（P0）

**文件：** `strategy/zscore_reversion_strategy.py` L91

```python
if z > -self.exit_z or z <= -self.stop_z:   # or → 几乎立即平仓
```

应拆分为正常退出（z 回中性区）和止损退出两个独立条件。

### BUG-05：MACD on_bar_fast_multi 提前返回 None（P1）

**文件：** `strategy/macd_strategy.py` L160-161。一个标的缺数据导致所有后续标的被跳过。应用 `continue`。

### BUG-06：DatabaseAdapter save_data 覆盖全表（P1）

`if_exists="replace"` 每次保存删表重建，丢失其他标的数据。应用 upsert。

### BUG-07：DriftRegime 窗口计算错误（P1）

`np.diff` 产生 `lookback-1` 个元素，但分母用 `self.lookback`，比例偏差。

### BUG-08：协议与实现签名不一致（P1）

`IOrderManager.expire_stale(current_bar)` vs `DefaultOrderManager.expire_stale(current_bar, current_date)`。

### BUG-09：CircuitBreaker reject_count 永不递增（P1）

`_reject_count` 在 `summary()` 输出但从未递增，`total_rejects` 永远为 0。

### BUG-10：RiskManagedBroker 不传递 current_prices（P1）

`RiskGate.validate()` 需要 `current_prices` 但调用处未传入。

---

## 4. 第二轮：性能瓶颈（8 个问题）

| ID | 问题 | 影响 | 位置 |
|----|------|------|------|
| PERF-01 | `kernels.py` 单文件 3,215 行 | Numba 编译慢、IDE 卡顿 | `backtest/kernels.py` |
| PERF-02 | 策略每 bar 重算完整指标 O(n²) | KAMA/MESA/MomBreakout/DriftRegime | 4 个策略文件 |
| PERF-03 | CacheManager 每次 get 复制 DataFrame | 内存翻倍 | `cache_manager.py` |
| PERF-04 | FileAdapter.get_latest_date 加载全量 | 大数据集极慢 | `file_adapter.py` |
| PERF-05 | TCA 使用 iterrows | pandas 最慢遍历 | `tca.py` |
| PERF-06 | Lorentzian 缓存无限增长 | 内存泄漏 | `lorentzian_strategy.py` |
| PERF-07 | Dataset.load_lazy 触发 schema 扫描 | 抵消 Polars lazy 优势 | `dataset.py` L126 |
| PERF-08 | 退出逻辑 18 内核重复 ~270 行 | 维护成本高 | `kernels.py` |

---

## 5. 第三轮：架构设计（8 个问题）

| ID | 问题 | 位置 |
|----|------|------|
| ARCH-01 | BaseStrategy 混合信号/仓位/RAG/组合跟踪 | `base_strategy.py` |
| ARCH-02 | backtest 反向依赖 live 模块 | `backtest_engine.py` → `live.replay` |
| ARCH-03 | `kernels_backup.py` 1,342 行定位不清 | 与 `kernels.py` 600 行重叠 |
| ARCH-04 | Dataset 仅用 FileAdapter，忽略 Api/Database | `dataset.py` |
| ARCH-05 | RAG Pipeline 工作线程无优雅关闭 | `rag/pipeline.py` |
| ARCH-06 | TradingRunner 656 行单类职责过重 | `live/trading_runner.py` |
| ARCH-07 | model 模块为空壳 | `model/__init__.py` |
| ARCH-08 | Config 字段未完全传播到内核成本 | `config.py` → `config_to_kernel_costs` |

---

## 6. 第四轮：代码质量与可维护性（8 个问题）

| ID | 问题 | 影响范围 |
|----|------|---------|
| QUAL-01 | 数据解包模式在 8+ 策略中重复 ~10 行 | 8 个策略 |
| QUAL-02 | `risk_percent=0.95` 名称误导 | MA/MACD/RSI/KAMA 等 |
| QUAL-03 | magic number 散布 9+ 处 | 多个模块 |
| QUAL-04 | 指标在 5 策略中重复实现 | KAMA/ATR/RSI/EMA/Z-Score |
| QUAL-05 | 错误处理 `except Exception: pass` | cache/storage |
| QUAL-06 | `_TruncatedDFView` 死代码 | `backtest_engine.py` |
| QUAL-07 | `_index` 字典写入但从未查询 | `order_manager.py` |
| QUAL-08 | 中英文混用 | `manifest.py` |

---

## 7. 第五轮：安全性与测试覆盖

### 安全问题

| ID | 问题 | 严重度 |
|----|------|--------|
| SEC-01 | `pickle.load()` 反序列化可致远程代码执行 | 高 |
| SEC-02 | RAG ingestion 路径未验证 | 中 |
| SEC-03 | WebSocket URL 硬编码无 TLS 验证 | 中 |
| SEC-04 | 交易日志路径可预测 | 低 |

### 零覆盖模块

| 模块 | 行数 |
|------|------|
| `visualization/plotter.py` | ~150 |
| `dashboard/app.py` | ~636 |
| `dashboard/charts.py` | ~859 |
| `live/trading_runner.py` | ~656 |
| `live/price_feed.py` | ~755 |
| `live/trade_journal.py` | ~200 |
| `model/__init__.py` | ~30 |

### 伪测试文件（非 pytest 格式）

- `test_lorentzian_optimization.py` — 优化脚本，依赖网络
- `test_rag_full_chain.py` — 自定义框架，被 conftest 排除
- `test_rag_realtime_stress.py` — 压力测试脚本

### 缺失关键测试

多标的状态隔离 | 非日线年化 | 鲁棒性扫描精度 | 数据库多标的 | RAG 生命周期 | 做空 P&L

---

# Part III — 十维极致优化方案

## 8. 回测引擎极致性能

### 8.1 内核模块拆分

**当前：** 单文件 3,215 行，18 个内核耦合。  
**目标：** 按策略族拆分，Numba 按需编译，编译提速 3-5x。

```
backtest/kernels/
├── __init__.py           # KERNEL_REGISTRY + 统一导出
├── _common.py            # 共享退出逻辑（止盈/止损/展仓 inline）
├── _precompute.py        # precompute_all_ma/ema/rsi/rolling_max
├── trend.py              # MA, EMA, MACD, KAMA, MESA
├── momentum.py           # RSI, MomentumBreakout, Drift, RaMom
├── mean_reversion.py     # BB, ZScore, Donchian
├── advanced.py           # Lorentzian, Consensus, Turtle
├── detailed.py           # eval_kernel_detailed + position 系列
└── scan.py               # scan_all_kernels + _SCANNERS
```

### 8.2 共享退出逻辑提取（消除 270 行冗余）

```python
@njit(cache=True, fastmath=True, inline="always")
def _check_exit(pos, c_i, entry_px, sl_frac, tp_frac):
    if pos == 1 and c_i <= entry_px * sl_frac:
        return -1
    if pos == 1 and tp_frac > 0 and c_i >= entry_px * tp_frac:
        return -1
    if pos == -1 and c_i >= entry_px * (2.0 - sl_frac):
        return 1
    return 0
```

### 8.3 `_rolling_vol` Welford Numba 化（10-50x）

```python
@njit(cache=True, fastmath=True)
def _rolling_vol_welford(ret, window):
    n = len(ret)
    out = np.full(n, np.nan, dtype=np.float64)
    s = 0.0; s2 = 0.0
    for i in range(n):
        s += ret[i]; s2 += ret[i] * ret[i]
        if i >= window:
            s -= ret[i - window]; s2 -= ret[i - window] * ret[i - window]
        if i >= window - 1:
            m = s / window
            out[i] = np.sqrt(max(0.0, s2 / window - m * m))
    return out
```

### 8.4 预计算 prange 并行化（2-4x）

```python
@njit(cache=True, fastmath=True, parallel=True)
def _precompute_macd_lines(emas, pairs, n):
    out = np.empty((pairs.shape[0], n), dtype=np.float64)
    for p in prange(pairs.shape[0]):
        out[p, :] = emas[int(pairs[p, 0]), :] - emas[int(pairs[p, 1]), :]
    return out
```

### 8.5 零拷贝 DataFrame 视图

启用已定义但未使用的 `_TruncatedDFView`，OHLC 数组去除不必要的 `.copy()`。

### 8.6 扫描级多核并行化

```python
def run_robust_scan(symbols, strategies, config, n_workers=None):
    n_workers = n_workers or min(len(symbols), os.cpu_count())
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_scan_symbol, sym, strategies, config): sym for sym in symbols}
        return {futures[f]: f.result() for f in as_completed(futures)}
```

### 8.7 内存池化 + PortfolioTracker 优化

预分配 `BacktestBufferPool`；`record_bar` 仅在持仓变化时做字典快照。

---

## 9. 数据层极致 I/O

### 9.1 真正的零拷贝 mmap 路径

**当前问题：** `BinaryMmapStorage.load()` 使用 `np.copy(arr[i0:i1])` 浪费 mmap；`load_arrays` 完全忽略 Binary/Arrow。

```python
class BinaryMmapStorage:
    def load_arrays(self, path, start_date=None, end_date=None):
        mm = np.memmap(path, dtype=self._dtype, mode='r')
        i0, i1 = self._date_range_indices(mm, start_date, end_date)
        block = mm[i0:i1]  # 视图，不拷贝
        return {col: np.ascontiguousarray(block[col]) for col in self._columns}
```

### 9.2 Arrow IPC 过滤前置

**当前问题：** 先 `to_pandas()` 再过滤日期，浪费 50-80% 转换。

```python
import pyarrow.compute as pc
table = feather.read_feather(path, memory_map=True)
if start_date:
    table = table.filter(pc.greater_equal(table['date'], start_date))
df = table.to_pandas()  # 仅转换过滤后的数据
```

### 9.3 Dataset.load_arrays 优先快速路径

```python
def load_arrays(self, symbol, start_date, end_date):
    if self._fast_io:
        if (r := self._binary_storage.load_arrays(symbol, start_date, end_date)):
            return r
        if (r := self._arrow_storage.load_arrays(symbol, start_date, end_date)):
            return r
    return self._load_parquet_arrays(symbol, start_date, end_date)
```

### 9.4 分层缓存 L1/L2/L3

```
L1: numpy 数组 in-memory（~100MB，微秒级）
L2: Arrow IPC mmap（毫秒级，零拷贝）
L3: Parquet on disk（Zstd 压缩，十毫秒级）
命中时自动 promote：L3 → L2 → L1
```

### 9.5 指标融合单次遍历（5-8x）

将 MA5/MA10/MA20/RSI/ATR 融合为一次遍历，从 ~10 pass 降至 1 pass。

```python
@njit(cache=True, fastmath=True)
def fused_compute_all_indicators(close, high, low, ma_windows, rsi_period=14):
    n = len(close)
    mas = np.full((len(ma_windows), n), np.nan)
    rsi = np.full(n, np.nan)
    ma_sums = np.zeros(len(ma_windows))
    gain_ema = 0.0; loss_ema = 0.0
    for i in range(n):
        for j in range(len(ma_windows)):
            w = ma_windows[j]; ma_sums[j] += close[i]
            if i >= w: ma_sums[j] -= close[i - w]
            if i >= w - 1: mas[j, i] = ma_sums[j] / w
        # RSI + ATR 增量更新...
    return mas, rsi
```

### 9.6 其他数据层优化

- Parquet 默认 Zstd 压缩（比 Snappy 提升 30%）
- CacheManager 添加 `copy=False` view 模式
- FileAdapter.get_latest_date 使用元数据查询
- Dataset.load_lazy 移除 `collect_schema()` 调用
- CCI/Williams/Stochastic 从 O(n×period) 优化为 O(n)

---

## 10. 策略执行模型升级

### 10.1 事件驱动架构

```python
class EventBus:
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
    async def publish(self, event: BarEvent):
        q = self._queues.setdefault(event.symbol, asyncio.Queue(maxsize=1000))
        await q.put(event)
```

### 10.2 Symbol Actor 隔离

每个标的独立执行器，故障隔离——一个标的的策略异常不影响其他标的。

### 10.3 End-to-end JIT 信号管道

`KernelAdapter.generate_signal_from_arrays()` 跳过 DataFrame，直接从环形缓冲区获取数组。

### 10.4 环形缓冲区替代 RollingWindow

`RingBuffer` 零分配 O(1) append + O(1) view，替代每 bar 分配 DataFrame 的 `_RollingWindow`。

---

## 11. 实盘交易 sub-ms 延迟

### 延迟对比

| 阶段 | 当前 | 优化后 |
|------|------|--------|
| RollingWindow → DataFrame | ~0.1-0.5 ms | RingBuffer view: ~0.001 ms |
| generate_signal (含 DF 转换) | ~0.5-2 ms | 纯数组 Numba: ~0.1-0.3 ms |
| RiskGate + broker | ~0.06 ms | ~0.06 ms |
| **总计** | **~1-3 ms** | **~0.3-0.5 ms** |

### 全链路延迟追踪

```python
class LatencyTrace:
    __slots__ = ('t_recv', 't_signal_start', 't_signal_end', 't_risk', 't_ack')
    def breakdown(self) -> Dict[str, float]:
        return {"signal_us": (self.t_signal_end - self.t_signal_start) / 1000, ...}
```

---

## 12. 机构级风控体系

### 当前 vs 目标

| 能力 | 当前 | 目标 |
|------|:----:|:----:|
| 单标的限制 | ✓ | 保留 |
| 熔断器 | ✓ | 增强（API/CLI 控制） |
| 组合 VaR | ✗ | **新增**（参数法 + 历史法） |
| 压力测试 | ✗ | **新增**（6 场景） |
| 集中度监控 | ✗ | **新增** |
| 日内最大回撤 | ✗ | **新增** |
| 执行算法 | ✗ | **新增**（TWAP/VWAP） |

### 关键实现

- `PortfolioVaR`：基于收益矩阵的参数法/历史法 VaR 计算
- `StressTestEngine`：flash_crash / vol_spike / crypto_crash / liquidity_crisis 等 6 场景
- `ConcentrationMonitor`：单资产/行业集中度上限
- `ExecutionAlgorithm` 框架：TWAP/VWAP 大单拆分 + `SmartOrderBroker` 集成

---

## 13. RAG 金融专属增强

### 13.1 FAISS/HNSW 向量搜索（30x）

暴力搜索 `mat @ q` 从 ~20-50ms → FAISS HNSW ~0.5-2ms。

### 13.2 金融情感分析

FinBERT 对文本打分 -1..1，存入 `chunk.metadata["sentiment"]`，检索时加权。

### 13.3 Ticker NER + 时间衰减

正则提取 Ticker 实体；检索评分乘以半衰期衰减因子 `0.5^(age/halflife)`。

### 13.4 SEC/财报文档解析

10-K Item 段落提取，结构化为独立 Document。

### 13.5 安全存储

`pickle.load()` → `numpy.save/load` + JSON 元数据。

---

## 14. Alpha 研究工作流

| 功能 | 当前 | 新增 |
|------|:----:|:----:|
| IC/ICIR | ✓ | 保留 |
| 正交因子选择 | ✓ | 保留 |
| IC 衰减曲线 | ✗ | 因子时效性评估 |
| 换手率分析 | ✗ | 成本敏感度评估 |
| 因子拥挤度检测 | ✗ | 与基准因子相关性 > 0.7 告警 |
| Walk-forward IC | ✗ | 稳定性验证 |

---

## 15. 生产级可观测性

| 层面 | 当前 | 升级方案 |
|------|------|---------|
| 日志 | `logging.info` | structlog 结构化 JSON |
| 指标 | 无 | Prometheus（bars_total, signal_latency, portfolio_value） |
| 追踪 | 无 | OpenTelemetry span（on_bar → signal → risk → order） |
| 告警 | 无 | 熔断器触发 / VaR 超限告警 |

---

## 16. Dashboard 实时化

### WebSocket 替代轮询

当前 `dcc.Interval(2000ms)` → WebSocket 推送 sub-second 更新。

### 缺失可视化组件

| 组件 | 价值 | 复杂度 |
|------|------|--------|
| 订单簿深度图 | 高 | 中 |
| 相关性热力图 | 高 | 低 |
| 实时 P&L 瀑布图 | 高 | 中 |
| 回撤水下图 | 中 | 低 |
| 策略信号时间线 | 高 | 中 |

---

## 17. 缺失的顶级特性

| 特性 | 建议 |
|------|------|
| 交易日历（多市场） | P1 — 全球市场支持 |
| 矩阵布局 (symbol×time) | P1 — 多标的向量化回测 |
| 回测结果缓存 | P1 — hash(策略+参数+数据) → 缓存 |
| Alpha 因子 DSL | P2 — 表达式引擎快速定义因子 |
| 在线学习 | P2 — river/incremental 集成 |
| A/B 测试 | P2 — 策略变体路由 + 对比指标 |
| 参数服务器 | P2 — Redis 实时参数热更新 |
| 事件溯源 | P1 — 确定性回放 + 崩溃恢复 |
| 数据验证 | P1 — Pandera OHLCV schema 验证 |

---

# Part IV — 统一实施计划

## 18. 八阶段实施路线图

### 阶段 0：Critical Bug 修复（Week 1-2）— 15h

| # | 任务 | 文件 | 工时 |
|---|------|------|------|
| 1 | 5 个策略多标的状态隔离 | 5 个策略 | 4h |
| 2 | robust.py 线性插值 → 真实权益 | `robust.py` | 2h |
| 3 | bars_per_year 传入 kernel costs | `config.py`, `robust_scan.py` | 2h |
| 4 | ZScore 退出逻辑修正 | `zscore_reversion_strategy.py` | 1h |
| 5 | MACD return None → continue | `macd_strategy.py` | 0.5h |
| 6 | DatabaseAdapter upsert | `database_adapter.py` | 2h |
| 7 | 协议签名对齐 | `protocols.py`, `order_manager.py` | 1h |
| 8 | CircuitBreaker reject_count 修复 | `risk.py` | 0.5h |
| 9 | DriftRegime 窗口修正 | `drift_regime_strategy.py` | 1h |
| 10 | RiskManagedBroker 传递 current_prices | `risk.py` | 1h |

### 阶段 1：核心引擎极致性能（Week 3-5）— 40h

| # | 任务 | 预期提升 | 工时 |
|---|------|---------|------|
| 11 | kernels.py → kernels/ 模块 | 编译 3-5x | 8h |
| 12 | 共享退出逻辑 + inline | 消除 270 行 | 3h |
| 13 | `_rolling_vol` Welford Numba 化 | 10-50x | 2h |
| 14 | 预计算 parallel + prange | 多核 2-4x | 4h |
| 15 | 零拷贝 `_TruncatedDFView` | 大 DF 零拷贝 | 1h |
| 16 | 指标融合单次遍历 | 指标 5-8x | 6h |
| 17 | OHLC 数组去除 `.copy()` | 内存 -50% | 1h |
| 18 | PortfolioTracker 持仓变化时才快照 | 减少分配 | 2h |
| 19 | robust_scan 多核并行 | 近线性 | 6h |
| 20 | BacktestBufferPool 内存池化 | GC 压力降 | 3h |
| 21 | 删除 kernels_backup.py + 死代码 | -1,342 行 | 2h |
| 22 | fill_simulator 按标的索引订单 | O(1) 查找 | 2h |

### 阶段 2：数据层极致 I/O（Week 5-7）— 30h

| # | 任务 | 预期提升 | 工时 |
|---|------|---------|------|
| 23 | BinaryMmap.load_arrays 零拷贝 | 加载 10x+ | 4h |
| 24 | Arrow 过滤前置 | 转换减 50-80% | 3h |
| 25 | Dataset.load_arrays 快速路径 | 热路径加速 | 3h |
| 26 | 分层缓存 L1/L2/L3 | 命中率 + 效率 | 8h |
| 27 | Parquet Zstd 压缩 | 压缩率 +30% | 1h |
| 28 | CacheManager view 模式 | 消除 copy | 3h |
| 29 | FileAdapter 元数据查询 | 消除全量加载 | 2h |
| 30 | load_lazy 移除 collect_schema | Polars lazy | 1h |
| 31 | OHLCV Pandera 验证 | 数据质量 | 3h |
| 32 | DatabaseAdapter upsert + 安全 | 完整性 | 2h |

### 阶段 3：策略层重构 + 实盘升级（Week 7-10）— 45h

| # | 任务 | 预期提升 | 工时 |
|---|------|---------|------|
| 33 | 策略公共代码提取 | -640 行重复 | 4h |
| 34 | 指标统一（含 CCI/ADX） | 消除 5 处重复 | 6h |
| 35 | risk_percent → capital_fraction | 语义修正 | 2h |
| 36 | RingBuffer 替代 RollingWindow | 零分配 | 4h |
| 37 | generate_signal_from_arrays | sub-ms | 3h |
| 38 | EventBus + SymbolActor | 故障隔离 | 8h |
| 39 | LatencyTrace 全链路 | 瓶颈定位 | 2h |
| 40 | RAG Pipeline shutdown() | 安全释放 | 2h |
| 41 | TradingRunner 拆分 | 可维护性 | 6h |
| 42 | backtest→live 依赖消除 | 架构清晰 | 2h |
| 43 | 协议签名对齐 | 类型安全 | 2h |
| 44 | magic numbers 常量化 | 可维护性 | 4h |

### 阶段 4：机构级风控 + 执行（Week 10-12）— 35h

| # | 任务 | 工时 |
|---|------|------|
| 45 | PortfolioVaR（参数法 + 历史法） | 6h |
| 46 | StressTestEngine（6 场景） | 4h |
| 47 | ConcentrationMonitor | 3h |
| 48 | CircuitBreaker 增强（API/CLI） | 3h |
| 49 | ExecutionAlgorithm 框架 (TWAP/VWAP) | 8h |
| 50 | SmartOrderBroker 集成 | 4h |
| 51 | Broker Protocol 标准化 | 2h |
| 52 | 日内最大回撤监控 | 3h |
| 53 | 做空 P&L 测试 + 修复 | 2h |

### 阶段 5：RAG 金融增强 + Alpha 工作流（Week 12-14）— 30h

| # | 任务 | 工时 |
|---|------|------|
| 54 | FAISS/HNSW 向量搜索 | 6h |
| 55 | 金融情感分析 (FinBERT) | 4h |
| 56 | Ticker NER + 实体提取 | 3h |
| 57 | 时间衰减检索评分 | 2h |
| 58 | SEC/财报文档解析器 | 3h |
| 59 | pickle → numpy+JSON 安全存储 | 3h |
| 60 | IC 衰减曲线 | 2h |
| 61 | 换手率分析 | 2h |
| 62 | 因子拥挤度检测 | 2h |
| 63 | Walk-forward IC 分析 | 3h |

### 阶段 6：生产化 + Dashboard（Week 14-16）— 35h

| # | 任务 | 工时 |
|---|------|------|
| 64 | 结构化日志 (structlog) | 3h |
| 65 | Prometheus 指标导出 | 4h |
| 66 | OpenTelemetry 链路追踪 | 4h |
| 67 | 事件溯源 EventStore | 6h |
| 68 | 状态检查点/恢复 | 4h |
| 69 | Dashboard WebSocket 推送 | 4h |
| 70 | 相关性热力图 | 2h |
| 71 | 订单簿深度图 | 3h |
| 72 | 回撤水下图 | 1h |
| 73 | pyproject.toml + 分组依赖 | 2h |
| 74 | pytest-cov + 覆盖率 70% | 2h |

### 阶段 7：差异化顶级特性（Week 16-20）— 40h

| # | 任务 | 工时 |
|---|------|------|
| 75 | 交易日历（多市场） | 6h |
| 76 | 矩阵布局 (symbol×time) | 8h |
| 77 | 回测结果缓存 | 4h |
| 78 | Alpha 因子 DSL | 8h |
| 79 | 在线学习 (river) | 6h |
| 80 | A/B 测试（策略变体路由） | 4h |
| 81 | 参数服务器 (Redis) | 4h |

---

## 19. 预期效果对比矩阵

### 性能提升

| 场景 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 单标的回测 10K bars | ~50ms | ~5ms | **10x** |
| 参数扫描 1000 组合 | ~15s | ~2s | **7x** |
| 多标的扫描 50 标的 | ~120s | ~15s | **8x** |
| 鲁棒性扫描 (MC+WF+CPCV) | ~300s | ~40s | **7x** |
| 数据加载 100K bars | ~200ms | ~5ms | **40x** |
| 指标计算 10×100K | ~100ms | ~15ms | **7x** |
| RAG 检索 100K docs | ~30ms | ~1ms | **30x** |
| 实盘信号延迟 | ~1-3ms | ~0.3-0.5ms | **4x** |

### 能力矩阵

| 能力 | 当前 | 优化后 |
|------|:----:|:------:|
| 组合 VaR | ✗ | ✓ |
| 压力测试 | ✗ | ✓ |
| TWAP/VWAP 执行算法 | ✗ | ✓ |
| 确定性回放 | ✗ | ✓ |
| 崩溃恢复 | ✗ | ✓ |
| 金融情感分析 | ✗ | ✓ |
| FAISS HNSW 检索 | ✗ | ✓ |
| structlog 结构化日志 | ✗ | ✓ |
| Prometheus 指标 | ✗ | ✓ |
| OpenTelemetry 追踪 | ✗ | ✓ |
| WebSocket Dashboard | ✗ | ✓ |
| OHLCV 数据验证 | ✗ | ✓ |
| 交易日历 | ✗ | ✓ |
| 因子 DSL | ✗ | ✓ |

### 总工时

| 阶段 | 工时 | 周期 |
|------|------|------|
| 阶段 0：Bug 修复 | 15h | Week 1-2 |
| 阶段 1：引擎性能 | 40h | Week 3-5 |
| 阶段 2：数据 I/O | 30h | Week 5-7 |
| 阶段 3：策略 + 实盘 | 45h | Week 7-10 |
| 阶段 4：风控 + 执行 | 35h | Week 10-12 |
| 阶段 5：RAG + Alpha | 30h | Week 12-14 |
| 阶段 6：生产化 | 35h | Week 14-16 |
| 阶段 7：差异化 | 40h | Week 16-20 |
| **总计** | **270h** | **~20 周** |

---

# Part V — 附录

## 附录 A：文件级问题清单

| 文件 | 问题数 | P0 | P1 | P2 |
|------|:------:|:--:|:--:|:--:|
| `backtest/kernels.py` | 8 | 0 | 4 | 4 |
| `backtest/robust.py` | 2 | 1 | 1 | 0 |
| `backtest/robust_scan.py` | 3 | 1 | 1 | 1 |
| `backtest/backtest_engine.py` | 4 | 0 | 2 | 2 |
| `backtest/protocols.py` | 2 | 0 | 1 | 1 |
| `backtest/order_manager.py` | 3 | 0 | 1 | 2 |
| `backtest/portfolio.py` | 3 | 0 | 0 | 3 |
| `backtest/tca.py` | 3 | 0 | 1 | 2 |
| `backtest/fill_simulator.py` | 2 | 0 | 0 | 2 |
| `strategy/momentum_breakout_strategy.py` | 3 | 1 | 1 | 1 |
| `strategy/drift_regime_strategy.py` | 3 | 1 | 1 | 1 |
| `strategy/zscore_reversion_strategy.py` | 2 | 1 | 0 | 1 |
| `strategy/macd_strategy.py` | 2 | 1 | 0 | 1 |
| `strategy/adaptive_regime_ensemble.py` | 3 | 1 | 1 | 1 |
| `strategy/microstructure_momentum.py` | 2 | 1 | 0 | 1 |
| `strategy/lorentzian_strategy.py` | 3 | 0 | 2 | 1 |
| `strategy/base_strategy.py` | 2 | 0 | 1 | 1 |
| `data/cache_manager.py` | 3 | 0 | 1 | 2 |
| `data/dataset.py` | 3 | 0 | 1 | 2 |
| `data/adapters/database_adapter.py` | 2 | 1 | 0 | 1 |
| `data/adapters/file_adapter.py` | 1 | 0 | 1 | 0 |
| `data/adapters/api_adapter.py` | 2 | 0 | 0 | 2 |
| `data/storage/binary_mmap_storage.py` | 3 | 0 | 1 | 2 |
| `data/storage/arrow_ipc_storage.py` | 2 | 0 | 1 | 1 |
| `data/storage/parquet_storage.py` | 3 | 0 | 1 | 2 |
| `rag/pipeline.py` | 2 | 0 | 1 | 1 |
| `rag/store/vector_store.py` | 1 | 0 | 0 | 1 |
| `live/risk.py` | 3 | 0 | 2 | 1 |
| `live/trading_runner.py` | 3 | 0 | 0 | 3 |
| `live/price_feed.py` | 2 | 0 | 0 | 2 |

**总计：75+ 个问题（10 个 P0，22+ 个 P1，43+ 个 P2）**

---

## 附录 B：依赖分析与 pyproject.toml 建议

### 当前问题

1. `aiohttp>=3.8.0`（注释）与 `aiohttp>=3.9.0`（未注释）版本冲突
2. 缺少 Python 版本约束
3. 无 pyproject.toml
4. 无 pytest-cov
5. Polars 代码中使用但 requirements 中为注释

### 建议

```toml
[project]
name = "quant-framework"
requires-python = ">=3.9"
dependencies = [
    "pandas>=1.5.0", "numpy>=1.21.0", "pyarrow>=10.0.0",
    "scipy>=1.9.0", "scikit-learn>=1.0.0", "matplotlib>=3.5.0", "yfinance>=0.2.0",
]
[project.optional-dependencies]
numba = ["numba>=0.56.0"]
rag = ["sentence-transformers>=2.2.0"]
live = ["websockets>=12.0", "aiohttp>=3.9.0"]
dashboard = ["dash>=2.14.0", "plotly>=5.18.0", "dash-bootstrap-components>=1.5.0"]
db = ["sqlalchemy>=1.4.0"]
polars = ["polars>=0.19.0"]
dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0"]
all = ["quant-framework[numba,rag,live,dashboard,db,polars,dev]"]
```

---

## 附录 C：代码规模与复杂度热点

| 文件 | 行数 | 复杂度风险 |
|------|------|-----------|
| `backtest/kernels.py` | 3,215 | **极高** — 应拆分 |
| `examples/walk_forward_robust_scan.py` | 2,067 | 高 |
| `examples/long_short_leveraged_scan.py` | 1,807 | 高 |
| `backtest/kernels_backup.py` | 1,342 | **应删除** |
| `tests/test_rag_full_chain.py` | 1,024 | 中 — 非 pytest |
| `dashboard/charts.py` | 859 | 中 |
| `backtest/__init__.py` | 824 | 中 |
| `live/price_feed.py` | 755 | 中 |
| `backtest/robust_scan.py` | 715 | 高 |
| `backtest/backtest_engine.py` | 704 | 高 |

---

> 本计划将框架从「功能完备的量化框架」升级为「机构级高性能量化 AI 平台」，在回测速度、实盘延迟、风控深度、RAG 金融专属能力、生产可观测性等维度全面对标甚至超越 Qlib / VectorBT / ArcticDB 等顶级框架。总计 81 项具体任务，270 工时，20 周完成。
