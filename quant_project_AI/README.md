# Quant Framework：极速数据处理 + AI 融合的量化框架

本仓库以**极快数据处理**与**AI 融合**为并列目标：行情与指标等结构化数据走多层极速 I/O（Binary Mmap / Arrow IPC / Parquet），新闻与研报等非结构化数据通过 RAG 与策略在同一时间轴参与决策，支持回测与实盘下的时间一致性与实时性。设计原则见 [docs/DESIGN.md](docs/DESIGN.md)。

---

## 一、框架概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Quant Framework                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  数据层 (data)                                                                │
│  ├── 结构化：DataManager、适配器(API/File/DB)、Dataset、Cache、Parquet 存储   │
│  └── 非结构化：RagContextProvider + RAGPipeline（实时接入 → 检索 → 上下文）   │
├─────────────────────────────────────────────────────────────────────────────┤
│  策略层 (strategy)                                                            │
│  BaseStrategy(on_bar, get_rag_context) + MA/MACD/RSI 等；可选注入 rag_provider │
├─────────────────────────────────────────────────────────────────────────────┤
│  执行层 (backtest)                                                            │
│  BacktestEngine + BacktestConfig：手续费/滑点、多标的、限价/止损（下一 bar OHLC） │
├─────────────────────────────────────────────────────────────────────────────┤
│  分析与展示 (analysis, visualization)                                         │
│  绩效分析、回测曲线与交易图表                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **极速数据处理**：读路径 Binary Mmap → Arrow IPC → Parquet（`fast_io=True`），回测热路径 NumPy 化、指标向量化；可选 CacheManager 内存+磁盘缓存。详见 [docs/IO_FAST.md](docs/IO_FAST.md)、[docs/PERFORMANCE.md](docs/PERFORMANCE.md)。  
- **AI 融合**：非结构化文本实时接入、分块、嵌入、混合检索与重排；策略通过 `RagContextProvider` 在 `on_bar` 中调用 `get_rag_context(..., as_of_date=current_date)`，与行情同 bar 决策，回测不偷看未来。详见 [docs/RAG.md](docs/RAG.md)。  
- **文档即规范**：架构与数据流、性能与 I/O、RAG 使用以 [docs/](docs/) 为准，见 [docs/INDEX.md](docs/INDEX.md)。

---

## 二、目录结构

```
Project_1/
├── README.md                 # 本文件：框架概览与快速开始
├── requirements.txt
├── docs/                     # 文档集中目录
│   ├── INDEX.md              # 文档索引与阅读顺序
│   ├── ARCHITECTURE.md       # 整体架构说明（量化 + RAG）
│   ├── DESIGN.md             # 设计原则：极速数据 + AI 融合
│   ├── RAG.md                # RAG 模块使用与结构
│   ├── OPTIMIZATION.md       # 优化说明（易懂版 + 循环 + 极限）
│   ├── COMPARISON.md         # 与主流 RAG 框架对比
│   └── ...                   # 更多专题文档见 INDEX.md
├── examples/
│   ├── multi_tf_backtest_example.py       # 多 TF 融合回测示例
│   ├── comprehensive_multi_tf_analysis.py # 多 TF 综合分析（参数网格搜索）
│   ├── full_strategy_analysis.py          # 18 策略全面分析
│   ├── example_ma_strategy.py             # 纯量化回测示例
│   ├── example_rag_with_backtest.py       # RAG + 回测融合
│   ├── paper_trading.py                   # 纸盘交易（单/多 TF）
│   └── ...                                # 更多示例见 examples/
├── reports/                  # 自动生成的分析报告
└── quant_framework/
    ├── __init__.py           # 统一入口与版本
    ├── data/                 # 数据层（DataManager + RagContextProvider）
    ├── strategy/             # 策略基类与示例策略
    ├── backtest/             # 回测引擎（单 TF + 多 TF 融合 + 组合回测）
    ├── live/                 # 实盘信号适配（KernelAdapter + MultiTFAdapter）
    ├── analysis/             # 绩效分析
    ├── visualization/        # 绘图
    ├── model/                # 预留
    └── rag/                  # RAG 管道（ingestion / processing / store / retrieval）
```

---

## 三、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 若使用 RAG 真实嵌入与重排，可选：pip install sentence-transformers
# 若需数值热路径加速，可选：pip install numba
```

### 2. 纯量化回测（可选极速 I/O）

```python
from quant_framework import DataManager, BacktestEngine
from quant_framework.strategy import MovingAverageStrategy

# fast_io=True：读优先 binary → arrow → parquet，写时同步写 binary/arrow，回测数据路径最快
dm = DataManager(data_dir="data", fast_io=True)
engine = BacktestEngine(dm, commission_rate=0.001)
strategy = MovingAverageStrategy(name="MA", initial_capital=1_000_000, short_window=5, long_window=20)
result = engine.run(strategy, "STOCK", "2020-01-01", "2023-12-31")
```

### 3. RAG + 回测（策略内用非结构化上下文）

```python
from quant_framework import DataManager, RagContextProvider, BacktestEngine
from quant_framework.rag import RAGPipeline, RAGConfig, Document
from quant_framework.strategy.base_strategy import BaseStrategy

rag = RAGPipeline(config=RAGConfig(chunk_size=256), start_worker=True)
rag.add_documents([Document(content="某公司业绩超预期。", source="news")])
provider = RagContextProvider(pipeline=rag)
# 策略构造时注入 rag_provider，在 on_bar 中调用 self.get_rag_context(..., as_of_date=current_date)
strategy = YourStrategy(name="RAG策略", initial_capital=1_000_000, rag_provider=provider)
engine = BacktestEngine(DataManager(data_dir="data"))
result = engine.run(strategy, symbol="AAPL", start_date="2020-01-01", end_date="2020-12-31")
```

### 4. 多时间框架融合回测（Multi-TF Fusion Backtest）

```python
from quant_framework.backtest import backtest_multi_tf, BacktestConfig

result = backtest_multi_tf(
    tf_configs={
        "1h": ("MA", (10, 50)),
        "4h": ("RSI", (14, 25, 75)),
        "1d": ("MACD", (28, 112, 3)),
    },
    tf_data={"1h": df_1h, "4h": df_4h, "1d": df_1d},
    mode="trend_filter",   # trend_filter | consensus | primary
)
print(result)              # ret, dd, sharpe, trades, equity curve
```

三种融合模式：
- **trend_filter**：最高 TF 设定趋势方向，最低 TF 提供入场时机
- **consensus**：多 TF 仓位多数表决
- **primary**：指定主 TF 决策，其余 TF 仅供观察

### 5. 高保真执行诊断（回测 vs 实盘）

```python
from quant_framework import BacktestConfig, BacktestEngine

cfg = BacktestConfig(
    max_participation_rate=0.2,
    impact_bps_buy_coeff=150.0,
    impact_bps_sell_coeff=120.0,
    impact_exponent=1.2,
    adaptive_impact=True,
    auto_export_execution_report=True,
    execution_report_path="docs/execution_divergence_report.md",
)
engine = BacktestEngine(dm, config=cfg)
result = engine.run(strategy, "STOCK", "2024-01-01", "2024-06-30", live_fills=live_fills_df)
# 结果包含 execution_divergence / execution_report_path / execution_bundle
```

更多用法见 `examples/` 与 **docs/** 下的文档。

---

## 四、文档索引

| 文档 | 内容 |
|------|------|
| [docs/INDEX.md](docs/INDEX.md) | 文档总索引与阅读顺序 |
| [docs/BEGINNER_GUIDE.md](docs/BEGINNER_GUIDE.md) | **初学者完全指南**（从代码到设计到优化，步步讲解） |
| [docs/PROJECT_KNOWLEDGE.md](docs/PROJECT_KNOWLEDGE.md) | **项目知识库**（装饰器/LRU/ABC/typing 等概念与用法） |
| [docs/DESIGN.md](docs/DESIGN.md) | **设计原则：极速数据处理 + AI 融合（必读）** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 整体架构（量化 + RAG 连贯设计） |
| [docs/STRATEGY_ARSENAL.md](docs/STRATEGY_ARSENAL.md) | **18 策略全览**（MA/RSI/MACD/BB/…/Consensus） |
| [docs/BACKTEST_VS_LIVE.md](docs/BACKTEST_VS_LIVE.md) | 回测与实盘差异分析 |
| [docs/LIVE_TRADING_AND_LATENCY.md](docs/LIVE_TRADING_AND_LATENCY.md) | 实盘对接与延迟（Broker 抽象、纸交易） |
| [docs/EXECUTION_DIAGNOSTICS.md](docs/EXECUTION_DIAGNOSTICS.md) | 执行偏差诊断：容量、冲击、自动报告 |
| [docs/IO_FAST.md](docs/IO_FAST.md) | 极速 I/O：Binary Mmap / Arrow IPC |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | 性能与热路径：NumPy / Numba 向量化 |
| [docs/RAG.md](docs/RAG.md) | RAG 模块：使用方式、与量化融合 |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | 优化说明：循环、极限、场景与边界 |
| [docs/COMPARISON.md](docs/COMPARISON.md) | 与 LangChain / LlamaIndex / Haystack 对比 |
| [docs/QUANT_FRAMEWORK_COMPARISON.md](docs/QUANT_FRAMEWORK_COMPARISON.md) | 与 Backtrader / VectorBT / Qlib 等对比 |

---

## 五、版本与引用

- 版本：见 `quant_framework/__init__.py` 中的 `__version__`。
- 量化与 RAG 为同一代码库内模块，统一以 `quant_framework` 引用。
