# Architecture: Quant Framework v2.0

Production-grade quantitative trading framework with RAG integration,
Numba-accelerated backtesting, and SOLID-decomposed engine.

---

## Design Principles

1. **SOLID Decomposition**: Each module has a single responsibility. The backtest
   engine is an orchestrator, not a monolith.
2. **Protocol-Based Interfaces**: Components communicate through `Protocol` classes
   (structural subtyping), enabling seamless swap between backtest and live trading.
3. **Numba-First Hot Paths**: All indicator computations use `@njit(cache=True, fastmath=True)`.
   No Python for-loops in the hot path.
4. **Polars Lazy-Scan I/O**: Parquet reads use `pl.scan_parquet()` with predicate pushdown
   when Polars is available — only row groups matching the date range are decoded.
5. **Anti-Overfitting by Design**: TCA, BiasDetector, and walk-forward are built-in,
   not afterthoughts.

---

## Module Architecture

```
quant_framework/
├── __init__.py              # v2.0.0 — public API surface
├── py.typed                 # PEP 561 typed package marker
│
├── alpha/                   # Next-gen feature engineering
│   ├── order_flow.py        #   OFI, VPIN, Trade Imbalance (@njit)
│   ├── cross_asset.py       #   Rolling Beta, Correlation, Lead-Lag (@njit)
│   ├── volatility.py        #   Yang-Zhang Vol, Vol-of-Vol, Vol Ratio (@njit)
│   └── evaluator.py         #   IC/IR analysis, PCA, orthogonal feature selection
│
├── backtest/                # SOLID-decomposed backtest engine
│   ├── protocols.py         #   Value objects: Order, Fill, BarData; Protocols: IFillSimulator, IOrderManager, IPortfolioTracker
│   ├── config.py            #   BacktestConfig (frozen dataclass): fees, slippage, impact, fill mode
│   ├── fill_simulator.py    #   CostModelFillSimulator: Almgren-Chriss impact, slippage, liquidity cap
│   ├── order_manager.py     #   DefaultOrderManager: submit, cancel, expire, audit trail
│   ├── portfolio.py         #   PortfolioTracker: positions, cash, equity curve (pre-allocated numpy)
│   ├── backtest_engine.py   #   BacktestEngine: slim orchestrator (~280 lines)
│   ├── tca.py               #   TransactionCostAnalyzer: cost decomposition, gross/net Sharpe
│   ├── bias_detector.py     #   BiasDetector: look-ahead, survivorship, data snooping (Bonferroni)
│   ├── robust.py            #   run_robust_backtest: multi-window × multi-param
│   └── manifest.py          #   Run manifest for reproducibility
│
├── broker/                  # Order execution adapters
│   ├── base.py              #   Broker ABC
│   └── paper.py             #   PaperBroker for simulation
│
├── data/                    # Unified data layer
│   ├── data_manager.py      #   DataManager: load/save/cache with LRU or CacheManager
│   ├── dataset.py           #   Dataset: fast_io cascade (binary → arrow → parquet → adapter)
│   ├── indicators.py        #   VectorizedIndicators: all @njit(cache=True, fastmath=True)
│   ├── cache_manager.py     #   CacheManager: in-memory LRU + disk Parquet, per-symbol invalidation
│   ├── rag_context.py       #   RagContextProvider: bridge between RAG pipeline and strategies
│   ├── adapters/            #   Data source adapters (file, API, database)
│   └── storage/             #   Parquet (Polars lazy-scan), Arrow IPC, Binary Mmap
│
├── live/                    # Live trading components
│   ├── risk.py              #   RiskGate, CircuitBreaker, RiskManagedBroker
│   ├── latency.py           #   LatencyTracker
│   └── replay.py            #   Execution divergence analysis
│
├── rag/                     # RAG pipeline (unstructured data)
│   ├── pipeline.py          #   RAGPipeline: ingest → chunk → embed → store → retrieve
│   ├── core.py              #   Vector search hot path (Numba-optional)
│   ├── ingestion/           #   Queue, stream, file watcher, directory adapter
│   ├── processing/          #   Chunker, embedder, normalizer
│   ├── store/               #   Vector store, keyword index
│   └── retrieval/           #   Hybrid retriever, reranker
│
├── strategy/                # Trading strategies
│   ├── base_strategy.py     #   BaseStrategy ABC with on_bar, on_bar_fast, RAG support
│   ├── ma_strategy.py       #   Moving Average crossover
│   ├── rsi_strategy.py      #   RSI mean-reversion
│   ├── macd_strategy.py     #   MACD trend-following
│   ├── lorentzian_strategy.py # Lorentzian Classification (ML-inspired)
│   └── ...                  #   Drift, ZScore, KAMA, MESA, MomentumBreakout
│
├── analysis/                # Post-backtest analysis
│   └── performance.py       #   PerformanceAnalyzer: Sharpe, drawdown, win rate
│
├── visualization/           # Plotting
│   └── plotter.py           #   Equity curve, trade markers
│
└── model/                   # ML model integration (placeholder)
    └── __init__.py
```

---

## Data Flow

```
  Market Data Sources
       │
       ▼
  DataManager  (three access tiers, fastest → most compatible)
       │
       ├── load_arrays(sym, start, end)  → Dict[str, np.ndarray]  (Numba-ready)
       │      └── Polars zero-copy → contiguous float64 arrays
       │
       ├── load_lazy(sym, start, end)    → pl.LazyFrame  (predicate pushdown)
       │      └── pl.scan_parquet() — only decode matching row groups
       │
       └── load_data(sym, start, end)    → pd.DataFrame  (classic, cached)
              │  ┌─ CacheManager (LRU + disk) ─── hit → return
              │  └─ miss → Dataset.load() ──┐
              │                              │
              │    ┌─────────────────────────┘
              │    │  Binary Mmap → Arrow IPC → Parquet (Polars scan) → File Adapter
              │    └─────────────────────────→ DataFrame
       │
       ▼
  BacktestEngine.run()
       │
       ├── _prepare_data() → BarData (contiguous float64 arrays)
       │
       │   For each bar i:
       │   ├── FillSimulator.try_fill_pending()  ← pending orders vs OHLCV
       │   ├── strategy.on_bar() / on_bar_fast() ← generate signals
       │   ├── OrderManager.submit() / cancel()  ← new orders
       │   ├── FillSimulator.execute_market()     ← immediate fills
       │   ├── PortfolioTracker.apply_fill()      ← update positions
       │   └── PortfolioTracker.record_bar()      ← snapshot equity
       │
       ▼
  Results Dict
       │
       ├── TransactionCostAnalyzer.analyze()  ← cost decomposition
       ├── BiasDetector.full_audit()          ← bias checks
       └── PerformanceAnalyzer.analyze()      ← metrics
```

---

## Backtest Engine: Before & After

| Aspect | v1.0 (Monolith) | v2.0 (SOLID) |
|--------|-----------------|--------------|
| `backtest_engine.py` | 759 lines | 280 lines (orchestrator) |
| Order management | Inline dicts + list manipulation | `DefaultOrderManager` with typed `Order` objects |
| Fill logic | Mixed into main loop | `CostModelFillSimulator` (single responsibility) |
| Portfolio | Split between `BaseStrategy` + engine | `PortfolioTracker` (single source of truth) |
| Cost analysis | None | `TransactionCostAnalyzer` with per-trade decomposition |
| Bias detection | None | `BiasDetector` (look-ahead, survivorship, snooping) |
| Live safety | Basic `RiskGate` | `CircuitBreaker` + enhanced `RiskManagedBroker` |
| Indicators | `@jit` (allows Python fallback) | `@njit(cache=True, fastmath=True)` (strict nopython) |
| Parquet I/O | `pl.read_parquet` (eager) | `pl.scan_parquet` (lazy, predicate pushdown) |
| Data access | Pandas-only | 3-tier: arrays (Numba) / LazyFrame (Polars) / DataFrame |
| Strategy loops | Python for-loops | Numba `@njit` compiled (EMA, RSI, CCI, ATR, ADX, KAMA, Hilbert) |
| Alpha features | OHLCV only | OrderFlow + CrossAsset + Volatility + FeatureEvaluator |
| Tests | 120 | 162 |

---

## Key Design Decisions

### Why Protocol over ABC for component interfaces
Structural subtyping allows a live `ExchangeFillAdapter` to satisfy
`IFillSimulator` without inheriting from a common base class. This keeps
the backtest and live trading dependency graphs completely separate.

### Why @njit(fastmath=True) is safe for indicators
Indicator computations are not sensitive to NaN-propagation edge cases
(NaN handled explicitly via `np.full` initialization). The associativity
relaxation only affects the last 1-2 ULP of precision — well within the
noise floor of financial data. The 15-25% speedup from fused-multiply-add
instructions is worth this trade-off.

### Why CircuitBreaker requires manual reset
Automatic reset would defeat the purpose of human oversight during
anomalous conditions. A bug generating erroneous orders at high frequency
could re-trip immediately after auto-reset, causing oscillating losses.
Manual reset forces a human to investigate before resuming.
