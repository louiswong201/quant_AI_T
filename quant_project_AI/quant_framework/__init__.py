"""
Quant Framework v3.0 — Production-grade quantitative trading & RAG framework.

Architecture (SOLID-decomposed):
  quant_framework/
  ├── alpha/         — Feature engineering (OrderFlow, CrossAsset, Volatility, Evaluator)
  ├── analysis/      — PerformanceAnalyzer, unified metrics (Sharpe, Sortino, VaR, DSR, ...)
  ├── backtest/      — Engine, FillSimulator, OrderManager, Portfolio (leverage/short/SL), TCA
  ├── broker/        — Paper & live broker adapters
  ├── data/          — DataManager, indicators (Numba-compiled), cache, storage
  ├── live/          — Latency tracking, risk management, replay analysis
  ├── rag/           — RAG pipeline (ingest, embed, retrieve, rerank)
  ├── strategy/      — BaseStrategy + implementations (MA, RSI, MACD, ARE, MSM, Lorentzian, ...)
  └── visualization/ — Plotting utilities
"""

__version__ = "3.0.0"

# Platform-specific Numba configuration MUST run before any numba import.
from .platform_config import configure as _configure_platform
_configure_platform()

from .data import DataManager, RagContextProvider
from .backtest import (
    backtest,
    optimize,
    backtest_portfolio,
    BacktestConfig,
    BacktestResult,
    OptimizeResult,
    BestStrategy,
    PortfolioResult,
    CPCVResult,
    DetailedKernelResult,
    run_cpcv_scan,
    run_kernel_detailed,
    DEFAULT_PARAM_GRIDS,
    KERNEL_NAMES,
    BacktestEngine,
    BiasDetector,
    PortfolioTracker,
    CostModelFillSimulator,
    TransactionCostAnalyzer,
    run_robust_backtest,
    build_run_manifest,
)
from .analysis import (
    PerformanceAnalyzer,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    conditional_var,
)
from .alpha import (
    OrderFlowFeatures,
    CrossAssetFeatures,
    VolatilityFeatures,
    FeatureEvaluator,
)
from .strategy import (
    BaseStrategy,
    MovingAverageStrategy,
    RSIStrategy,
    MACDStrategy,
    AdaptiveRegimeEnsemble,
    MicrostructureMomentum,
)
from .broker import Broker, PaperBroker
from .live import (
    BarEvent,
    BinanceFeed,
    KernelAdapter,
    LatencyTracker,
    PriceFeedManager,
    RiskConfig,
    RiskGate,
    RiskManagedBroker,
    TradeJournal,
    TradingRunner,
    YFinanceFeed,
    analyze_execution_divergence,
    export_execution_diagnostics_bundle,
    export_execution_divergence_report,
)

__all__ = [
    "__version__",
    # data
    "DataManager",
    "RagContextProvider",
    # backtest — primary API
    "backtest",
    "optimize",
    "backtest_portfolio",
    "BacktestConfig",
    "BacktestResult",
    "OptimizeResult",
    "BestStrategy",
    "PortfolioResult",
    "CPCVResult",
    "DetailedKernelResult",
    "run_cpcv_scan",
    "run_kernel_detailed",
    "DEFAULT_PARAM_GRIDS",
    "KERNEL_NAMES",
    # backtest — internal (backward compat)
    "BacktestEngine",
    "BiasDetector",
    "PortfolioTracker",
    "CostModelFillSimulator",
    "TransactionCostAnalyzer",
    "run_robust_backtest",
    "build_run_manifest",
    # analysis
    "PerformanceAnalyzer",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
    "conditional_var",
    # alpha
    "OrderFlowFeatures",
    "CrossAssetFeatures",
    "VolatilityFeatures",
    "FeatureEvaluator",
    # strategy
    "BaseStrategy",
    "MovingAverageStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "AdaptiveRegimeEnsemble",
    "MicrostructureMomentum",
    # broker
    "Broker",
    "PaperBroker",
    # live
    "BarEvent",
    "BinanceFeed",
    "KernelAdapter",
    "LatencyTracker",
    "PriceFeedManager",
    "RiskConfig",
    "RiskGate",
    "RiskManagedBroker",
    "TradeJournal",
    "TradingRunner",
    "YFinanceFeed",
    "analyze_execution_divergence",
    "export_execution_diagnostics_bundle",
    "export_execution_divergence_report",
]
