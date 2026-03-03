# Project ZIP Packages — MANIFEST

Unzip all packages into the same root directory to reconstruct the full project.
Directory structure is preserved inside each zip.

Total packages: 20

## 01_backtest_core.zip  (10 files, 54 KB)

  ✓ quant_framework/__init__.py
  ✓ quant_framework/backtest/__init__.py
  ✓ quant_framework/backtest/config.py
  ✓ quant_framework/backtest/kernels.py
  ✓ quant_framework/backtest/robust_scan.py
  ✓ quant_framework/backtest/backtest_engine.py
  ✓ quant_framework/backtest/portfolio.py
  ✓ quant_framework/backtest/fill_simulator.py
  ✓ quant_framework/backtest/order_manager.py
  ✓ quant_framework/backtest/protocols.py

## 02_backtest_support.zip  (8 files, 24 KB)

  ✓ quant_framework/backtest/bias_detector.py
  ✓ quant_framework/backtest/manifest.py
  ✓ quant_framework/backtest/robust.py
  ✓ quant_framework/backtest/tca.py
  ✓ quant_framework/analysis/__init__.py
  ✓ quant_framework/analysis/performance.py
  ✓ quant_framework/visualization/__init__.py
  ✓ quant_framework/visualization/plotter.py

## 03_strategy_part1.zip  (10 files, 22 KB)

  ✓ quant_framework/strategy/__init__.py
  ✓ quant_framework/strategy/base_strategy.py
  ✓ quant_framework/strategy/ma_strategy.py
  ✓ quant_framework/strategy/rsi_strategy.py
  ✓ quant_framework/strategy/macd_strategy.py
  ✓ quant_framework/strategy/mesa_strategy.py
  ✓ quant_framework/strategy/kama_strategy.py
  ✓ quant_framework/strategy/drift_regime_strategy.py
  ✓ quant_framework/strategy/lorentzian_strategy.py
  ✓ quant_framework/strategy/zscore_reversion_strategy.py

## 04_strategy_broker_alpha.zip  (10 files, 21 KB)

  ✓ quant_framework/strategy/momentum_breakout_strategy.py
  ✓ quant_framework/strategy/microstructure_momentum.py
  ✓ quant_framework/strategy/adaptive_regime_ensemble.py
  ✓ quant_framework/broker/__init__.py
  ✓ quant_framework/broker/base.py
  ✓ quant_framework/broker/paper.py
  ✓ quant_framework/alpha/__init__.py
  ✓ quant_framework/alpha/cross_asset.py
  ✓ quant_framework/alpha/evaluator.py
  ✓ quant_framework/alpha/order_flow.py

## 05_alpha_live.zip  (9 files, 23 KB)

  ✓ quant_framework/alpha/volatility.py
  ✓ quant_framework/live/__init__.py
  ✓ quant_framework/live/kernel_adapter.py
  ✓ quant_framework/live/price_feed.py
  ✓ quant_framework/live/risk.py
  ✓ quant_framework/live/trade_journal.py
  ✓ quant_framework/live/trading_runner.py
  ✓ quant_framework/live/latency.py
  ✓ quant_framework/live/replay.py

## 06_data_module.zip  (10 files, 21 KB)

  ✓ quant_framework/data/__init__.py
  ✓ quant_framework/data/data_manager.py
  ✓ quant_framework/data/cache_manager.py
  ✓ quant_framework/data/dataset.py
  ✓ quant_framework/data/indicators.py
  ✓ quant_framework/data/rag_context.py
  ✓ quant_framework/data/adapters/__init__.py
  ✓ quant_framework/data/adapters/api_adapter.py
  ✓ quant_framework/data/adapters/base_adapter.py
  ✓ quant_framework/data/adapters/database_adapter.py

## 07_data_storage_dashboard.zip  (9 files, 19 KB)

  ✓ quant_framework/data/adapters/file_adapter.py
  ✓ quant_framework/data/storage/__init__.py
  ✓ quant_framework/data/storage/arrow_ipc_storage.py
  ✓ quant_framework/data/storage/binary_mmap_storage.py
  ✓ quant_framework/data/storage/parquet_storage.py
  ✓ quant_framework/dashboard/__init__.py
  ✓ quant_framework/dashboard/app.py
  ✓ quant_framework/dashboard/charts.py
  ✓ quant_framework/model/__init__.py

## 08_rag_core.zip  (10 files, 17 KB)

  ✓ quant_framework/rag/__init__.py
  ✓ quant_framework/rag/config.py
  ✓ quant_framework/rag/core.py
  ✓ quant_framework/rag/pipeline.py
  ✓ quant_framework/rag/prompts.py
  ✓ quant_framework/rag/types.py
  ✓ quant_framework/rag/README.md
  ✓ quant_framework/rag/store/__init__.py
  ✓ quant_framework/rag/store/keyword_index.py
  ✓ quant_framework/rag/store/vector_store.py

## 09_rag_retrieval_processing.zip  (10 files, 10 KB)

  ✓ quant_framework/rag/retrieval/__init__.py
  ✓ quant_framework/rag/retrieval/reranker.py
  ✓ quant_framework/rag/retrieval/retriever.py
  ✓ quant_framework/rag/processing/__init__.py
  ✓ quant_framework/rag/processing/chunker.py
  ✓ quant_framework/rag/processing/embedder.py
  ✓ quant_framework/rag/processing/normalizer.py
  ✓ quant_framework/rag/processing/pipeline.py
  ✓ quant_framework/rag/ingestion/__init__.py
  ✓ quant_framework/rag/ingestion/base.py

## 10_rag_ingestion_tests1.zip  (10 files, 12 KB)

  ✓ quant_framework/rag/ingestion/directory_adapter.py
  ✓ quant_framework/rag/ingestion/file_watcher.py
  ✓ quant_framework/rag/ingestion/queue.py
  ✓ quant_framework/rag/ingestion/stream.py
  ✓ tests/__init__.py
  ✓ tests/conftest.py
  ✓ tests/test_backtest.py
  ✓ tests/test_broker.py
  ✓ tests/test_elite_strategies.py
  ✓ tests/test_indicators.py

## 11_tests2.zip  (10 files, 53 KB)

  ✓ tests/test_kernel_regression.py
  ✓ tests/test_live.py
  ✓ tests/test_lorentzian_optimization.py
  ✓ tests/test_new_modules.py
  ✓ tests/test_performance.py
  ✓ tests/test_rag_backtest.py
  ✓ tests/test_rag_full_chain.py
  ✓ tests/test_rag_realtime_stress.py
  ✓ tests/test_replay.py
  ✓ tests/test_strategies.py

## 12_examples1.zip  (10 files, 29 KB)

  ✓ examples/demo_backtest_api.py
  ✓ examples/example_ma_strategy.py
  ✓ examples/example_rag.py
  ✓ examples/example_rag_with_backtest.py
  ✓ examples/example_robust_backtest.py
  ✓ examples/example_execution_diagnostics.py
  ✓ examples/realistic_full_scan.py
  ✓ examples/unified_backtest.py
  ✓ examples/paper_trading.py
  ✓ examples/full_strategy_analysis.py

## 13_examples2.zip  (10 files, 60 KB)

  ✓ examples/advanced_strategies_benchmark.py
  ✓ examples/benchmark_backtest_performance.py
  ✓ examples/benchmark_multi_tf.py
  ✓ examples/comprehensive_multi_tf_analysis.py
  ✓ examples/comprehensive_robust_scan.py
  ✓ examples/crypto_full_benchmark.py
  ✓ examples/cutting_edge_strategies.py
  ✓ examples/dashboard_preview.py
  ✓ examples/download_5m_data.py
  ✓ examples/download_multi_tf_data.py

## 14_examples3.zip  (10 files, 85 KB)

  ✓ examples/intraday_analysis.py
  ✓ examples/investment_research.py
  ✓ examples/leverage_analysis.py
  ✓ examples/long_short_leveraged_scan.py
  ✓ examples/multi_tf_backtest_example.py
  ✓ examples/multi_timeframe_analysis.py
  ✓ examples/next_open_full_scan.py
  ✓ examples/optimize_all_strategies_next_open.py
  ✓ examples/param_decay_study.py
  ✓ examples/param_scan_benchmark.py

## 15_examples4_scripts.zip  (10 files, 63 KB)

  ✓ examples/realtime_signal_analysis.py
  ✓ examples/robust_backtest_api.py
  ✓ examples/run_real_market_backtest.py
  ✓ examples/technical_benchmark.py
  ✓ examples/walk_forward_robust_scan.py
  ✓ audit_returns.py
  ✓ bench_backtest.py
  ✓ bench_full.py
  ✓ run_full_scan.py
  ✓ run_full_scan_v2.py

## 16_root_docs_core.zip  (10 files, 53 KB)

  ✓ _pack.py
  ✓ pack_project.py
  ✓ README.md
  ✓ requirements.txt
  ✓ docs/ARCHITECTURE.md
  ✓ docs/BEGINNER_GUIDE.md
  ✓ docs/DESIGN.md
  ✓ docs/INDEX.md
  ✓ docs/PROJECT_KNOWLEDGE.md
  ✓ docs/coding_guide.md

## 17_docs_technical.zip  (10 files, 39 KB)

  ✓ docs/AUDIT.md
  ✓ docs/BACKTEST_VS_LIVE.md
  ✓ docs/COMPARISON.md
  ✓ docs/EXECUTION_DIAGNOSTICS.md
  ✓ docs/FRAMEWORK_COMPARISON.md
  ✓ docs/FREQUENCY_SUPPORT.md
  ✓ docs/IO_FAST.md
  ✓ docs/LIVE_TRADING_AND_LATENCY.md
  ✓ docs/OPTIMIZATION.md
  ✓ docs/PERFORMANCE.md

## 18_docs_strategy_rag.zip  (10 files, 64 KB)

  ✓ docs/POLARS_VS_PANDAS.md
  ✓ docs/QUANT_FRAMEWORK_COMPARISON.md
  ✓ docs/RAG.md
  ✓ docs/RAG_TEST_REPORT.md
  ✓ docs/ROBUST_ANTI_OVERFIT_REPORT.md
  ✓ docs/ROBUST_BACKTEST.md
  ✓ docs/STRATEGY_ARSENAL.md
  ✓ docs/UPGRADE_PLAN_V1.md
  ✓ docs/VALIDATION_REPORT.md
  ✓ docs/PARAM_DECAY_STUDY.md

## 19_docs_reports.zip  (10 files, 98 KB)

  ✓ docs/ALL_STRATEGIES_NEXT_OPEN_REPORT.md
  ✓ docs/COMPREHENSIVE_BACKTEST_ANALYSIS_V3.md
  ✓ docs/COMPREHENSIVE_ROBUST_REPORT.md
  ✓ docs/CRYPTO_BENCHMARK_REPORT.md
  ✓ docs/FREQTRADE_BACKTEST_LIVE_ANALYSIS.md
  ✓ docs/INSTITUTIONAL_QUANT_VS_FRAMEWORK.md
  ✓ docs/INVESTMENT_RESEARCH_REPORT.md
  ✓ docs/LONG_SHORT_LEVERAGED_REPORT.md
  ✓ docs/NEXT_OPEN_FULL_SCAN_REPORT.md
  ✓ reports/full_analysis_report.md

## 20_reports_remaining.zip  (6 files, 34 KB)

  ✓ reports/intraday_5m_analysis_report.md
  ✓ reports/leverage_analysis_report.md
  ✓ reports/multi_tf_fusion_analysis_v2.md
  ✓ reports/multi_timeframe_analysis_report.md
  ✓ reports/technical_benchmark_report.md
  ✓ results/robust_scan/robust_report.md

---
Total: 193 files in 20 packages