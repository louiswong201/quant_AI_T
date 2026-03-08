# Benchmark Report

Generated: 2026-03-08

## Scope

This report benchmarks the `Ultimate Comprehensive Backtest` run completed on this machine and translates the raw runtime into reusable performance metrics:

- full-pipeline wall-clock runtime
- unique-parameter throughput
- effective evaluation throughput
- per-symbol throughput
- phase-level bottleneck analysis
- horizontal comparison versus major open-source quant frameworks

The goal is not to produce a marketing claim, but to establish a fair engineering baseline for this codebase.

## Source Run

Primary run summary:

- Total runtime: `2311.9s` (`38.5 min`)
- Unique parameter combinations: `132,442`
- Phase 1 runtime: `2102.1s`
- Phase 2 runtime: `63.0s`
- Phase 3 runtime: `93.0s`
- Phase 2 tested combinations: `37,050`
- Phase 3 CPCV combinations: `1,899,000`

Phase 1 segmented scan logs used for throughput analysis:

- `1d / 1x / crypto (10 sym) ... 63.1s (7,946,520 combos)`
- `1d / 1x / stock  (20 sym) ... 210.6s (15,893,040 combos)`
- `1d / 2x / crypto (10 sym) ... 34.9s (7,946,520 combos)`
- `1d / 2x / stock  (20 sym) ... 228.6s (15,893,040 combos)`
- `1d / 3x / crypto (10 sym) ... 33.5s (7,946,520 combos)`
- `1d / 3x / stock  (20 sym) ... 211.1s (15,893,040 combos)`
- `4h / 1x / crypto (9 sym)  ... 53.4s (7,151,868 combos)`
- `4h / 1x / stock  (22 sym) ... 48.7s (17,482,344 combos)`
- `4h / 2x / crypto (9 sym)  ... 53.2s (7,151,868 combos)`
- `4h / 2x / stock  (22 sym) ... 49.4s (17,482,344 combos)`
- `4h / 3x / crypto (9 sym)  ... 52.1s (7,151,868 combos)`
- `4h / 3x / stock  (22 sym) ... 46.2s (17,482,344 combos)`
- `1h / 1x / crypto (9 sym)  ... 219.4s (7,151,868 combos)`
- `1h / 1x / stock  (22 sym) ... 121.7s (17,482,344 combos)`
- `1h / 2x / crypto (9 sym)  ... 216.5s (7,151,868 combos)`
- `1h / 2x / stock  (22 sym) ... 122.8s (17,482,344 combos)`
- `1h / 3x / crypto (9 sym)  ... 215.7s (7,151,868 combos)`
- `1h / 3x / stock  (22 sym) ... 121.4s (17,482,344 combos)`

## Important Measurement Note

There are two valid throughput lenses:

1. `Unique parameter throughput`
   Measures how many unique parameter tuples from the search grid are processed per second.
   This is the strictest and easiest metric to compare to optimization tools.

2. `Effective evaluation throughput`
   Measures how many actual evaluation events the engine performs per second after expanding across:
   - symbols
   - timeframes
   - leverage buckets
   - robustness validation paths

For serious research systems, the second metric is usually the more honest representation of engine work.

## Derived Metrics

### 1. Full Pipeline Throughput

Formula:

`unique_params_per_sec = unique_params / total_runtime`

Result:

- `132,442 / 2311.9 = 57.29 unique params/s`
- `57.29 * 60 = 3,437.2 unique params/min`

Interpretation:

- As a full research pipeline, this system completes about `3.4k` unique hyperparameter candidates per minute while also doing multi-TF fusion, CPCV, ranking, and report export.

### 2. Phase 1 Effective Evaluation Throughput

Summed Phase 1 scan load:

- Total effective Phase 1 evaluation count: `219,323,952`
- Phase 1 runtime: `2102.1s`

Formula:

`phase1_effective_eval_rate = total_phase1_combos / phase1_seconds`

Result:

- `219,323,952 / 2102.1 = 104,335.6 combo-evals/s`

Interpretation:

- The core scan engine is sustaining about `104k` effective evaluations per second during the dominant workload.

### 3. Phase 1 Per-Symbol Throughput

Formula:

`combo_evals_per_symbol_sec = total_phase1_combos / sum(symbol_count * segment_seconds)`

Result:

- `6,678.9 combo-evals/s/symbol`

Interpretation:

- After normalizing by symbol-count exposure, each symbol receives roughly `6.7k` effective strategy evaluations per second.

### 4. Phase 2 Throughput

Formula:

`phase2_tests_per_sec = phase2_tested_combinations / phase2_seconds`

Result:

- `37,050 / 63.0 = 588.1 multi-TF tests/s`

Interpretation:

- Multi-timeframe fusion is not the main bottleneck. It is materially lighter than Phase 1.

### 5. Phase 3 Throughput

Formula:

`phase3_combo_rate = cpcv_combos / phase3_seconds`

Result:

- `1,899,000 / 93.0 = 20,419.4 CPCV combos/s`
- Per symbol: `1,361.3 CPCV combos/s/symbol`

Interpretation:

- CPCV validation is computationally meaningful, but still far from the dominant runtime driver.

## Bottleneck Decomposition

Phase share of full runtime:

- Phase 1: `90.9%`
- Phase 2: `2.7%`
- Phase 3: `4.0%`
- Remaining reporting / finalization overhead: about `2.4%`

Conclusion:

- Optimization effort should focus almost entirely on `Phase 1`.

## Throughput by Timeframe

| Timeframe | Effective Combos | Runtime (s) | Combos/s | Combos/s/symbol |
|-----------|------------------|-------------|----------|-----------------|
| `1d` | 71,518,680 | 781.8 | 91,479.5 | 4,994.0 |
| `4h` | 73,902,636 | 303.0 | 243,903.1 | 16,055.7 |
| `1h` | 73,902,636 | 1017.5 | 72,631.6 | 5,311.3 |

Interpretation:

- `4h` is the fastest lane by a wide margin.
- `1h` is the slowest lane and the strongest candidate for kernel-path optimization.
- `1d` sits in the middle, but is still materially slower than `4h`.

## Throughput by Asset Bucket

| Bucket | Effective Combos | Runtime (s) | Combos/s | Combos/s/symbol |
|--------|------------------|-------------|----------|-----------------|
| `crypto` | 66,750,768 | 941.8 | 70,875.7 | 7,754.8 |
| `stock`  | 152,573,184 | 1160.5 | 131,471.9 | 6,296.8 |

Interpretation:

- Aggregate stock throughput is higher because more stock symbols are being processed in the fast `4h` lane.
- Per-symbol crypto throughput is slightly higher after normalization.

## Engineering Assessment

This run demonstrates that the framework is not behaving like a typical slow event-driven Python backtester.

On this machine, it behaves much closer to a specialized research engine:

- Numba-accelerated kernel execution
- large-grid scanning
- multi-symbol expansion
- robustness testing
- validation and export in the same run

The most meaningful internal baseline from this run is:

- `104.3k` effective evaluations per second during the core search workload

That is the number to carry forward as the practical engine-speed benchmark for future regressions or optimizations.

## Horizontal Comparison

The comparison below is intentionally careful: different frameworks optimize for different objectives and expose different benchmark units.

### 1. vectorbt

Public positioning:

- official docs emphasize testing `many thousands of strategies in seconds`
- public examples often cite `1,000,000 backtest simulations in 20 seconds`

Approximate public headline:

- about `50,000 simulations/s` in a highly vectorizable scenario

How this benchmark compares:

- On simple brute-force signal sweeps, `vectorbt` remains one of the strongest public Python baselines.
- On this workload, the present engine sustained `57.3 unique params/s` at full pipeline level and `104.3k effective combo-evals/s` in Phase 1.
- Because `vectorbt` benchmarks usually refer to simpler fully vectorized sweeps, the fairest statement is:
  - this engine is competitive with high-performance research-oriented Python frameworks
  - direct apples-to-apples claims require the exact same strategy family, data shape, and validation stack

Professional conclusion:

- `vectorbt` is likely still the better benchmark for pure vectorized parameter sweeps.
- this framework is already in the same general performance tier for complex research pipelines.

### 2. Backtrader

Public benchmark examples:

- about `14,713 candles/s` in one CPython benchmark
- roughly `12,743 candles/s` with trading enabled
- around `30k-35k candles/s` in some PyPy configurations

How this benchmark compares:

- `Backtrader` is event-driven and broker-simulation focused.
- It is not optimized for huge hyperparameter search spaces with robustness validation.
- For this kind of large research scan, this framework is materially faster and more suitable.

Professional conclusion:

- For research optimization throughput, this framework is clearly ahead of `Backtrader`.
- `Backtrader` remains valuable for event-driven modeling and broker semantics, but not for this throughput class.

### 3. LEAN / QuantConnect

Public positioning:

- LEAN publishes official engine benchmarks in `data points per second`
- public community material also notes Python strategies are materially slower than C# due to interop overhead

How this benchmark compares:

- Against `LEAN Python`, this framework is likely faster for local parameter-search and scan-heavy research workloads.
- Against `LEAN C#`, direct comparison is harder because LEAN's strengths are:
  - event-driven execution
  - production parity
  - data-engine integration
  - platform infrastructure

Professional conclusion:

- For local single-machine quant research throughput, this framework is probably stronger than `LEAN Python`.
- For full production ecosystem and event-driven platform semantics, LEAN remains a top-tier reference.

## Summary Verdict

Within the open-source Python quant ecosystem, this run places the framework approximately here:

- clearly above traditional event-driven research stacks like `Backtrader` for large optimization workloads
- likely above `LEAN Python` for local brute-force research scanning
- competitive with top-tier Python research frameworks such as `vectorbt`, though not proven superior on fully vectorized toy workloads
- below what is theoretically achievable with institution-grade distributed or low-level `C++/Rust/GPU` research engines

Short version:

- This is already a high-performance Python quant research engine.
- It is not a generic backtester pretending to be fast.
- The measured bottleneck is concentrated in `Phase 1`, especially the `1h` lane.

## Recommended Benchmark KPI Set

For future optimization work, track these KPIs on every benchmark run:

1. total runtime
2. unique params/s
3. Phase 1 effective combo-evals/s
4. Phase 1 combo-evals/s/symbol
5. Phase 1 throughput by timeframe (`1d`, `4h`, `1h`)
6. Phase 2 tests/s
7. Phase 3 CPCV combos/s
8. peak memory usage

## Optimization Priority

If the objective is to improve runtime by `20%-50%`, the best next targets are:

1. optimize Phase 1 `1h` execution path
2. reduce repeated per-segment overhead in Phase 1
3. improve symbol batching / data locality
4. reduce unnecessary Python-layer object construction in result aggregation
5. profile robustness dispatch overhead inside the scan loop

## References

- `quant_project_AI/reports/ultimate_backtest_report_20260308_005414.md`
- `ultimate_scan_log.txt`
- [vectorbt docs](https://vectorbt.dev/)
- [PyQuant News vectorbt benchmark example](https://pyquantnews.com/1000000-backtest-simulations-20-seconds-vectorbt)
- [Backtrader performance article](https://www.backtrader.com/blog/2019-10-25-on-backtesting-performance-and-out-of-memory/on-backtesting-performance-and-out-of-memory/)
- [QuantConnect LEAN performance page](https://www.quantconnect.com/performance)
- [QuantConnect engine performance docs](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/engine-performance)
