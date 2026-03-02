# 执行诊断与一致性闭环

本文档说明如何使用框架中的执行诊断能力，缩小 **回测 / 纸盘 / 实盘** 差异。

---

## 一、目标

- 将“回测结果好但实盘差”拆解为可观测指标：延迟、价格偏差、成交缺失。
- 通过统一成本参数（手续费/滑点/冲击/流动性约束）提升一致性。
- 产出可审计报告与原始 CSV，便于复盘与团队协作。

---

## 二、关键配置（BacktestConfig）

执行相关配置位于 `quant_framework.backtest.BacktestConfig`：

- `max_participation_rate`：单根 bar 最大参与率（容量约束）。
- `impact_bps_buy_coeff` / `impact_bps_sell_coeff`：市场冲击系数。
- `impact_exponent`：冲击非线性指数。
- `adaptive_impact`：是否按波动率动态调整冲击。
- `impact_vol_window`：自适应冲击的波动率窗口。
- `impact_vol_ref`：波动率参考值（用于缩放）。
- `auto_export_execution_report`：回测结束自动导出执行偏差报告。
- `execution_report_path`：报告输出路径（默认 `docs/execution_divergence_report.md`）。

---

## 三、自动导出诊断包

当 `auto_export_execution_report=True` 且 `engine.run(..., live_fills=...)` 传入实盘成交后，
回测将自动输出：

- `execution_divergence_report.md`
- `backtest_trades.csv`
- `live_fills.csv`

并在结果中返回：

- `execution_divergence`
- `execution_report_path`
- `execution_bundle`

---

## 四、示例代码

可直接运行：

- `examples/example_execution_diagnostics.py`

该示例演示了：

1. 开启流动性约束 + 非线性冲击 + 自适应冲击；
2. 传入 `live_fills`；
3. 自动导出诊断报告与 CSV 包。

---

## 五、调参建议（实践）

- 先固定 `max_participation_rate`（例如 0.1~0.3）再调冲击系数。
- 先用较保守 `impact_exponent`（1.0~1.5），避免过度惩罚。
- 实盘成交偏差大时，优先检查：
  - 是否有系统性延迟（`mean/p95 delay`）
  - 是否大量 `missing_in_live`
  - 是否价格偏差长尾（`p95_price_diff_bps`）
- 若波动放大期偏差更大，可开启 `adaptive_impact=True`。

---

## 六、常见误区

- 只加滑点，不加容量与冲击：会低估实盘摩擦。
- 只看均值，不看分位数：长尾风险被掩盖。
- 回测与纸盘成本参数不一致：会制造伪差异。

