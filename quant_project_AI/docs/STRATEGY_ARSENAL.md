# 18 大策略终极排名报告

> 测试数据: AAPL / GOOGL / TSLA · 3 年日线数据 (2022-02~2025-02)
> 条件: 扣除滑点 (5bps) + 手续费 (15bps) · 密集参数扫描
> 引擎: Numba @njit 全编译 · 782,127 次回测 · 4.6 秒完成 · 168,878 combos/sec

---

## 一、策略分类总览

| 编号 | 策略 | 类型 | 来源 | 框架类 |
|:---:|:---|:---|:---|:---|
| 1 | MA Crossover | 趋势跟踪 | 经典技术分析 | `MovingAverageStrategy` |
| 2 | RSI | 超买超卖 | Welles Wilder | `RSIStrategy` |
| 3 | MACD | 趋势+动量 | Gerald Appel | `MACDStrategy` |
| 4 | DriftRegime | 均值回归 | arxiv 2511.12490 (Sharpe 13) | `DriftRegimeStrategy` |
| 5 | RAMOM | 风险调整动量 | SSRN 2457647 | Numba benchmark |
| 6 | Turtle | 通道突破 | 海龟交易系统 | Numba benchmark |
| 7 | Bollinger | 均值回归 | John Bollinger | Numba benchmark |
| 8 | Keltner | 通道突破 | Chester Keltner | Numba benchmark |
| 9 | MultiFactor | 多因子融合 | 动量+价值+波动 | Numba benchmark |
| 10 | VolRegime | 状态自适应 | 波动率分段切换 | Numba benchmark |
| 11 | ConnorsRSI2 | 均值回归 | Larry Connors | Numba benchmark |
| 12 | MESA (MAMA/FAMA) | 自适应趋势 | John Ehlers Hilbert | `MESAStrategy` |
| 13 | KAMA | 自适应趋势 | Perry Kaufman | `KAMAStrategy` |
| 14 | Donchian+ATR | 突破+跟踪止损 | Richard Donchian | Numba benchmark |
| 15 | Dual Thrust | 日内突破 | TB 经典 | Numba benchmark |
| 16 | Z-Score Reversion | 统计套利 | 经典统计学 | `ZScoreReversionStrategy` |
| 17 | Momentum Breakout | 动量突破 | 52周高点动量 | `MomentumBreakoutStrategy` |
| 18 | Regime Switch EMA | 状态切换 | 波动率自适应EMA | Numba benchmark |

---

## 二、终极排名 (3 标的平均收益)

| 排名 | 策略 | 平均 3 年收益 | AAPL | GOOGL | TSLA |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **RSI** | **+395.53%** | +157.3% | +289.0% | +740.3% |
| 2 | **DriftRegime** | **+347.18%** | +90.0% | +76.8% | +874.7% |
| 3 | **MACD** | **+207.05%** | +89.8% | +200.9% | +330.5% |
| 4 | Turtle | +156.96% | +88.8% | +148.3% | +233.8% |
| 5 | MultiFactor | +151.93% | +112.4% | +251.5% | +91.9% |
| 6 | MA Crossover | +144.87% | +71.5% | +179.3% | +183.8% |
| 7 | MomBreakout | +128.65% | +67.8% | +186.9% | +131.3% |
| 8 | RAMOM | +127.02% | +54.6% | +162.0% | +164.4% |
| 9 | DonchianATR | +122.97% | +41.5% | +197.6% | +129.8% |
| 10 | KAMA | +96.13% | +36.8% | +132.2% | +119.3% |
| 11 | MESA | +93.21% | +10.2% | +170.5% | +98.9% |
| 12 | RegimeEMA | +83.85% | +36.5% | +123.3% | +91.7% |
| 13 | VolRegime | +71.38% | +52.6% | +111.3% | +50.2% |
| 14 | Keltner | +65.90% | +18.6% | +101.5% | +77.6% |
| 15 | ZScoreRev | +58.96% | +34.6% | +32.3% | +110.0% |
| 16 | Bollinger | +51.83% | +22.5% | +32.7% | +100.2% |
| 17 | ConnorsRSI2 | +49.68% | +31.2% | +31.3% | +86.5% |

> 注：Dual Thrust 因使用同一根 K 线的 Open/Close 存在日内偏差，单独评价不与日线策略对比。
> 其在日内框架下表现极强 (TSLA 3 年 2781 万%)，适合分钟级框架。

---

## 三、各标的最优参数

### AAPL 最优参数

| 策略 | 收益 | 最优参数 |
|:---|:---:|:---|
| RSI | +157.3% | period=9, oversold=15, overbought=80 |
| MultiFactor | +112.4% | rsi=5, mom=35, vol=30, lt=0.6, st=0.3 |
| DriftRegime | +90.0% | lookback=10, threshold=0.52, hold=19 |
| MACD | +89.8% | fast=14, slow=76, signal=50 |
| Turtle | +88.8% | entry=5, exit=30, atr_p=20, stop=3.5 |

### GOOGL 最优参数

| 策略 | 收益 | 最优参数 |
|:---|:---:|:---|
| RSI | +289.0% | period=9, oversold=35, overbought=85 |
| MultiFactor | +251.5% | rsi=21, mom=30, vol=35, lt=0.55, st=0.2 |
| MACD | +200.9% | fast=26, slow=166, signal=34 |
| DonchianATR | +197.6% | entry=8, atr_p=10, mult=4.0 |
| MomBreakout | +186.9% | high_p=40, proximity=8%, atr_p=10, trail=3.5 |

### TSLA 最优参数

| 策略 | 收益 | 最优参数 |
|:---|:---:|:---|
| DriftRegime | +874.7% | lookback=15, threshold=0.62, hold=27 |
| RSI | +740.3% | period=28, oversold=35, overbought=75 |
| MACD | +330.5% | fast=16, slow=18, signal=16 |
| Turtle | +233.8% | entry=5, exit=39, atr_p=10, stop=1.0 |
| MA Crossover | +183.8% | short=4, long=39 |

---

## 四、策略特性深度分析

### 冠军: RSI (+395.53% 平均)

- **类型**: 超买超卖均值回归
- **优势**: 短周期 RSI (period=9~28) 捕捉短期价格极端，配合宽松阈值实现高频交易
- **适用市场**: 震荡市与趋势市均表现良好，在 TSLA 的高波动环境下尤其突出
- **风险提示**: 参数敏感性高，需要稳健性检验 (跨时段/跨标的)

### 亚军: DriftRegime (+347.18% 平均)

- **类型**: 学术论文级均值回归
- **优势**: TSLA 上获得 +874.7% 的极端收益，在高波动标的上表现出色
- **核心逻辑**: 当上涨/下跌天数比例极端时，预期反转
- **风险提示**: 在低波动环境 (如 GOOGL) 表现一般

### 季军: MACD (+207.05% 平均)

- **类型**: 趋势+动量复合
- **优势**: 三个标的表现均衡 (+89.8% / +200.9% / +330.5%)
- **核心逻辑**: 经典的快慢 EMA 差异 + 信号线交叉
- **风险提示**: 参数空间大 (fast/slow/signal 三维)，扫描成本高

### 新策略亮点

| 策略 | 独特价值 |
|:---|:---|
| **MomBreakout** | GOOGL 上 +186.9%，纯动量+ATR跟踪止损，适合趋势强的标的 |
| **DonchianATR** | GOOGL 上 +197.6%，经典通道突破+动态止损，稳健性好 |
| **KAMA** | 三标的均正收益，自适应特性天然抗噪声 |
| **MESA** | GOOGL 上 +170.5%，零参数调优需求，适合懒人交易 |
| **ZScoreRev** | TSLA 上 +110.0%，纯统计方法，可扩展到配对交易 |

---

## 五、性能基准

### 吞吐量排名

| 策略 | 耗时 (3 标的) | 组合数 | 速度 (combos/sec) |
|:---|:---:|:---:|:---:|
| MA Crossover | 0.07s | 59,103 | **894,092** |
| ZScoreRev | 0.01s | 5,280 | 458,290 |
| MACD | 1.18s | 488,775 | 414,511 |
| Keltner | 0.01s | 2,415 | 298,323 |
| RSI | 0.17s | 33,432 | 192,158 |
| ConnorsRSI2 | 0.04s | 7,200 | 169,529 |
| RegimeEMA | 0.03s | 5,220 | 164,079 |
| DualThrust | 0.01s | 1,536 | 162,557 |
| VolRegime | 0.20s | 25,056 | 126,998 |
| KAMA | 0.04s | 3,888 | 101,293 |

> 全部策略: 782,127 次回测 · 总耗时 4.6 秒 · 平均 168,878/sec

### 关键优化技术

1. **Numba @njit(cache=True)**: 所有策略内核编译为机器码，无 Python 解释器开销
2. **预计算指标**: MA/EMA/RSI 等一次性预计算全周期数组，扫描时 O(1) 查表
3. **Single-pass O(n)**: 所有回测内核单次遍历价格数组，无重复计算
4. **标量循环**: 纯 float64 标量运算，CPU 缓存友好
5. **零内存分配**: 回测循环内无 numpy 数组创建，避免 GC 暂停

---

## 六、框架集成

### 已集成为策略类 (可直接 import)

```python
from quant_framework.strategy import (
    MovingAverageStrategy,   # MA 交叉
    RSIStrategy,             # RSI 超买超卖
    MACDStrategy,            # MACD 趋势动量
    DriftRegimeStrategy,     # 漂移状态均值回归
    ZScoreReversionStrategy, # Z-Score 均值回归
    MomentumBreakoutStrategy,# 动量突破
    KAMAStrategy,            # Kaufman 自适应均线
    MESAStrategy,            # Ehlers MAMA/FAMA
)
```

### 极速 Numba 版本 (用于参数扫描)

```python
# examples/cutting_edge_strategies.py — 8 个新策略
from cutting_edge_strategies import (
    bt_connors_rsi2,        # Connors RSI-2
    bt_mesa_adaptive,       # MESA MAMA/FAMA
    bt_kama_crossover,      # KAMA + ATR
    bt_donchian_atr,        # Donchian + ATR Trailing
    bt_dual_thrust,         # Dual Thrust 日内突破
    bt_zscore_reversion,    # Z-Score 均值回归
    bt_momentum_breakout,   # 动量突破
    bt_regime_switch_ema,   # 波动率分段 EMA
)

# examples/advanced_strategies_benchmark.py — 7 个高级策略
from advanced_strategies_benchmark import (
    bt_drift_regime,        # 漂移状态均值回归
    bt_ramom,               # 风险调整动量
    bt_turtle,              # 海龟突破
    bt_bollinger,           # 布林带均值回归
    bt_keltner,             # 肯特纳通道
    bt_multifactor,         # 多因子融合
    bt_vol_regime,          # 波动率状态自适应
)

# examples/param_scan_benchmark.py — 3 个经典策略
from param_scan_benchmark import (
    _bt_ma,                 # MA 交叉
    _bt_rsi,                # RSI
    _bt_macd,               # MACD
)
```

---

## 七、注意事项

1. **过拟合风险**: 密集参数扫描的最优参数可能过拟合历史数据。实际使用应:
   - 跨时段验证 (Walk-forward)
   - 多标的均表现稳定的参数优先
   - 对 Top-10 参数取中位数而非最优
   
2. **成本假设**: 滑点 5bps + 手续费 15bps 为保守估计，高频策略实际成本可能更高

3. **不含做空**: 框架策略类当前仅支持多头。Numba 版本支持多空双向

4. **Dual Thrust 特殊性**: 使用同日 Open/Close 作为入场/出场价格，在日线级别存在不可执行性。
   适合分钟级数据框架使用
