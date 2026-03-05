# V4 Research Engine 指标计算详解

> 面向初学者的完整技术指南：每个指标是什么、为什么用它、代码怎么算的

---

## 目录

- [总览：四大引擎与指标地图](#总览四大引擎与指标地图)
- [第一章：Monitor Engine（监控引擎）](#第一章monitor-engine监控引擎)
  - [1.1 Rolling Sharpe Ratio（滚动夏普比率）](#11-rolling-sharpe-ratio滚动夏普比率)
  - [1.2 Drawdown（回撤）](#12-drawdown回撤)
  - [1.3 Win Rate EMA（指数加权胜率）](#13-win-rate-ema指数加权胜率)
  - [1.4 Trade Frequency（交易频率）](#14-trade-frequency交易频率)
  - [1.5 Health Status（健康状态判定）](#15-health-status健康状态判定)
  - [1.6 Regime Detection（市场状态识别）](#16-regime-detection市场状态识别)
  - [1.7 Performance Attribution（表现归因分析）](#17-performance-attribution表现归因分析)
- [第二章：Optimizer Engine（优化引擎）](#第二章optimizer-engine优化引擎)
  - [2.1 Bayesian Parameter Update（贝叶斯参数融合）](#21-bayesian-parameter-update贝叶斯参数融合)
  - [2.2 Composite Gate Score（复合门控评分）](#22-composite-gate-score复合门控评分)
  - [2.3 Parameter Neighborhood Stability（参数邻域稳定性）](#23-parameter-neighborhood-stability参数邻域稳定性)
  - [2.4 Champion/Challenger Protocol（冠军/挑战者协议）](#24-championchallenger-protocol冠军挑战者协议)
- [第三章：Portfolio Engine（投资组合引擎）](#第三章portfolio-engine投资组合引擎)
  - [3.1 Correlation Matrix（相关性矩阵）](#31-correlation-matrix相关性矩阵)
  - [3.2 Weight Optimization（权重优化）](#32-weight-optimization权重优化)
  - [3.3 Portfolio Metrics（组合层级指标）](#33-portfolio-metrics组合层级指标)
- [第四章：Discover Engine（发现引擎）](#第四章discover-engine发现引擎)
  - [4.1 Autocorrelation Shift（自相关结构变化）](#41-autocorrelation-shift自相关结构变化)
  - [4.2 Volatility Regime Change（波动率状态变化）](#42-volatility-regime-change波动率状态变化)
  - [4.3 Skewness Shift（偏度变化）](#43-skewness-shift偏度变化)
  - [4.4 Cross-Asset Correlation（跨资产相关性漂移）](#44-cross-asset-correlation跨资产相关性漂移)
- [附录：年化因子与跨时间框架处理](#附录年化因子与跨时间框架处理)

---

## 总览：四大引擎与指标地图

```
┌─────────────────────────────────────────────────────────────────┐
│                   V4 Research System                            │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   Monitor    │  Optimizer   │  Portfolio   │    Discover       │
│  (每天运行)   │ (每周/触发)   │  (每周运行)   │   (每月运行)       │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│ Sharpe 比率   │ 贝叶斯融合    │ 相关性矩阵    │ 自相关结构变化     │
│ 回撤 / 回撤   │ 门控评分      │ 权重优化      │ 波动率变化         │
│  持续时间     │ 邻域稳定性    │ Sharpe/Sortino│ 偏度变化          │
│ 胜率 EMA     │ 冠军/挑战者   │ 分散化比率     │ 跨资产相关性漂移   │
│ 交易频率      │              │ 边际风险贡献   │                   │
│ 状态识别      │              │              │                   │
│ 表现归因      │              │              │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

---

## 第一章：Monitor Engine（监控引擎）

Monitor 引擎是"体检医生"——每天对每个正在运行的策略做健康检查。

### 1.1 Rolling Sharpe Ratio（滚动夏普比率）

#### 这个指标是什么？

Sharpe Ratio 衡量的是**"每承担一单位风险，能获得多少回报"**。它是量化交易中最重要的单一指标。

用生活中的比喻：假设你开车去上班，有两条路：
- A 路：平均 30 分钟，但有时 20 分钟有时 50 分钟（波动大）
- B 路：平均 32 分钟，但总是 31-33 分钟之间（波动小）

Sharpe Ratio 就是在说：B 路虽然平均慢一点，但因为**稳定**，实际体验更好。

#### 为什么要用它？

- 纯看收益率会被一次暴利扭曲（比如策略亏了 11 个月，最后一个月暴涨 200%，总收益很高但过程痛苦）
- Sharpe 同时考虑了回报和波动，避免选出"坐过山车"的策略
- 业界标准：Sharpe > 1 算不错，> 2 算优秀，> 3 算顶级

#### 代码逐行解读

```python
def _rolling_sharpe(equity: np.ndarray, window: int = 30,
                    bars_per_year: int = 252) -> float:
```

**参数说明：**
- `equity`：策略的净值曲线，比如 `[1.0, 1.01, 0.99, 1.02, ...]`，每个元素代表一根K线结束时策略的总资产
- `window`：回看窗口，默认 30 根K线（对于日线就是 30 天）
- `bars_per_year`：一年有多少根K线（日线=252，4小时线=1512，1小时线=6048）

```python
    if len(equity) < window + 1:
        return 0.0
```
如果数据不够 30 根K线，没法算，直接返回 0。

```python
    tail = equity[-(window + 1):]
```
取最后 31 个净值点。为什么是 31 不是 30？因为 31 个点才能算出 30 个收益率。

```python
    rets = np.diff(tail) / np.maximum(tail[:-1], 1e-12)
```
**这是核心步骤：计算每根K线的收益率。**

`np.diff(tail)` 算相邻两点的差值：`[1.0, 1.01, 0.99]` → `[0.01, -0.02]`

然后除以前一个点的净值，得到百分比收益率：`[0.01/1.0, -0.02/1.01]` = `[1%, -1.98%]`

`np.maximum(tail[:-1], 1e-12)` 是防止除以零（如果净值跌到 0，除法会报错）。

```python
    mu = np.mean(rets)     # 平均每根K线的收益率
    sig = np.std(rets)      # 收益率的标准差（波动率）
```

```python
    return float(mu / sig * math.sqrt(bars_per_year)) if sig > 1e-12 else 0.0
```

**Sharpe 公式：**

$$\text{Sharpe} = \frac{\mu}{\sigma} \times \sqrt{N}$$

其中：
- $\mu$ = 平均每根K线收益率（比如每天赚 0.05%）
- $\sigma$ = 收益率标准差（比如每天波动 0.5%）
- $N$ = 一年有多少根K线

乘以 $\sqrt{N}$ 是**年化**操作。原理：如果每天的收益率是独立的，那年化回报 = 日回报 × 252，年化波动 = 日波动 × $\sqrt{252}$，所以年化 Sharpe = 日 Sharpe × $\sqrt{252}$。

**具体数字例子：** 如果日均收益 0.05%，日波动 0.5%，则：
- 日 Sharpe = 0.05 / 0.5 = 0.1
- 年化 Sharpe = 0.1 × √252 = 0.1 × 15.87 = **1.59**

---

### 1.2 Drawdown（回撤）

#### 这个指标是什么？

回撤 = **"从最高点跌了多少"**。

想象你股票账户最高到过 100 万，现在是 85 万，那你的回撤就是 15%。

回撤持续时间 = **"从最高点到现在已经过了多久"**。如果你账户 3 月 1 日到达 100 万高点，现在是 4 月 1 日还没回到 100 万，那回撤持续时间就是 30 天。

#### 为什么要用它？

- Sharpe 告诉你整体表现，但回撤告诉你**最痛苦的时刻**有多痛苦
- 交易者心理极限：通常 20% 以上的回撤会让人忍不住关掉策略
- 回撤持续时间也重要：跌 10% 但一周恢复 vs 跌 10% 但三个月不恢复，心理压力完全不同

#### 代码逐行解读

```python
def _drawdown_info(equity: np.ndarray) -> Tuple[float, int]:
```
返回两个值：(当前回撤百分比, 回撤持续了多少根K线)。

```python
    peak = np.maximum.accumulate(equity)
```
**关键操作：计算"历史最高值"曲线。**

`np.maximum.accumulate` 的意思是：到每个时间点为止，净值最高是多少。

```
equity = [1.0, 1.05, 1.03, 1.08, 1.02, 1.10]
peak   = [1.0, 1.05, 1.05, 1.08, 1.08, 1.10]
```

注意 peak 只会上升或持平，永远不会下降。

```python
    dd_series = (peak - equity) / np.maximum(peak, 1e-12)
```

每个时间点的回撤 = (历史最高 - 当前净值) / 历史最高

```
peak   = [1.0, 1.05, 1.05, 1.08, 1.08, 1.10]
equity = [1.0, 1.05, 1.03, 1.08, 1.02, 1.10]
dd     = [0%,  0%,   1.9%, 0%,   5.6%, 0%  ]
```

```python
    dd_now = float(dd_series[-1])
```
取最后一个时间点的回撤值。

```python
    for i in range(len(equity) - 1, -1, -1):
        if equity[i] >= peak[i] * (1.0 - 1e-9):
            last_peak_bar = i
            break
    dd_duration = len(equity) - 1 - last_peak_bar
```
从后往前找最近一次净值等于历史最高的位置，与当前位置的距离就是回撤持续时间。

---

### 1.3 Win Rate EMA（指数加权胜率）

#### 这个指标是什么？

胜率 = 赚钱的交易占所有交易的比例。但这里用了 **EMA（指数移动平均）** 方法，让**近期的胜负**比**远期的胜负**权重更大。

#### 为什么要用它？

- 普通胜率把所有交易一视同仁。但如果一个策略"以前经常赢，最近开始经常输"，普通胜率还是显示很高，而 EMA 胜率会迅速下降
- 及时发现策略"失效"的信号

#### 代码逐行解读

```python
def _win_rate_ema(equity: np.ndarray, span: int = 20) -> float:
```

```python
    alpha = 2.0 / (len(rets) + 1)
```
EMA 的衰减系数。span=20 时 alpha ≈ 0.095，意味着最新的一次胜负占约 10% 的权重。

```python
    wr = 0.5  # 初始值：假设胜率 50%
    for r in rets:
        win = 1.0 if r > 0 else 0.0   # 这根K线赚钱了吗？
        wr = alpha * win + (1 - alpha) * wr  # EMA 更新
```

**EMA 更新公式：**

$$WR_{new} = \alpha \times \text{本次结果} + (1 - \alpha) \times WR_{old}$$

假设 alpha=0.1，当前 WR=0.6，然后连输 3 次：
- 输1: WR = 0.1×0 + 0.9×0.6 = 0.54
- 输2: WR = 0.1×0 + 0.9×0.54 = 0.486
- 输3: WR = 0.1×0 + 0.9×0.486 = 0.437

可以看到 EMA 胜率迅速下降，反映了"策略最近在输钱"。

---

### 1.4 Trade Frequency（交易频率）

#### 这个指标是什么？

年化交易次数 = 策略在过去一段时间内的交易次数，折算成"一年会交易多少次"。

#### 为什么要用它？

- 如果一个策略突然不交易了（频率骤降），可能是市场条件变了
- 如果交易频率突然暴增，可能是策略在频繁止损、来回被打脸
- 用于健康状态判断：频率偏离历史均值 2 个标准差就触发警告

#### 代码解读

```python
trade_freq = nt / max(len(c), 1) * bars_per_year
```

公式：`年化频率 = 交易次数 / K线数量 × 一年K线数`

例：日线 90 根K线里交易了 5 次 → 5/90 × 252 = **14 次/年**

---

### 1.5 Health Status（健康状态判定）

#### 三种状态是什么意思？

| 状态 | 含义 | 类比 |
|------|------|------|
| **HEALTHY** | 策略运行正常 | 体检报告全绿 |
| **WATCH** | 出现一些异常信号 | 某些指标黄灯 |
| **ALERT** | 策略可能已失效 | 红灯，需要立即处理 |

#### 判定逻辑

```python
def assess_status(current, history, original_sharpe):
```

三个触发条件：
1. **declining**：最近 3 次检查，Sharpe 连续下降
2. **dd_exceeds**：当前回撤超过历史最大回撤的 80%
3. **freq_deviation**：交易频率偏离历史均值超过 2 个标准差

判定规则：
- Sharpe ≤ 0 **或者** 3 个触发条件中有 2 个 → **ALERT**
- Sharpe 不到原始值的 80% **或者** 有 1 个触发条件 → **WATCH**
- 其余 → **HEALTHY**

---

### 1.6 Regime Detection（市场状态识别）

#### 四种市场状态是什么？

| 状态 | 英文 | 特征 | 适合的策略 |
|------|------|------|-----------|
| **趋势** | trending | 价格持续一个方向移动 | 趋势跟踪（MA、Turtle） |
| **均值回归** | mean_reverting | 价格在一个范围内来回震荡 | 均值回归（RSI、Bollinger） |
| **高波动** | high_vol | 价格剧烈波动 | 降低仓位，加宽止损 |
| **压缩** | compression | 价格波动很小，在蓄力 | 等待突破 |

#### 为什么要识别市场状态？

一个策略不可能在所有市场环境下都赚钱。趋势策略在震荡市场亏钱，均值回归策略在趋势市场亏钱。知道当前市场处于什么状态，才能选择正确的策略。

#### 三个技术指标

代码用三个技术指标来判断市场状态：

**1. ADX（平均方向指数）—— 衡量趋势强度**

```python
def _adx(h, l, c, period=14):
    plus_dm = np.maximum(np.diff(h[-period - 1:]), 0)   # 向上运动
    minus_dm = np.maximum(-np.diff(l[-period - 1:]), 0)  # 向下运动
    tr_arr = ...  # True Range
    plus_di = np.sum(plus_dm) / tr_sum    # 上升方向指标
    minus_di = np.sum(minus_dm) / tr_sum  # 下降方向指标
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
```

直觉：
- ADX > 25：有趋势（不管涨还是跌）
- ADX < 20：没趋势（在震荡）

`plus_dm` 看最高价有没有创新高（上升运动），`minus_dm` 看最低价有没有创新低（下降运动）。如果上升运动和下降运动差距很大，就说明有趋势。

**2. ATR Ratio（波动比率）—— 衡量波动变化**

```python
recent_vol = np.std(rets[-20:])   # 最近 20 根K线的波动率
full_vol = np.std(rets)            # 全部 60 根K线的波动率
atr_ratio = recent_vol / full_vol
```

- ATR Ratio > 1.3：最近波动明显加大 → 高波动状态
- ATR Ratio < 0.7：最近波动明显减小 → 压缩状态

**3. Bollinger Bandwidth（布林带宽度）—— 衡量价格压缩程度**

```python
sma = np.mean(tail[-20:])
std_20 = np.std(tail[-20:])
bb_width = std_20 / sma
```

布林带宽度 = 20 日标准差 / 20 日均价。宽度很小说明价格在很窄的范围内波动（压缩状态）。

#### Sigmoid 评分与归一化

```python
trend_raw = _sigmoid(adx_val, center=25, scale=10)
mr_raw = _sigmoid(-adx_val, center=-20, scale=10)
vol_raw = _sigmoid(atr_ratio, center=1.3, scale=0.3)
comp_raw = _sigmoid(-bb_width, center=-0.02, scale=0.01)
```

Sigmoid 函数将任意数值压缩到 (0, 1) 范围：

$$\sigma(x) = \frac{1}{1 + e^{-(x - \text{center}) / \text{scale}}}$$

- 当 x 远大于 center 时，输出接近 1
- 当 x 远小于 center 时，输出接近 0
- 当 x = center 时，输出恰好 0.5

最后归一化让四个概率加起来等于 1：

```python
total = trend_raw + mr_raw + vol_raw + comp_raw
return {
    "trending": trend_raw / total,
    "mean_reverting": mr_raw / total,
    "high_vol": vol_raw / total,
    "compression": comp_raw / total,
}
```

**数字例子：** 假设 ADX=35（强趋势），ATR_ratio=1.0（波动正常），BB_width=0.03（一般）

- trend_raw = sigmoid((35-25)/10) = sigmoid(1.0) = 0.73
- mr_raw = sigmoid((-35-(-20))/10) = sigmoid(-1.5) = 0.18
- vol_raw = sigmoid((1.0-1.3)/0.3) = sigmoid(-1.0) = 0.27
- comp_raw = sigmoid((-0.03-(-0.02))/0.01) = sigmoid(-1.0) = 0.27

归一化后：趋势=50%, 均值回归=12%, 高波动=19%, 压缩=19% → **趋势市场**

---

### 1.7 Performance Attribution（表现归因分析）

#### 这个指标是什么？

当策略表现不好时，归因分析告诉你：**是谁的错？**

把策略的收益分解成三个因素：
1. **市场因素**：整个市场涨/跌了多少（大盘好坏）
2. **策略因素**：策略本身相对市场是超额表现还是落后
3. **参数因素**：如果换更好的参数，还能提升多少

#### 为什么需要它？

假设你的 BTC 趋势策略这个月亏了 10%：
- 如果 BTC 本身跌了 12%，而策略只亏 10%，说明**策略其实表现不错**，只是市场不行
- 如果 BTC 涨了 5%，而策略亏了 10%，说明**策略出了问题**
- 如果同一策略换一组参数能赚 3%，说明**参数需要更新**

#### 代码核心逻辑

```python
# Beta = 策略与市场的联动系数
beta = _estimate_beta(equity, close)
market_factor = beta * market_return

# 策略因素 = 策略实际收益 - 市场给的收益
strategy_factor = strategy_return - market_factor

# 参数因素 = 最优参数的收益 - 当前参数的收益
param_factor = best_possible_return - strategy_return
```

Beta 的计算用的是经典的协方差/方差公式：

$$\beta = \frac{Cov(R_{strategy}, R_{market})}{Var(R_{market})}$$

如果 beta=1.5，说明市场涨 1% 策略大约涨 1.5%。

---

## 第二章：Optimizer Engine（优化引擎）

Optimizer 引擎是"参数进化器"——不是简单暴力搜索最佳参数，而是像人类学习一样，结合历史经验逐步改进。

### 2.1 Bayesian Parameter Update（贝叶斯参数融合）

#### 这个指标是什么？

每次重新扫描找到"新的最佳参数"时，不是直接用新参数替换旧参数，而是把新旧参数**加权混合**。

#### 为什么这样做？

直接用新参数有两个风险：
1. **过拟合**：新参数可能只是恰好在最近的数据上表现好（运气成分）
2. **不稳定**：每次参数跳变很大会导致实盘表现不连贯

贝叶斯融合的思想：**"新证据很重要，但历史经验也有价值"**

#### 代码逐行解读

```python
for i, entry in enumerate(param_history):
    age_weight = decay ** i               # 越老的参数，权重越低
    sharpe = max(entry.get("sharpe", 0), 0)
    w = age_weight * (1.0 + sharpe)       # 表现好的参数，权重更高
    weighted_sum += np.array(p) * w
    weight_total += w
```

**衰减权重** `decay ** i`：`decay=0.9` 时：
- 最近一次参数：权重 0.9⁰ = 1.0
- 上一次：0.9¹ = 0.9
- 上上次：0.9² = 0.81
- 5 次前：0.9⁵ = 0.59

**Sharpe 加权** `(1 + sharpe)`：Sharpe=2 的参数比 Sharpe=0 的参数权重高 3 倍。

```python
new_w = 1.0 + new_scan_sharpe
weighted_sum += np.array(new_scan_params) * new_w
blended = weighted_sum / weight_total
```

最终公式（简化）：

$$\text{blended}_j = \frac{\sum_{i} w_i \times p_{i,j} + w_{new} \times p_{new,j}}{\sum_{i} w_i + w_{new}}$$

**数字例子：** 参数是 RSI 的 [period, oversold, overbought]

- 历史1 (sharpe=1.5): [14, 30, 70]，权重 = 1.0 × 2.5 = 2.5
- 历史2 (sharpe=0.8): [20, 25, 75]，权重 = 0.9 × 1.8 = 1.62
- 新扫描 (sharpe=2.0): [16, 35, 65]，权重 = 3.0

period 的融合结果 = (2.5×14 + 1.62×20 + 3.0×16) / (2.5+1.62+3.0)
= (35 + 32.4 + 48) / 7.12 = 16.2 → 四舍五入到 **16**

---

### 2.2 Composite Gate Score（复合门控评分）

#### 这个指标是什么？

Gate Score 是一个 0~1 之间的综合评分，回答一个问题：**"这个策略的参数组合，到底有多靠谱？"**

0 = 完全不可靠，1 = 极度可靠。只有 Gate Score > 0.6 的参数才有资格成为"挑战者"。

#### 六个维度与权重

| 维度 | 权重 | 衡量什么 | 为什么需要 |
|------|------|----------|-----------|
| **统计显著性** | 25% | Sharpe 高不高 + DSR 检验通不通过 | 防止 Sharpe 是凑巧算出来的 |
| **Walk-Forward** | 20% | 样本外表现是否一致 | 防止策略只在训练数据上好 |
| **邻域稳定性** | 20% | 参数微调后表现是否崩溃 | 防止参数过拟合到一个"尖峰" |
| **Monte Carlo** | 15% | 随机打乱后还能赚钱吗 | 防止策略靠运气 |
| **Deflated Sharpe** | 10% | 扣除多重测试偏差后的 Sharpe | 防止"测试了 1000 个策略总有一个好的"偏差 |
| **尾部风险** | 10% | 最大回撤和极端损失 | 防止"平时赚小钱，一次亏光" |

#### 每个子评分的计算

**统计显著性：**

```python
sharpe_score = min(sharpe / 3.0, 1.0)  # Sharpe=3 得满分
dsr_score = 1.0 - min(dsr_p, 1.0)      # DSR p值越小越好
return 0.6 * sharpe_score + 0.4 * dsr_score
```

DSR (Deflated Sharpe Ratio) 的 p 值：如果 p < 0.05，说明 Sharpe 在统计上显著（不是运气）。

**Walk-Forward：**

```python
wf_norm = min(wf_score / 100, 1.0)      # WF得分满分100
oos_score = min(oos_ret / 0.5, 1.0)      # 样本外收益50%得满分
return 0.7 * wf_norm + 0.3 * oos_score
```

Walk-Forward = 用前 70% 数据训练参数，在后 30% 数据上测试。如果在"没见过的数据"上也表现好，说明策略是真的有效。

**Monte Carlo：**

```python
return min(mc_pct_positive / 100, 1.0)  # 90% 路径盈利 → 0.9 分
```

Monte Carlo 模拟：随机打乱交易顺序/添加噪声，跑 1000 次。如果 90% 的情况下还是赚钱，说明策略很稳健。

**尾部风险：**

```python
dd_score = max(0, 1.0 - abs(max_dd) / 0.5)  # DD=0%→1分, DD=50%→0分
cvar_score = max(0, 1.0 - abs(cvar_95) / 0.1)
return 0.6 * dd_score + 0.4 * cvar_score
```

CVaR (Conditional Value at Risk) = 在最差的 5% 情况下，平均亏损多少。比 VaR 更保守。

---

### 2.3 Parameter Neighborhood Stability（参数邻域稳定性）

#### 这个指标是什么？

把参数微调 ±10%，看 Sharpe 是否暴跌。如果微调后 Sharpe 基本不变，说明参数是稳定的；如果微调一点就崩溃，说明这个参数是"过拟合的尖峰"。

#### 为什么需要它？

想象你在山上找最高点：
- 找到一座"宽厚的山顶"（plateau）：往任何方向走几步还是很高 → **稳定**
- 找到一个"针尖"（spike）：稍微偏一点就悬崖 → **不稳定（过拟合）**

#### 代码核心逻辑

```python
for i, p in enumerate(params):
    for direction in [1 + 0.10, 1 - 0.10]:  # +10% 和 -10%
        test_params[i] = p * direction
        neighbor_sharpe = _sharpe(test_params)
        drop = (base_sharpe - neighbor_sharpe) / base_sharpe
        if drop < 0.30:  # Sharpe 下降不超过 30%
            stable_count += 1

stability = stable_count / total_tests  # 稳定比例，0~1
```

例如 RSI 策略参数 [14, 30, 70]，会测试 6 个邻居：
- [15, 30, 70], [13, 30, 70]（period ±10%）
- [14, 33, 70], [14, 27, 70]（oversold ±10%）
- [14, 30, 77], [14, 30, 63]（overbought ±10%）

如果 6 个邻居中 5 个的 Sharpe 下降不超过 30% → 稳定性 = 5/6 = 0.83

---

### 2.4 Champion/Challenger Protocol（冠军/挑战者协议）

#### 这是什么？

类似于公司选拔人才的流程：
1. **Champion（冠军）**：当前正在实盘运行的参数
2. **Challenger（挑战者）**：新找到的、Gate Score 更高的参数
3. **纸上交易**：挑战者先用模拟盘跑一段时间
4. **晋升**：如果挑战者的 Gate Score 比冠军高 5% 以上 → 替换冠军上实盘

#### 为什么需要这个流程？

防止**频繁切换参数**。即使新参数在回测中更好，也可能是过拟合。通过模拟阶段验证后再上线，大大降低风险。

---

## 第三章：Portfolio Engine（投资组合引擎）

Portfolio 引擎回答一个问题：**同时运行多个策略时，每个策略应该分配多少资金？**

### 3.1 Correlation Matrix（相关性矩阵）

#### 这个指标是什么？

相关性衡量两个策略的收益是否**同步涨跌**。

| 相关系数 | 含义 | 例子 |
|----------|------|------|
| +1.0 | 完全同步 | BTC 趋势策略 A 和 BTC 趋势策略 B |
| 0.0 | 完全无关 | BTC 趋势策略 和 黄金均值回归策略 |
| -1.0 | 完全相反 | 做多策略 和 做空策略 |

#### 为什么需要它？

如果你运行 5 个策略但它们全部高度正相关（相关性 > 0.7），那本质上只是在"下同一个注"——市场跌的时候 5 个策略一起亏。

分散化的核心就是选择**低相关性**的策略组合。

#### 代码核心

```python
corr = np.corrcoef(matrix)  # matrix 每行是一个策略的收益序列
```

Pearson 相关系数公式：

$$\rho_{A,B} = \frac{Cov(R_A, R_B)}{\sigma_A \times \sigma_B}$$

---

### 3.2 Weight Optimization（权重优化）

#### 三步走策略

**第一步：反波动率权重（Risk Parity 风险平价）**

```python
weights = {k: 1.0 / vol_map[k] for k in returns_dict}
```

波动率低的策略分配更多资金。直觉：稳定的策略值得更多信任。

例如：策略 A 年化波动 10%，策略 B 年化波动 30%
- A 的权重 = 1/0.10 = 10
- B 的权重 = 1/0.30 = 3.33
- 归一化后：A = 75%, B = 25%

**第二步：Sharpe 倾斜**

```python
weights[k] *= max(sharpe_map.get(k, 0), 0.01)
```

在风险平价的基础上，给 Sharpe 更高的策略更多权重。

**第三步：相关性惩罚**

```python
if abs(correlation) > 0.7:
    weights[worse_strategy] *= 0.5  # 减半
```

如果两个策略高度相关，降低表现较差那个的权重，避免重复下注。

---

### 3.3 Portfolio Metrics（组合层级指标）

#### Sortino Ratio（索提诺比率）

```python
downside = port_rets[port_rets < 0]  # 只看亏损的日子
down_std = np.std(downside)
port_sortino = mu / down_std * sqrt(252)
```

和 Sharpe 的区别：Sharpe 把"上涨波动"也当成风险，但投资者其实**不介意向上的波动**。Sortino 只用**下行波动**做分母，更能反映真实风险。

#### Diversification Ratio（分散化比率）

```python
div_ratio = weighted_avg_individual_vol / portfolio_vol
```

- 如果各策略完全不相关：组合波动 < 个体波动之和 → 比率 > 1 → 分散化效果好
- 如果各策略完全相关：组合波动 = 个体波动之和 → 比率 ≈ 1 → 没有分散化效果

**比率越高越好**，说明组合中的策略真正在互相"对冲"。

#### Marginal Contribution to Risk（边际风险贡献）

```python
# 移除策略 k，看组合波动变化了多少
sub_vol = 不含策略k的组合波动
marginal[k] = portfolio_vol - sub_vol
```

- 正值：这个策略**增加**了组合风险
- 负值：这个策略**降低**了组合风险（它是好的分散化工具）

---

## 第四章：Discover Engine（发现引擎）

Discover 引擎是"市场侦探"——扫描市场数据，发现统计特征的**结构性变化**。

### 4.1 Autocorrelation Shift（自相关结构变化）

#### 这个指标是什么？

自相关 = "今天涨了，明天更可能涨还是跌？"

```python
def _autocorrelation(rets, lag=1):
    r1 = rets[:-1]   # 今天的收益
    r2 = rets[1:]     # 明天的收益
    return mean((r1 - mu) * (r2 - mu)) / var(rets)
```

- AC > 0：**动量效应**（涨了继续涨）→ 趋势策略有效
- AC < 0：**均值回归**（涨了会跌回来）→ 均值回归策略有效
- AC ≈ 0：随机游走 → 两种策略都不太好使

#### 为什么检测变化？

如果一个资产从 AC > 0 变成 AC < 0，意味着以前有效的趋势策略现在可能开始亏钱。检测这种变化可以**提前调整策略类型**。

```python
ac_full = _autocorrelation(全年数据)
ac_recent = _autocorrelation(最近60天数据)
if abs(ac_recent - ac_full) > 0.15:
    flag("自相关结构发生了变化！")
```

---

### 4.2 Volatility Regime Change（波动率状态变化）

```python
recent_vol = np.std(最近20天收益) * sqrt(252)  # 年化
hist_vol = np.std(过去不含最近20天的收益) * sqrt(252)
vol_ratio = recent_vol / hist_vol
```

- vol_ratio > 1.5：**波动剧增**（比如突发事件），应该减仓
- vol_ratio < 0.6：**波动骤降**（比如长期横盘），均值回归策略可能更好

---

### 4.3 Skewness Shift（偏度变化）

#### 偏度是什么？

偏度描述收益分布的**不对称性**：

```python
skew = mean(((rets - mu) / sigma) ** 3)
```

- 偏度 > 0：**正偏**（偶尔有大涨，但小跌更常见）→ "彩票型"
- 偏度 < 0：**负偏**（偶尔有大跌，但小涨更常见）→ "卖保险型"
- 偏度 ≈ 0：对称分布

#### 为什么检测变化？

偏度从正变负意味着**尾部风险增加**了——虽然平均收益可能没变，但出现极端亏损的概率增大了。需要检查止损是否够宽。

---

### 4.4 Cross-Asset Correlation（跨资产相关性漂移）

```python
for 每一对资产 (A, B):
    full_corr = corrcoef(A的全年收益, B的全年收益)
    recent_corr = corrcoef(A的最近60天收益, B的最近60天收益)
    if abs(recent_corr - full_corr) > 0.3:
        flag("相关性发生了变化！")
```

实际案例：
- META/MSFT 全年相关 +0.49，最近 60 天相关 -0.12 → **"解耦"**
- 这意味着以前 META 涨 MSFT 也涨，但最近不是了
- 如果你同时做多两者来"分散风险"，现在反而达到了更好的分散效果

---

## 附录：年化因子与跨时间框架处理

### 为什么需要年化？

不同时间频率的原始指标不可比：
- 日线策略每天赚 0.05% → 看起来很少
- 1 小时线策略每小时赚 0.002% → 看起来更少

但年化后：
- 日线：0.05% × 252 = 12.6% 年回报
- 1 小时：0.002% × 6048 = 12.1% 年回报

原来差不多！

### 年化因子对照表

| 时间框架 | 每年K线数 (bars_per_year) | 每天K线数 (bars_per_day) | √(bars_per_year) |
|----------|-------------------------|------------------------|-------------------|
| 1d (日线) | 252 | 1 | 15.87 |
| 4h (4小时) | 1,512 | 6 | 38.88 |
| 1h (1小时) | 6,048 | 24 | 77.77 |

### 对 Sharpe Window 的影响

为了让所有时间框架的 Sharpe 都基于约 30 个**日历日**的数据：

```python
scaled_window = sharpe_window * bars_per_day
```

| 时间框架 | sharpe_window (原始) | bars_per_day | scaled_window (实际) |
|----------|---------------------|--------------|---------------------|
| 1d | 30 | 1 | 30 根K线 |
| 4h | 30 | 6 | 180 根K线 |
| 1h | 30 | 24 | 720 根K线 |

这样日线和小时线的 Sharpe 都是基于"最近 30 天的表现"来计算的，是可以直接比较的。

---

> **文档版本：** v1.0  
> **对应代码版本：** V4 Intelligent Strategy Research System  
> **最后更新：** 2026-03-04
