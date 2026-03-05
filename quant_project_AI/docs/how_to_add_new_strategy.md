# 如何添加新策略 — 完整指南

本框架支持两种策略模式。根据你的需求选择：

| 模式 | 速度 | 适用场景 | 需要改的文件数 |
|------|------|----------|--------------|
| **Kernel 模式** | 极快（Numba JIT） | 纯价格信号策略，支持批量参数扫描 | 2 个 |
| **Python 模式** | 较慢 | 复杂逻辑、外部数据、ML 模型 | 1 个 |

---

## 方式一：Kernel 模式（推荐）

这是框架的核心路径，所有内置策略（MA、RSI、Bollinger 等）都用这种方式。
优点：Numba 编译为机器码，速度是 Python 的 100-1000 倍，且能自动进行参数网格搜索。

### 总共需要改 2 个文件：

```
quant_framework/
├── strategy/
│   ├── __init__.py              ← 第6步：注册导出
│   └── my_new_strategy.py       ← 第5步：写 Python 策略类（可选）
└── backtest/
    └── kernels.py               ← 第1-4步：写 Numba kernel
```

---

### 第 1 步：写 Kernel 函数

在 `kernels.py` 中找到最后一个 `bt_xxx_ls` 函数的后面，添加你的 kernel。

以一个简单的 **双均线 + RSI 过滤** 策略为例：

```python
@njit(cache=True, fastmath=_SAFE_FASTMATH)
def bt_marsi_ls(c, o, ma_s, ma_l, rsi, rsi_thresh,
                sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    """MA crossover filtered by RSI.

    做多条件: 短均线上穿长均线 AND RSI < rsi_thresh (超卖确认)
    做空条件: 短均线下穿长均线 AND RSI > (100 - rsi_thresh)
    """
    n = len(c)
    pos = 0       # 当前仓位: 0=空仓, 1=做多, -1=做空
    ep = 0.0      # 入场价格
    tr = 1.0      # 总资产 (从 1.0 开始, 即 100%)
    pend = 0      # 挂单: 0=无, 1=买, -1=卖, 2=平仓, 3=反手买, -3=反手卖
    pk = 1.0      # 历史最高资产 (用于算回撤)
    mdd = 0.0     # 最大回撤 (%)
    nt = 0        # 交易次数

    for i in range(1, n):
        # ---- 1. 执行上一根K线产生的挂单 ----
        pos, ep, tr, tc, liq = _fx_lev(
            pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac
        )
        nt += tc
        pend = 0

        # 如果被清算 (杠杆爆仓), 重置仓位
        if liq:
            pos = 0
            ep = 0.0
            continue

        # ---- 2. 检查止损 ----
        pos, ep, tr, tc2 = _sl_exit(
            pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip
        )
        nt += tc2

        # ---- 3. 你的策略信号逻辑 ----
        s0 = ma_s[i-1]; l0 = ma_l[i-1]  # 前一根K线的均线值
        s1 = ma_s[i];   l1 = ma_l[i]    # 当前K线的均线值
        r = rsi[i]                        # 当前 RSI

        # 检查 NaN (Numba 中 NaN != NaN)
        if s0 != s0 or l0 != l0 or s1 != s1 or l1 != l1 or r != r:
            pass
        # 短均线上穿长均线 + RSI 低于阈值 → 做多
        elif s0 <= l0 and s1 > l1 and r < rsi_thresh:
            if pos == 0:
                pend = 1     # 开多
            elif pos == -1:
                pend = 3     # 空翻多 (先平空再开多)
        # 短均线下穿长均线 + RSI 高于阈值 → 做空
        elif s0 >= l0 and s1 < l1 and r > (100.0 - rsi_thresh):
            if pos == 0:
                pend = -1    # 开空
            elif pos == 1:
                pend = -3    # 多翻空

        # ---- 4. 记录净值和回撤 ----
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        if eq > pk:
            pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd:
            mdd = dd_

    # ---- 5. 收盘时平掉剩余仓位 ----
    if pos == 1 and ep > 0:
        raw = (c[n-1] * ss * (1-cm)) / (ep * (1+cm))
        dep = _deploy(tr, pfrac)
        tr += dep * ((raw - 1.0) * lev)
        tr = max(0.01, tr)
        nt += 1
    elif pos == -1 and ep > 0:
        raw = (ep * (1-cm)) / (c[n-1] * sb * (1+cm))
        dep = _deploy(tr, pfrac)
        tr += dep * ((raw - 1.0) * lev)
        tr = max(0.01, tr)
        nt += 1

    return (tr - 1.0) * 100.0, mdd, nt
```

**关键变量解释：**

| 变量 | 含义 |
|------|------|
| `pend` | 挂单指令。`1`=下根K线开盘买入，`-1`=卖出，`2`=平仓，`3`=反手（先平再反向开仓），`-3`=反向反手 |
| `_fx_lev()` | 执行挂单，处理滑点和手续费，返回新仓位、入场价、资产、成交数、是否被清算 |
| `_sl_exit()` | 检查止损，如果触发则平仓 |
| `_mtm_lev()` | 按当前价格计算账户净值（mark-to-market） |
| `_deploy()` | 计算实际投入的资金比例（pfrac 控制） |
| `sb, ss` | 买入/卖出滑点乘数（如 1.001 表示 0.1% 滑点） |
| `cm` | 手续费率 |
| `lev` | 杠杆倍数 |
| `dc` | 资金费率（crypto 永续合约） |
| `sl` | 止损百分比 |

---

### 第 2 步：写 Equity 版本（用于 Monitor 引擎）

Equity 版本和上面几乎一模一样，多返回两个数组：`eq_arr`（净值曲线）和 `pos_arr`（仓位曲线）。

```python
@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _eq_marsi(c, o, ma_s, ma_l, rsi, rsi_thresh,
              sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n = len(c)
    pos = 0; ep = 0.0; tr = 1.0; pend = 0; pk = 1.0; mdd = 0.0; nt = 0
    eq_arr = np.ones(n, dtype=np.float64)      # ← 新增
    pos_arr = np.zeros(n, dtype=np.int64)       # ← 新增

    for i in range(1, n):
        pos, ep, tr, tc, liq = _fx_lev(pend, pos, ep, o[i], tr, sb, ss, cm, lev, dc, pfrac)
        nt += tc; pend = 0
        if liq:
            pos = 0; ep = 0.0
            eq_arr[i] = tr; pos_arr[i] = 0     # ← 新增
            continue
        pos, ep, tr, tc2 = _sl_exit(pos, ep, tr, c[i], sb, ss, cm, lev, sl, pfrac, sl_slip)
        nt += tc2

        # ---- 你的信号逻辑（同上） ----
        s0 = ma_s[i-1]; l0 = ma_l[i-1]
        s1 = ma_s[i];   l1 = ma_l[i]
        r = rsi[i]
        if s0 != s0 or l0 != l0 or s1 != s1 or l1 != l1 or r != r:
            pass
        elif s0 <= l0 and s1 > l1 and r < rsi_thresh:
            if pos == 0: pend = 1
            elif pos == -1: pend = 3
        elif s0 >= l0 and s1 < l1 and r > (100.0 - rsi_thresh):
            if pos == 0: pend = -1
            elif pos == 1: pend = -3

        pos_arr[i] = pos                        # ← 新增
        eq = _mtm_lev(pos, tr, c[i], ep, sb, ss, cm, lev, sl, pfrac)
        eq_arr[i] = eq                           # ← 新增
        if eq > pk: pk = eq
        dd_ = (pk - eq) / pk * 100.0 if pk > 0 else 0.0
        if dd_ > mdd: mdd = dd_

    fpos = pos                                   # ← 新增
    tr, tc = _close_pos(pos, ep, tr, c[n-1], sb, ss, cm, lev, pfrac)
    nt += tc
    eq_arr[n-1] = tr

    return (tr - 1.0) * 100.0, mdd, nt, eq_arr, fpos, pos_arr
```

---

### 第 3 步：注册到 KERNEL_REGISTRY 和 DEFAULT_PARAM_GRIDS

在 `kernels.py` 中找到 `KERNEL_REGISTRY` 字典，添加一行：

```python
KERNEL_REGISTRY: Dict[str, Callable] = {
    "MA": bt_ma_ls,
    "RSI": bt_rsi_ls,
    # ... 其他策略 ...
    "MARSI": bt_marsi_ls,          # ← 添加这行
}
```

然后在 `DEFAULT_PARAM_GRIDS` 中定义参数搜索空间：

```python
DEFAULT_PARAM_GRIDS: Dict[str, List[tuple]] = {
    # ... 其他策略 ...
    "MARSI": [
        (s, lg, rsi_thresh)
        for s in range(5, 50, 5)           # 短均线周期: 5, 10, 15, ..., 45
        for lg in range(s + 10, 200, 10)   # 长均线周期: 比短均线大10+
        for rsi_thresh in range(25, 45, 5) # RSI 阈值: 25, 30, 35, 40
    ],
}
```

---

### 第 4 步：添加 eval 分发和 scan 函数

**在 `eval_kernel_detailed` 函数中添加分发：**

```python
def eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    p = params
    # ... 其他 if 分支 ...
    if name == "MARSI":
        return _eq_marsi(
            c, o,
            _rolling_mean(c, int(p[0])),    # 短均线
            _rolling_mean(c, int(p[1])),    # 长均线
            _rsi_wilder(c, 14),             # RSI (固定14周期)
            float(p[2]),                    # RSI 阈值
            sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
        )
```

**在 `eval_kernel` 函数中也添加对应分发：**

```python
def eval_kernel(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl=0.80, pfrac=1.0, sl_slip=0.0):
    p = params
    # ... 其他 if 分支 ...
    if name == "MARSI":
        return bt_marsi_ls(
            c, o,
            _rolling_mean(c, int(p[0])),
            _rolling_mean(c, int(p[1])),
            _rsi_wilder(c, 14),
            float(p[2]),
            sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
        )
```

**写 scan 函数（用于参数网格搜索）：**

```python
@njit(cache=True, fastmath=_SAFE_FASTMATH, parallel=True)
def _scan_marsi_njit(grid, c, o, mas, rsis, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    ng = grid.shape[0]
    _sc = np.empty(ng); _r = np.empty(ng)
    _d = np.empty(ng);  _n = np.empty(ng, dtype=np.int64)
    for k in prange(ng):
        r, d, nt = bt_marsi_ls(
            c, o,
            mas[int(grid[k, 0])],      # 预计算的短均线
            mas[int(grid[k, 1])],      # 预计算的长均线
            rsis[14],                   # 预计算的 RSI-14
            grid[k, 2],                # RSI 阈值
            sb, ss, cm, lev, dc, sl, pfrac, sl_slip,
        )
        _sc[k] = _score(r, d, nt)
        _r[k] = r; _d[k] = d; _n[k] = nt
    bi = 0
    for k in range(1, ng):
        if _sc[k] > _sc[bi]: bi = k
    return bi, _sc[bi], _r[bi], _d[bi], _n[bi], ng
```

**在 `scan_all_kernels` 的 `_scan_dispatch` 字典中注册：**

```python
_scan_dispatch = {
    # ... 其他策略 ...
    "MARSI": lambda ga: _scan_marsi_njit(ga, c, o, mas, rsis, *_cost),
}
```

**在 `_INDICATOR_DEPS` 中声明指标依赖：**

```python
_INDICATOR_DEPS: Dict[str, frozenset] = {
    # ... 其他策略 ...
    "MARSI": frozenset({"mas", "rsis"}),
}
```

---

### 第 5 步（可选）：写 Python 策略类

如果你想在**实盘交易**或 **Python 回测引擎**中使用这个策略，需要写一个策略类。

创建 `quant_framework/strategy/marsi_strategy.py`：

```python
"""双均线 + RSI 过滤策略"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
from .base_strategy import BaseStrategy
from ..data.indicators import _rsi_numba


class MARSIStrategy(BaseStrategy):
    """MA Crossover filtered by RSI."""

    DEFAULT_PARAM_GRID = [
        (s, lg, rt)
        for s in range(5, 50, 5)
        for lg in range(s + 10, 200, 10)
        for rt in range(25, 45, 5)
    ]

    def __init__(
        self,
        name: str = "MARSI策略",
        initial_capital: float = 1000000,
        ma_short: int = 10,
        ma_long: int = 50,
        rsi_thresh: float = 30,
    ):
        super().__init__(name, initial_capital)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_thresh = rsi_thresh
        self._min_lookback = max(ma_long, 14) + 1

    @property
    def kernel_name(self) -> str:
        """必须与 KERNEL_REGISTRY 中的 key 一致"""
        return "MARSI"

    @property
    def kernel_params(self) -> tuple:
        """参数顺序必须与 DEFAULT_PARAM_GRIDS 一致"""
        return (self.ma_short, self.ma_long, self.rsi_thresh)

    @property
    def fast_columns(self) -> Tuple[str, ...]:
        return ("close",)

    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, list]:
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "STOCK")

        if len(df) < self._min_lookback:
            return {"action": "hold"}

        close = df["close"]
        ma_s = close.rolling(self.ma_short).mean()
        ma_l = close.rolling(self.ma_long).mean()

        arr = np.ascontiguousarray(close.values, dtype=np.float64)
        rsi_arr = _rsi_numba(arr, 14)
        rsi = float(rsi_arr[-1])

        prev_s, prev_l = float(ma_s.iloc[-2]), float(ma_l.iloc[-2])
        curr_s, curr_l = float(ma_s.iloc[-1]), float(ma_l.iloc[-1])
        price = float(close.iloc[-1])

        if prev_s <= prev_l and curr_s > curr_l and rsi < self.rsi_thresh:
            holdings = self.positions.get(symbol, 0)
            if holdings == 0:
                shares = self.calculate_position_size(price, capital_fraction=0.95)
                if shares > 0 and self.can_buy(symbol, price, shares):
                    return {"action": "buy", "symbol": symbol, "shares": shares}

        elif prev_s >= prev_l and curr_s < curr_l and rsi > (100 - self.rsi_thresh):
            holdings = self.positions.get(symbol, 0)
            if holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        return {"action": "hold"}
```

---

### 第 6 步：注册导出

在 `quant_framework/strategy/__init__.py` 中添加：

```python
from .marsi_strategy import MARSIStrategy

__all__ = [
    # ... 现有策略 ...
    "MARSIStrategy",
]
```

---

### 完成! 验证方式

添加完以上代码后，你的新策略自动支持：

```python
# 1. 单次回测
from quant_framework.strategy import MARSIStrategy
from quant_framework.backtest import BacktestEngine, BacktestConfig

strategy = MARSIStrategy(ma_short=10, ma_long=50, rsi_thresh=30)
engine = BacktestEngine(BacktestConfig.stock_ibkr())
result = engine.run(strategy, ["AAPL"], "2020-01-01", "2024-01-01")

# 2. 参数网格搜索 (自动并行)
from quant_framework.backtest.kernels import scan_all_kernels
results = scan_all_kernels(c, o, h, l, config, strategies=["MARSI"])

# 3. Research 引擎自动监控
# run_monitor, run_optimizer 等会自动识别 "MARSI" kernel
```

---

## 方式二：纯 Python 模式（无 Kernel）

适用于逻辑太复杂无法用 Numba 的场景（比如用到 pandas、sklearn 等）。

只需要创建一个继承 `BaseStrategy` 的类，**不定义** `kernel_name`：

```python
from .base_strategy import BaseStrategy

class MyMLStrategy(BaseStrategy):
    """用 ML 模型做信号的策略"""

    def __init__(self, name="ML策略", initial_capital=1000000):
        super().__init__(name, initial_capital)
        self._min_lookback = 100
        # 可以加载 sklearn 模型等

    # 不定义 kernel_name 和 kernel_params
    # 回测引擎会自动走 Python 事件循环路径

    def on_bar(self, data, current_date, current_prices=None):
        # 这里可以用任何 Python 库
        # pandas, numpy, sklearn, xgboost 等都可以
        df = data if not isinstance(data, dict) else next(iter(data.values()))
        symbol = df.attrs.get("symbol", "STOCK") if not isinstance(data, dict) else next(iter(data))

        # 你的信号逻辑
        prediction = self._predict(df)

        if prediction > 0.6:
            return {"action": "buy", "symbol": symbol,
                    "shares": self.calculate_position_size(float(df["close"].iloc[-1]))}
        elif prediction < 0.4:
            holdings = self.positions.get(symbol, 0)
            if holdings > 0:
                return {"action": "sell", "symbol": symbol, "shares": holdings}

        return {"action": "hold"}
```

**局限性：** 无法使用 `scan_all_kernels` 做参数搜索，无法被 Research 引擎的 Monitor/Optimizer 自动追踪。速度比 Kernel 模式慢 100-1000 倍。

---

## 速查：Kernel 信号指令一览

| `pend` 值 | 含义 | 场景 |
|-----------|------|------|
| `0` | 无操作 | 保持当前状态 |
| `1` | 下根K线开盘做多 | 空仓 → 做多 |
| `-1` | 下根K线开盘做空 | 空仓 → 做空 |
| `2` | 下根K线开盘平仓 | 有仓位 → 平仓 |
| `3` | 反手做多 | 做空 → 平空 → 做多 |
| `-3` | 反手做空 | 做多 → 平多 → 做空 |

## 速查：Kernel 内置工具函数

| 函数 | 作用 |
|------|------|
| `_fx_lev(pend, pos, ep, open_price, tr, ...)` | 执行挂单，处理杠杆/手续费/清算 |
| `_sl_exit(pos, ep, tr, close_price, ...)` | 检查止损条件并执行 |
| `_mtm_lev(pos, tr, close_price, ep, ...)` | 计算当前净值（Mark-to-Market） |
| `_deploy(tr, pfrac)` | 计算实际投入金额（按 pfrac 比例） |
| `_close_pos(pos, ep, tr, close_price, ...)` | 收盘强制平仓 |
| `_rolling_mean(arr, window)` | 预计算移动平均线数组 |
| `_rsi_wilder(arr, period)` | 预计算 RSI 数组（Wilder 平滑法） |
| `_ema(arr, span)` | 预计算 EMA 数组 |
| `_score(ret, dd, nt)` | 参数搜索评分函数（Sharpe - DD 惩罚） |

---

> **提示：** 代码写完后，第一次运行会触发 Numba JIT 编译（约 2-5 秒），之后的运行都是毫秒级。编译结果会被缓存（`cache=True`），重启 Python 后也不需要重新编译。
