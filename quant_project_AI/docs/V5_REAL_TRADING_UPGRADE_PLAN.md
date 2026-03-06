# V5 Real Trading Upgrade Plan

## 总体目标

将框架从「回测 + 纸交易模拟」升级为「回测 ↔ 纸交易 ↔ 实盘」三层一致的生产交易系统，同时支持**加密货币**（Binance 现货/合约）和**美股**（IBKR）。

### 三层架构定义

```
┌──────────────────────────────────────────────────────────────────┐
│                      统一核心抽象层                                │
│  MarginModel · CostModel · SymbolSpec · OrderProtocol           │
│  (一套模型，三层共用，保证数学行为一致)                              │
└────────┬──────────────────┬──────────────────┬──────────────────┘
         │                  │                  │
  ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
  │   回测层     │   │  纸交易层    │   │   实盘层     │
  │ Backtest    │   │  Paper      │   │  Exchange   │
  │ Engine      │   │  Broker     │   │  Broker     │
  │             │   │             │   │             │
  │ 历史数据     │   │ 实时行情     │   │ 实时行情     │
  │ 模拟撮合     │   │ 模型模拟     │   │ 交易所撮合   │
  │ 零延迟       │   │ 模拟延迟     │   │ 真实延迟     │
  └─────────────┘   └─────────────┘   └─────────────┘
```

| 层 | 目的 | 价格来源 | 撮合方式 | 资金 |
|----|------|----------|----------|------|
| 回测 | 历史验证策略表现 | 历史 OHLCV | 内核/引擎模拟 | 虚拟 |
| 纸交易 | 用真实行情验证策略 | 实时行情 | MarginModel 模拟 | 虚拟 |
| 实盘 | 真实交易 | 交易所 API | 交易所撮合引擎 | 真实 |

---

## Phase 0: 统一核心抽象层

**目标**: 建立资产类型感知的保证金、成本、精度模型，三层共用。

### 0.1 资产类型枚举

**新增文件**: `quant_framework/core/asset_types.py`

```python
class AssetClass(Enum):
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"      # USDT-M 永续合约
    CRYPTO_INVERSE = "crypto_inverse" # COIN-M 反向合约
    US_EQUITY = "us_equity"
    US_EQUITY_MARGIN = "us_equity_margin"  # Reg T 融资
```

### 0.2 保证金模型

**新增文件**: `quant_framework/core/margin.py`

三种实现对应真实规则：

| 实现类 | 适用于 | 初始保证金 | 维持保证金 | 强平机制 |
|--------|--------|-----------|-----------|----------|
| `CryptoSpotMargin` | 加密现货 | 100%（无杠杆） | N/A | 不强平 |
| `CryptoFuturesMargin` | 永续合约 | notional / leverage | 阶梯制 (按仓位层级) | equity < 维持 → 交易所强平 |
| `RegTMargin` | 美股融资 | 50% notional | 25% market_value | Margin Call → T+2~5 补仓 |

**`CryptoFuturesMargin` 设计细节**:

```python
class CryptoFuturesMargin(MarginModel):
    # Binance 阶梯保证金 (以 BTC 为例)
    TIERS = [
        # (max_notional, maintenance_rate, maintenance_amount)
        (50_000,     0.004, 0),
        (250_000,    0.005, 50),
        (1_000_000,  0.01,  1_050),
        (10_000_000, 0.025, 16_050),
        (20_000_000, 0.05,  266_050),
        (50_000_000, 0.10,  1_266_050),
        (100_000_000,0.125, 2_516_050),
        (200_000_000,0.15,  5_016_050),
        (float('inf'), 0.25, 25_016_050),
    ]

    def initial_margin(self, notional: float, leverage: float) -> float:
        return notional / leverage

    def maintenance_margin(self, notional: float) -> float:
        for max_n, rate, amount in self.TIERS:
            if notional <= max_n:
                return notional * rate - amount
        return notional * 0.25

    def liquidation_price_long(self, entry: float, leverage: float,
                                maint_rate: float) -> float:
        return entry * (1 - 1/leverage + maint_rate)

    def liquidation_price_short(self, entry: float, leverage: float,
                                 maint_rate: float) -> float:
        return entry * (1 + 1/leverage - maint_rate)
```

**`RegTMargin` 设计细节**:

```python
class RegTMargin(MarginModel):
    INITIAL_RATE = 0.50          # Reg T: 50%
    MAINTENANCE_RATE = 0.25      # NYSE/FINRA minimum
    PDT_EQUITY_THRESHOLD = 25000 # Pattern Day Trader threshold

    def initial_margin(self, notional: float, leverage: float = 2.0) -> float:
        return notional * self.INITIAL_RATE

    def maintenance_margin(self, market_value: float) -> float:
        return market_value * self.MAINTENANCE_RATE

    def check_margin_call(self, equity: float, market_value: float) -> bool:
        return equity < market_value * self.MAINTENANCE_RATE

    def margin_call_deadline_days(self) -> int:
        return 2  # T+2 补仓期限 (severe cases: next business day)
```

### 0.3 成本模型

**新增文件**: `quant_framework/core/costs.py`

```python
class CryptoFuturesCost(CostModel):
    # --- 手续费 ---
    def commission(self, side, notional, is_maker=False):
        rate = 0.0002 if is_maker else 0.0004  # VIP0 默认
        return notional * rate

    # --- 持仓成本 (每 8 小时) ---
    def holding_cost(self, position_notional, funding_rate, direction):
        # funding_rate > 0: 多头付费给空头
        # funding_rate < 0: 空头付费给多头
        if direction == "long":
            return position_notional * funding_rate   # 正=支出, 负=收入
        else:
            return -position_notional * funding_rate

class USEquityCost(CostModel):
    # --- 手续费 (IBKR Fixed) ---
    def commission(self, side, notional, shares):
        return max(1.0, shares * 0.005)  # $0.005/股, 最低$1

    # --- 持仓成本 ---
    def holding_cost_long(self, borrowed_amount, annual_rate=0.0583):
        # 融资利率: IBKR Benchmark + 1.5%, 每日计息
        return borrowed_amount * annual_rate / 365

    def holding_cost_short(self, market_value, borrow_rate):
        # 借券费: 按股票不同, 0.25% ~ 100%+ 年化
        return market_value * borrow_rate / 365

    # --- 股票特有 ---
    def settlement_delay_days(self) -> int:
        return 1  # T+1 结算 (2024年起)
```

### 0.4 交易对精度规则

**新增文件**: `quant_framework/core/symbol_spec.py`

```python
@dataclass
class SymbolSpec:
    symbol: str
    asset_class: AssetClass
    base_asset: str          # BTC, AAPL
    quote_asset: str         # USDT, USD
    tick_size: float         # 价格步长: BTC=0.10, AAPL=0.01
    step_size: float         # 数量步长: BTC=0.001, AAPL=1
    min_notional: float      # 最小名义: BTC=5 USDT, AAPL≈$1
    min_qty: float           # 最小数量
    max_qty: float           # 最大数量
    max_leverage: float      # 最大杠杆: BTC=125x, AAPL=4x(PDT)

    def round_price(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def round_quantity(self, qty: float) -> float:
        return int(qty / self.step_size) * self.step_size

    def validate_order(self, qty: float, price: float) -> Optional[str]:
        if qty < self.min_qty:
            return f"qty {qty} < min_qty {self.min_qty}"
        if qty > self.max_qty:
            return f"qty {qty} > max_qty {self.max_qty}"
        notional = qty * price
        if notional < self.min_notional:
            return f"notional {notional} < min_notional {self.min_notional}"
        return None
```

**SymbolSpec 数据源**:
- 加密货币: `GET /fapi/v1/exchangeInfo` → filters 字段
- 美股: IBKR `reqContractDetails()` 或硬编码常见规则

### 0.5 Broker 接口扩展

**修改文件**: `quant_framework/broker/base.py`

```python
class Broker(ABC):
    # --- 现有 (保留) ---
    @abstractmethod
    def submit_order(self, signal: Dict) -> Dict: ...
    @abstractmethod
    def get_positions(self) -> Dict: ...
    @abstractmethod
    def get_cash(self) -> float: ...

    # --- 新增: 异步版本 (实盘必须) ---
    async def submit_order_async(self, signal: Dict) -> Dict:
        return self.submit_order(signal)  # 默认回退到同步

    async def cancel_order_async(self, order_id: str) -> Dict:
        raise NotImplementedError

    async def get_order_status_async(self, order_id: str) -> Dict:
        raise NotImplementedError

    async def get_open_orders_async(self, symbol: str = "") -> List[Dict]:
        return []

    # --- 新增: 持仓/余额同步 (实盘必须) ---
    async def sync_positions(self) -> Dict:
        return self.get_positions()

    async def sync_balance(self) -> Dict:
        return {"cash": self.get_cash()}

    # --- 新增: 保证金查询 ---
    def get_available_margin(self) -> float:
        return self.get_cash()

    def get_margin_ratio(self) -> float:
        return 1.0
```

**涉及文件清单**:

| 文件 | 操作 | 说明 |
|------|------|------|
| `quant_framework/core/__init__.py` | 新增 | 核心抽象包 |
| `quant_framework/core/asset_types.py` | 新增 | 资产类型枚举 |
| `quant_framework/core/margin.py` | 新增 | 保证金模型 (3 个实现) |
| `quant_framework/core/costs.py` | 新增 | 成本模型 (2 个实现) |
| `quant_framework/core/symbol_spec.py` | 新增 | 交易对精度规则 |
| `quant_framework/broker/base.py` | 修改 | 扩展 Broker 抽象接口 |
| `quant_framework/backtest/config.py` | 修改 | 增加 `asset_class` 字段 |

---

## Phase 1: 回测层改进 — 逼近真实

**目标**: 让回测结果更真实可信，消除回测和实际交易的已知偏差。

### 1.1 PortfolioTracker 保证金化

**修改文件**: `quant_framework/backtest/portfolio.py`

当前问题: 买入时 `cash -= price × shares + commission`（全额扣现金），不符合杠杆/保证金交易的真实行为。

改造后:

```python
class PortfolioTracker:
    def __init__(self, ..., margin_model: MarginModel, cost_model: CostModel):
        self._margin_model = margin_model
        self._cost_model = cost_model
        self._frozen_margins: Dict[str, float] = {}  # 已冻结保证金
        self._position_entries: Dict[str, PositionEntry] = {}

    def apply_fill(self, fill: Fill):
        notional = fill.exec_price * shares
        leverage = self._config.leverage

        if fill.order.side == OrderSide.BUY:
            if leverage > 1.0:
                margin = self._margin_model.initial_margin(notional, leverage)
                self._cash -= margin + commission
                self._frozen_margins[sym] = margin
            else:
                self._cash -= notional + commission  # 无杠杆: 全额 (和现在一样)
        else:  # SELL (平仓)
            if sym in self._frozen_margins:
                # 归还保证金 + 盈亏
                pnl = self._calc_realized_pnl(sym, fill.exec_price, shares)
                self._cash += self._frozen_margins.pop(sym) + pnl - commission
            else:
                self._cash += notional - commission  # 无杠杆

    def process_bar_costs(self, prices, bar_index, date):
        for sym, qty in self._positions.items():
            px = prices[sym]
            notional = abs(qty) * px
            direction = "long" if qty > 0 else "short"

            # 持仓成本 (按资产类型不同)
            cost = self._cost_model.holding_cost(notional, self._funding_rate, direction)
            self._cash -= cost

            # 强平检查 (按保证金模型)
            margin = self._frozen_margins.get(sym, 0)
            unrealized = self._calc_unrealized_pnl(sym, px)
            equity = margin + unrealized
            maint = self._margin_model.maintenance_margin(notional)
            if equity < maint:
                close_signals.append(...)  # 强平
```

### 1.2 内核路径止损改进

**修改文件**: `quant_framework/backtest/kernels.py`

当前问题: `_sl_exit` 只用 close 价检查止损，遗漏 bar 内穿越。

```python
# 改造前 (只用 close)
@njit
def _sl_exit(pos, ep, tr, ci, sb, ss, cm, lev, sl, pfrac, sl_slip):
    if pos == 1:
        raw = (ci * ss * (1.0 - cm)) / (ep * (1.0 + cm))
        pnl = (raw - 1.0) * lev
    ...

# 改造后 (加入 high/low intra-bar 检查)
@njit
def _sl_exit_v2(pos, ep, tr, hi, lo, ci, sb, ss, cm, lev, sl, pfrac, sl_slip):
    if pos == 0 or ep <= 0:
        return pos, ep, tr, 0
    if pos == 1:
        # 用 bar 的 low 检查 (bar 内可能触及止损价)
        worst = lo * ss * (1.0 - cm) / (ep * (1.0 + cm))
        pnl_worst = (worst - 1.0) * lev
        if pnl_worst < -sl:
            # 止损触发, 以止损价成交 (不是 close)
            actual_loss = sl + sl_slip
            deployed = _deploy(tr, pfrac)
            tr -= deployed * actual_loss
            return 0, 0.0, max(0.01, tr), 1
        # 正常路径: 用 close 计算当前 PnL
        raw = (ci * ss * (1.0 - cm)) / (ep * (1.0 + cm))
        pnl = (raw - 1.0) * lev
    elif pos == -1:
        # 用 bar 的 high 检查空头止损
        worst = (ep * (1.0 - cm)) / (hi * sb * (1.0 + cm))
        pnl_worst = (worst - 1.0) * lev
        if pnl_worst < -sl:
            actual_loss = sl + sl_slip
            deployed = _deploy(tr, pfrac)
            tr -= deployed * actual_loss
            return 0, 0.0, max(0.01, tr), 1
    ...
```

需要同步修改所有 18 个内核的 `_sl_exit` 调用，传入 `h[i]`, `l[i]`。

### 1.3 历史 Funding Rate 数据

**新增文件**: `quant_framework/data/funding_rates.py`

```python
class FundingRateLoader:
    """下载并缓存 Binance 历史 funding rate"""

    BINANCE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

    async def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # 返回 DataFrame: [timestamp, funding_rate]
        # funding_rate 每 8 小时一条, 典型范围 -0.003 ~ +0.003
        ...

    def load_cached(self, symbol: str, interval: str) -> np.ndarray:
        # 将 8h funding rate 映射到回测 bar 级别
        # 例如: 1d bar → 一天3次 funding rate 求和
        # 例如: 4h bar → 分配最近的 funding rate
        ...
```

**改造 `config_to_kernel_costs`**:

```python
def config_to_kernel_costs(config, funding_rates=None):
    if funding_rates is not None:
        # 用真实历史费率 (每 bar 不同)
        dc_array = funding_rates  # 传入内核
    else:
        # 回退到固定费率 (现有行为)
        dc = config.daily_funding_rate / bpd
```

### 1.4 真实费率阶梯

**修改文件**: `quant_framework/backtest/config.py`

```python
@dataclass(frozen=True)
class BacktestConfig:
    # 新增字段
    asset_class: str = "crypto_perp"

    # 加密货币: maker/taker 区分
    commission_pct_maker: float = 0.0002    # Binance VIP0 maker
    commission_pct_taker: float = 0.0004    # Binance VIP0 taker
    maker_ratio: float = 0.3               # 假设 30% 成交为 maker

    # 美股: IBKR 费率
    commission_per_share: float = 0.005     # $0.005/股
    commission_min: float = 1.0             # 最低 $1
    exchange_fee_per_share: float = 0.003   # 交易所费用

    # 美股: 融资利率
    margin_interest_rate: float = 0.0583    # ~5.83% 年化
```

### 1.5 跳空 (Gap) 处理

**修改 `_fx_lev`**: 开仓时如果 open 价和信号价差距过大，成交价用 open（而非假设以信号价成交）。这已经是现有行为。

**修改 `_sl_exit_v2`**: 如果 open 已经跳过止损价，以 open 价止损（而非止损价），模拟滑点跳空：

```python
if pos == 1 and oi < sl_price:
    # 开盘已跳空低于止损价, 实际亏损大于 sl
    actual_loss = min(1.0, (ep - oi) / ep * lev)
    ...
```

### Phase 1 涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `quant_framework/backtest/portfolio.py` | 修改 | 保证金模型集成 |
| `quant_framework/backtest/kernels.py` | 修改 | `_sl_exit` → `_sl_exit_v2` + 18 个内核 |
| `quant_framework/backtest/config.py` | 修改 | 新增 asset_class, maker/taker, 股票费率字段 |
| `quant_framework/data/funding_rates.py` | 新增 | 历史 funding rate 下载器 |
| `tests/test_margin_model.py` | 新增 | 保证金模型单元测试 |
| `tests/test_sl_v2.py` | 新增 | intra-bar 止损测试 |

---

## Phase 2: 纸交易层改进 — 高仿真模拟

**目标**: PaperBroker 使用与实盘相同的保证金/成本模型，模拟结果高度逼近实盘。

### 2.1 PaperBroker 重构

**修改文件**: `quant_framework/broker/paper.py`

```python
class PaperBroker(Broker):
    def __init__(self,
                 initial_cash: float,
                 margin_model: MarginModel,
                 cost_model: CostModel,
                 symbol_specs: Dict[str, SymbolSpec] = None,
                 ...):
        self._margin_model = margin_model
        self._cost_model = cost_model
        self._symbol_specs = symbol_specs or {}
        self._frozen_margins: Dict[str, float] = {}
        self._leverage_settings: Dict[str, float] = {}

    def submit_order(self, signal):
        symbol = signal["symbol"]
        spec = self._symbol_specs.get(symbol)

        # 1. 精度过滤
        if spec:
            shares = spec.round_quantity(raw_shares)
            error = spec.validate_order(shares, price)
            if error:
                return {"status": "rejected", "message": error}

        # 2. 保证金检查
        notional = price * shares
        leverage = self._leverage_settings.get(symbol, 1.0)
        margin_req = self._margin_model.initial_margin(notional, leverage)
        available = self._available_margin()
        if margin_req > available:
            return {"status": "rejected", "message": "insufficient margin"}

        # 3. 执行 (模拟成交, 含滑点)
        exec_price = self._apply_slippage(price, action)
        commission = self._cost_model.commission(action, notional, shares)

        # 4. 更新状态 (保证金模式)
        if leverage > 1.0:
            self._frozen_margins[symbol] = margin_req
            self._cash -= margin_req + commission
        else:
            self._cash -= notional + commission
        ...

    def process_funding(self, current_time, prices):
        """每 8 小时调用 (加密货币) 或每日 (股票融资利息)"""
        for sym, qty in self._positions.items():
            notional = abs(qty) * prices[sym]
            direction = "long" if qty > 0 else "short"
            cost = self._cost_model.holding_cost(notional, self._funding_rate, direction)
            self._cash -= cost
```

### 2.2 TradingRunner 适配

**修改文件**: `quant_framework/live/trading_runner.py`

```python
class TradingRunner:
    def __init__(self, ..., margin_model=None, cost_model=None):
        # 定时任务
        self._funding_interval = 8 * 3600  # 加密: 每 8 小时
        self._last_funding_ts = 0

    async def _on_bar(self, bar):
        # ... 现有逻辑 ...
        # 新增: 检查是否需要处理资金费
        await self._maybe_process_funding()
        # 新增: 检查保证金率
        self._check_margin_levels()

    async def _maybe_process_funding(self):
        now = time.time()
        if now - self._last_funding_ts >= self._funding_interval:
            prices = self._get_prices()
            self._broker.process_funding(now, prices)
            self._last_funding_ts = now

    def _check_margin_levels(self):
        """检查所有持仓的保证金率, 接近强平时告警"""
        ratio = self._broker.get_margin_ratio()
        if ratio < 0.1:  # <10% 缓冲
            logger.critical("MARGIN RATIO CRITICAL: %.1f%%", ratio * 100)
            # 触发减仓或告警

    def _calc_shares(self, sig):
        """改造: 用保证金计算可买数量"""
        price = float(sig.get("price", 0))
        if price <= 0:
            return 0.0

        available_margin = self._broker.get_available_margin()
        cfg = self._get_config(sig.get("symbol", ""))
        leverage = cfg.leverage if cfg else 1.0

        # 保证金模式: 可开仓位 = available_margin × leverage × pos_size_pct
        max_notional = available_margin * leverage * self._pos_size_pct
        raw = max_notional / price

        # 精度过滤
        spec = self._broker._symbol_specs.get(sig.get("symbol", ""))
        if spec:
            raw = spec.round_quantity(raw)
        return max(0, raw)
```

### 2.3 交易时间限制

**新增文件**: `quant_framework/core/market_hours.py`

```python
class MarketCalendar:
    """交易时间检查, 按资产类型不同"""

    def is_tradable(self, asset_class: AssetClass, dt: datetime) -> bool:
        if asset_class in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_PERP):
            return True  # 24/7

        if asset_class in (AssetClass.US_EQUITY, AssetClass.US_EQUITY_MARGIN):
            return self._is_us_market_open(dt)

    def _is_us_market_open(self, dt: datetime) -> bool:
        et = dt.astimezone(ZoneInfo("US/Eastern"))
        if et.weekday() >= 5: return False
        if self._is_us_holiday(et.date()): return False
        t = et.time()
        return time(9, 30) <= t <= time(16, 0)

    def next_open(self, asset_class: AssetClass, dt: datetime) -> datetime:
        """下次开盘时间"""
        ...

    def _is_us_holiday(self, date) -> bool:
        # NYSE holidays: New Year, MLK, Presidents Day, Good Friday,
        # Memorial Day, Juneteenth, Independence Day, Labor Day,
        # Thanksgiving, Christmas
        ...
```

### Phase 2 涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `quant_framework/broker/paper.py` | 重构 | 保证金模式 + 精度过滤 |
| `quant_framework/live/trading_runner.py` | 修改 | 资金费处理 + 保证金检查 + 仓位计算 |
| `quant_framework/core/market_hours.py` | 新增 | 交易时间日历 |
| `tests/test_paper_margin.py` | 新增 | PaperBroker 保证金测试 |

---

## Phase 3: 实盘层 — 交易所对接

**目标**: 实现真实的交易所连接，支持 Binance 合约和 IBKR 美股。

### 3.1 安全基础设施

**新增文件**: `quant_framework/broker/credentials.py`

```python
class CredentialManager:
    """API 密钥安全管理"""

    def load(self, exchange: str) -> Tuple[str, str]:
        # 优先级: 环境变量 > .env 文件 > 配置文件
        # 1. os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET"]
        # 2. .env 文件 (python-dotenv)
        # 3. config/credentials.yaml (gitignore'd)
        ...

    def sign_request(self, params: dict, secret: str) -> str:
        """Binance HMAC-SHA256 签名"""
        ...
```

**新增文件**: `quant_framework/broker/rate_limiter.py`

```python
class RateLimiter:
    """Token bucket 限速器"""

    def __init__(self, max_weight: int = 1200, window_seconds: int = 60):
        self._max_weight = max_weight
        self._window = window_seconds
        self._used: Deque[Tuple[float, int]] = deque()

    async def acquire(self, weight: int = 1):
        """等待直到有足够的配额"""
        while self._current_usage() + weight > self._max_weight:
            await asyncio.sleep(0.1)
        self._used.append((time.time(), weight))

    def _current_usage(self) -> int:
        now = time.time()
        while self._used and now - self._used[0][0] > self._window:
            self._used.popleft()
        return sum(w for _, w in self._used)
```

### 3.2 Binance 合约 Broker

**新增文件**: `quant_framework/broker/binance_futures.py`

```python
class BinanceFuturesBroker(Broker):
    """Binance USDT-M 永续合约实盘 Broker"""

    BASE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"
    WS_URL = "wss://fstream.binance.com"

    def __init__(self, credentials: CredentialManager,
                 testnet: bool = False,
                 margin_type: str = "ISOLATED"):
        self._cred = credentials
        self._base = self.TESTNET_URL if testnet else self.BASE_URL
        self._margin_type = margin_type
        self._rate_limiter = RateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._positions: Dict[str, float] = {}
        self._balances: Dict[str, float] = {}
        self._order_updates: asyncio.Queue = asyncio.Queue()
        self._symbol_specs: Dict[str, SymbolSpec] = {}

    # --- 初始化 ---
    async def initialize(self):
        """启动时: 加载交易对规则 + 同步持仓"""
        await self._load_exchange_info()
        await self.sync_positions()
        await self.sync_balance()
        asyncio.create_task(self._user_data_stream())

    async def _load_exchange_info(self):
        """GET /fapi/v1/exchangeInfo → 填充 SymbolSpec"""
        data = await self._request("GET", "/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            filters = {f["filterType"]: f for f in s["filters"]}
            lot = filters.get("LOT_SIZE", {})
            price = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", {})
            self._symbol_specs[s["symbol"]] = SymbolSpec(
                symbol=s["symbol"],
                asset_class=AssetClass.CRYPTO_PERP,
                base_asset=s["baseAsset"],
                quote_asset=s["quoteAsset"],
                tick_size=float(price.get("tickSize", 0.01)),
                step_size=float(lot.get("stepSize", 0.001)),
                min_notional=float(notional.get("notional", 5)),
                min_qty=float(lot.get("minQty", 0.001)),
                max_qty=float(lot.get("maxQty", 1000)),
                max_leverage=125.0,
            )

    # --- 杠杆设置 ---
    async def set_leverage(self, symbol: str, leverage: int):
        await self._request("POST", "/fapi/v1/leverage",
                           {"symbol": symbol, "leverage": leverage})

    async def set_margin_type(self, symbol: str, margin_type: str):
        try:
            await self._request("POST", "/fapi/v1/marginType",
                               {"symbol": symbol, "marginType": margin_type})
        except Exception:
            pass  # 已经是该模式则会报错, 忽略

    # --- 下单 ---
    async def submit_order_async(self, signal: Dict) -> Dict:
        symbol = signal["symbol"]
        spec = self._symbol_specs.get(symbol)
        if not spec:
            return {"status": "rejected", "message": f"unknown symbol {symbol}"}

        qty = spec.round_quantity(float(signal["shares"]))
        error = spec.validate_order(qty, float(signal.get("price", 0)))
        if error:
            return {"status": "rejected", "message": error}

        side = "BUY" if signal["action"] == "buy" else "SELL"
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
        }

        try:
            result = await self._request("POST", "/fapi/v1/order", params)
            return {
                "status": "filled" if result["status"] == "FILLED" else "submitted",
                "order_id": str(result["orderId"]),
                "fill_price": float(result.get("avgPrice", 0)),
                "filled_shares": float(result.get("executedQty", 0)),
                "commission": float(result.get("commission", 0)),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # --- 止损单 ---
    async def place_stop_loss(self, symbol: str, side: str,
                               qty: float, stop_price: float):
        spec = self._symbol_specs[symbol]
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": spec.round_quantity(qty),
            "stopPrice": spec.round_price(stop_price),
            "closePosition": "false",
        }
        return await self._request("POST", "/fapi/v1/order", params)

    # --- 同步 ---
    async def sync_positions(self) -> Dict:
        data = await self._request("GET", "/fapi/v2/positionRisk")
        self._positions = {}
        for p in data:
            qty = float(p["positionAmt"])
            if abs(qty) > 1e-10:
                self._positions[p["symbol"]] = qty
        return dict(self._positions)

    async def sync_balance(self) -> Dict:
        data = await self._request("GET", "/fapi/v2/balance")
        self._balances = {}
        for b in data:
            bal = float(b["balance"])
            if bal > 0:
                self._balances[b["asset"]] = bal
        return dict(self._balances)

    def get_positions(self) -> Dict:
        return dict(self._positions)

    def get_cash(self) -> float:
        return self._balances.get("USDT", 0.0)

    # --- WebSocket 用户数据流 ---
    async def _user_data_stream(self):
        """监听订单更新、账户更新"""
        listen_key = await self._get_listen_key()
        url = f"{self.WS_URL}/ws/{listen_key}"
        while True:
            try:
                async with websockets.connect(url) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("e") == "ORDER_TRADE_UPDATE":
                            await self._handle_order_update(data["o"])
                        elif data.get("e") == "ACCOUNT_UPDATE":
                            self._handle_account_update(data["a"])
            except Exception as e:
                logger.warning("User data stream error: %s, reconnecting...", e)
                await asyncio.sleep(5)
                listen_key = await self._get_listen_key()

    # --- 签名请求 ---
    async def _request(self, method, path, params=None):
        await self._rate_limiter.acquire(weight=1)
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._cred.sign_request(params, self._secret)
        headers = {"X-MBX-APIKEY": self._api_key}
        url = f"{self._base}{path}"
        async with self._session.request(method, url, params=params,
                                          headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise Exception(f"Binance API error: {data}")
            return data
```

### 3.3 IBKR 股票 Broker

**新增文件**: `quant_framework/broker/ibkr_broker.py`

```python
class IBKRBroker(Broker):
    """Interactive Brokers 实盘 Broker (基于 ib_insync)"""

    def __init__(self, host="127.0.0.1", port=7497,  # 7497=TWS纸交易, 7496=实盘
                 client_id=1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: Optional[IB] = None
        self._positions: Dict[str, float] = {}
        self._cash: float = 0
        self._pdt_tracker = PDTTracker()

    async def connect(self):
        from ib_insync import IB
        self._ib = IB()
        await self._ib.connectAsync(self._host, self._port, self._client_id)
        await self.sync_positions()
        await self.sync_balance()

    async def submit_order_async(self, signal: Dict) -> Dict:
        from ib_insync import Stock, MarketOrder, LimitOrder

        symbol = signal["symbol"]
        action = "BUY" if signal["action"] == "buy" else "SELL"
        shares = int(signal["shares"])  # 美股通常整股

        # PDT 检查
        if signal["action"] == "sell" and self._pdt_tracker.would_violate(symbol):
            return {"status": "rejected", "message": "PDT rule violation"}

        # 交易时间检查
        if not MarketCalendar().is_tradable(AssetClass.US_EQUITY, datetime.now()):
            return {"status": "rejected", "message": "market closed"}

        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder(action, shares)

        trade = self._ib.placeOrder(contract, order)
        # 等待成交 (最多 30 秒)
        timeout = 30
        while not trade.isDone() and timeout > 0:
            await asyncio.sleep(0.5)
            timeout -= 0.5

        if trade.orderStatus.status == "Filled":
            return {
                "status": "filled",
                "order_id": str(trade.order.orderId),
                "fill_price": trade.orderStatus.avgFillPrice,
                "filled_shares": trade.orderStatus.filled,
                "commission": sum(f.commission for f in trade.fills),
            }
        else:
            return {
                "status": trade.orderStatus.status.lower(),
                "message": trade.orderStatus.whyHeld or "",
            }

    async def sync_positions(self) -> Dict:
        positions = self._ib.positions()
        self._positions = {}
        for p in positions:
            sym = p.contract.symbol
            self._positions[sym] = float(p.position)
        return dict(self._positions)

    async def sync_balance(self) -> Dict:
        account = self._ib.accountSummary()
        for item in account:
            if item.tag == "TotalCashValue":
                self._cash = float(item.value)
        return {"cash": self._cash}

    def get_positions(self) -> Dict:
        return dict(self._positions)

    def get_cash(self) -> float:
        return self._cash
```

### 3.4 PDT 规则追踪 (美股特有)

**新增文件**: `quant_framework/core/pdt_tracker.py`

```python
class PDTTracker:
    """Pattern Day Trader 规则追踪 (账户 < $25k 时生效)"""

    def __init__(self, equity_threshold=25000):
        self._threshold = equity_threshold
        self._day_trades: Deque[Tuple[str, date]] = deque()  # (symbol, date)

    def record_round_trip(self, symbol: str, dt: datetime):
        """记录一次日内往返交易"""
        self._day_trades.append((symbol, dt.date()))
        self._cleanup_old(dt.date())

    def day_trade_count(self, current_date: date) -> int:
        """最近 5 个交易日的日内交易次数"""
        self._cleanup_old(current_date)
        return len(self._day_trades)

    def would_violate(self, equity: float, current_date: date) -> bool:
        if equity >= self._threshold:
            return False  # $25k 以上不受限
        return self.day_trade_count(current_date) >= 3

    def _cleanup_old(self, current_date: date):
        cutoff = current_date - timedelta(days=7)  # ~5 个交易日
        while self._day_trades and self._day_trades[0][1] < cutoff:
            self._day_trades.popleft()
```

### Phase 3 涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `quant_framework/broker/credentials.py` | 新增 | API 密钥管理 |
| `quant_framework/broker/rate_limiter.py` | 新增 | API 限速器 |
| `quant_framework/broker/binance_futures.py` | 新增 | Binance 合约 Broker |
| `quant_framework/broker/ibkr_broker.py` | 新增 | IBKR 股票 Broker |
| `quant_framework/core/pdt_tracker.py` | 新增 | PDT 规则追踪 |
| `config/credentials.yaml.example` | 新增 | 密钥配置模板 (无真实密钥) |
| `.env.example` | 新增 | 环境变量模板 |

---

## Phase 4: 订单管理 + 持仓对账

**目标**: 实盘环境下的订单全生命周期管理和状态一致性保证。

### 4.1 实盘订单管理器

**新增文件**: `quant_framework/broker/live_order_manager.py`

```python
class LiveOrderManager:
    """实盘订单全生命周期管理"""

    class OrderState(Enum):
        PENDING = "pending"         # 等待提交
        SUBMITTED = "submitted"     # 已提交到交易所
        PARTIAL = "partial"         # 部分成交
        FILLED = "filled"           # 完全成交
        CANCELLING = "cancelling"   # 取消中
        CANCELLED = "cancelled"     # 已取消
        EXPIRED = "expired"         # 已过期
        ERROR = "error"             # 错误

    async def submit(self, broker, signal, timeout_seconds=30):
        """提交订单并等待确认"""
        # 1. 预检 (RiskGate)
        # 2. 异步提交
        result = await broker.submit_order_async(signal)
        # 3. 如果未立即成交, 等待 + 超时处理
        if result["status"] == "submitted":
            result = await self._wait_fill(broker, result["order_id"], timeout_seconds)
        # 4. 超时未成交 → 取消
        if result["status"] not in ("filled", "partial"):
            await broker.cancel_order_async(result["order_id"])
        return result

    async def _wait_fill(self, broker, order_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = await broker.get_order_status_async(order_id)
            if status["status"] in ("filled", "cancelled", "expired", "error"):
                return status
            await asyncio.sleep(0.5)
        return {"status": "timeout", "order_id": order_id}
```

### 4.2 持仓对账器

**新增文件**: `quant_framework/broker/reconciler.py`

```python
class PositionReconciler:
    """定期对账: 内部状态 vs 交易所真实状态"""

    def __init__(self, broker: Broker, journal: TradeJournal,
                 alert_manager=None, interval_seconds=60):
        self._broker = broker
        self._journal = journal
        self._alert = alert_manager
        self._interval = interval_seconds

    async def run(self):
        while True:
            await asyncio.sleep(self._interval)
            await self.reconcile()

    async def reconcile(self):
        exchange_positions = await self._broker.sync_positions()
        internal_positions = self._broker.get_positions()
        exchange_balance = await self._broker.sync_balance()

        discrepancies = []
        all_symbols = set(exchange_positions) | set(internal_positions)
        for sym in all_symbols:
            ex_qty = exchange_positions.get(sym, 0)
            in_qty = internal_positions.get(sym, 0)
            if abs(ex_qty - in_qty) > 1e-8:
                discrepancies.append({
                    "symbol": sym,
                    "exchange": ex_qty,
                    "internal": in_qty,
                    "diff": ex_qty - in_qty,
                })

        if discrepancies:
            logger.warning("Position discrepancies: %s", discrepancies)
            if self._alert:
                await self._alert.send(
                    level="WARNING",
                    title="持仓不一致",
                    body=f"发现 {len(discrepancies)} 个不一致: {discrepancies}"
                )
            # 以交易所为准, 修正内部状态
            self._broker._positions = exchange_positions
```

### 4.3 崩溃恢复

**修改文件**: `quant_framework/live/trading_runner.py`

```python
class TradingRunner:
    async def run(self, lookback=200):
        # 启动前: 恢复状态
        await self._recover_state()
        # ... 正常启动 ...

    async def _recover_state(self):
        """从交易所 + SQLite 恢复运行状态"""
        # 1. 从交易所同步真实持仓
        positions = await self._broker.sync_positions()
        balance = await self._broker.sync_balance()
        logger.info("Recovered: %d positions, cash=%.2f",
                     len(positions), balance.get("cash", 0))

        # 2. 从 TradeJournal 恢复 entry_prices
        trades = self._journal.get_trades(limit=500)
        for sym in positions:
            sym_trades = trades[trades["symbol"] == sym].sort_values("timestamp")
            if not sym_trades.empty:
                last_buy = sym_trades[sym_trades["side"] == "buy"].iloc[-1]
                self._entry_prices[sym] = (last_buy["price"], positions[sym])

        # 3. 从交易所恢复挂单
        open_orders = await self._broker.get_open_orders_async()
        logger.info("Recovered %d open orders", len(open_orders))
```

---

## Phase 5: 告警 + 审计

### 5.1 多渠道告警

**新增文件**: `quant_framework/live/alerts.py`

```python
class AlertManager:
    """多渠道告警: Telegram / Discord / Email"""

    def __init__(self, config: Dict):
        self._telegram_token = config.get("telegram_bot_token")
        self._telegram_chat = config.get("telegram_chat_id")
        self._discord_webhook = config.get("discord_webhook_url")

    async def send(self, level: str, title: str, body: str):
        tasks = []
        if self._telegram_token:
            tasks.append(self._send_telegram(level, title, body))
        if self._discord_webhook:
            tasks.append(self._send_discord(level, title, body))
        await asyncio.gather(*tasks, return_exceptions=True)

    # 告警场景:
    # - 订单成交/失败
    # - 熔断器触发
    # - 持仓对账异常
    # - 日亏损接近限额
    # - 系统错误/重连
    # - 保证金率低于阈值
```

### 5.2 审计追踪

**新增文件**: `quant_framework/live/audit.py`

```python
class AuditTrail:
    """全链路审计追踪"""

    def record_order_lifecycle(self,
                                internal_id: str,
                                exchange_order_id: str,
                                signal_ts: datetime,
                                submit_ts: datetime,
                                ack_ts: Optional[datetime],
                                fill_ts: Optional[datetime],
                                fill_price: Optional[float],
                                latency_ms: float):
        # 写入 SQLite audit 表
        ...

    def generate_daily_audit_report(self, date: date) -> str:
        # 生成当日审计报告:
        # - 总订单数, 成交率, 平均延迟
        # - 拒绝原因分布
        # - 滑点统计 (信号价 vs 成交价)
        ...
```

---

## Phase 6: Testnet 验证 + 集成测试

### 6.1 Testnet 配置

**新增文件**: `quant_framework/broker/testnet.py`

```python
class TestnetConfig:
    BINANCE_FUTURES = {
        "rest": "https://testnet.binancefuture.com",
        "ws": "wss://stream.binancefuture.com",
        "ws_user": "wss://stream.binancefuture.com",
    }
    IBKR_PAPER = {
        "port": 7497,  # TWS Paper Trading port
    }
```

### 6.2 集成测试矩阵

**新增文件**: `tests/test_integration_exchange.py`

```
测试矩阵:
┌────────────────────┬────────────┬──────────────┐
│ 测试场景            │ Binance    │ IBKR         │
│                    │ Testnet    │ Paper        │
├────────────────────┼────────────┼──────────────┤
│ 连接 + 认证         │ ✓          │ ✓            │
│ 查余额              │ ✓          │ ✓            │
│ 查持仓              │ ✓          │ ✓            │
│ 市价买入            │ ✓          │ ✓            │
│ 市价卖出            │ ✓          │ ✓            │
│ 限价挂单            │ ✓          │ ✓            │
│ 取消挂单            │ ✓          │ ✓            │
│ 止损单              │ ✓          │ ✓            │
│ 设置杠杆            │ ✓          │ N/A          │
│ 资金费结算           │ ✓          │ N/A          │
│ 持仓对账            │ ✓          │ ✓            │
│ 崩溃恢复            │ ✓          │ ✓            │
│ 熔断器测试           │ ✓          │ ✓            │
│ 精度过滤            │ ✓          │ ✓            │
│ PDT 规则            │ N/A        │ ✓            │
│ 盘前盘后限制         │ N/A        │ ✓            │
└────────────────────┴────────────┴──────────────┘
```

---

## Phase 7 (可选): 高级执行算法

**新增文件**: `quant_framework/broker/execution_algo.py`

| 算法 | 用途 | 逻辑 |
|------|------|------|
| TWAP | 大单均匀分片 | 将 N 个币/股分成 K 片, 每隔 T 秒下一片 |
| Limit Chase | 优化成交价 | 先挂 limit 单, 未成交则逐步追价 |
| Iceberg | 隐藏真实数量 | 只显示 10% 的量, 成交后自动补挂 |

---

## 完整文件变更清单

### 新增文件 (18 个)

| 文件路径 | Phase | 说明 |
|----------|-------|------|
| `quant_framework/core/__init__.py` | 0 | 核心抽象包 |
| `quant_framework/core/asset_types.py` | 0 | 资产类型 |
| `quant_framework/core/margin.py` | 0 | 保证金模型 (3 实现) |
| `quant_framework/core/costs.py` | 0 | 成本模型 (2 实现) |
| `quant_framework/core/symbol_spec.py` | 0 | 交易对精度 |
| `quant_framework/core/market_hours.py` | 2 | 交易日历 |
| `quant_framework/core/pdt_tracker.py` | 3 | PDT 规则 |
| `quant_framework/data/funding_rates.py` | 1 | 历史 funding rate |
| `quant_framework/broker/credentials.py` | 3 | 密钥管理 |
| `quant_framework/broker/rate_limiter.py` | 3 | API 限速 |
| `quant_framework/broker/binance_futures.py` | 3 | Binance 合约 |
| `quant_framework/broker/ibkr_broker.py` | 3 | IBKR 股票 |
| `quant_framework/broker/live_order_manager.py` | 4 | 实盘 OMS |
| `quant_framework/broker/reconciler.py` | 4 | 持仓对账 |
| `quant_framework/live/alerts.py` | 5 | 告警系统 |
| `quant_framework/live/audit.py` | 5 | 审计追踪 |
| `quant_framework/broker/testnet.py` | 6 | Testnet 配置 |
| `quant_framework/broker/execution_algo.py` | 7 | 执行算法 (可选) |

### 修改文件 (7 个)

| 文件路径 | Phase | 改动范围 |
|----------|-------|----------|
| `quant_framework/broker/base.py` | 0 | 新增异步方法 + 保证金查询 |
| `quant_framework/backtest/config.py` | 0+1 | 新增 asset_class, maker/taker, 股票字段 |
| `quant_framework/backtest/portfolio.py` | 1 | 保证金化重构 |
| `quant_framework/backtest/kernels.py` | 1 | `_sl_exit_v2` + 18 内核适配 |
| `quant_framework/broker/paper.py` | 2 | 保证金模式 + 精度过滤 |
| `quant_framework/live/trading_runner.py` | 2+4 | 资金费 + 保证金检查 + 恢复 |
| `run_live_trading.py` | 3 | 支持选择 Broker 类型 |

### 新增测试 (6 个)

| 文件路径 | Phase | 说明 |
|----------|-------|------|
| `tests/test_margin_models.py` | 0 | 保证金模型 |
| `tests/test_cost_models.py` | 0 | 成本模型 |
| `tests/test_sl_v2.py` | 1 | intra-bar 止损 |
| `tests/test_paper_margin.py` | 2 | PaperBroker 保证金 |
| `tests/test_integration_exchange.py` | 6 | 交易所集成 |
| `tests/test_reconciler.py` | 4 | 对账测试 |

---

## 时间估计

| Phase | 内容 | 预估时间 | 依赖 |
|-------|------|----------|------|
| 0 | 核心抽象层 | 2-3 天 | 无 |
| 1 | 回测改进 | 3-4 天 | Phase 0 |
| 2 | 纸交易改进 | 2-3 天 | Phase 0 |
| 3 | 交易所对接 | 5-7 天 | Phase 0 |
| 4 | OMS + 对账 | 2-3 天 | Phase 3 |
| 5 | 告警 + 审计 | 1-2 天 | Phase 3 |
| 6 | Testnet 验证 | 2-3 天 | Phase 3+4 |
| 7 | 执行算法 (可选) | 2-3 天 | Phase 3 |

**总计**: Phase 0-6 约 **17-25 天**, Phase 7 可选 +2-3 天。

Phase 1 和 Phase 2 可以和 Phase 3 并行开发。

---

## 关键设计原则

1. **三层一致**: 回测、纸交易、实盘共用 MarginModel + CostModel, 保证数学行为一致
2. **资产感知**: 加密货币和美股的保证金/费率/规则完全独立建模, 不混淆
3. **Testnet First**: 实盘代码必须先在测试网/纸交易账户验证通过
4. **向后兼容**: 现有的回测脚本和策略代码无需修改即可运行
5. **安全第一**: API 密钥不入库, 有熔断, 有对账, 有告警
6. **可观测**: 全链路延迟追踪 + 审计日志 + 实时告警
7. **渐进式**: 先 Binance Testnet → Binance Mainnet → IBKR Paper → IBKR Live

---

## V5.1 深度优化增补

> 基于对现有代码库 (`kernels.py`, `portfolio.py`, `fill_simulator.py`, `paper.py`,
> `trading_runner.py`, `risk.py`, `config.py`) 的逐行审查，以及对计划中代码设计的
> 极致性能分析，以下增补分四大类：**安全关键**、**正确性修复**、**性能极致**、
> **功能补全**。

---

### A. 安全关键缺失 (实盘必须，否则可能亏损真金白银)

#### A.1 紧急平仓 / Kill Switch

**原始计划缺失**: 没有「一键平仓」机制。实盘必须有。

```python
# quant_framework/live/kill_switch.py

class KillSwitch:
    """Emergency position flattener — the nuclear option."""

    def __init__(self, broker: Broker, alert: AlertManager):
        self._broker = broker
        self._alert = alert
        self._armed = True

    async def flatten_all(self, reason: str):
        """Cancel all open orders, close all positions at market."""
        if not self._armed:
            return
        self._armed = False  # prevent re-entry

        # 1. Cancel all pending orders
        open_orders = await self._broker.get_open_orders_async()
        cancel_tasks = [
            self._broker.cancel_order_async(o["order_id"])
            for o in open_orders
        ]
        await asyncio.gather(*cancel_tasks, return_exceptions=True)

        # 2. Market-close all positions
        positions = await self._broker.sync_positions()
        for sym, qty in positions.items():
            if abs(qty) < 1e-10:
                continue
            side = "sell" if qty > 0 else "buy"
            await self._broker.submit_order_async({
                "symbol": sym,
                "action": side,
                "shares": abs(qty),
                "reduce_only": True,  # A.2
            })

        # 3. Alert
        await self._alert.send("CRITICAL", "KILL SWITCH",
                                f"All positions flattened. Reason: {reason}")
        logger.critical("KILL SWITCH activated: %s", reason)
```

**触发条件** (集成到 `TradingRunner`):
- 日亏损超限 (CircuitBreaker trip)
- 保证金率 < 5%
- 手动 HTTP endpoint / Unix signal
- 连续 N 次 API 错误 (交易所可能在维护)

#### A.2 Reduce-Only 订单

**原始计划缺失**: `BinanceFuturesBroker.submit_order_async` 下平仓单时没有使用 `reduceOnly`，
如果信号方向错误可能意外开反向仓位。

```python
# binance_futures.py submit_order_async 改造
async def submit_order_async(self, signal: Dict) -> Dict:
    ...
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
    }
    # 平仓单必须 reduceOnly，防止信号错误导致反向开仓
    if signal.get("reduce_only") or signal.get("action") in ("close_long", "close_short"):
        params["reduceOnly"] = "true"
    ...
```

#### A.3 订单幂等性 / 去重

**原始计划缺失**: 网络超时后重试可能导致双重下单。

```python
# binance_futures.py — 使用 Binance newClientOrderId
import uuid

async def submit_order_async(self, signal: Dict) -> Dict:
    # 生成唯一 client order ID，用于幂等性
    client_oid = signal.get("client_order_id") or f"qf_{uuid.uuid4().hex[:16]}"
    params = {
        ...
        "newClientOrderId": client_oid,
    }
    try:
        result = await self._request("POST", "/fapi/v1/order", params)
        ...
    except asyncio.TimeoutError:
        # 超时不代表失败！查询该 client_oid 的真实状态
        await asyncio.sleep(1)
        status = await self._query_by_client_oid(symbol, client_oid)
        if status and status["status"] == "FILLED":
            return {"status": "filled", ...}
        return {"status": "timeout", "client_order_id": client_oid}
```

**去重机制** (在 `LiveOrderManager` 添加):

```python
class LiveOrderManager:
    def __init__(self):
        self._recent_signals: Dict[str, float] = {}  # signal_hash → timestamp
        self._dedup_window = 5.0  # seconds

    def _is_duplicate(self, signal: Dict) -> bool:
        key = f"{signal['symbol']}_{signal['action']}_{signal.get('shares', 0)}"
        now = time.time()
        if key in self._recent_signals:
            if now - self._recent_signals[key] < self._dedup_window:
                logger.warning("Duplicate signal suppressed: %s", key)
                return True
        self._recent_signals[key] = now
        return False
```

#### A.4 优雅关机

**原始计划缺失**: 进程被 kill 时，挂单仍留在交易所。

```python
# trading_runner.py 改造
class TradingRunner:
    async def run(self, lookback=200):
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(
                self._graceful_shutdown(s)))
        await self._recover_state()
        ...

    async def _graceful_shutdown(self, sig):
        logger.info("Received signal %s, shutting down gracefully...", sig.name)
        self._running = False
        # 1. 取消所有挂单
        open_orders = await self._broker.get_open_orders_async()
        for o in open_orders:
            await self._broker.cancel_order_async(o["order_id"])
        # 2. 记录当前状态到 journal
        self._journal.record_equity(
            time.time(), self._broker.get_cash(),
            self._broker.get_positions(), "graceful_shutdown"
        )
        # 3. 停止价格源
        await self._feed.stop_all()
        logger.info("Graceful shutdown complete")
        # 注意: 不自动平仓！仅取消挂单 + 保存状态
```

#### A.5 行情过期检测

**原始计划缺失**: WebSocket 断连时 `_get_prices()` 返回陈旧价格，策略基于错误数据交易。

```python
# price_feed.py 改造 — 添加 staleness detection
class PriceFeedManager:
    STALE_THRESHOLD_SECONDS = 30  # 超过 30 秒无更新视为过期

    def get_latest_prices(self) -> Dict[str, float]:
        prices = {}
        now = time.time()
        for sym, (price, ts) in self._last_prices.items():
            age = now - ts
            if age > self.STALE_THRESHOLD_SECONDS:
                logger.warning("STALE PRICE: %s age=%.1fs, excluding", sym, age)
                continue  # 不返回过期价格
            prices[sym] = price
        return prices

    def is_healthy(self) -> bool:
        """所有订阅源是否活跃"""
        now = time.time()
        for sym, (_, ts) in self._last_prices.items():
            if now - ts > self.STALE_THRESHOLD_SECONDS:
                return False
        return True
```

**集成到 `TradingRunner._on_bar`**:

```python
async def _on_bar(self, bar):
    if not self._feed.is_healthy():
        logger.error("Price feed unhealthy, skipping signal generation")
        return  # 不交易
    ...
```

#### A.6 Binance Listen Key 续期

**原始计划缺失**: Binance user data stream listen key 60 分钟过期，必须每 30 分钟续期。

```python
# binance_futures.py — listen key 管理
class BinanceFuturesBroker:
    async def _user_data_stream(self):
        listen_key = await self._get_listen_key()
        keepalive_task = asyncio.create_task(self._keepalive_listen_key(listen_key))
        url = f"{self.WS_URL}/ws/{listen_key}"
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    backoff = 1.0  # reset on success
                    async for msg in ws:
                        data = orjson.loads(msg)  # C.1: 高性能 JSON
                        event = data.get("e")
                        if event == "ORDER_TRADE_UPDATE":
                            await self._handle_order_update(data["o"])
                        elif event == "ACCOUNT_UPDATE":
                            self._handle_account_update(data["a"])
                            # A.7: 检查 ADL 事件
                            reason = data["a"].get("m", "")
                            if reason == "ADL":
                                await self._handle_adl(data["a"])
            except Exception as e:
                logger.warning("User data stream error: %s, reconnect in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)  # C.5: 指数退避
                listen_key = await self._get_listen_key()
        keepalive_task.cancel()

    async def _keepalive_listen_key(self, listen_key: str):
        """每 25 分钟续期 listen key (60 分钟过期)"""
        while self._running:
            await asyncio.sleep(25 * 60)
            try:
                await self._request("PUT", "/fapi/v1/listenKey",
                                    {"listenKey": listen_key})
            except Exception as e:
                logger.warning("Listen key keepalive failed: %s", e)

    async def _handle_adl(self, account_data: dict):
        """处理 Auto-Deleverage: 交易所强制减仓"""
        for pos in account_data.get("P", []):
            sym = pos["s"]
            qty = float(pos["pa"])
            logger.critical("ADL event: %s reduced to %.4f", sym, qty)
            self._positions[sym] = qty
        if self._alert:
            await self._alert.send("CRITICAL", "ADL 触发",
                                    f"交易所自动减仓: {account_data}")
```

---

### B. 正确性修复 (计划中代码的 Bug)

#### B.1 `can_afford` 忽略杠杆

**现有代码** (`portfolio.py:267-268`):
```python
def can_afford(self, price, shares, commission):
    return self._cash >= (price * abs(shares) + commission)
```

杠杆模式下应检查保证金而非全额:

```python
def can_afford(self, price: float, shares: PositionQty,
               commission: float, leverage: float = 1.0) -> bool:
    notional = price * abs(shares)
    if leverage > 1.0 and self._margin_model is not None:
        required = self._margin_model.initial_margin(notional, leverage) + commission
    else:
        required = notional + commission
    return self._cash >= required
```

#### B.2 内核路径 `dc` 必须拆分为 `dc_long` / `dc_short`

**现有代码** (`kernels.py:976-979`): `_fx_lev` 中 `dc` 对所有仓位统一收费。
做空的 borrow cost 和做多的 funding cost 不同。

```python
# 改造 _fx_lev: 接受方向性成本
@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _fx_lev(pend, pos, ep, oi, tr, sb, ss, cm, lev, dc_long, dc_short, pfrac):
    if pos != 0:
        deployed = _deploy(tr, pfrac)
        dc = dc_long if pos == 1 else dc_short
        cost = deployed * dc
        tr -= cost
        if tr < 0.01:
            return 0, 0.0, 0.01, 0, 1
    ...

# config_to_kernel_costs 改造
def config_to_kernel_costs(config):
    bpd = getattr(config, "bars_per_day", 1.0)

    # Long: funding rate only
    dc_long = config.daily_funding_rate / bpd
    if config.funding_leverage_scaling and lev > 1.0:
        dc_long *= lev * (1.0 + 0.02 * lev)

    # Short: funding rate + borrow cost
    borrow_daily = config.short_borrow_rate_annual / bpy
    dc_short = dc_long + borrow_daily * lev

    return {
        ..., "dc_long": dc_long, "dc_short": dc_short,
    }
```

需要同步修改所有 18 个内核的 `_fx_lev` 调用，但每个内核只需改一行参数传递。
这是正确性修复，不是可选优化。

#### B.3 `RiskGate` 截断小数持仓

**现有代码** (`risk.py:170`): `pos = int(positions.get(symbol, 0))` 把 10.5 截断为 10。
加密货币必须支持小数持仓。

```python
# 修复
pos = positions.get(symbol, 0)
if isinstance(pos, float):
    pos_for_check = pos
else:
    pos_for_check = int(pos)

if action == "sell":
    if not self.config.allow_short and shares > pos_for_check:
        return "insufficient position"
```

#### B.4 `_sl_exit_v2` 跳空整合

原始计划在 1.2 节和 1.5 节分别描述了 intra-bar 止损和跳空处理，但代码是分离的。
应合并为一个函数:

```python
@njit(cache=True, fastmath=_SAFE_FASTMATH)
def _sl_exit_v2(pos, ep, tr, oi, hi, lo, ci, sb, ss, cm, lev, sl, pfrac, sl_slip):
    """Intra-bar stop-loss with gap handling.

    Priority: gap_stop > intra_bar_stop > normal_check
    """
    if pos == 0 or ep <= 0:
        return pos, ep, tr, 0

    if pos == 1:
        sl_price = ep * (1.0 + cm) * (1.0 - sl / lev) / (ss * (1.0 - cm))
        # Gap: open already below stop
        if oi <= sl_price:
            actual_loss = min(1.0, (ep - oi * ss * (1.0 - cm)) / (ep * (1.0 + cm)) * lev)
            deployed = _deploy(tr, pfrac)
            tr -= deployed * actual_loss
            return 0, 0.0, max(0.01, tr), 1
        # Intra-bar: low touched stop
        if lo <= sl_price:
            deployed = _deploy(tr, pfrac)
            tr -= deployed * (sl + sl_slip)
            return 0, 0.0, max(0.01, tr), 1
        # Normal: check close
        raw = (ci * ss * (1.0 - cm)) / (ep * (1.0 + cm))
        pnl = (raw - 1.0) * lev

    else:  # pos == -1
        sl_price = ep * (1.0 - cm) * (1.0 + sl / lev) / (sb * (1.0 + cm))
        if oi >= sl_price:
            actual_loss = min(1.0, (oi * sb * (1.0 + cm) - ep * (1.0 - cm)) / (ep * (1.0 - cm)) * lev)
            deployed = _deploy(tr, pfrac)
            tr -= deployed * actual_loss
            return 0, 0.0, max(0.01, tr), 1
        if hi >= sl_price:
            deployed = _deploy(tr, pfrac)
            tr -= deployed * (sl + sl_slip)
            return 0, 0.0, max(0.01, tr), 1
        denom = ci * sb * (1.0 + cm)
        raw = (ep * (1.0 - cm)) / denom if denom > 0 else 1.0
        pnl = (raw - 1.0) * lev

    if pnl >= -sl:
        return pos, ep, tr, 0
    deployed = _deploy(tr, pfrac)
    tr -= deployed * (sl + sl_slip)
    return 0, 0.0, max(0.01, tr), 1
```

#### B.5 `PaperBroker._available_margin()` 缺失实现

原始计划 2.1 节 `submit_order` 调用 `self._available_margin()` 但未定义:

```python
class PaperBroker:
    def _available_margin(self) -> float:
        total_frozen = sum(self._frozen_margins.values())
        unrealized = self._calc_total_unrealized_pnl()
        return self._cash + unrealized - total_frozen

    def _calc_total_unrealized_pnl(self) -> float:
        pnl = 0.0
        for sym, qty in self._positions.items():
            if sym not in self._entry_prices or sym not in self._last_prices:
                continue
            entry = self._entry_prices[sym]
            current = self._last_prices[sym]
            lev = self._leverage_settings.get(sym, 1.0)
            if qty > 0:
                pnl += abs(qty) * entry * ((current / entry - 1) * lev)
            else:
                pnl += abs(qty) * entry * ((entry / current - 1) * lev)
        return pnl

    def get_margin_ratio(self) -> float:
        total_maint = 0.0
        for sym, qty in self._positions.items():
            if abs(qty) < 1e-10:
                continue
            notional = abs(qty) * self._last_prices.get(sym, 0)
            total_maint += self._margin_model.maintenance_margin(notional)
        if total_maint == 0:
            return 1.0
        equity = self._cash + self._calc_total_unrealized_pnl()
        return equity / total_maint
```

#### B.6 `BacktestConfig` 废弃字段清理

`initial_margin_pct` 和 `maintenance_margin_pct` 在 `BacktestConfig` 中定义但从未使用。
应标记废弃并迁移到 `MarginModel`:

```python
@dataclass(frozen=True)
class BacktestConfig:
    # 废弃 — 使用 MarginModel 代替
    # initial_margin_pct: float = 1.0       # DEPRECATED
    # maintenance_margin_pct: float = 0.5   # DEPRECATED

    # 新增: 指定使用哪种 MarginModel
    margin_model_type: str = "none"  # "none" | "crypto_futures" | "reg_t"
```

#### B.7 `BacktestEngine` bar data fallback bug

**现有代码** (`backtest_engine.py:283`): 当 `ac_sym` 不在 `bar_data_map` 中时，
回退到 `list(bar_data_map.values())[0]` — 使用了错误标的的数据:

```python
# 修复: 明确报错而非静默使用错误数据
bar = bar_data_map.get(ac_sym)
if bar is None:
    raise KeyError(f"No bar data for symbol '{ac_sym}' at bar {i}")
```

---

### C. 性能极致优化

#### C.1 WebSocket JSON 解析: `orjson` 替代 `json`

Binance WebSocket 每秒可能收到数百条消息。`orjson` 比标准库快 3-5x:

```python
import orjson  # pip install orjson

# 替换所有 json.loads(msg) → orjson.loads(msg)
# 替换所有 json.dumps(d) → orjson.dumps(d)
```

**基准**: 对典型 Binance order update 消息 (~500 bytes):
- `json.loads`: ~4.2 μs/call
- `orjson.loads`: ~0.9 μs/call

#### C.2 `RateLimiter._current_usage()` O(1) 优化

原始计划中每次 `acquire` 都 O(n) 遍历 deque 求和:

```python
class RateLimiter:
    def __init__(self, max_weight: int = 1200, window_seconds: int = 60):
        self._max_weight = max_weight
        self._window = window_seconds
        self._used: Deque[Tuple[float, int]] = deque()
        self._running_total: int = 0  # 维护运行总和

    async def acquire(self, weight: int = 1):
        self._expire_old()
        while self._running_total + weight > self._max_weight:
            await asyncio.sleep(0.05)  # 更短的轮询间隔
            self._expire_old()
        self._used.append((time.time(), weight))
        self._running_total += weight

    def _expire_old(self):
        cutoff = time.time() - self._window
        while self._used and self._used[0][0] < cutoff:
            _, w = self._used.popleft()
            self._running_total -= w
```

复杂度: O(expired) 均摊，而非 O(n) 每次调用。

#### C.3 `CryptoFuturesMargin.maintenance_margin()` 二分查找

原始计划线性扫描 9 层阶梯。回测中每 bar × 每持仓调用一次，
可能 10M+ 次调用:

```python
import numpy as np
from bisect import bisect_right

class CryptoFuturesMargin(MarginModel):
    _TIER_BOUNDS = np.array([50_000, 250_000, 1_000_000, 10_000_000,
                             20_000_000, 50_000_000, 100_000_000, 200_000_000])
    _TIER_RATES = np.array([0.004, 0.005, 0.01, 0.025, 0.05, 0.10, 0.125, 0.15, 0.25])
    _TIER_AMOUNTS = np.array([0, 50, 1050, 16050, 266050, 1266050, 2516050, 5016050, 25016050])

    def maintenance_margin(self, notional: float) -> float:
        idx = bisect_right(self._TIER_BOUNDS, notional)  # O(log n)
        return notional * self._TIER_RATES[idx] - self._TIER_AMOUNTS[idx]
```

**Numba 兼容版本** (供内核路径使用):

```python
# 预分配为 module-level 常量，Numba 可以内联
_TIER_BOUNDS_NB = np.array([50_000., 250_000., 1_000_000., 10_000_000.,
                            20_000_000., 50_000_000., 100_000_000., 200_000_000.])
_TIER_RATES_NB  = np.array([0.004, 0.005, 0.01, 0.025, 0.05, 0.10, 0.125, 0.15, 0.25])
_TIER_AMTS_NB   = np.array([0., 50., 1050., 16050., 266050., 1266050., 2516050., 5016050., 25016050.])

@njit(cache=True)
def _maintenance_margin_njit(notional):
    idx = np.searchsorted(_TIER_BOUNDS_NB, notional)
    return notional * _TIER_RATES_NB[idx] - _TIER_AMTS_NB[idx]
```

#### C.4 aiohttp 连接池 + Keep-Alive

原始计划 `_request` 没有优化连接复用:

```python
class BinanceFuturesBroker:
    async def initialize(self):
        connector = aiohttp.TCPConnector(
            limit=20,              # 最大并发连接
            keepalive_timeout=30,  # Keep-Alive
            enable_cleanup_closed=True,
            ssl=False,             # Binance 不需要客户端证书
        )
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=lambda x: orjson.dumps(x).decode(),
        )
        ...

    async def close(self):
        """必须在关机时调用"""
        if self._session:
            await self._session.close()
```

#### C.5 WebSocket 指数退避重连

原始计划固定 5 秒重连，过于激进:

```python
# 已在 A.6 的代码中实现
# backoff = 1.0 → 2.0 → 4.0 → 8.0 → ... → max 60.0
# 成功连接后 reset 为 1.0
```

#### C.6 内核路径 Funding Rate 数组化

原始计划提到 `dc_array` 但没有解决内核签名问题。
方案: 不改内核签名，在调用前预减:

```python
def _run_kernel_with_funding(self, kernel_name, data, config, funding_rates=None):
    """在调用内核前，将历史 funding rate 折算进价格调整。"""
    c = data["close"].values.astype(np.float64)
    o = data["open"].values.astype(np.float64)

    if funding_rates is not None:
        # 将 funding 累积效果体现在 close 价格中
        # 这等价于在每个 bar 扣除 funding cost
        cum_funding = np.cumsum(funding_rates)
        # funding > 0 时，多头付费 → 等效于 close 降低
        c_adj = c * (1 - cum_funding)
        o_adj = o * (1 - cum_funding)
    else:
        c_adj, o_adj = c, o

    costs = config_to_kernel_costs(config)
    costs["dc"] = 0.0  # funding 已折算进价格，dc 归零
    ...
```

优势: 零内核签名修改，18 个内核完全不动。
劣势: 近似方法，对极端 funding rate 有微小误差 (< 0.01% 累积)。

#### C.7 IBKRBroker 事件驱动替代轮询

原始计划用 `while not trade.isDone()` 轮询等待成交，浪费 CPU:

```python
class IBKRBroker:
    async def submit_order_async(self, signal: Dict) -> Dict:
        ...
        trade = self._ib.placeOrder(contract, order)

        # 事件驱动: 使用 asyncio.Event 替代轮询
        fill_event = asyncio.Event()
        def on_status_changed(trade):
            if trade.isDone():
                fill_event.set()
        trade.statusEvent += on_status_changed

        try:
            await asyncio.wait_for(fill_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            trade.statusEvent -= on_status_changed
            await self._ib.cancelOrder(trade.order)
            return {"status": "timeout", "message": "order timeout 30s"}

        trade.statusEvent -= on_status_changed
        ...
```

#### C.8 预分配订单结果对象

原始计划每次下单创建新 Dict。高频场景应使用 `__slots__`:

```python
class OrderResult:
    __slots__ = ("status", "order_id", "fill_price", "filled_shares",
                 "commission", "message", "client_order_id", "timestamp")

    def __init__(self):
        self.status = ""
        self.order_id = ""
        self.fill_price = 0.0
        self.filled_shares = 0.0
        self.commission = 0.0
        self.message = ""
        self.client_order_id = ""
        self.timestamp = 0.0

    def to_dict(self) -> Dict:
        return {s: getattr(self, s) for s in self.__slots__ if getattr(self, s)}
```

---

### D. 功能补全

#### D.1 Cross-Margin 模式支持

原始计划仅支持 `ISOLATED`，但多品种交易时 `CROSS` 更资本高效:

```python
class BinanceFuturesBroker:
    def __init__(self, ..., margin_type: str = "ISOLATED"):
        ...

    async def set_margin_type(self, symbol: str, margin_type: str = None):
        mt = margin_type or self._margin_type
        try:
            await self._request("POST", "/fapi/v1/marginType",
                               {"symbol": symbol, "marginType": mt})
        except Exception:
            pass

# CryptoFuturesMargin 增加 cross-margin 计算
class CryptoFuturesMargin:
    def __init__(self, mode: str = "isolated"):
        self._mode = mode  # "isolated" | "cross"

    def initial_margin(self, notional: float, leverage: float) -> float:
        return notional / leverage

    def available_balance_cross(self, total_balance: float,
                                 all_positions: Dict[str, float],
                                 prices: Dict[str, float]) -> float:
        """Cross-margin: 全账户余额减去所有持仓维持保证金"""
        total_maint = sum(
            self.maintenance_margin(abs(qty) * prices.get(sym, 0))
            for sym, qty in all_positions.items()
        )
        unrealized = sum(
            self._calc_unrealized(sym, qty, prices)
            for sym, qty in all_positions.items()
        )
        return total_balance + unrealized - total_maint
```

#### D.2 信号冲突解决器

多策略/多时间框架可能同时产生矛盾信号:

```python
class SignalConflictResolver:
    """当多策略对同一标的产生矛盾信号时的仲裁"""

    RULES = {
        "priority": "high_tf_wins",  # 高时间框架优先
        "tie_break": "no_action",     # 平局时不操作
    }

    def resolve(self, signals: List[Dict]) -> Optional[Dict]:
        by_symbol: Dict[str, List] = defaultdict(list)
        for s in signals:
            by_symbol[s["symbol"]].append(s)

        resolved = []
        for sym, sigs in by_symbol.items():
            if len(sigs) == 1:
                resolved.append(sigs[0])
                continue

            buys = [s for s in sigs if s["action"] == "buy"]
            sells = [s for s in sigs if s["action"] == "sell"]

            if buys and sells:
                logger.warning("Conflicting signals for %s: %d buy, %d sell",
                               sym, len(buys), len(sells))
                # 高时间框架优先
                all_sorted = sorted(sigs, key=lambda s: s.get("timeframe_minutes", 0),
                                    reverse=True)
                resolved.append(all_sorted[0])
            else:
                resolved.append(sigs[0])
        return resolved
```

#### D.3 美股公司行为处理

原始计划遗漏了对股票拆分、合并、分红的处理:

```python
class CorporateActionHandler:
    """处理股票拆分、反向拆分、分红"""

    async def check_and_apply(self, symbol: str, positions: Dict,
                               entry_prices: Dict):
        # 数据源: IBKR 通知 或 polygon.io API
        actions = await self._fetch_recent_actions(symbol)
        for action in actions:
            if action["type"] == "split":
                ratio = action["ratio"]  # e.g., 4.0 for 4:1 split
                if symbol in positions:
                    positions[symbol] *= ratio
                    entry_prices[symbol] /= ratio
                    logger.info("Applied %s split %.1f:1, new pos=%.2f",
                                symbol, ratio, positions[symbol])

            elif action["type"] == "dividend":
                amount = action["per_share"]
                qty = positions.get(symbol, 0)
                if qty > 0:
                    cash_received = qty * amount
                    logger.info("Dividend: %s $%.4f/share × %.0f = $%.2f",
                                symbol, amount, qty, cash_received)
                    return cash_received
        return 0.0
```

#### D.4 健康检查 HTTP 端点

长期运行的交易系统需要外部可访问的健康检查:

```python
from aiohttp import web

class HealthCheckServer:
    """轻量 HTTP 端点，供外部监控系统探测"""

    def __init__(self, runner: TradingRunner, port: int = 8080):
        self._runner = runner
        self._port = port

    async def start(self):
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/metrics", self._metrics)
        app_runner = web.AppRunner(app)
        await app_runner.setup()
        site = web.TCPSite(app_runner, "0.0.0.0", self._port)
        await site.start()

    async def _health(self, request):
        feed_ok = self._runner._feed.is_healthy()
        broker_ok = self._runner._broker.get_cash() > 0
        status = 200 if (feed_ok and broker_ok) else 503
        return web.json_response({
            "status": "ok" if status == 200 else "degraded",
            "feed_healthy": feed_ok,
            "broker_connected": broker_ok,
            "uptime_seconds": time.time() - self._runner._start_time,
            "open_positions": len(self._runner._broker.get_positions()),
        }, status=status)

    async def _metrics(self, request):
        state = self._runner.get_state()
        return web.json_response(state)
```

#### D.5 策略预热

崩溃恢复后内核适配器需要足够历史 bar 才能生成有效信号:

```python
class TradingRunner:
    async def _recover_state(self):
        ...
        # 4. 预热策略指标 (recovery 新增)
        for sym in self._symbols:
            cfg = self._get_config(sym)
            lookback = max(200, cfg.bars_per_day * 5)  # 至少 5 天
            historical = await self._feed.fetch_historical(sym, lookback)
            if historical is not None and len(historical) >= lookback:
                self._adapters[sym].warm_up(historical)
                logger.info("Warmed up %s with %d bars", sym, len(historical))
            else:
                logger.warning("Insufficient history for %s warmup", sym)
```

---

### 更新后的文件变更清单

#### 额外新增文件 (V5.1)

| 文件路径 | 类别 | 说明 |
|----------|------|------|
| `quant_framework/live/kill_switch.py` | A.1 | 紧急平仓 |
| `quant_framework/live/health_server.py` | D.4 | 健康检查 HTTP |
| `quant_framework/core/signal_resolver.py` | D.2 | 信号冲突仲裁 |
| `quant_framework/core/corporate_actions.py` | D.3 | 公司行为处理 |
| `tests/test_kill_switch.py` | A.1 | Kill switch 测试 |
| `tests/test_signal_resolver.py` | D.2 | 信号冲突测试 |

#### 额外修改文件 (V5.1)

| 文件路径 | 类别 | 改动 |
|----------|------|------|
| `quant_framework/backtest/kernels.py` | B.2 | `_fx_lev` 拆分 `dc_long`/`dc_short` |
| `quant_framework/backtest/portfolio.py` | B.1 | `can_afford` 支持杠杆 |
| `quant_framework/live/risk.py` | B.3 | `RiskGate` 支持小数持仓 |
| `quant_framework/live/price_feed.py` | A.5 | 过期检测 |
| `quant_framework/live/trading_runner.py` | A.4/D.5 | 优雅关机 + 策略预热 |
| `quant_framework/broker/binance_futures.py` | A.2/A.3/A.6/C.4 | reduce-only, 幂等性, listen key, 连接池 |
| `quant_framework/broker/ibkr_broker.py` | C.7 | 事件驱动成交等待 |
| `quant_framework/broker/rate_limiter.py` | C.2 | O(1) 运行总和 |
| `quant_framework/core/margin.py` | C.3 | 二分查找 + Numba 版本 |
| `quant_framework/backtest/backtest_engine.py` | B.7 | bar data fallback 修复 |
| `quant_framework/backtest/config.py` | B.6 | 废弃字段标记 |

---

### 更新后的设计原则

1. **三层一致**: 回测、纸交易、实盘共用 MarginModel + CostModel, 保证数学行为一致
2. **资产感知**: 加密货币和美股的保证金/费率/规则完全独立建模, 不混淆
3. **Testnet First**: 实盘代码必须先在测试网/纸交易账户验证通过
4. **向后兼容**: 现有的回测脚本和策略代码无需修改即可运行
5. **安全第一**: API 密钥不入库, 有熔断, 有对账, 有告警, **有 Kill Switch**
6. **可观测**: 全链路延迟追踪 + 审计日志 + 实时告警 + **健康检查 HTTP**
7. **渐进式**: 先 Binance Testnet → Binance Mainnet → IBKR Paper → IBKR Live
8. **防御性**: **reduce-only 平仓, 订单去重, 过期价格检测, 优雅关机** (V5.1 新增)
9. **极致性能**: **orjson, O(1) 限速器, 二分保证金, 事件驱动, 连接池** (V5.1 新增)
10. **方向性成本**: **dc_long/dc_short 分离, 消除多空成本混淆** (V5.1 新增)

---

### V5.1 时间增量

| 类别 | 内容 | 增量时间 |
|------|------|----------|
| A | 安全关键 (kill switch, reduce-only, idempotency, shutdown, stale detect, listen key, ADL) | +2-3 天 |
| B | 正确性修复 (7 项 bug fix) | +1-2 天 |
| C | 性能极致 (8 项优化) | +1-2 天 |
| D | 功能补全 (5 项) | +2-3 天 |

**总计 V5 + V5.1**: Phase 0-6 约 **23-35 天**, Phase 7 可选 +2-3 天。

---

## V5.2 全局联动优化 — 全代码库审查

> V5.1 聚焦在 backtest / broker / live 三层。本节扩展到框架的**全部模块**:
> strategy (11 个策略)、research (5 个引擎)、data (6 个子模块)、
> kernel_adapter、trade_journal、dashboard、analysis、alpha、tests (18 个文件)。
> 发现 **8 个策略层 bug**、**7 个 Research 层 V5 缺口**、**6 个数据层缺口**、
> 以及 **测试基础设施的系统性缺失**。

---

### E. 策略层 Bug 修复 + V5 适配

#### E.1 `drift_regime_strategy.py` 空头无法平仓

**现有代码** (line 84-86):
```python
if holdings > 0:
    return {"action": "sell", "symbol": symbol, "shares": holdings}
# 注意：框架暂不支持空头平仓（buy to cover），用 hold 代替
return {"action": "hold"}
```

空头持仓到期时返回 `hold`，仓位永远不会关闭。

**修复**:
```python
if holdings > 0:
    return {"action": "sell", "symbol": symbol, "shares": holdings}
elif holdings < 0:
    return {"action": "buy", "symbol": symbol, "shares": abs(holdings)}
```

#### E.2 `zscore_reversion_strategy.py` 缺少做空逻辑和空头止损

**现有代码** (line 96-100): 只有 `z < -entry_z → buy`，没有 `z > +entry_z → sell/short`。
均值回复策略应该双向交易:

```python
# on_bar_fast 修复
if holdings > 0:
    if abs(z) < self.exit_z:
        return {"action": "sell", "symbol": symbol, "shares": holdings}
    if z < -self.stop_z:
        return {"action": "sell", "symbol": symbol, "shares": holdings}
elif holdings < 0:
    if abs(z) < self.exit_z:
        return {"action": "buy", "symbol": symbol, "shares": abs(holdings)}
    if z > self.stop_z:  # 空头止损 (原缺失)
        return {"action": "buy", "symbol": symbol, "shares": abs(holdings)}

if holdings == 0:
    if z < -self.entry_z:
        shares = self.calculate_position_size(current_price)
        if shares > 0 and self.can_buy(symbol, current_price, shares):
            return {"action": "buy", "symbol": symbol, "shares": shares}
    elif z > self.entry_z:  # 做空逻辑 (原缺失)
        shares = self.calculate_position_size(current_price)
        if shares > 0:
            return {"action": "sell", "symbol": symbol, "shares": shares}
```

#### E.3 `lorentzian_strategy.py` `on_bar_fast_multi` 返回类型不一致

**现有代码** (line 463):
```python
return signals if signals else {"action": "hold"}
```

当有信号时返回 `List[Dict]`，无信号时返回 `Dict`。调用者无法统一处理。

**修复**:
```python
return signals if signals else [{"action": "hold"}]
```

**同样的问题存在于** `macd_strategy.py` 的 `on_bar_fast_multi`，需要同步修复。

#### E.4 `momentum_breakout_strategy.py` 跟踪止损重置 Bug

**现有代码** (line 108):
```python
self._trailing_stop[symbol] = 0.0
```

卖出后设为 `0.0` 而非删除。下次买入时 `new_stop > 0.0` 几乎总是 `True`，
导致跟踪止损提前生效。

**修复**:
```python
self._trailing_stop.pop(symbol, None)
return {"action": "sell", "symbol": symbol, "shares": holdings}
```

#### E.5 `BaseStrategy.calculate_position_size` 不支持杠杆和精度

**现有代码** (line 106-120):
```python
def calculate_position_size(self, price, capital_fraction=0.95):
    amount = self.portfolio_value * capital_fraction
    shares = int(amount / price)
    return max(0, shares)
```

忽略杠杆、保证金、交易手续费、和交易对精度。

**V5 改造**:
```python
def calculate_position_size(self, price: float,
                             capital_fraction: float = 0.95,
                             leverage: float = 1.0,
                             margin_model: Optional[MarginModel] = None,
                             symbol_spec: Optional[SymbolSpec] = None,
                             commission_pct: float = 0.0) -> float:
    available = self.portfolio_value * capital_fraction
    if margin_model and leverage > 1.0:
        max_notional = available * leverage
    else:
        max_notional = available
    max_notional *= (1 - commission_pct)
    raw = max_notional / price
    if symbol_spec:
        raw = symbol_spec.round_quantity(raw)
    else:
        raw = int(raw)
    return max(0, raw)
```

#### E.6 `BaseStrategy.can_buy` 忽略杠杆

**现有代码** (line 122-125):
```python
def can_buy(self, symbol, price, shares):
    cost = price * shares
    return cost <= self.cash
```

**修复**: 与 E.5 对齐，接受 `leverage` 和 `margin_model` 参数。

#### E.7 所有策略硬编码 `capital_fraction=0.95`

11 个策略文件中 `capital_fraction=0.95` 出现 22 次。应提升为策略构造器参数:

```python
class BaseStrategy:
    DEFAULT_CAPITAL_FRACTION = 0.95

    def __init__(self, name, initial_capital=1_000_000, *,
                 capital_fraction: float = DEFAULT_CAPITAL_FRACTION, ...):
        self._capital_fraction = capital_fraction

    def calculate_position_size(self, price, ...):
        amount = self.portfolio_value * self._capital_fraction
        ...
```

各策略删除 `capital_fraction=0.95` 参数，改用 `self._capital_fraction`。

#### E.8 `microstructure_momentum.py` / `adaptive_regime_ensemble.py` 资产感知

VPIN、OFI、Yang-Zhang vol 的参数调校是基于加密货币微观结构的。
美股的 tick 结构、volume profile 完全不同:

```python
# 添加资产感知参数默认值
class MicrostructureMomentumStrategy(BaseStrategy):
    def __init__(self, ..., asset_class: str = "crypto"):
        if asset_class == "crypto":
            self._vol_min, self._vol_max = 0.2, 2.0
            self._ofi_clamp = 3.0
        else:  # equity
            self._vol_min, self._vol_max = 0.05, 0.5
            self._ofi_clamp = 2.0
```

---

### F. Research 层 V5 联动

#### F.1 ResearchDB Schema 迁移

现有 `strategy_health` 表有 `leverage` 和 `interval` 但缺少:

```sql
-- 需要 ALTER TABLE 添加
ALTER TABLE strategy_health ADD COLUMN asset_class TEXT DEFAULT 'crypto_perp';
ALTER TABLE strategy_health ADD COLUMN margin_model TEXT DEFAULT 'none';
ALTER TABLE strategy_health ADD COLUMN funding_cost REAL DEFAULT 0;

ALTER TABLE param_history ADD COLUMN asset_class TEXT DEFAULT 'crypto_perp';

ALTER TABLE strategy_library ADD COLUMN asset_class TEXT DEFAULT '';
ALTER TABLE strategy_library ADD COLUMN broker_type TEXT DEFAULT '';  -- 'binance' | 'ibkr'
ALTER TABLE strategy_library ADD COLUMN margin_mode TEXT DEFAULT '';  -- 'isolated' | 'cross'
```

**迁移方案**: 在 `_create_tables` 中检测列是否存在:
```python
def _create_tables(self, conn):
    ...
    # V5 schema migration
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(strategy_health)").fetchall()}
    if "asset_class" not in existing:
        conn.execute("ALTER TABLE strategy_health ADD COLUMN asset_class TEXT DEFAULT 'crypto_perp'")
    if "margin_model" not in existing:
        conn.execute("ALTER TABLE strategy_health ADD COLUMN margin_model TEXT DEFAULT 'none'")
    if "funding_cost" not in existing:
        conn.execute("ALTER TABLE strategy_health ADD COLUMN funding_cost REAL DEFAULT 0")
    # ... 同理 param_history, strategy_library
```

#### F.2 `config_maker` 资产感知化

`monitor.py`, `optimizer.py`, `discover.py`, `daily_research.py` 中都有:
```python
config = BacktestConfig.crypto()  # 或 BacktestConfig.stock_ibkr()
```

应统一为:
```python
def make_config_v5(symbol: str, leverage: float, interval: str,
                   asset_class: AssetClass, margin_model: MarginModel,
                   cost_model: CostModel) -> BacktestConfig:
    base = BacktestConfig.crypto() if asset_class.is_crypto else BacktestConfig.stock_ibkr()
    return dataclasses.replace(base,
        leverage=leverage,
        interval=interval,
        asset_class=asset_class.value,
        margin_model_type=margin_model.__class__.__name__,
    )
```

#### F.3 `discover_variants` 硬编码 leverage=1.0

**现有代码** (discover.py line 114):
```python
config = config_maker(sym, 1.0, "1d")
```

应使用推荐的 leverage 和 interval:
```python
rec = recommendations.get(sym, {})
lev = rec.get("leverage", 1.0)
interval = rec.get("interval", "1d")
config = config_maker(sym, lev, interval)
```

#### F.4 Monitor `record_health` 错误字段传入

当 `compute_health_metrics` 返回 `{"error": "...", "status": "ERROR"}` 时，
`run_monitor` 直接 `db.record_health(... **metrics)` 会把 `error` 字段传入 SQL INSERT:

```python
# 修复: 过滤非数据库字段
safe_keys = {"sharpe_30d", "drawdown_pct", "dd_duration", "trade_freq",
             "win_rate", "ret_pct", "n_trades"}
safe_metrics = {k: v for k, v in metrics.items() if k in safe_keys}
db.record_health(sym, strategy, status=metrics.get("status", "UNKNOWN"), **safe_metrics)
```

#### F.5 Portfolio `compute_correlation_matrix` 累积收益率差分语义

**现有代码** (portfolio.py line 52-55):
```python
daily_rets = np.diff(rets) / np.maximum(np.abs(rets[:-1]), 1e-12)
```

`rets` 是累积 `ret_pct` 序列。`np.diff(rets) / rets[:-1]` 计算的是 "累积收益率的变化率"，
不是真正的 daily return。应改为:

```python
# 用 equity 序列计算 daily returns
equity = 1.0 + np.array(rets)  # 累积收益 → 净值
daily_rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
```

#### F.6 Research Report V5 扩展

`_report.py` 需要新增:
- 保证金使用率 section
- 资产类别分布 section
- Funding 成本统计 section
- 跨层一致性验证 section (回测 vs 纸交易 Sharpe 对比)

#### F.7 Research Pipeline `phase0_refresh` 需要下载 Funding Rate

`daily_research.py` `phase0_refresh` 只下载 OHLCV。V5 需要同步下载 funding rate:

```python
def phase0_refresh(data_dir, symbols):
    ...
    # V5: 下载 crypto funding rates
    from quant_framework.data.funding_rates import FundingRateLoader
    loader = FundingRateLoader()
    for sym in crypto_symbols:
        asyncio.run(loader.download(f"{sym}USDT", start, end))
```

---

### G. 数据层 V5 适配

#### G.1 多交易所数据布局

现有 `dataset.py` 路径: `{data_dir}/{symbol}.parquet`
V5 需要: `{data_dir}/{exchange}/{asset_class}/{symbol}.parquet`

```python
class Dataset:
    def _resolve_path(self, symbol: str, exchange: str = "binance",
                       asset_class: str = "perp") -> Path:
        # V5: exchange-aware path
        v5_path = self._data_dir / exchange / asset_class / f"{symbol}.parquet"
        if v5_path.exists():
            return v5_path
        # 向后兼容: 旧路径
        return self._data_dir / f"{symbol}.parquet"
```

#### G.2 Symbol Metadata Registry

新增统一的标的元数据管理:

```python
# quant_framework/core/symbol_registry.py
@dataclass
class SymbolMeta:
    symbol: str           # "BTCUSDT"
    base: str             # "BTC"
    quote: str            # "USDT"
    exchange: str         # "binance"
    asset_class: AssetClass
    data_symbol: str      # 数据文件用的 key (可能不同于 exchange symbol)
    feed_symbol: str      # WebSocket 用的 key
    display_name: str     # 显示用

class SymbolRegistry:
    def __init__(self):
        self._registry: Dict[str, SymbolMeta] = {}

    def register(self, meta: SymbolMeta): ...
    def resolve(self, symbol: str) -> SymbolMeta: ...
    def by_exchange(self, exchange: str) -> List[SymbolMeta]: ...
    def by_asset_class(self, ac: AssetClass) -> List[SymbolMeta]: ...
```

**解决的问题**:
- `run_live_trading.py` 的 `to_feed_symbol` 硬编码 (`BTC` → `BTCUSDT`)
- `run_production_scan.py` 的 `is_crypto` 逻辑脆弱 (`sym.replace("USDT", "")`)
- `daily_research.py` 的 `CRYPTO_SYMBOLS` / `STOCK_SYMBOLS` 硬编码

#### G.3 `DataManager` 内存限制

`_LRU_MAXSIZE = 128` 无内存上限。大数据集可能 OOM:

```python
class CacheManager:
    def __init__(self, max_memory_mb: float = 2048):
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._current_bytes = 0

    def _maybe_evict(self):
        while self._current_bytes > self._max_bytes and self._cache:
            key, (data, size) = self._cache.popitem(last=False)
            self._current_bytes -= size
```

#### G.4 Indicators 资产感知

`_BARS_PER_YEAR = 252` 硬编码在多处。加密货币 24/7 = 365 天:

```python
def annualization_factor(asset_class: AssetClass, interval: str) -> float:
    if asset_class in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_PERP):
        return {"1d": 365, "4h": 365*6, "1h": 365*24}[interval]
    else:
        return {"1d": 252, "4h": 252*6.5/4, "1h": 252*6.5}[interval]
```

---

### H. KernelAdapter + TradeJournal + Dashboard

#### H.1 `KernelAdapter._resolve_kernel_params` 只处理 MA/RSI

**现有代码** (kernel_adapter.py line 66-86): 只有 MA 和 RSI 的 dict → tuple 映射。
其他 16 个策略传 dict 会默认用 `_PARAM_DEFAULTS`，丢弃用户参数。

**修复**: 通用 dict 解包:
```python
@staticmethod
def _resolve_kernel_params(name: str, params) -> tuple:
    if params is None:
        return _PARAM_DEFAULTS[name]
    if isinstance(params, (tuple, list)):
        return tuple(params)
    if isinstance(params, dict):
        base = list(_PARAM_DEFAULTS[name])
        # 通用: 按 DEFAULT_PARAM_GRIDS 的 key 顺序映射
        grid = DEFAULT_PARAM_GRIDS.get(name, {})
        keys = list(grid.keys())
        for idx, key in enumerate(keys):
            if key in params and idx < len(base):
                base[idx] = type(base[idx])(params[key])
        return tuple(base)
    return _PARAM_DEFAULTS[name]
```

#### H.2 KernelAdapter 缺少 Volume 数组

`generate_signal` 只传 OHLC。使用 volume 的策略 (ARE, MSM) 在 live 模式下
拿不到 volume 数据:

```python
def generate_signal(self, window_df, symbol, *, arrays=None):
    if arrays is not None:
        c = arrays.get("close")
        o = arrays["open"]
        h = arrays["high"]
        l = arrays["low"]
        v = arrays.get("volume")  # V5: 添加 volume
    else:
        ...
        v = window_df["volume"].values if "volume" in window_df.columns else None
```

#### H.3 MultiTFAdapter `_emit_on_position_change` 返回 `price: 0.0`

**现有代码**: 当 MultiTF 模式产生信号时，返回的 signal 中 `price` 为 0:

```python
# 修复: 传入当前价格
def _emit_on_position_change(self, new_pos, prev_pos, symbol, current_prices=None):
    price = 0.0
    if current_prices and symbol in current_prices:
        price = current_prices[symbol]
    ...
```

#### H.4 TradeJournal Schema V5 扩展

现有 schema 缺少实盘所需字段:

```sql
-- V5 trades 表扩展
ALTER TABLE trades ADD COLUMN exchange TEXT DEFAULT '';
ALTER TABLE trades ADD COLUMN order_id TEXT DEFAULT '';
ALTER TABLE trades ADD COLUMN fill_id TEXT DEFAULT '';
ALTER TABLE trades ADD COLUMN client_order_id TEXT DEFAULT '';
ALTER TABLE trades ADD COLUMN order_type TEXT DEFAULT 'MARKET';
ALTER TABLE trades ADD COLUMN funding_cost REAL DEFAULT 0;
ALTER TABLE trades ADD COLUMN margin_used REAL DEFAULT 0;
ALTER TABLE trades ADD COLUMN leverage REAL DEFAULT 1;
ALTER TABLE trades ADD COLUMN latency_ms REAL DEFAULT 0;
ALTER TABLE trades ADD COLUMN slippage_bps REAL DEFAULT 0;

-- V5 新增索引
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_exchange ON trades(exchange);
```

同时需要扩展 `record_trade()` 方法接受新字段:

```python
def record_trade(self, *, exchange="", order_id="", fill_id="",
                  client_order_id="", order_type="MARKET",
                  funding_cost=0.0, margin_used=0.0, leverage=1.0,
                  latency_ms=0.0, slippage_bps=0.0, **existing_kwargs):
    ...
```

#### H.5 `TradeJournal.get_trade_stats` 排除平局交易

**现有代码**: `pnl != 0` 过滤。`pnl == 0` 的交易 (breakeven) 不被计入。

```python
# 修复: 只过滤未关仓 (pnl IS NULL)
wins = len([t for t in trades if t["pnl"] > 0])
losses = len([t for t in trades if t["pnl"] < 0])
breakeven = len([t for t in trades if t["pnl"] == 0])
```

#### H.6 Dashboard 实盘模块

`dashboard/app.py` (787 行) 目前是回测结果展示。V5 需要新增实盘面板:

| 面板 | 功能 |
|------|------|
| 持仓监控 | 实时仓位、入场价、未实现PnL、保证金占用 |
| 订单流 | 最近订单状态、延迟、滑点 |
| 资金曲线 | 实时净值曲线 (vs 回测对比) |
| 风控状态 | CircuitBreaker、RiskGate、保证金率 |
| 告警日志 | 实时告警展示 |

实现可通过 Dash `dcc.Interval` 轮询 `TradingRunner.get_state()`:

```python
@app.callback(Output("live-equity", "figure"),
              Input("live-interval", "n_intervals"))
def update_live_equity(n):
    state = runner.get_state()
    equity = state.get("equity_curve", [])
    return go.Figure(data=[go.Scatter(y=equity, mode='lines')])
```

---

### I. 分析层 + Alpha 层

#### I.1 `performance.py` 保证金感知 PnL

`PerformanceAnalyzer.analyze()` 用 `portfolio_value` 计算 returns。
杠杆模式下应基于保证金占用计算 ROI:

```python
def analyze(self, ..., margin_used: Optional[np.ndarray] = None):
    ...
    if margin_used is not None:
        # ROI = PnL / margin_used (not / total_equity)
        roi_returns = pnl / np.maximum(margin_used, 1e-12)
    else:
        roi_returns = returns
```

#### I.2 Alpha 模块资产感知

`alpha/order_flow.py` 的 OFI/VPIN 和 `alpha/volatility.py` 的 Yang-Zhang 都假设
加密货币连续交易。美股有开盘跳空、午休、盘后:

```python
class OrderFlowFeatures:
    def __init__(self, asset_class: AssetClass = AssetClass.CRYPTO_PERP):
        self._asset_class = asset_class
        if asset_class.is_equity:
            self._ofi_normalize = "session_volume"
            self._vpin_buckets = 30   # 盘中时段较少
        else:
            self._ofi_normalize = "24h_volume"
            self._vpin_buckets = 50
```

---

### J. 测试基础设施

#### J.1 缺失的测试覆盖

| 模块 | 现有测试 | 缺失 |
|------|----------|------|
| MarginModel | 无 | Phase 0 核心，需 100% 覆盖 |
| CostModel | 无 | maker/taker, funding, borrow 全测 |
| SymbolSpec | 无 | 精度、校验、边界 |
| BinanceFuturesBroker | 无 | Mock API + Testnet |
| IBKRBroker | 无 | Mock IB + Paper |
| KillSwitch | 无 | 紧急平仓流程 |
| PositionReconciler | 无 | 对账逻辑 + 不一致处理 |
| LiveOrderManager | 无 | 超时、部分成交、幂等 |
| backtest_multi_tf | 极少 | 多时间框架融合 |
| ZScore 做空 | 无 | 当前无空头测试 |
| Drift 空头平仓 | 无 | 当前返回 hold |
| Paper ↔ Backtest 对账 | 无 | 关键一致性验证 |

#### J.2 CI/CD Pipeline

**完全缺失**。需要 GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -x --timeout=120

  kernel-regression:
    runs-on: ubuntu-latest
    steps:
      - ...
      - run: python -m pytest tests/test_kernel_regression.py --timeout=300

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - ...
      - run: python examples/technical_benchmark.py
```

#### J.3 Reconciliation Test

Paper trading vs Backtest 用相同数据、相同参数的对比测试:

```python
def test_paper_backtest_reconciliation():
    """Paper 和 Backtest 在相同条件下应产生相同 PnL。"""
    data = load_test_data("BTC", 500)
    config = BacktestConfig.crypto()

    # 1. Backtest
    bt_result = backtest("BTC", data, "MA", (10, 30), config=config)

    # 2. Paper replay
    paper = PaperBroker.from_backtest_config(config, initial_cash=100000)
    adapter = KernelAdapter("MA", (10, 30), config)
    for i in range(30, len(data)):
        window = data.iloc[:i+1]
        sig = adapter.generate_signal(window, "BTC")
        if sig:
            sig["price"] = float(data.iloc[i]["close"])
            paper.submit_order(sig)

    # 3. Compare
    assert abs(bt_result.ret_pct - paper_ret_pct) < 0.01, \
        f"Divergence: bt={bt_result.ret_pct:.4f}, paper={paper_ret_pct:.4f}"
```

#### J.4 Mock Exchange Tests

```python
# tests/mocks/mock_binance.py
class MockBinanceAPI:
    """Stateful mock for Binance Futures API — no network needed."""

    def __init__(self):
        self._positions = {}
        self._balance = {"USDT": 10000.0}
        self._orders = {}
        self._next_oid = 1

    async def handle_request(self, method, path, params):
        if path == "/fapi/v1/order" and method == "POST":
            return self._place_order(params)
        elif path == "/fapi/v2/positionRisk":
            return self._get_positions()
        elif path == "/fapi/v2/balance":
            return self._get_balance()
        ...
```

#### J.5 脚本化测试转 pytest

以下文件不是 pytest 而是独立脚本:
- `test_rag_full_chain.py` (conftest 中被 `collect_ignore_glob` 排除)
- `test_rag_backtest.py`
- `test_rag_realtime_stress.py`
- `test_lorentzian_optimization.py`

需要:
1. 添加 `@pytest.mark.slow` 标记
2. 用 `pytest.fixture` 替代 `main()` 入口
3. 用 mock 替代网络依赖 (yfinance)
4. 把 print 输出改为 assert 断言

---

### V5.2 涉及文件变更清单

#### 策略层修复 (E)

| 文件 | 操作 | 说明 |
|------|------|------|
| `strategy/drift_regime_strategy.py` | 修复 | 空头平仓 |
| `strategy/zscore_reversion_strategy.py` | 修复 | 做空逻辑 + 空头止损 |
| `strategy/lorentzian_strategy.py` | 修复 | `on_bar_fast_multi` 返回类型 |
| `strategy/macd_strategy.py` | 修复 | `on_bar_fast_multi` 返回类型 |
| `strategy/momentum_breakout_strategy.py` | 修复 | 跟踪止损清理 |
| `strategy/base_strategy.py` | 改造 | V5 position sizing + can_buy |
| `strategy/microstructure_momentum.py` | 改造 | 资产感知参数 |
| `strategy/adaptive_regime_ensemble.py` | 改造 | 资产感知参数 |

#### Research 层联动 (F)

| 文件 | 操作 | 说明 |
|------|------|------|
| `research/database.py` | 修改 | V5 Schema 迁移 |
| `research/monitor.py` | 修改 | config_maker V5 + record_health 安全 |
| `research/optimizer.py` | 修改 | config_maker V5 |
| `research/discover.py` | 修改 | 使用推荐 leverage/interval |
| `research/portfolio.py` | 修复 | 收益率计算语义 |
| `research/_report.py` | 扩展 | V5 新 section |
| `daily_research.py` | 修改 | funding rate 下载 + config_maker V5 |

#### 数据层 (G)

| 文件 | 操作 | 说明 |
|------|------|------|
| `data/dataset.py` | 修改 | 多交易所路径 |
| `data/data_manager.py` | 修改 | 内存限制 |
| `core/symbol_registry.py` | 新增 | 标的元数据 |

#### Adapter / Journal / Dashboard (H)

| 文件 | 操作 | 说明 |
|------|------|------|
| `live/kernel_adapter.py` | 修复 | 通用 param 解析 + volume + price |
| `live/trade_journal.py` | 扩展 | V5 schema + stats 修复 |
| `dashboard/app.py` | 扩展 | 实盘面板 |

#### 分析 / Alpha (I)

| 文件 | 操作 | 说明 |
|------|------|------|
| `analysis/performance.py` | 修改 | 保证金 ROI |
| `alpha/order_flow.py` | 修改 | 资产感知 |
| `alpha/volatility.py` | 修改 | 资产感知 |

#### 测试 (J)

| 文件 | 操作 | 说明 |
|------|------|------|
| `tests/test_margin_models.py` | 新增 | MarginModel |
| `tests/test_cost_models.py` | 新增 | CostModel |
| `tests/test_reconciliation.py` | 新增 | Paper ↔ Backtest |
| `tests/mocks/mock_binance.py` | 新增 | Mock Binance API |
| `tests/mocks/mock_ibkr.py` | 新增 | Mock IBKR |
| `.github/workflows/test.yml` | 新增 | CI/CD |

---

### V5.2 时间增量

| 类别 | 内容 | 增量时间 |
|------|------|----------|
| E | 策略层 bug 修复 + V5 适配 (8 个策略文件) | +2-3 天 |
| F | Research 层 V5 联动 (7 个文件 + schema 迁移) | +2-3 天 |
| G | 数据层 V5 适配 (多交易所 + symbol registry) | +1-2 天 |
| H | KernelAdapter + TradeJournal + Dashboard | +2-3 天 |
| I | 分析层 + Alpha 层资产感知 | +1 天 |
| J | 测试基础设施 (CI/CD + 对账 + mock + 迁移) | +3-4 天 |

---

### 最终总时间表

| 阶段 | 内容 | 时间 |
|------|------|------|
| V5 Phase 0-6 | 原始三层架构 | 17-25 天 |
| V5 Phase 7 | 高级执行算法 (可选) | 2-3 天 |
| V5.1 A-D | 安全 + 正确性 + 性能 + 功能 | 6-10 天 |
| V5.2 E-J | 全局联动 + 测试 | 11-16 天 |
| **合计** | **完整 V5 升级** | **34-51 天** |

### 建议执行顺序

```
Week 1-2:  Phase 0 (核心抽象) + E (策略 bug 修复) + J.2 (CI/CD)
Week 3-4:  Phase 1 (回测改进) + F (Research 联动) + G (数据层)
Week 5-6:  Phase 2 (纸交易) + Phase 3 (交易所对接) + H (Adapter/Journal)
Week 7-8:  Phase 4-5 (OMS/告警) + V5.1 A-D (安全/性能)
Week 9:    Phase 6 (Testnet 验证) + J (测试完善) + I (分析层)
Week 10:   Phase 7 (可选) + 集成测试 + 压力测试
```
