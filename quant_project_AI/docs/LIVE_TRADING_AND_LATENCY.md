# 实盘对接与延迟：如何解决

本框架**不内置**实盘柜台或券商 API，但可以通过「Broker 抽象 + 自实现」完成实盘对接；延迟则通过**策略设计**和**执行层设计**来缓解与度量。本文说明可行做法与注意事项。

---

## 一、实盘对接：两种思路

### 1.1 思路 A：策略不变，外加「执行层」

- **回测**：`BacktestEngine.run(strategy, ...)` 内部根据策略返回的 signal（action/symbol/shares/order_type/limit_price/stop_price）模拟成交。
- **实盘**：同一套 `strategy.on_bar(data, current_date)` 照常调用，得到的 signal 不交给回测引擎，而是交给**你自己写的执行模块**，由该模块调用券商 API 下单。

这样策略代码与回测一致，只差「谁消费 signal」：回测里是引擎记账，实盘里是你的执行层发单。

### 1.2 思路 B：使用框架提供的 Broker 抽象（推荐）

框架提供** Broker 接口**与**纸交易 Broker**，便于你：

- 用同一套「策略 → 信号 → Broker」写回测与实盘逻辑；
- 先接 `PaperBroker`（只记单、不发实盘）验证流程与延迟假设；
- 再实现一个「对接你券商 API」的 Broker，替换 `PaperBroker` 即可。

Broker 约定见 `quant_framework.broker`：`submit_order(signal) -> order_result`、`get_positions()`、`get_cash()` 等。你只需实现这些方法，内部调券商 SDK/REST。

---

## 二、实盘对接具体步骤

### 2.1 定义「实盘驱动循环」（与回测类似）

实盘没有历史 bar 循环，需要你自己驱动，例如：

- **日频**：每日收盘后定时任务（如 16:05）拉取当日 K 线或日线 close，构造 `data = df.iloc[: 当前]`，调用 `strategy.on_bar(data, current_date=今天)`，拿到信号后调用 `broker.submit_order(signal)`。
- **分钟频**：每分钟或每 5 分钟拉最新 K 线，同上，`on_bar` 后 `submit_order`。

要点：**传给策略的 data 与 current_date 必须是「当前已可得的」**，不能含未来，与回测时的约定一致。

**回测与实盘一致性（框架保证与你的配合）**：

- **历史 K 线数**：策略有 `min_lookback` 属性（如 MA 为 long_window，RSI 为 rsi_period+1）。回测时若数据 bar 数小于 min_lookback 会直接报错；**实盘拉取的历史 K 线数应 ≥ strategy.min_lookback**（建议略多，如 +20），否则指标与回测不一致。
- **标的处理顺序**：多标的回测时引擎按 `symbol_list` 顺序处理；**实盘驱动中处理标的与信号的顺序建议与回测一致**，避免同一 bar 内笔数或顺序分叉。
- **成本**：回测建议用 `BacktestConfig.conservative()`（佣金 0.15%、滑点 5 bps）或更高，与实盘预期对齐，避免回测过于乐观。

### 2.2 实现 Broker（对接券商）

在 `quant_framework.broker` 中定义抽象接口，你实现一个类，例如：

```python
from quant_framework.broker import Broker

class YourBroker(Broker):
    def submit_order(self, signal: dict) -> dict:
        # 将 signal (action, symbol, shares, order_type, limit_price, stop_price)
        # 转成券商 API 的委托请求，发单，返回 { "order_id", "status", "fill_price", "filled_shares", ... }
        ...
    def get_positions(self) -> dict:
        # 调券商接口查持仓，返回 { symbol: shares }
        ...
    def get_cash(self) -> float:
        # 调券商接口查可用资金
        ...
```

券商侧通常提供 Python SDK 或 REST，你只需在以上方法里调用并做字段映射（如 symbol 转合约代码、数量取整等）。

### 2.3 纸交易（模拟盘）先行

在真正上实盘前，建议：

- 用 **PaperBroker**（或自写一个「只记单、不真发」的 Broker）跑一段时间；
- 驱动循环与实盘一致（同一时间点、同一 data 源），仅把 `submit_order` 改为写库/写日志；
- 对比「纸交易记录」与「若当时回测」的结果，检查逻辑与延迟假设是否合理。

---

## 三、延迟：成因与应对

### 3.1 延迟从哪里来

| 来源 | 说明 |
|------|------|
| 网络 | 你的服务器到券商/交易所的 RTT。 |
| 柜台 | 券商系统排队、风控、验资。 |
| 撮合 | 订单进入交易所到成交的时间。 |
| 策略/代码 | 若在收到行情后再算指标再发单，计算本身也有耗时。 |

回测里是「瞬时成交」，实盘必然存在上述延迟，可能造成：**信号发出时价格是 P，实际成交时已变成 P'**。

### 3.2 策略设计上「吸收」延迟（推荐）

- **T 日收盘出信号、T+1 日执行**：回测里已经按「下一 bar」成交，实盘里你在 T 日收盘后算信号、T+1 日集合竞价或开盘市价发单，相当于把「几秒到几百毫秒」的延迟收进「隔日」里，对日频策略影响很小。
- **避免「当前 tick 触发、当前 tick 就要成交」**：若做日内或高频，延迟会直接吃掉利润，需要在架构上做低延迟（见下），本框架面向的是日频/低频，不解决高频延迟。

### 3.3 执行层减轻延迟（实盘代码）

- **异步/非阻塞发单**：主循环里不要「发单并同步等成交」阻塞；应「提交委托即返回」，用订单号或回调再查成交。这样网络延迟不会拖住下一笔计算。
- **先算信号、再批量发单**：若同一时刻多标的信号，可先算完所有 signal，再一次性提交，减少「算一个发一个」的串行延迟。
- **预连接与心跳**：与券商/行情的长连接提前建好，避免每次发单前现建连接。

### 3.4 度量延迟（可选）

若你想量化「信号到成交」的延迟：

- 在发出 signal 时打时间戳 `t_signal`；
- 在收到券商回报（成交）时打时间戳 `t_fill`；
- 记录 `t_fill - t_signal` 的分布（如 p50/p99），用于评估滑点假设或调整策略频率。

---

## 四、本框架提供的 Broker 模块（简要）

- **`quant_framework.broker.Broker`**：抽象基类，定义 `submit_order(signal) -> result`、`get_positions()`、`get_cash()` 等，你实现后即可对接任意券商。
- **`quant_framework.broker.PaperBroker`**：纸交易实现，不向券商发单，只在内存（或可选写文件）中记录订单与持仓，用于实盘前验证流程与延迟假设。

这样你可以：

1. 回测：仍用 `BacktestEngine.run(strategy, ...)`，无需 Broker。  
2. 纸交易 / 实盘：用同一策略 + 同一驱动逻辑，仅把 Broker 从 `PaperBroker` 换成你的实盘 Broker。

---

## 五、小结

| 问题 | 解决方向 |
|------|----------|
| **实盘对接** | 使用框架的 Broker 抽象，实现 `submit_order` / `get_positions` / `get_cash`，内部调券商 API；实盘驱动循环自己写（按日/按分钟拉 data、调 `on_bar`、再 `broker.submit_order`）。 |
| **延迟** | 策略上采用「T 信号 → T+1 执行」；执行层异步发单、预连接；可选记录信号时间与成交时间以度量延迟；日频策略下延迟影响可通过设计压到较小。 |

框架负责**策略与信号的语义一致**（回测与实盘用同一套 `on_bar` 与 signal 结构）；**具体券商对接与延迟优化**由你在 Broker 实现与实盘驱动中完成。更多回测与实盘差异见 [BACKTEST_VS_LIVE.md](BACKTEST_VS_LIVE.md)。
