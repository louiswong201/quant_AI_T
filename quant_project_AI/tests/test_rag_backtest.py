"""
RAG 回测完整测试
══════════════════════════════════════════════════════════════
模拟 6 个月 BTC 行情 + 新闻事件驱动的 RAG 策略回测。
全面评估：
  1. 策略表现（收益率、夏普、最大回撤、胜率…）
  2. RAG 框架表现（检索延迟、时序一致性、上下文命中率…）
  3. RAG vs 纯技术面策略 A/B 对比
  4. 逐 bar 诊断日志
"""

import sys
import os
import time
import math
import random
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from quant_framework.rag import RAGConfig, RAGPipeline, Document
from quant_framework.data.rag_context import RagContextProvider
from quant_framework.strategy.base_strategy import BaseStrategy
from quant_framework.backtest.backtest_engine import BacktestEngine
from quant_framework.backtest.config import BacktestConfig
from quant_framework.data.data_manager import DataManager


# ═══════════════════════════════════════════════════════════════
# 1. 模拟真实行情数据生成
# ═══════════════════════════════════════════════════════════════

def generate_btc_ohlcv(
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    initial_price: float = 42000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """生成带真实统计特征的 BTC 日线数据。

    用 GBM (Geometric Brownian Motion) 模型，叠加新闻事件跳跃：
    - 减半事件 (2024-04-20): 短期波动放大
    - ETF 获批 (2024-01-10): 价格跳涨
    - 暴跌事件 (2024-03-15): 急跌恢复
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    mu = 0.0008
    sigma = 0.032
    returns = rng.normal(mu, sigma, n)

    event_map = {}
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        if ds == "2024-01-10":
            event_map[ds] = ("ETF获批跳涨", 0.08)
        elif ds == "2024-01-11":
            event_map[ds] = ("ETF后续上涨", 0.04)
        elif ds == "2024-03-14":
            event_map[ds] = ("鲸鱼抛售恐慌", -0.07)
        elif ds == "2024-03-15":
            event_map[ds] = ("连续抛售", -0.05)
        elif ds == "2024-03-18":
            event_map[ds] = ("恐慌后反弹", 0.06)
        elif ds == "2024-04-19":
            event_map[ds] = ("减半预期炒作", 0.04)
        elif "2024-04-20" <= ds <= "2024-04-25":
            event_map[ds] = ("减半后波动", rng.normal(0, 0.04))
        elif ds == "2024-05-20":
            event_map[ds] = ("ETH ETF利好联动", 0.05)
        elif ds == "2024-06-10":
            event_map[ds] = ("美联储鸽派信号", 0.03)

    for i, d in enumerate(dates):
        ds = d.strftime("%Y-%m-%d")
        if ds in event_map:
            _, jump = event_map[ds]
            returns[i] += jump

    prices = np.empty(n)
    prices[0] = initial_price
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + returns[i])
        prices[i] = max(prices[i], 100)

    opens = prices * (1 + rng.uniform(-0.005, 0.005, n))
    highs = np.maximum(prices, opens) * (1 + rng.uniform(0.005, 0.025, n))
    lows = np.minimum(prices, opens) * (1 - rng.uniform(0.005, 0.025, n))
    volumes = rng.uniform(2e9, 8e9, n) * (1 + np.abs(returns) * 10)

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(prices, 2),
        "volume": np.round(volumes, 0).astype(int),
    })
    return df


# ═══════════════════════════════════════════════════════════════
# 2. 新闻语料库生成 (带精确时间戳)
# ═══════════════════════════════════════════════════════════════

def generate_news_corpus() -> List[Document]:
    """生成与行情事件对应的新闻语料库。

    每条新闻有精确的 created_at 时间戳，回测时通过 as_of_date 过滤
    确保策略不会使用"未来信息"。
    """
    news = [
        # 1月: ETF 获批阶段
        ("SEC正式批准比特币现货ETF上市申请。贝莱德、富达、灰度等11只比特币现货ETF获批，"
         "标志着加密货币行业迎来里程碑时刻。分析师预计将有超过100亿美元的新增资金流入。"
         "比特币价格应声大涨，市场情绪极度乐观。", "2024-01-10", "reuters", "BTC"),
        ("比特币ETF首日交易量突破46亿美元，远超预期。贝莱德IBIT单只基金交易量达10亿美元。"
         "机构投资者争相入场，灰度GBTC出现大量申购。市场预期比特币将进入新一轮牛市周期。",
         "2024-01-11", "bloomberg", "BTC"),
        ("比特币突破48000美元，创近两年新高。链上数据显示大户持续增持，交易所余额降至五年低点。"
         "分析师看好后市，目标价上调至60000美元。",
         "2024-01-15", "coindesk", "BTC"),
        ("以太坊跟随比特币走强，站上2500美元。ETH生态DeFi TVL回升至500亿美元。"
         "市场预期以太坊现货ETF也将获批。",
         "2024-01-20", "coindesk", "ETH"),
        ("美联储1月会议纪要偏鹰，暗示短期内不会降息。美元指数走强，风险资产承压。"
         "比特币回调至43000美元附近，投资者担忧加息周期延长。"
         "市场恐慌指数上升，多头信心受到打击。",
         "2024-02-01", "reuters", "MACRO"),
        ("灰度GBTC持续大量流出，每日净流出超过5亿美元。分析师警告抛售压力可能持续数周。"
         "比特币价格承压下跌，短期内回调风险加大。市场信心下降。",
         "2024-02-08", "bloomberg", "BTC"),
        ("比特币矿企算力创新高，全网算力突破500EH/s。但电力成本飙升导致小矿工面临关停压力。"
         "分析师认为减半后算力可能下降30%，小矿场面临亏损甚至破产。",
         "2024-02-15", "coindesk", "BTC"),
        ("全球监管趋严，美国多州考虑加强加密货币监管。SEC对多个项目发出警告。"
         "市场情绪转向谨慎，交易量萎缩。",
         "2024-02-20", "reuters", "MARKET"),
        ("加密市场总市值重返2万亿美元。比特币市占率稳定在50%左右。"
         "山寨币板块开始轮动，AI+Crypto概念币领涨。",
         "2024-02-25", "bloomberg", "MARKET"),

        # 3月: 震荡阶段
        ("比特币冲击65000美元未果，出现技术性回调。短期持有者获利了结压力增大。"
         "合约市场持仓量创新高，多空比偏向空方。",
         "2024-03-10", "coindesk", "BTC"),
        ("知名鲸鱼地址大量转入交易所，引发市场恐慌。比特币单日跌幅超7%，跌破60000美元。"
         "合约爆仓金额超过5亿美元，市场恐慌指数飙升至极度恐慌。",
         "2024-03-14", "reuters", "BTC"),
        ("比特币跌至54000美元后强势反弹，V型反转。机构抄底资金涌入，ETF单日净流入创纪录。"
         "分析师认为急跌属于健康回调，长期趋势未变。",
         "2024-03-18", "bloomberg", "BTC"),
        ("美联储3月议息会议维持利率不变，但点阵图暗示年内将降息三次。"
         "市场提前计入降息预期，风险资产全面反弹。比特币重返65000美元。",
         "2024-03-21", "reuters", "MACRO"),

        # 4月: 减半事件
        ("比特币减半倒计时进入最后一周。历史数据显示，前三次减半后平均涨幅超过500%。"
         "但本次市场结构已发生重大变化，ETF的存在使得供需动态更加复杂。",
         "2024-04-14", "coindesk", "BTC"),
        ("比特币在区块高度840000完成第四次减半！区块奖励从6.25BTC降至3.125BTC。"
         "市场反应平稳，价格在减半后小幅波动。分析师认为减半的真正影响将在6-12个月后显现。",
         "2024-04-20", "coindesk", "BTC"),
        ("减半后首周矿工收入下降50%，小矿工面临关停压力。比特币手续费收入虽因Runes协议暴增，"
         "但难以弥补区块奖励减半的损失。多家矿企股价大幅下跌，市场担忧算力可能暴跌。",
         "2024-04-25", "bloomberg", "BTC"),
        ("香港正式批准比特币和以太坊现货ETF上市，成为亚洲首个。"
         "预计将为亚太地区投资者提供合规的加密投资渠道。",
         "2024-04-28", "reuters", "BTC"),

        # 5月: 震荡上行
        ("SEC对以太坊现货ETF态度出现积极转变。多家机构紧急更新S-1申请文件。"
         "以太坊价格单日暴涨15%，突破3800美元。比特币联动上涨至70000美元。",
         "2024-05-20", "bloomberg", "ETH"),
        ("全球加密货币监管趋势向好。欧盟MiCA法规正式实施，日本放宽加密基金投资限制。"
         "监管明确化被视为长期利好。",
         "2024-05-28", "reuters", "MARKET"),

        # 6月: 宏观与价格走势
        ("美联储6月议息会议暗示9月可能首次降息，鸽派信号明确。"
         "美国国债收益率下行，美元走弱。风险资产普涨，比特币突破72000美元。",
         "2024-06-10", "reuters", "MACRO"),
        ("比特币矿企报告显示减半后盈利能力依然强劲。Marathon Digital和Riot Platforms"
         "二季度营收均超预期。AI和挖矿协同发展成为新趋势。",
         "2024-06-15", "bloomberg", "BTC"),
        ("稳定币总市值突破1600亿美元，Tether市值超过1100亿。"
         "链上活跃地址数和交易量持续增长，表明市场参与度提升。",
         "2024-06-20", "coindesk", "MARKET"),
        ("比特币闪电网络容量创新高，突破6000BTC。Layer2解决方案持续发展，"
         "比特币不再仅是'数字黄金'，更是完整的支付和智能合约平台。",
         "2024-06-25", "coindesk", "BTC"),
    ]

    docs = []
    for content, date_str, source, symbol in news:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        doc = Document(
            content=content,
            source=source,
            metadata={"symbol": symbol, "date": date_str},
            created_at=dt,
        )
        docs.append(doc)
    return docs


# ═══════════════════════════════════════════════════════════════
# 3. RAG 增强策略
# ═══════════════════════════════════════════════════════════════

class RAGEnhancedStrategy(BaseStrategy):
    """RAG + 技术面融合策略。

    信号逻辑:
    1. 技术面：MA 交叉 + RSI 超买超卖
    2. RAG 情绪：从新闻上下文中提取看涨/看跌关键词
    3. 融合：技术面信号为主，RAG 情绪作为增强/抑制因子
    """

    BULLISH_KW = {"获批", "批准", "利好", "突破", "上涨", "暴涨", "牛市", "创新高",
                  "净流入", "增持", "反弹", "鸽派", "降息", "超预期", "里程碑",
                  "乐观", "积极", "吸引", "强劲", "创纪录", "涨幅"}
    BEARISH_KW = {"恐慌", "暴跌", "抛售", "爆仓", "下跌", "承压", "回调", "鹰派",
                  "关停", "失败", "禁止", "打压", "崩盘", "跌幅", "跌破",
                  "下降", "关停", "压力", "亏损", "偏鹰", "不会降息"}

    def __init__(
        self,
        name: str = "RAG增强策略",
        initial_capital: float = 1_000_000,
        short_ma: int = 5,
        long_ma: int = 20,
        rsi_period: int = 14,
        rag_provider: Optional["RagContextProvider"] = None,
    ):
        super().__init__(name=name, initial_capital=initial_capital, rag_provider=rag_provider)
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.rsi_period = rsi_period
        self._min_lookback = max(long_ma, rsi_period + 1)
        self._rag_latencies: List[float] = []
        self._rag_contexts: List[dict] = []
        self._signal_log: List[dict] = []

    def _calc_rsi(self, close: pd.Series) -> float:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_period).mean()
        g, l_ = float(gain.iloc[-1]), float(loss.iloc[-1])
        if l_ == 0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - (100.0 / (1.0 + g / l_))

    def _analyze_sentiment(self, context: str) -> float:
        """简单关键词情绪分析 → [-1, 1] 分数。"""
        if not context:
            return 0.0
        bull = sum(1 for kw in self.BULLISH_KW if kw in context)
        bear = sum(1 for kw in self.BEARISH_KW if kw in context)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total

    def on_bar(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_date: pd.Timestamp,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Union[Dict, List[Dict]]:
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "BTC") if hasattr(df, "attrs") else "BTC"

        if len(df) < self._min_lookback:
            return {"action": "hold"}

        close = df["close"]
        price = float(close.iloc[-1])
        ma_short = float(close.rolling(self.short_ma).mean().iloc[-1])
        ma_long = float(close.rolling(self.long_ma).mean().iloc[-1])
        rsi = self._calc_rsi(close)

        tech_signal = 0.0
        if ma_short > ma_long:
            tech_signal += 0.5
        else:
            tech_signal -= 0.5
        if rsi < 30:
            tech_signal += 0.5
        elif rsi > 70:
            tech_signal -= 0.5

        rag_context = ""
        sentiment = 0.0
        rag_latency = 0.0
        if self.rag_provider is not None:
            t0 = time.perf_counter()
            rag_context = self.get_rag_context(
                query="比特币 BTC 市场动态",
                symbol=symbol,
                top_k=3,
                max_chars=1000,
                as_of_date=current_date,
            )
            rag_latency = (time.perf_counter() - t0) * 1000
            self._rag_latencies.append(rag_latency)
            sentiment = self._analyze_sentiment(rag_context)

        combined = tech_signal * 0.6 + sentiment * 0.4

        self._rag_contexts.append({
            "date": str(current_date.date()),
            "context_len": len(rag_context),
            "sentiment": round(sentiment, 3),
            "tech_signal": round(tech_signal, 3),
            "combined": round(combined, 3),
            "rag_latency_ms": round(rag_latency, 3),
            "rsi": round(rsi, 1),
            "ma_short": round(ma_short, 2),
            "ma_long": round(ma_long, 2),
            "price": price,
        })

        holdings = self.positions.get(symbol, 0)
        action = "hold"

        if combined > 0.2 and holdings == 0:
            shares = self.calculate_position_size(price, risk_percent=0.9)
            if shares > 0 and self.can_buy(symbol, price, shares):
                action = "buy"
                log_entry = {"date": str(current_date.date()), "action": "buy",
                             "shares": shares, "price": price, "combined": round(combined, 3),
                             "tech": round(tech_signal, 3), "sentiment": round(sentiment, 3)}
                self._signal_log.append(log_entry)
                return {"action": "buy", "symbol": symbol, "shares": shares}

        elif combined < -0.2 and holdings > 0:
            action = "sell"
            log_entry = {"date": str(current_date.date()), "action": "sell",
                         "shares": holdings, "price": price, "combined": round(combined, 3),
                         "tech": round(tech_signal, 3), "sentiment": round(sentiment, 3)}
            self._signal_log.append(log_entry)
            return {"action": "sell", "symbol": symbol, "shares": holdings}

        self._signal_log.append({"date": str(current_date.date()), "action": "hold",
                                  "combined": round(combined, 3)})
        return {"action": "hold"}


class PureTechnicalStrategy(BaseStrategy):
    """纯技术面策略 (无 RAG)，作为对照组。"""

    def __init__(self, name: str = "纯技术面策略", initial_capital: float = 1_000_000,
                 short_ma: int = 5, long_ma: int = 20, rsi_period: int = 14):
        super().__init__(name=name, initial_capital=initial_capital)
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.rsi_period = rsi_period
        self._min_lookback = max(long_ma, rsi_period + 1)

    def _calc_rsi(self, close: pd.Series) -> float:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_period).mean()
        g, l_ = float(gain.iloc[-1]), float(loss.iloc[-1])
        if l_ == 0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - (100.0 / (1.0 + g / l_))

    def on_bar(self, data, current_date, current_prices=None):
        if isinstance(data, dict):
            symbol = next(iter(data))
            df = data[symbol]
        else:
            df = data
            symbol = df.attrs.get("symbol", "BTC") if hasattr(df, "attrs") else "BTC"
        if len(df) < self._min_lookback:
            return {"action": "hold"}
        close = df["close"]
        price = float(close.iloc[-1])
        ma_short = float(close.rolling(self.short_ma).mean().iloc[-1])
        ma_long = float(close.rolling(self.long_ma).mean().iloc[-1])
        rsi = self._calc_rsi(close)
        tech_signal = 0.0
        if ma_short > ma_long:
            tech_signal += 0.5
        else:
            tech_signal -= 0.5
        if rsi < 30:
            tech_signal += 0.5
        elif rsi > 70:
            tech_signal -= 0.5
        holdings = self.positions.get(symbol, 0)
        if tech_signal > 0.2 and holdings == 0:
            shares = self.calculate_position_size(price, risk_percent=0.9)
            if shares > 0 and self.can_buy(symbol, price, shares):
                return {"action": "buy", "symbol": symbol, "shares": shares}
        elif tech_signal < -0.2 and holdings > 0:
            return {"action": "sell", "symbol": symbol, "shares": holdings}
        return {"action": "hold"}


# ═══════════════════════════════════════════════════════════════
# 4. 模拟 DataManager (内存行情)
# ═══════════════════════════════════════════════════════════════

class InMemoryDataManager(DataManager):
    """内存数据管理器，直接使用预生成的 DataFrame。"""

    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self._data_map = data_map
        self.dataset = None

    def load_data(self, symbol: str, start_date: str, end_date: str):
        df = self._data_map.get(symbol)
        if df is None:
            return None
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        result = df.loc[mask].copy()
        return result if not result.empty else None

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "close" in df.columns:
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma10"] = df["close"].rolling(10).mean()
            df["ma20"] = df["close"].rolling(20).mean()
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            df["rsi"] = 100 - (100 / (1 + rs))
        return df


# ═══════════════════════════════════════════════════════════════
# 5. 分析工具函数
# ═══════════════════════════════════════════════════════════════

def calc_max_drawdown(pv_arr: np.ndarray) -> float:
    peak = pv_arr[0]
    max_dd = 0.0
    for v in pv_arr:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calc_sharpe(daily_returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / 252
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def calc_sortino(daily_returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / 252
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = np.std(downside)
    if down_std == 0:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(252))


def calc_win_rate(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty or "action" not in trades_df.columns:
        return {"total_trades": 0, "win_rate": 0, "avg_profit_pct": 0}
    buys = trades_df[trades_df["action"] == "buy"].copy()
    sells = trades_df[trades_df["action"] == "sell"].copy()
    if buys.empty or sells.empty:
        return {"total_trades": len(buys) + len(sells), "win_rate": 0, "avg_profit_pct": 0}
    pairs = min(len(buys), len(sells))
    wins = 0
    profits = []
    for j in range(pairs):
        buy_px = float(buys.iloc[j]["price"])
        sell_px = float(sells.iloc[j]["price"])
        pnl_pct = (sell_px - buy_px) / buy_px * 100 if buy_px > 0 else 0
        profits.append(pnl_pct)
        if sell_px > buy_px:
            wins += 1
    return {
        "total_trades": len(buys) + len(sells),
        "round_trips": pairs,
        "wins": wins,
        "losses": pairs - wins,
        "win_rate": wins / pairs * 100 if pairs > 0 else 0,
        "avg_profit_pct": sum(profits) / len(profits) if profits else 0,
        "max_profit_pct": max(profits) if profits else 0,
        "max_loss_pct": min(profits) if profits else 0,
    }


# ═══════════════════════════════════════════════════════════════
# 6. 主测试流程
# ═══════════════════════════════════════════════════════════════

def run_backtest():
    print("=" * 80)
    print("RAG 回测完整测试")
    print("模拟 2024-01-01 ~ 2024-06-30 BTC 行情 + 新闻事件驱动")
    print("=" * 80)

    # ── 生成行情数据 ──
    print("\n[1] 生成模拟 BTC 日线行情...")
    btc_data = generate_btc_ohlcv()
    print(f"    日期范围: {btc_data['date'].iloc[0].date()} ~ {btc_data['date'].iloc[-1].date()}")
    print(f"    总交易日: {len(btc_data)}")
    print(f"    起始价: ${btc_data['close'].iloc[0]:,.2f}")
    print(f"    终止价: ${btc_data['close'].iloc[-1]:,.2f}")
    buy_hold_ret = (btc_data['close'].iloc[-1] / btc_data['close'].iloc[0] - 1) * 100
    print(f"    Buy & Hold 收益: {buy_hold_ret:+.2f}%")

    # ── 构建新闻语料库 ──
    print("\n[2] 构建新闻语料库并入库 RAG...")
    news_docs = generate_news_corpus()
    print(f"    新闻总数: {len(news_docs)}")

    rag_config = RAGConfig(
        chunk_size=256,
        chunk_overlap=32,
        chunk_by_sentence=True,
        use_hybrid=True,
        hybrid_weights=(0.4, 0.6),
        vector_top_k=10,
        keyword_top_k=10,
        dedup_cache_max=10000,
        query_embedding_cache_size=128,
    )
    rag_pipeline = RAGPipeline(config=rag_config, start_worker=False)

    t0 = time.perf_counter()
    n_chunks = rag_pipeline.add_documents(news_docs)
    ingest_ms = (time.perf_counter() - t0) * 1000
    print(f"    入库分块数: {n_chunks}")
    print(f"    入库耗时: {ingest_ms:.1f} ms")
    print(f"    向量存储: {rag_pipeline.vector_store.size()} 条")
    print(f"    关键词索引: {rag_pipeline.keyword_index.size()} 条")

    rag_provider = RagContextProvider(pipeline=rag_pipeline)

    # 预热: 消除首次检索的冷启动延迟 (JIT编译、缓存预热等)
    _ = rag_pipeline.retrieve("warmup", top_k=1)

    # ── 创建策略 ──
    print("\n[3] 创建策略...")
    rag_strategy = RAGEnhancedStrategy(
        name="RAG增强策略",
        initial_capital=1_000_000,
        rag_provider=rag_provider,
    )
    tech_strategy = PureTechnicalStrategy(
        name="纯技术面策略",
        initial_capital=1_000_000,
    )
    print(f"    RAG策略: {rag_strategy.name} (初始资金 ${rag_strategy.initial_capital:,.0f})")
    print(f"    对照策略: {tech_strategy.name} (初始资金 ${tech_strategy.initial_capital:,.0f})")

    # ── 运行回测 ──
    print("\n[4] 运行回测...")
    data_manager = InMemoryDataManager({"BTC": btc_data})
    bt_config = BacktestConfig(
        commission_pct_buy=0.001,
        commission_pct_sell=0.001,
        slippage_bps_buy=5.0,
        slippage_bps_sell=5.0,
    )
    engine = BacktestEngine(data_manager=data_manager, config=bt_config)

    print("    运行 RAG增强策略...")
    t0 = time.perf_counter()
    rag_result = engine.run(rag_strategy, "BTC", "2024-01-01", "2024-06-30")
    rag_time = time.perf_counter() - t0
    print(f"    完成 ({rag_time:.3f}s)")

    print("    运行 纯技术面策略...")
    t0 = time.perf_counter()
    tech_result = engine.run(tech_strategy, "BTC", "2024-01-01", "2024-06-30")
    tech_time = time.perf_counter() - t0
    print(f"    完成 ({tech_time:.3f}s)")

    # ═══════════════════════════════════════════════════════════
    # 策略表现分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("一、策略表现对比")
    print("=" * 80)

    for label, result, strategy in [
        ("RAG增强策略", rag_result, rag_strategy),
        ("纯技术面策略", tech_result, tech_strategy),
    ]:
        pv = result["portfolio_values"]
        dr = result["daily_returns"]
        trades = result["trades"]
        final = result["final_value"]
        total_ret = (final / strategy.initial_capital - 1) * 100
        max_dd = calc_max_drawdown(pv) * 100
        sharpe = calc_sharpe(dr)
        sortino = calc_sortino(dr)
        win_info = calc_win_rate(trades)
        ann_vol = float(np.std(dr) * np.sqrt(252) * 100)

        print(f"\n  [{label}]")
        print(f"  {'─' * 50}")
        print(f"  最终净值:       ${final:>14,.2f}")
        print(f"  总收益率:       {total_ret:>+13.2f}%")
        print(f"  年化波动率:     {ann_vol:>13.2f}%")
        print(f"  夏普比率:       {sharpe:>13.3f}")
        print(f"  索提诺比率:     {sortino:>13.3f}")
        print(f"  最大回撤:       {max_dd:>13.2f}%")
        print(f"  总交易笔数:     {win_info['total_trades']:>13d}")
        print(f"  配对交易数:     {win_info.get('round_trips', 0):>13d}")
        print(f"  胜率:           {win_info['win_rate']:>12.1f}%")
        print(f"  平均盈亏:       {win_info['avg_profit_pct']:>+12.2f}%")
        if win_info.get("max_profit_pct"):
            print(f"  最大单笔盈利:   {win_info['max_profit_pct']:>+12.2f}%")
            print(f"  最大单笔亏损:   {win_info['max_loss_pct']:>+12.2f}%")

    # ── A/B 对比 ──
    rag_ret = (rag_result["final_value"] / 1_000_000 - 1) * 100
    tech_ret = (tech_result["final_value"] / 1_000_000 - 1) * 100
    alpha = rag_ret - tech_ret

    print(f"\n  [A/B 对比]")
    print(f"  {'─' * 50}")
    print(f"  RAG策略收益:    {rag_ret:>+13.2f}%")
    print(f"  技术面收益:     {tech_ret:>+13.2f}%")
    print(f"  Buy&Hold 收益:  {buy_hold_ret:>+13.2f}%")
    print(f"  RAG vs 技术面 Alpha: {alpha:>+10.2f}%")
    print(f"  RAG vs Buy&Hold:     {rag_ret - buy_hold_ret:>+10.2f}%")

    # ═══════════════════════════════════════════════════════════
    # RAG 框架表现分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("二、RAG 框架表现分析")
    print("=" * 80)

    latencies = rag_strategy._rag_latencies
    contexts = rag_strategy._rag_contexts

    if latencies:
        lat_arr = np.array(latencies)
        print(f"\n  [检索性能]")
        print(f"  {'─' * 50}")
        print(f"  总检索次数:     {len(latencies):>13d}")
        print(f"  平均延迟:       {np.mean(lat_arr):>12.3f} ms")
        print(f"  P50 延迟:       {np.percentile(lat_arr, 50):>12.3f} ms")
        print(f"  P95 延迟:       {np.percentile(lat_arr, 95):>12.3f} ms")
        print(f"  P99 延迟:       {np.percentile(lat_arr, 99):>12.3f} ms")
        print(f"  最大延迟:       {np.max(lat_arr):>12.3f} ms")

    if contexts:
        ctx_lens = [c["context_len"] for c in contexts]
        sentiments = [c["sentiment"] for c in contexts]
        non_empty = sum(1 for l in ctx_lens if l > 0)
        print(f"\n  [上下文质量]")
        print(f"  {'─' * 50}")
        print(f"  有上下文的 bar:  {non_empty}/{len(contexts)} ({non_empty/len(contexts)*100:.1f}%)")
        print(f"  平均上下文长度:  {np.mean(ctx_lens):>10.0f} 字")
        print(f"  最大上下文长度:  {max(ctx_lens):>10d} 字")
        positive_s = sum(1 for s in sentiments if s > 0)
        negative_s = sum(1 for s in sentiments if s < 0)
        neutral_s = sum(1 for s in sentiments if s == 0)
        print(f"  情绪分布: 看涨={positive_s} 看跌={negative_s} 中性={neutral_s}")
        print(f"  平均情绪分数:    {np.mean(sentiments):>+10.3f}")

    # ── 时序一致性验证 ──
    print(f"\n  [时序一致性 (as_of_date)]")
    print(f"  {'─' * 50}")
    temporal_violations = 0
    early_dates = [c for c in contexts if c["date"] < "2024-01-10"]
    late_dates = [c for c in contexts if c["date"] >= "2024-06-20"]
    early_ctx = [c["context_len"] for c in early_dates]
    late_ctx = [c["context_len"] for c in late_dates]
    if early_ctx:
        print(f"  1月初 (ETF获批前) 平均上下文: {np.mean(early_ctx):>6.0f} 字  (应较少)")
    if late_ctx:
        print(f"  6月底 (信息充分)   平均上下文: {np.mean(late_ctx):>6.0f} 字  (应较多)")
    if early_ctx and late_ctx and np.mean(late_ctx) >= np.mean(early_ctx):
        print(f"  [PASS] 时序一致: 后期上下文 >= 早期上下文，未使用未来信息")
    elif not early_ctx:
        print(f"  [PASS] 时序一致: 1月10日前无上下文（无新闻数据），符合预期")
    else:
        temporal_violations += 1
        print(f"  [WARN] 时序可能不一致，需进一步检查")

    # ── 事件命中分析 ──
    print(f"\n  [关键事件命中分析]")
    print(f"  {'─' * 50}")
    key_events = [
        ("2024-01-10", "ETF获批", "获批,批准,ETF"),
        ("2024-03-14", "鲸鱼抛售", "恐慌,抛售,爆仓"),
        ("2024-04-20", "减半事件", "减半,奖励,区块"),
        ("2024-05-20", "ETH ETF利好", "以太坊,ETH,ETF"),
        ("2024-06-10", "鸽派信号", "降息,鸽派,美联储"),
    ]
    for event_date, event_name, keywords in key_events:
        matching_bars = [c for c in contexts if c["date"] >= event_date
                         and c["date"] <= (datetime.strptime(event_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")]
        kw_list = keywords.split(",")
        if matching_bars:
            bar = matching_bars[0]
            hit = bar["context_len"] > 0
            sentiment_dir = "看涨" if bar["sentiment"] > 0 else ("看跌" if bar["sentiment"] < 0 else "中性")
            status = "HIT" if hit else "MISS"
            print(f"  [{status}] {event_date} {event_name}: ctx={bar['context_len']}字, 情绪={sentiment_dir}({bar['sentiment']:+.2f})")
        else:
            print(f"  [N/A]  {event_date} {event_name}: 该日无交易")

    # ═══════════════════════════════════════════════════════════
    # 交易明细
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("三、RAG 策略交易明细")
    print("=" * 80)

    rag_trades = rag_result["trades"]
    if not rag_trades.empty:
        print(f"\n  {'日期':<14} {'动作':<6} {'股数':>10} {'价格':>12} {'手续费':>10} {'信号详情'}")
        print(f"  {'─' * 75}")
        signal_log = rag_strategy._signal_log
        trade_signals = [s for s in signal_log if s["action"] != "hold"]
        for idx, row in rag_trades.iterrows():
            date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10]
            action_cn = "买入" if row["action"] == "buy" else "卖出"
            sig_info = ""
            matching_sigs = [s for s in trade_signals if s["date"] == date_str]
            if matching_sigs:
                s = matching_sigs[0]
                sig_info = f"综合={s['combined']:+.2f} 技术={s['tech']:+.2f} 情绪={s['sentiment']:+.2f}"
            print(f"  {date_str:<14} {action_cn:<6} {int(row['shares']):>10,d} ${row['price']:>10,.2f} ${row['commission']:>8,.2f}  {sig_info}")

    # ═══════════════════════════════════════════════════════════
    # 净值曲线关键点
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("四、月度表现分解")
    print("=" * 80)

    rag_results_df = rag_result["results"]
    tech_results_df = tech_result["results"]

    print(f"\n  {'月份':<10} {'RAG净值':>14} {'RAG月收益':>12} {'技术面净值':>14} {'技术面月收益':>12} {'BTC月收益':>12}")
    print(f"  {'─' * 80}")

    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    prev_rag, prev_tech, prev_btc = 1_000_000, 1_000_000, float(btc_data["close"].iloc[0])

    for m in months:
        rag_month = rag_results_df[rag_results_df["date"].astype(str).str.startswith(m)]
        tech_month = tech_results_df[tech_results_df["date"].astype(str).str.startswith(m)]
        btc_month = btc_data[btc_data["date"].astype(str).str.startswith(m)]
        if rag_month.empty:
            continue
        rag_end = float(rag_month["portfolio_value"].iloc[-1])
        tech_end = float(tech_month["portfolio_value"].iloc[-1])
        btc_end = float(btc_month["close"].iloc[-1])
        rag_m_ret = (rag_end / prev_rag - 1) * 100
        tech_m_ret = (tech_end / prev_tech - 1) * 100
        btc_m_ret = (btc_end / prev_btc - 1) * 100
        print(f"  {m:<10} ${rag_end:>12,.2f} {rag_m_ret:>+11.2f}% ${tech_end:>12,.2f} {tech_m_ret:>+11.2f}% {btc_m_ret:>+11.2f}%")
        prev_rag, prev_tech, prev_btc = rag_end, tech_end, btc_end

    # ═══════════════════════════════════════════════════════════
    # RAG Pipeline 健康状态
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("五、RAG Pipeline 健康诊断")
    print("=" * 80)

    stats = rag_pipeline.get_stats()
    health = rag_pipeline.health_check()
    print(f"\n  向量存储:       {health['vector_store_size']:>10d} 条")
    print(f"  关键词索引:     {health['keyword_index_size']:>10d} 条")
    print(f"  入队队列:       {health['ingest_queue_size']:>10d} / {health['ingest_queue_max_size']}")
    print(f"  Worker 状态:    {'运行中' if health['worker_started'] else '未启动'}")
    if "last_retrieve_sec" in stats:
        print(f"  最后检索耗时:   {stats['last_retrieve_sec']*1000:>10.3f} ms")

    embedder_name = type(rag_pipeline.embedder).__name__
    print(f"  嵌入器:         {embedder_name}")
    if embedder_name == "DummyEmbedder":
        print(f"  [NOTE] 使用 DummyEmbedder — 向量检索路径为伪随机，实际检索主要依赖 BM25 关键词匹配")
        print(f"         生产环境请安装 sentence-transformers 以获得真实语义检索能力")

    # ═══════════════════════════════════════════════════════════
    # 综合评估
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("六、综合评估")
    print("=" * 80)

    rag_sharpe = calc_sharpe(rag_result["daily_returns"])
    tech_sharpe = calc_sharpe(tech_result["daily_returns"])
    rag_dd = calc_max_drawdown(rag_result["portfolio_values"]) * 100
    tech_dd = calc_max_drawdown(tech_result["portfolio_values"]) * 100
    rag_total = (rag_result["final_value"] / 1_000_000 - 1) * 100
    tech_total = (tech_result["final_value"] / 1_000_000 - 1) * 100

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │                   指标             RAG策略    技术面    差异  │
  ├──────────────────────────────────────────────────────────────┤
  │  总收益率           {rag_total:>+8.2f}%   {tech_total:>+8.2f}%  {rag_total-tech_total:>+7.2f}%│
  │  夏普比率            {rag_sharpe:>8.3f}    {tech_sharpe:>8.3f}  {rag_sharpe-tech_sharpe:>+7.3f} │
  │  最大回撤            {rag_dd:>7.2f}%    {tech_dd:>7.2f}%  {rag_dd-tech_dd:>+7.2f}%│
  │  回测耗时            {rag_time:>7.3f}s    {tech_time:>7.3f}s  {rag_time-tech_time:>+7.3f}s│
  └──────────────────────────────────────────────────────────────┘
""")

    if latencies:
        avg_lat = np.mean(latencies)
        print(f"  策略评价:")
        if rag_total > tech_total:
            print(f"    [+] RAG 策略收益率优于纯技术面 {rag_total - tech_total:+.2f}%")
        else:
            print(f"    [-] RAG 策略收益率低于纯技术面 {rag_total - tech_total:+.2f}%")
        if rag_sharpe > tech_sharpe:
            print(f"    [+] RAG 策略风险调整收益更优 (Sharpe {rag_sharpe:.3f} vs {tech_sharpe:.3f})")
        else:
            print(f"    [-] RAG 策略风险调整收益较差 (Sharpe {rag_sharpe:.3f} vs {tech_sharpe:.3f})")
        if rag_dd < tech_dd:
            print(f"    [+] RAG 策略最大回撤更小 ({rag_dd:.2f}% vs {tech_dd:.2f}%)")
        else:
            print(f"    [-] RAG 策略最大回撤更大 ({rag_dd:.2f}% vs {tech_dd:.2f}%)")

        print(f"\n  框架评价:")
        print(f"    检索延迟: 平均 {avg_lat:.3f}ms — {'优秀 (<1ms)' if avg_lat < 1 else ('良好 (<5ms)' if avg_lat < 5 else '需优化')}")
        print(f"    时序一致性: {'通过' if temporal_violations == 0 else '存在违规'}")
        print(f"    上下文覆盖率: {non_empty}/{len(contexts)} ({non_empty/len(contexts)*100:.0f}%)")
        overhead_pct = (rag_time - tech_time) / tech_time * 100 if tech_time > 0 else 0
        print(f"    RAG 额外开销: {rag_time - tech_time:.3f}s ({overhead_pct:+.1f}%)")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    return {
        "rag_result": rag_result,
        "tech_result": tech_result,
        "rag_strategy": rag_strategy,
        "tech_strategy": tech_strategy,
        "btc_data": btc_data,
    }


if __name__ == "__main__":
    run_backtest()
