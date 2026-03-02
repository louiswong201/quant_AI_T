"""
RAG 与回测融合示例：策略在 on_bar 中使用 RAG 上下文（按标的与当前日期）
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from quant_framework.data import DataManager, RagContextProvider
from quant_framework.rag import Document, RAGPipeline, RAGConfig
from quant_framework.strategy.base_strategy import BaseStrategy
from quant_framework.backtest.backtest_engine import BacktestEngine


class RAGAwareStrategy(BaseStrategy):
    """示例：在 on_bar 中获取当前标的的 RAG 上下文并参与决策（此处仅打印）。"""

    def on_bar(self, data, current_date, current_prices=None):
        if getattr(data, "__len__", lambda: 0)() < 20:
            return {"action": "hold", "symbol": "AAPL"}
        df = data if isinstance(data, pd.DataFrame) else (list(data.values())[0] if data else pd.DataFrame())
        if len(df) < 20:
            return {"action": "hold", "symbol": "AAPL"}
        symbol = "AAPL"  # 示例
        # 若注入了 rag_provider，可获取与该标的相关的非结构化上下文（回测时 as_of_date=current_date）
        ctx = self.get_rag_context(
            query="近期新闻与业绩",
            symbol=symbol,
            top_k=3,
            max_chars=500,
            as_of_date=current_date,
        )
        if ctx:
            pass  # 此处可将 ctx 送入模型或规则做信号增强
        # 简单返回 hold，实际可结合 ctx 与 data 生成 buy/sell
        return {"action": "hold", "symbol": symbol}


def main():
    # 1) RAG 管道与上下文提供者（与 DataManager 平级）
    rag_config = RAGConfig(chunk_size=256, use_hybrid=True)
    rag_pipeline = RAGPipeline(config=rag_config, start_worker=True)
    rag_provider = RagContextProvider(pipeline=rag_pipeline)

    # 2) 写入若干文档（可来自新闻/研报接口）
    rag_provider.pipeline.add_documents([
        Document(content="苹果公司发布新款iPhone，销量超预期。", source="news_1"),
        Document(content="美联储维持利率不变，市场波动加剧。", source="news_2"),
    ])

    # 3) 策略注入 rag_provider，回测时 get_rag_context(as_of_date=current_date) 可用
    strategy = RAGAwareStrategy(name="RAG策略", initial_capital=1000000, rag_provider=rag_provider)
    data_manager = DataManager(data_dir="data", use_parquet=True)
    engine = BacktestEngine(data_manager=data_manager)

    # 4) 运行回测（需有对应标的与日期范围的行情数据）
    try:
        result = engine.run(strategy, "AAPL", "2020-01-01", "2020-12-31")
        print("回测完成, 最终净值:", result["final_value"])
    except Exception as e:
        print("回测跳过（可能缺少数据）:", e)
    print("RAG 与回测融合示例结束。")


if __name__ == "__main__":
    main()
