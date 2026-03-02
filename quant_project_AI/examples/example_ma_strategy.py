"""
移动平均线策略回测示例
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_framework import DataManager, BacktestEngine
from quant_framework.strategy import MovingAverageStrategy
from quant_framework.analysis.performance import PerformanceAnalyzer
from quant_framework.visualization.plotter import Plotter


def main():
    """主函数"""
    # 初始化组件
    data_manager = DataManager(data_dir="data")
    strategy = MovingAverageStrategy(
        name="MA策略",
        initial_capital=1000000,
        short_window=5,
        long_window=20
    )
    backtest_engine = BacktestEngine(data_manager, commission_rate=0.001)
    analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
    plotter = Plotter()
    
    # 设置回测参数
    symbol = "STOCK"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"开始回测: {symbol} ({start_date} 至 {end_date})")
    print(f"策略: {strategy.name}")
    print(f"初始资金: {strategy.initial_capital:,.2f} 元\n")
    
    # 运行回测
    results = backtest_engine.run(strategy, symbol, start_date, end_date)
    
    # 性能分析
    performance = analyzer.analyze(
        results['portfolio_values'],
        results['daily_returns'],
        results['initial_capital']
    )
    
    # 交易分析
    trades_analysis = analyzer.analyze_trades(results['trades'])
    
    # 打印摘要
    analyzer.print_summary(performance, trades_analysis)
    
    # 绘制图表
    os.makedirs("output", exist_ok=True)
    print("正在生成图表...")
    plotter.plot_backtest_results(
        results['results'],
        results['portfolio_values'],
        save_path="output/backtest_results.png"
    )
    
    plotter.plot_trades(
        data_manager.load_data(symbol, start_date, end_date),
        results['trades'],
        save_path="output/trades_chart.png"
    )
    
    print("回测完成！")


if __name__ == "__main__":
    main()
