"""
可视化模块
绘制回测结果图表
"""

import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class Plotter:
    """图表绘制器"""
    
    def __init__(self, figsize: tuple = (15, 10), style: str = 'seaborn-v0_8'):
        """
        初始化绘图器
        
        Args:
            figsize: 图表大小
            style: matplotlib样式
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_backtest_results(self, results_df: pd.DataFrame, 
                             portfolio_values: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        绘制回测结果
        
        Args:
            results_df: 回测结果DataFrame
            portfolio_values: 投资组合价值数组
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        dates = pd.to_datetime(results_df['date'])
        
        # 1. 投资组合价值曲线
        axes[0].plot(dates, portfolio_values, label='投资组合价值', linewidth=2)
        axes[0].axhline(y=results_df.iloc[0]['portfolio_value'], 
                       color='r', linestyle='--', alpha=0.5, label='初始资金')
        axes[0].set_ylabel('价值 (元)', fontsize=12)
        axes[0].set_title('投资组合价值曲线', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 累计收益率曲线
        initial_value = portfolio_values[0]
        cumulative_returns = (portfolio_values - initial_value) / initial_value * 100
        axes[1].plot(dates, cumulative_returns, label='累计收益率', 
                    linewidth=2, color='green')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('收益率 (%)', fontsize=12)
        axes[1].set_title('累计收益率曲线', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 回撤曲线
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max * 100
        axes[2].fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='回撤')
        axes[2].plot(dates, drawdown, linewidth=1, color='red')
        axes[2].set_ylabel('回撤 (%)', fontsize=12)
        axes[2].set_xlabel('日期', fontsize=12)
        axes[2].set_title('回撤曲线', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 格式化x轴日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("图表已保存到: %s", save_path)
        else:
            plt.show()
    
    def plot_trades(self, data_df: pd.DataFrame, trades_df: pd.DataFrame,
                   save_path: Optional[str] = None) -> None:
        """
        在价格图上标注买卖点
        
        Args:
            data_df: 价格数据DataFrame
            trades_df: 交易记录DataFrame
            save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        dates = pd.to_datetime(data_df['date'])
        prices = data_df['close']
        
        # 绘制价格曲线
        ax.plot(dates, prices, label='收盘价', linewidth=1.5, alpha=0.7)
        
        # 标注买卖点
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            if not buy_trades.empty:
                buy_dates = pd.to_datetime(buy_trades['date'])
                buy_prices = buy_trades['price']
                ax.scatter(buy_dates, buy_prices, color='green', marker='^', 
                          s=100, label='买入', zorder=5)
            
            if not sell_trades.empty:
                sell_dates = pd.to_datetime(sell_trades['date'])
                sell_prices = sell_trades['price']
                ax.scatter(sell_dates, sell_prices, color='red', marker='v', 
                          s=100, label='卖出', zorder=5)
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格 (元)', fontsize=12)
        ax.set_title('交易信号图', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("图表已保存到: %s", save_path)
        else:
            plt.show()
    
    def plot_comparison(self, portfolio_values: np.ndarray,
                       benchmark_values: np.ndarray,
                       dates: pd.DatetimeIndex,
                       save_path: Optional[str] = None) -> None:
        """
        绘制策略与基准的对比
        
        Args:
            portfolio_values: 策略投资组合价值
            benchmark_values: 基准投资组合价值
            dates: 日期索引
            save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 归一化到初始值
        portfolio_normalized = portfolio_values / portfolio_values[0] * 100
        benchmark_normalized = benchmark_values / benchmark_values[0] * 100
        
        ax.plot(dates, portfolio_normalized, label='策略', linewidth=2)
        ax.plot(dates, benchmark_normalized, label='基准', linewidth=2, 
               linestyle='--', alpha=0.7)
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('归一化价值 (初始=100)', fontsize=12)
        ax.set_title('策略 vs 基准', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("图表已保存到: %s", save_path)
        else:
            plt.show()
