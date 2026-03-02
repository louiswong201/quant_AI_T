"""Base data adapter (数据适配器基类).

Defines unified data access interface.
定义统一的数据访问接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime


class BaseDataAdapter(ABC):
    """Base data adapter (数据适配器基类). Defines unified data access interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化适配器
        
        Args:
            config: 适配器配置字典
        """
        self.config = config or {}
    
    @abstractmethod
    def load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        加载数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            如果数据不存在，返回None
        """
        pass
    
    @abstractmethod
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        保存数据
        
        Args:
            symbol: 股票代码
            data: 要保存的数据
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def check_connection(self) -> bool:
        """
        检查数据源连接是否正常
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    def get_available_symbols(self) -> list:
        """
        获取可用的股票代码列表
        
        Returns:
            股票代码列表
        """
        return []
    
    def get_latest_date(self, symbol: str) -> Optional[datetime]:
        """
        获取指定股票的最新数据日期
        
        Args:
            symbol: 股票代码
            
        Returns:
            最新日期，如果不存在返回None
        """
        return None

