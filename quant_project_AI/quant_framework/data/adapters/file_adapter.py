"""File data adapter (文件数据适配器).

Supports CSV, Parquet, HDF5. When polars is installed, parquet/csv use Polars for reading then convert to Pandas.
支持 CSV、Parquet、HDF5；若已安装 polars 则 parquet/csv 优先用 Polars 读再转 Pandas。
"""

import logging
import os

import pandas as pd
from typing import Optional
from datetime import datetime

from .base_adapter import BaseDataAdapter

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


class FileDataAdapter(BaseDataAdapter):
    """File data adapter (文件数据适配器). Supports CSV, Parquet, HDF5."""
    
    def __init__(self, data_dir: str = "data", preferred_format: str = "parquet", **kwargs):
        """Initialize file adapter (初始化文件适配器).

        Args:
            data_dir: Data directory (数据目录)
            preferred_format: Preferred format (首选格式) ('csv', 'parquet', 'hdf5')
        """
        super().__init__(kwargs)
        self.data_dir = data_dir
        self.preferred_format = preferred_format.lower()
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def _get_file_path(self, symbol: str, format: Optional[str] = None) -> str:
        """Get file path (获取文件路径)."""
        fmt = format or self.preferred_format
        filename = f"{symbol}.{fmt}"
        return os.path.join(self.data_dir, filename)
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data (加载数据). When polars is available, parquet/csv use Polars for reading then convert to Pandas."""
        formats = [self.preferred_format, "parquet", "csv", "hdf5"]
        for fmt in formats:
            file_path = self._get_file_path(symbol, fmt)
            if not os.path.exists(file_path):
                continue
            try:
                if fmt == "parquet" and _POLARS_AVAILABLE:
                    df = pl.read_parquet(file_path).to_pandas()
                elif fmt == "csv" and _POLARS_AVAILABLE:
                    df = pl.read_csv(file_path).to_pandas()
                elif fmt == "parquet":
                    df = pd.read_parquet(file_path)
                elif fmt == "hdf5":
                    df = pd.read_hdf(file_path, key=symbol)
                elif fmt == "csv":
                    df = pd.read_csv(file_path)
                else:
                    continue
                if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                else:
                    raise ValueError("无法找到日期列")
                df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
                df = df.sort_values("date").reset_index(drop=True)
                required_columns = ["date", "open", "high", "low", "close", "volume"]
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise ValueError(f"缺少必要的列: {missing}")
                return df[required_columns]
            except Exception:
                logger.debug("Failed to load data from %s", file_path, exc_info=True)
                continue
        return None
    
    def save_data(self, symbol: str, data: pd.DataFrame, format: Optional[str] = None) -> bool:
        """Save data (保存数据)."""
        try:
            fmt = format or self.preferred_format
            file_path = self._get_file_path(symbol, fmt)
            
            # 确保日期列存在
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                data['date'] = pd.to_datetime(data['date'])
            
            # 按格式保存
            if fmt == 'parquet':
                data.to_parquet(file_path, index=False, compression='snappy')
            elif fmt == 'hdf5':
                data.to_hdf(file_path, key=symbol, mode='w', format='table')
            elif fmt == 'csv':
                data.to_csv(file_path, index=False)
            else:
                # 默认使用parquet
                data.to_parquet(file_path, index=False, compression='snappy')
            
            return True
        except Exception as e:
            logger.exception("保存数据失败: %s", e)
            return False
    
    def check_connection(self) -> bool:
        """Check connection (检查连接，验证文件系统访问)."""
        return os.path.exists(self.data_dir) and os.access(self.data_dir, os.W_OK)
    
    def get_available_symbols(self) -> list:
        """Get available symbol list (获取可用的股票代码列表)."""
        symbols = []
        if not os.path.exists(self.data_dir):
            return symbols
        
        for filename in os.listdir(self.data_dir):
            # 移除扩展名
            base_name = os.path.splitext(filename)[0]
            if base_name not in symbols:
                symbols.append(base_name)
        
        return sorted(symbols)
    
    def get_latest_date(self, symbol: str) -> Optional[datetime]:
        """Get latest data date (获取最新数据日期)."""
        df = self.load_data(symbol, "2000-01-01", "2099-12-31")
        if df is not None and not df.empty:
            return df['date'].max()
        return None
