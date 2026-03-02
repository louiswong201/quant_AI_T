"""Database data adapter (数据库数据适配器).

Supports MySQL, PostgreSQL, SQLite and other databases.
支持 MySQL、PostgreSQL、SQLite 等数据库。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from .base_adapter import BaseDataAdapter


class DatabaseDataAdapter(BaseDataAdapter):
    """Database data adapter (数据库数据适配器). Supports MySQL, PostgreSQL, SQLite."""
    
    def __init__(self, connection_string: str, table_name: str = "stock_data", 
                 pool_size: int = 5, max_overflow: int = 10, **kwargs):
        """Initialize database adapter (初始化数据库适配器).

        Args:
            connection_string: Database connection string (数据库连接字符串)
                MySQL: mysql+pymysql://user:pass@host:port/db
                PostgreSQL: postgresql://user:pass@host:port/db
                SQLite: sqlite:///path/to/db.db
            table_name: Table name (数据表名)
            pool_size: Connection pool size (连接池大小)
            max_overflow: Max overflow connections (最大溢出连接数)
        """
        super().__init__(config=kwargs if kwargs else None)
        self.connection_string = connection_string
        self.table_name = table_name
        
        # 创建连接池
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # 连接前检查连接是否有效
            echo=False
        )
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from database (从数据库加载数据)."""
        try:
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM {self.table_name}
            WHERE symbol = :symbol
            AND date >= :start_date
            AND date <= :end_date
            ORDER BY date
            """
            
            df = pd.read_sql(
                text(query),
                self.engine,
                params={'symbol': symbol, 'start_date': start_date, 'end_date': end_date}
            )
            
            if df.empty:
                return None
            
            # 确保日期列为datetime类型
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except SQLAlchemyError as e:
            logger.exception("数据库查询错误: %s", e)
            return None
        except Exception as e:
            logger.exception("加载数据失败: %s", e)
            return None
    
    def save_data(self, symbol: str, data: pd.DataFrame, if_exists: str = "replace") -> bool:
        """Save data to database (保存数据到数据库)."""
        try:
            # 确保有symbol列
            if 'symbol' not in data.columns:
                data = data.copy()
                data['symbol'] = symbol
            
            # 确保日期列为datetime类型
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # 保存到数据库
            data.to_sql(
                self.table_name,
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi'  # 批量插入
            )
            
            return True
            
        except SQLAlchemyError as e:
            logger.exception("数据库保存错误: %s", e)
            return False
        except Exception as e:
            logger.exception("保存数据失败: %s", e)
            return False
    
    def check_connection(self) -> bool:
        """Check database connection (检查数据库连接)."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.debug("check_connection failed", exc_info=True)
            return False
    
    def get_available_symbols(self) -> list:
        """Get available symbol list (获取可用的股票代码列表)."""
        try:
            query = f"SELECT DISTINCT symbol FROM {self.table_name} ORDER BY symbol"
            df = pd.read_sql(text(query), self.engine)
            return df['symbol'].tolist() if not df.empty else []
        except Exception:
            logger.debug("get_available_symbols failed", exc_info=True)
            return []
    
    def get_latest_date(self, symbol: str) -> Optional[datetime]:
        """Get latest data date (获取最新数据日期)."""
        try:
            query = f"""
            SELECT MAX(date) as latest_date
            FROM {self.table_name}
            WHERE symbol = :symbol
            """
            df = pd.read_sql(
                text(query),
                self.engine,
                params={'symbol': symbol}
            )
            
            if not df.empty and df['latest_date'].iloc[0] is not None:
                return pd.to_datetime(df['latest_date'].iloc[0])
            return None
        except Exception:
            logger.debug("get_latest_date failed for symbol=%s", symbol, exc_info=True)
            return None

