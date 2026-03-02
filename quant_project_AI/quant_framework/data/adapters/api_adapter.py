"""
API数据适配器
支持REST API数据源，带连接池和重试机制
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
import aiohttp
import asyncio
from typing import Optional, Dict, Any, Callable
from .base_adapter import BaseDataAdapter


class APIDataAdapter(BaseDataAdapter):
    """API数据适配器"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 request_func: Optional[Callable] = None, 
                 max_retries: int = 3, timeout: int = 30,
                 pool_size: int = 10, **kwargs):
        """
        初始化API适配器
        
        Args:
            base_url: API基础URL
            api_key: API密钥（可选）
            request_func: 自定义请求函数（可选）
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            pool_size: 连接池大小
        """
        super().__init__(kwargs)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.request_func = request_func
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.pool_size = pool_size
        
        # 创建连接器（支持连接池）
        self.connector = aiohttp.TCPConnector(limit=pool_size)
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建会话"""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers=headers
            )
        return self.session
    
    async def _fetch_data_async(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """异步获取数据"""
        if self.request_func:
            # 使用自定义请求函数
            return await self.request_func(symbol, start_date, end_date)
        
        # 默认实现：假设API有标准接口
        url = f"{self.base_url}/data"
        params = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        }
        
        session = self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 假设返回的是JSON格式的数据
                        df = pd.DataFrame(data)
                        if not df.empty:
                            df['date'] = pd.to_datetime(df['date'])
                            return df
                    elif response.status == 429:  # 速率限制
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        response.raise_for_status()
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.exception("API请求失败: %s", e)
                return None
        
        return None
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载数据（同步接口）"""
        try:
            # 如果已有事件循环，使用它
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 运行异步函数
            if loop.is_running():
                # 如果循环正在运行，使用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._fetch_data_async(symbol, start_date, end_date)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._fetch_data_async(symbol, start_date, end_date)
                )
        except Exception as e:
            logger.exception("加载数据失败: %s", e)
            return None
    
    async def load_data_async(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """异步加载数据"""
        return await self._fetch_data_async(symbol, start_date, end_date)
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """保存数据（API通常不支持保存，需要实现自定义逻辑）"""
        # API适配器通常只读，保存功能需要自定义实现
        if self.request_func and hasattr(self.request_func, 'save'):
            return self.request_func.save(symbol, data)
        return False
    
    def check_connection(self) -> bool:
        """检查API连接"""
        try:
            session = self._get_session()
            # 尝试发送一个简单的请求
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，使用线程
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._check_connection_async()
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self._check_connection_async())
        except Exception:
            logger.debug("check_connection failed", exc_info=True)
            return False
    
    async def _check_connection_async(self) -> bool:
        """异步检查连接"""
        try:
            session = self._get_session()
            async with session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except Exception:
            logger.debug("_check_connection_async failed", exc_info=True)
            return False
    
    def close(self) -> None:
        """关闭会话和连接器"""
        if self.session and not self.session.closed:
            asyncio.get_event_loop().run_until_complete(self.session.close())
        if self.connector:
            asyncio.get_event_loop().run_until_complete(self.connector.close())
    
    def __del__(self) -> None:
        """析构函数"""
        self.close()

