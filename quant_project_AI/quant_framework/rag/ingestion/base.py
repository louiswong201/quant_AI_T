"""
接入适配器基类：统一非结构化数据源接口
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional

from ..types import Document


class BaseIngestAdapter(ABC):
    """非结构化数据接入适配器基类"""

    @abstractmethod
    def fetch(self) -> Iterator[Document]:
        """拉取一批/单个文档（同步迭代）"""
        pass

    async def fetch_async(self) -> AsyncIterator[Document]:
        """拉取文档（异步迭代），默认用 sync 包装"""
        for doc in self.fetch():
            yield doc
