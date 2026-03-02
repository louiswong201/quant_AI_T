"""Data module: DataManager, Dataset, RagContextProvider, CacheManager.

数据管理模块 — 统一数据访问层。
"""

from .cache_manager import CacheManager
from .data_manager import DataManager
from .dataset import Dataset
from .rag_context import RagContextProvider

__all__ = ["DataManager", "Dataset", "RagContextProvider", "CacheManager"]
