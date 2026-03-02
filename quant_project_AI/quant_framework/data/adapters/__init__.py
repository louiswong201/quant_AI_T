"""数据适配器模块"""

from .base_adapter import BaseDataAdapter
from .file_adapter import FileDataAdapter
from .database_adapter import DatabaseDataAdapter
from .api_adapter import APIDataAdapter

__all__ = [
    'BaseDataAdapter',
    'FileDataAdapter',
    'DatabaseDataAdapter',
    'APIDataAdapter'
]

