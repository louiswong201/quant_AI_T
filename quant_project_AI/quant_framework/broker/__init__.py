"""Broker 抽象与纸交易实现，用于实盘对接与延迟验证。"""

from .base import Broker
from .paper import PaperBroker

__all__ = ["Broker", "PaperBroker"]
