"""
缓存管理器（生产级）
内存 LRU + 按字节淘汰；DataFrame 磁盘缓存用 Parquet，与框架存储一致；支持按前缀失效。
"""

from __future__ import annotations

import logging
import os
import pickle
import hashlib
import threading
from typing import Iterator, Optional, Dict, Any, Callable, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _data_key(symbol: str, start_date: str, end_date: str) -> str:
    """稳定、可前缀匹配的 load_data 缓存键。"""
    return f"data|{symbol}|{start_date}|{end_date}"


def _key_to_filename(key: str) -> str:
    """键转安全文件名（用于磁盘）。"""
    return key.replace("|", "_").replace("/", "-")




class LRUCache:
    """LRU 缓存：按访问顺序淘汰，O(1) get/put。"""

    __slots__ = ("_cache", "_max_size")

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def remove(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)

    def keys_with_prefix(self, prefix: str) -> Iterator[str]:
        """Iterate keys starting with prefix (迭代 key 以 prefix 开头的键，用于按前缀失效)."""
        for k in list(self._cache.keys()):
            if k.startswith(prefix):
                yield k

    def pop_oldest(self) -> Optional[Tuple[str, Any]]:
        """弹出并返回最久未访问的 (key, value)，不触发访问；用于按 LRU 顺序淘汰。"""
        if not self._cache:
            return None
        key = next(iter(self._cache))
        value = self._cache[key]
        del self._cache[key]
        return (key, value)


class CacheManager:
    """
    缓存管理器：内存 LRU + 按字节上限淘汰；磁盘缓存 DataFrame 用 Parquet、其它用 pickle。
    线程安全；支持按 key 前缀失效（如按 symbol 失效）。
    """

    def __init__(
        self,
        max_memory_items: int = 128,
        max_memory_size_mb: float = 512.0,
        disk_cache_dir: Optional[str] = None,
        enable_disk_cache: bool = True,
    ):
        self._memory = LRUCache(max_memory_items)
        self._max_bytes = int(max_memory_size_mb * 1024 * 1024)
        self._current_bytes = 0
        self._enable_disk = bool(enable_disk_cache)
        self._disk_dir = os.path.join("cache", "disk") if disk_cache_dir is None else disk_cache_dir
        if self._enable_disk and not os.path.exists(self._disk_dir):
            os.makedirs(self._disk_dir, exist_ok=True)
        self._lock = threading.RLock()

    def _estimate_size(self, obj: Any) -> int:
        if isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(deep=True).sum())
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _disk_path(self, key: str, is_dataframe: bool) -> str:
        safe = _key_to_filename(key)
        return os.path.join(self._disk_dir, f"{safe}.parquet" if is_dataframe else f"{safe}.pkl")

    def get(self, key: str, copy: bool = True) -> Optional[Any]:
        with self._lock:
            val = self._memory.get(key)
            if val is not None:
                return val.copy() if (copy and isinstance(val, pd.DataFrame)) else val

            if not self._enable_disk:
                return None

            path_parquet = self._disk_path(key, True)
            path_pkl = self._disk_path(key, False)
            if os.path.isfile(path_parquet):
                try:
                    val = pd.read_parquet(path_parquet)
                    size = self._estimate_size(val)
                    if size + self._current_bytes <= self._max_bytes:
                        self._evict_until(size)
                        self._memory.put(key, val)
                        self._current_bytes += size
                    return val.copy() if (copy and isinstance(val, pd.DataFrame)) else val
                except Exception:
                    logger.debug("Cache operation failed", exc_info=True)
            if os.path.isfile(path_pkl):
                try:
                    with open(path_pkl, "rb") as f:
                        val = pickle.load(f)
                    size = self._estimate_size(val)
                    if size + self._current_bytes <= self._max_bytes:
                        self._evict_until(size)
                        self._memory.put(key, val)
                        self._current_bytes += size
                    return val.copy() if (copy and isinstance(val, pd.DataFrame)) else val
                except Exception:
                    logger.debug("Cache operation failed", exc_info=True)
            return None

    def _evict_until(self, need_bytes: int) -> None:
        """按 LRU 顺序淘汰内存项直到满足 need_bytes 空间。"""
        while self._current_bytes + need_bytes > self._max_bytes and self._memory.size() > 0:
            pair = self._memory.pop_oldest()
            if pair is not None:
                _, old_val = pair
                self._current_bytes -= self._estimate_size(old_val)

    def put(self, key: str, value: Any, to_disk: bool = True) -> None:
        size = self._estimate_size(value)
        with self._lock:
            if size > self._max_bytes:
                if self._enable_disk and to_disk and isinstance(value, pd.DataFrame):
                    try:
                        path = self._disk_path(key, True)
                        value.to_parquet(path, index=False)
                    except Exception:
                        logger.debug("Cache operation failed", exc_info=True)
                return

            self._evict_until(size)
            self._memory.put(key, value)
            self._current_bytes += size

            if self._enable_disk and to_disk:
                try:
                    if isinstance(value, pd.DataFrame):
                        value.to_parquet(self._disk_path(key, True), index=False)
                    else:
                        with open(self._disk_path(key, False), "wb") as f:
                            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    logger.debug("Cache operation failed", exc_info=True)

    def remove(self, key: str) -> None:
        with self._lock:
            val = self._memory.get(key)
            if val is not None:
                self._current_bytes -= self._estimate_size(val)
                self._memory.remove(key)
            for path in (self._disk_path(key, True), self._disk_path(key, False)):
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        logger.debug("Cache operation failed", exc_info=True)

    def remove_keys_with_prefix(self, prefix: str) -> int:
        """删除所有 key 以 prefix 开头的项（内存+磁盘）。返回删除条数。"""
        count = 0
        with self._lock:
            for k in list(self._memory.keys_with_prefix(prefix)):
                val = self._memory.get(k)
                if val is not None:
                    self._current_bytes -= self._estimate_size(val)
                self._memory.remove(k)
                count += 1
                for path in (self._disk_path(k, True), self._disk_path(k, False)):
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except Exception:
                            logger.debug("Cache operation failed", exc_info=True)
            if self._enable_disk and os.path.isdir(self._disk_dir):
                safe_prefix = _key_to_filename(prefix)
                for name in os.listdir(self._disk_dir):
                    if name.startswith(safe_prefix) and (name.endswith(".parquet") or name.endswith(".pkl")):
                        try:
                            os.remove(os.path.join(self._disk_dir, name))
                            count += 1
                        except Exception:
                            logger.debug("Cache operation failed", exc_info=True)
        return count

    def clear(self, clear_disk: bool = False) -> None:
        with self._lock:
            self._memory.clear()
            self._current_bytes = 0
            if clear_disk and self._enable_disk and os.path.isdir(self._disk_dir):
                for name in os.listdir(self._disk_dir):
                    if name.endswith(".parquet") or name.endswith(".pkl"):
                        try:
                            os.remove(os.path.join(self._disk_dir, name))
                        except Exception:
                            logger.debug("Cache operation failed", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats: Dict[str, Any] = {
                "memory_items": self._memory.size(),
                "memory_size_mb": round(self._current_bytes / (1024 * 1024), 4),
                "max_memory_mb": round(self._max_bytes / (1024 * 1024), 4),
                "disk_enabled": self._enable_disk,
            }
            if self._enable_disk and os.path.isdir(self._disk_dir):
                items = [f for f in os.listdir(self._disk_dir) if f.endswith(".parquet") or f.endswith(".pkl")]
                total = sum(os.path.getsize(os.path.join(self._disk_dir, f)) for f in items)
                stats["disk_items"] = len(items)
                stats["disk_size_mb"] = round(total / (1024 * 1024), 4)
            return stats

    def cached(self, key_func: Optional[Callable[..., str]] = None):
        """装饰器：用本 CacheManager 缓存函数结果。"""

        def deco(func: Callable[..., Any]):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                k = key_func(*args, **kwargs) if key_func else hashlib.md5(
                    (func.__name__ + str(args) + str(sorted(kwargs.items()))).encode()
                ).hexdigest()
                with self._lock:
                    val = self._memory.get(k)
                if val is not None:
                    return val.copy() if isinstance(val, pd.DataFrame) else val
                if self._enable_disk:
                    path_p = self._disk_path(k, True)
                    path_pkl = self._disk_path(k, False)
                    if os.path.isfile(path_p):
                        try:
                            out = pd.read_parquet(path_p)
                            self.put(k, out, to_disk=False)
                            return out
                        except Exception:
                            logger.debug("Cache operation failed", exc_info=True)
                    if os.path.isfile(path_pkl):
                        try:
                            with open(path_pkl, "rb") as f:
                                out = pickle.load(f)
                            self.put(k, out, to_disk=False)
                            return out
                        except Exception:
                            logger.debug("Cache operation failed", exc_info=True)
                out = func(*args, **kwargs)
                self.put(k, out)
                return out
            return wrapper
        return deco
