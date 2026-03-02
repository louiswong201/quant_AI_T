"""
二进制内存映射存储（极速 I/O）
固定记录布局：每行 date(int64) + open/high/low/close/volume(float32)，无压缩，mmap 按需换页。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
import struct
from typing import Optional, List

import numpy as np
import pandas as pd

_MAGIC = b"QFM\0"
_HEADER_SIZE = 32
_RECORD_DTYPE = np.dtype([
    ("date", np.int64),
    ("open", np.float32),
    ("high", np.float32),
    ("low", np.float32),
    ("close", np.float32),
    ("volume", np.float32),
])
_RECORD_SIZE = _RECORD_DTYPE.itemsize  # 8 + 4*5 = 28


class BinaryMmapStorage:
    """
    每标的一个 .bin 文件：32 字节头 + n_rows * 28 字节记录（date ns, o,h,l,c,v）。
    读：np.memmap 打开，searchsorted 按日期范围切片，零拷贝视图或小范围 copy。
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    def save(self, symbol: str, data: pd.DataFrame, **kwargs) -> bool:
        try:
            path = os.path.join(self.data_dir, f"{symbol}.bin")
            if "date" not in data.columns or "close" not in data.columns:
                return False
            n = len(data)
            if n == 0:
                return False
            ts = pd.to_datetime(data["date"]).astype(np.int64).values
            rec = np.empty(n, dtype=_RECORD_DTYPE)
            rec["date"] = ts
            rec["open"] = data["open"].astype(np.float32).values if "open" in data.columns else 0.0
            rec["high"] = data["high"].astype(np.float32).values if "high" in data.columns else 0.0
            rec["low"] = data["low"].astype(np.float32).values if "low" in data.columns else 0.0
            rec["close"] = data["close"].astype(np.float32).values
            rec["volume"] = data["volume"].astype(np.float32).values if "volume" in data.columns else 0.0
            header = _MAGIC + struct.pack("<HH", 1, 0) + struct.pack("<q", n) + struct.pack("<qq", int(ts[0]), int(ts[-1]))
            assert len(header) == _HEADER_SIZE, (len(header), _HEADER_SIZE)
            with open(path, "wb") as f:
                f.write(header)
                rec.tofile(f)
            return True
        except Exception as e:
            logger.exception("Binary Mmap 写入失败: %s", e)
            return False

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        path = os.path.join(self.data_dir, f"{symbol}.bin")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                head = f.read(_HEADER_SIZE)
            if head[:4] != _MAGIC:
                return None
            n_rows = struct.unpack("<q", head[8:16])[0]
            arr = np.memmap(path, dtype=_RECORD_DTYPE, mode="r", offset=_HEADER_SIZE, shape=(n_rows,))
            ts_start = pd.Timestamp(start_date).value if start_date else arr["date"][0]
            ts_end = pd.Timestamp(end_date).value if end_date else arr["date"][-1]
            i0 = np.searchsorted(arr["date"], ts_start, side="left")
            i1 = np.searchsorted(arr["date"], ts_end, side="right")
            i0, i1 = max(0, i0), min(n_rows, i1)
            if i0 >= i1:
                return pd.DataFrame()
            block = np.copy(arr[i0:i1])
            df = pd.DataFrame(block)
            df["date"] = pd.to_datetime(df["date"], unit="ns")
            col_order = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in col_order if c in df.columns]]
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            return df.reset_index(drop=True)
        except Exception as e:
            logger.exception("Binary Mmap 读取失败: %s", e)
            return None
