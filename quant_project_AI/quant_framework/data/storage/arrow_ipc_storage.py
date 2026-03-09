"""
Arrow IPC (Feather) 存储
读写使用 PyArrow Feather；读取时 memory_map=True 实现零拷贝/按需换页，适合本地极速 I/O。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
from typing import Optional, List

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather


class ArrowIpcStorage:
    """
    Arrow IPC (Feather v2) 存储。
    写：PyArrow Table 写 Feather；读：memory_map=True 零拷贝打开，再按日期过滤转 Pandas。
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    def save(self, symbol: str, data: pd.DataFrame, **kwargs) -> bool:
        try:
            path = os.path.join(self.data_dir, f"{symbol}.arrow")
            if "date" in data.columns:
                data = data.copy()
                data["date"] = pd.to_datetime(data["date"])
            table = pa.Table.from_pandas(data)
            feather.write_feather(table, path, compression="lz4", **kwargs)
            return True
        except Exception as e:
            logger.exception("Arrow IPC 写入失败: %s", e)
            return False

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        path = os.path.join(self.data_dir, f"{symbol}.arrow")
        if not os.path.isfile(path):
            return None
        try:
            import pyarrow.compute as pc

            source = pa.ipc.open_file(pa.memory_map(path, "r"))
            table = source.read_all()
            if columns is not None:
                existing = [c for c in columns if c in table.column_names]
                if "date" not in existing and "date" in table.column_names:
                    existing = ["date"] + existing
                table = table.select(existing)

            if "date" in table.column_names:
                date_col = pc.cast(table.column("date"), pa.timestamp("us"))
                if start_date is not None:
                    mask = pc.greater_equal(date_col, pa.scalar(pd.Timestamp(start_date), type=pa.timestamp("us")))
                    table = table.filter(mask)
                    date_col = pc.cast(table.column("date"), pa.timestamp("us"))
                if end_date is not None:
                    mask = pc.less_equal(date_col, pa.scalar(pd.Timestamp(end_date), type=pa.timestamp("us")))
                    table = table.filter(mask)

            df = table.to_pandas(self_destruct=True)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.exception("Arrow IPC 读取失败: %s", e)
            return None
