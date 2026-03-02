# 数据文件保存与读取：极速 I/O 方案

在保证兼容性的前提下，框架提供**多层存储**，读时按「最快可用」顺序尝试，写时可**同步写入多种格式**，便于下次极速读。

---

## 一、存储层级与顺序（读）

| 优先级 | 格式 | 实现 | 特点 |
|--------|------|------|------|
| 1 | **Binary Mmap** | `BinaryMmapStorage` | 固定记录（date int64 + OHLCV float32），无压缩；`np.memmap` + `searchsorted` 按日期切片，按需换页，同机读最快。 |
| 2 | **Arrow IPC (Feather)** | `ArrowIpcStorage` | 列存，`read_feather(..., memory_map=True)` 零拷贝/按需换页，LZ4 压缩可选。 |
| 3 | **Parquet** | `ParquetStorage` | 列存+压缩，可选 Polars 读；兼容性最好，适合归档与跨机。 |
| 4 | **适配器** | File/API/DB | CSV、API 等回源。 |

启用极速 I/O 时（`Dataset(fast_io=True)` 或 `DataManager(fast_io=True)`），**读**按 1→2→3→4 尝试；**写**时在原有保存逻辑上**同时写入** binary 与 arrow（若 `fast_io`），下次加载即可走 1 或 2。

---

## 二、各格式对比（为何这样选）

| 维度 | Binary Mmap | Arrow IPC | Parquet |
|------|-------------|-----------|---------|
| **读速度（同机）** | 最高：mmap 按需换页，无解压 | 高：memory_map 零拷贝 | 中：需解压，可选 Polars 加速 |
| **写速度** | 高：顺序写连续块 | 中高：列存序列化 | 中：压缩+编码 |
| **磁盘占用** | 大：无压缩 | 中：LZ4 | 小：列存+压缩 |
| **可移植性** | 同机/同架构 | 跨机 | 跨机、生态最好 |
| **日期范围查询** | searchsorted 切片，O(log n) 定位 | 读全表再过滤 | 谓词下推（Polars/PyArrow） |

- **Binary Mmap**：借鉴本地 tick/bar 常见做法（固定记录 + 内存映射），适合回测热路径、单机、对延迟敏感。
- **Arrow IPC**：与 Pandas/Arrow 生态一致，零拷贝读，适合「一次写入、多次按范围读」的本地分析。
- **Parquet**：保留为默认与归档格式，便于与 Polars、Spark、DuckDB 等互通。

---

## 三、使用方式

```python
from quant_framework.data import DataManager

# 启用极速 I/O：读优先 binary → arrow → parquet；写时同步写 binary + arrow
dm = DataManager(data_dir="data", use_parquet=True, fast_io=True)
df = dm.load_data("STOCK", "2020-01-01", "2023-12-31")
# 首次可能从 parquet 或适配器加载，并自动写入 .bin / .arrow，下次同区间即从 binary 或 arrow 读
```

仅用极速层（不写 Parquet）可单独使用存储类：

```python
from quant_framework.data.storage import BinaryMmapStorage, ArrowIpcStorage

binary = BinaryMmapStorage(data_dir="data")
binary.save("STOCK", df)
df2 = binary.load("STOCK", "2020-01-01", "2023-12-31")

arrow = ArrowIpcStorage(data_dir="data")
arrow.save("STOCK", df)
df3 = arrow.load("STOCK", "2020-01-01", "2023-12-31")
```

---

## 四、Binary 格式说明（便于自研或排查）

- **文件**：`{symbol}.bin`
- **头**：32 字节 = magic "QFM\0" (4) + version (2) + reserved (2) + n_rows (8) + start_ts (8) + end_ts (8)，均为 little-endian。
- **记录**：每行 28 字节 = date (int64, 纳秒时间戳) + open, high, low, close, volume (各 float32)。
- **读**：`np.memmap(..., dtype=record_dtype, offset=32, shape=(n_rows,))`，对 `arr["date"]` 做 `searchsorted` 得到 start/end 下标，再 `np.copy(arr[i0:i1])` 转 DataFrame。

---

## 五、可选扩展（未实现）

- **DuckDB 直查 Parquet**：不加载全表，`SELECT * FROM 'x.parquet' WHERE date BETWEEN ? AND ?` 返回 DataFrame，适合大表按区间取数。
- **Zarr**：分块数组，适合云存储与多维/多标的统一存储。
- **冷热分层**：近期数据用 binary/arrow，历史用 Parquet，按日期自动选择后端。

当前实现已在**不改上层 API** 的前提下，通过 `fast_io` 和三层存储把「能快的 I/O」做到极致；需要再快时可优先考虑加大 binary 命中率（先跑一遍预热写入）或按需加 DuckDB 查询路径。
