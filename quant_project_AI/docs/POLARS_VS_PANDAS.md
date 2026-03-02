# Polars 与 Pandas 对比及在本框架中的选用

本文档从**性能、内存、API、生态**四方面对比 Polars 与 Pandas，并说明本框架的改造策略：**I/O 与重计算用 Polars（可选），热路径与策略接口保持 NumPy/Pandas 兼容**。

---

## 一、对比总览

| 维度 | Polars | Pandas | 本框架更适用 |
|------|--------|--------|--------------|
| **实现** | Rust 内核 + Arrow，多线程 | C/Cython + NumPy，单线程为主 | Polars：I/O、join、过滤 |
| **读 Parquet** | 原生 Arrow，常比 Pandas 快一个数量级 | 经 PyArrow 再转 Pandas | Polars |
| **读 CSV** | 多线程解析，远快于 Pandas | 单线程 | Polars |
| **Join / 对齐** | 多线程 join，内存效率高 | 单线程，reindex 易产生临时表 | Polars（多标的对齐） |
| **过滤 / 筛选** | 惰性 + 谓词下推，约 4x+ | 布尔索引 | Polars |
| **逐行 / 切片** | 非强项，表达式与列为主 | iloc/loc 成熟 | Pandas（策略收切片） |
| **Rolling/EWM** | 有 rolling，API 不同 | 成熟，与 sklearn/指标库兼容好 | Pandas/NumPy（指标已用 NumPy） |
| **内存** | Arrow 列存，常更省、零拷贝多 | 行式 Block，拷贝多 | Polars（大表加载） |
| **生态** | 增长快，与 Arrow/DuckDB 互通 | 极广（sklearn、talib、回测库） | Pandas 做接口兼容 |
| **API 风格** | 表达式、链式、无 in-place | 命令式、in-place 多 | 策略层保留 Pandas 更易写 |

**结论（针对本框架）**：  
- **读文件、多标的按日期对齐**：用 Polars 更合适（更快、更省内存）。  
- **回测内循环、策略入参、指标计算**：已用 NumPy 或 Pandas 切片，且与现有策略/分析兼容，**保持 Pandas/NumPy**；若全面切 Polars，需改所有策略 API 与下游分析，收益主要在 I/O 已由 Polars 承担，内循环收益有限。

---

## 二、分项说明

### 2.1 读 Parquet / CSV

- **Polars**：`pl.read_parquet`、`pl.scan_parquet`（惰性）、`pl.read_csv` 多线程，大文件常比 Pandas 快约 **5–17x**，内存更稳。  
- **Pandas**：`pd.read_parquet` 依赖 PyArrow，再转 Pandas，多一次拷贝。  
- **本框架**：数据加载在 Dataset/Adapter/ParquetStorage，**不在**回测每 bar 调用，改用 Polars 读再 `.to_pandas()` 一次，即可获得 I/O 加速且下游无需改。

### 2.2 多标的按日期对齐

- **Polars**：`join(on=date, how="inner")` 多线程，中间结果列存。  
- **Pandas**：`reindex` + `ffill/bfill` 或 `merge`，易产生多份临时 DataFrame。  
- **本框架**：多标的回测里对齐只做一次，用 Polars 做 join 再转 Pandas，可减少时间和内存峰值。

### 2.3 回测内循环与策略

- **热路径**：已改为预提取 NumPy（OHLC/close）、预分配数组、传 DataFrame 的 iloc 切片（不 copy）。  
- **策略**：接收 `data.iloc[:i+1]`（Pandas）或 `Dict[symbol, DataFrame]`，内部用 `data['close']`、`.iloc[-1]`、`.rolling()` 等。若改为 Polars，需统一改为 `pl.col("close")`、`.row(-1)`、`.rolling()` 等，**所有策略都要改**。  
- **取舍**：内循环成本已主要在 NumPy；策略层保留 Pandas 可读性好、与现有代码一致。因此**不在策略层引入 Polars**，仅在 I/O 与对齐层用 Polars。

### 2.4 指标计算

- **现状**：已用 NumPy（及可选 Numba），`df['close'].values` 或 `.to_numpy()` 后算指标，再写回 DataFrame。  
- **若用 Polars**：可 `pl.col("close").to_numpy()` 再走同一套 NumPy/Numba；或写 Polars 表达式版 rolling/ewm，与现有策略/回测接口不一致。  
- **取舍**：指标层保持「NumPy in、NumPy/DataFrame 出」；数据来源可以是 Polars 读入后转的 Pandas，无需在指标内部用 Polars。

---

## 三、本框架的改造策略（采用「更好」的一方）

- **采用 Polars 的部分**（在「更好」的维度上）：  
  - **Parquet/CSV 读取**：在 `ParquetStorage.load`、`FileDataAdapter.load_data` 中，若已安装 Polars，则用 `pl.read_parquet` / `pl.read_csv`，再 `.to_pandas()` 返回，保证下游接口仍为 Pandas。  
  - **多标的对齐**（可选）：在回测引擎中，多标的时用 Polars 做按日 join，再转 Pandas，减少对齐开销。  

- **保留 Pandas/NumPy 的部分**：  
  - **回测热路径**：继续用预提取的 NumPy 数组与预分配序列。  
  - **策略 on_bar 入参**：继续传 Pandas DataFrame（或切片），不改为 Polars。  
  - **指标**：继续基于 NumPy（和现有 Numba），输入来自 DataFrame 的列。  
  - **分析/绘图**：继续消费 Pandas/NumPy，无需改。

- **依赖**：Polars 作为**可选**依赖（`pip install polars`）；未安装时自动退回纯 Pandas 路径，行为与改造前一致。

---

## 四、改造后的数据流（简要）

```
[ 磁盘 ] → Polars 读 (pl.read_parquet/read_csv) → .to_pandas() → Dataset/DataManager
                ↑ 可选；未装 Polars 则 Pandas 读
多标的对齐（可选）：多张 Pandas → 转 Polars → join(on=date) → .to_pandas() → 回测
回测循环：Pandas 切片（只读） + NumPy 数组（OHLC/close）→ 策略 → 信号
```

这样在**不改策略、不改回测热路径逻辑**的前提下，把「Polars 更好的部分」用上，其余继续用 Pandas/NumPy 以保持可读性和生态兼容。

---

## 五、若未来「全 Polars」会怎样

- **优点**：全链路列存、理论上更少拷贝、join/filter 全在 Polars。  
- **成本**：策略必须改为 Polars API（如 `.filter`、`pl.col`、`.row`）；与 sklearn、部分指标库、现有示例代码不兼容，需维护两套或全面迁移。  
- **建议**：当前不采用全 Polars；若后续有「仅 Polars」的研发管线，可单独建一分支或子模块，与现有「Pandas + NumPy 热路径」并存。

---

## 六、已实现的改造（可选 Polars I/O）

- **ParquetStorage.load**：若已安装 `polars`，则用 `pl.read_parquet` 读入，再按 `start_date`/`end_date` 过滤并 `.to_pandas()` 返回；未安装则沿用 PyArrow + Pandas。
- **FileDataAdapter.load_data**：对 parquet/csv，若已安装 `polars`，则用 `pl.read_parquet`/`pl.read_csv` 再 `.to_pandas()`；hdf5 仍用 Pandas。
- **依赖**：在 `requirements.txt` 中已将 `polars` 注释为可选；需要 I/O 加速时执行 `pip install polars` 即可，无需改下游代码。
- **多标的对齐**：当前仍用 Pandas 的 `reindex`/`ffill`/`bfill`；若希望进一步加速，可在回测引擎中增加「转 Polars → join(on=date) → to_pandas」的可选路径，见文档「改造后的数据流」一节。

---

## 七、小结

| 问题 | 结论 |
|------|------|
| Polars 和 Pandas 哪个更好？ | **看场景**：I/O、join、大表过滤 Polars 更好；小表切片、生态、策略可读性 Pandas 仍占优。 |
| 本框架应如何选？ | **混合**：I/O 与对齐用 Polars（可选），热路径与策略用 NumPy/Pandas。 |
| 具体改造 | ParquetStorage/FileAdapter 的 load 优先走 Polars 再 to_pandas；多标的对齐可选 Polars join；其余保持不变。 |
