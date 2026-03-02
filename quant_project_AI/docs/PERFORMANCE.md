# 性能与数据结构：Pandas 与 NumPy 的边界

本文档说明框架在**热路径**上为何用 NumPy、在**接口边界**上保留 Pandas，以及每一类优化的理由。

---

## 一、原则：热路径用 NumPy，接口边界用 Pandas

| 层级 | 推荐 | 原因 |
|------|------|------|
| **回测内循环（每 bar）** | NumPy 数组、O(1) 索引、零拷贝 | 避免每 bar 的 DataFrame.copy()、iloc 的 Python 层开销；预分配数组避免 list 扩容。 |
| **策略入参** | 仍传 DataFrame（或 iloc 切片） | 策略写法简单、与生态兼容；约定「只读不写」即可不 copy。 |
| **指标计算** | 输入/输出优先 NumPy，内部用 Numba/NumPy | 已用 .values 与 Numba；fallback 用 NumPy（如 convolve）替代 pd.Series.rolling。 |
| **最终结果** | 时间序列用 NumPy 数组，表格用 DataFrame | 与现有分析/绘图接口一致。 |

---

## 二、回测引擎热路径优化（逐项理由）

### 2.1 预提取 OHLC/close 为 NumPy 数组

- **原来**：每 bar 用 `df.iloc[i]["close"]`、`bar_df.iloc[i]["open"]` 等，即多次 Python 索引 + 可能的类型转换。
- **现在**：在 `run()` 开始时一次提取 `close_arrays[sym] = df["close"].to_numpy(dtype=np.float64)`（及 open/high/low），循环内用 `close_arrays[sym][i]`、`ohlc[sym][0][i]` 等。
- **理由**：单次 iloc 在 pandas 中有索引与 Block 查找；NumPy 下标是 O(1) 且连续内存，缓存友好；float64 连续数组也便于后续若用 Numba/C 扩展。

### 2.2 策略接收 iloc 切片且不 copy

- **原来**：`data_by_symbol[sym].iloc[: i + 1].copy()`，每 bar 复制整段历史，总代价 O(n²)。
- **现在**：传 `data_by_symbol[sym].iloc[: i + 1]`（不 copy）；约定策略**不得修改**传入的 data。
- **理由**：iloc 切片在 pandas 中多为视图；去掉 copy 后，内循环从「每 bar 一次大拷贝」降为「仅传引用」，大幅减内存与 CPU。

### 2.3 portfolio_values / daily_returns 预分配

- **原来**：`portfolio_values.append(...)`、`daily_returns.append(...)`，list 动态扩容可能多次 realloc。
- **现在**：`portfolio_values_arr = np.empty(n, dtype=np.float64)`，循环内 `portfolio_values_arr[i] = pv`。
- **理由**：一次分配、固定大小，无扩容；返回的即为 NumPy 数组，与 analysis 入参一致。

### 2.4 统一 float64 与连续性

- **现在**：通过 `_to_f64()` 保证 OHLC/close 为 `np.float64` 且 `c_contiguous`。
- **理由**：避免循环内隐式转换；若将来在 C/Numba 中做聚合或统计，连续 float64 是最小公约数。

### 2.5 策略层使用预计算指标列（极速路径）

- **原来**：策略在 `on_bar` 内每 bar 调用 `rolling(...).mean()`、`calculate_rsi()`、`calculate_macd()` 等，相当于每 bar 对整段历史重算，总代价 O(n²)。
- **现在**：引擎在回测开始前对 data 调用一次 `calculate_indicators(data)`，DataFrame 已含 `ma5`/`ma10`/`ma20`、`rsi`、`macd`/`macd_signal`/`macd_hist` 等列；MA/RSI/MACD 策略在 `on_bar` 内优先检测这些列是否存在且参数与默认一致，若存在则仅做 `df["ma5"].iloc[-1]`、`df["rsi"].iloc[-1]` 等 O(1) 读取（极速路径），否则才回退到按需计算。
- **理由**：指标在循环外用 Numba/向量化算一次；策略层只读最后一两个值，避免在热路径上重复 rolling/ewm 链，实现「极速计算」。

---

## 三、指标模块：Pandas 退场处

- **calculate_all**：仍对 DataFrame 按列赋值（如 `df['ma5'] = ...`），在**加载/预处理阶段**调用一次，不在回测每 bar 调用，故保留 DataFrame 便于与存储/特征表一致。
- **ma() / bollinger_bands() 无 Numba 时的 fallback**：原用 `pd.Series(prices).rolling(...)`，现改为纯 NumPy（如 `np.convolve` 做滚动均值、方差 = E[X²]-E[X]² 做滚动标准差），避免在无 Numba 环境下仍创建 Series 与 rolling 对象。
- **rsi / macd 等 fallback**：无 Numba 时已用纯 NumPy 单循环实现，热路径零 Pandas。

---

## 四、Numba 是最快的吗？还有更快方案吗？

**单线程、单标的**：在 Python 生态里，**Numba 已是接近最优的实用选择**——JIT 把循环编译成机器码，且易用（加装饰器即可）。同量级的替代方案：

| 方案 | 特点 | 适用场景 |
|------|------|----------|
| **Numba** | JIT、易集成、首跑有编译缓存 | 当前默认；循环型指标、单进程 |
| **Cython** | AOT 编译成 C，可关 boundscheck | 需要极致单核、可接受写 .pyx 与编译 |
| **Rust (PyO3)** | 内存/线程安全、无 GIL 时可多线程 | 新模块或对安全/部署要求高 |
| **纯 NumPy 向量化** | 无 JIT、无新依赖 | 能写成 array 运算的指标（如 MA 用 convolve） |
| **GPU (CuPy/Numba CUDA)** | 大规模并行 | 大批量标的/长序列一次性算 |

结论：**不必为了「再快一点」强行换 Cython/Rust**，除非你要压榨最后几个百分点或做 C/Rust 库对接。更划算的加速是**多标的并行**。

### 4.1 多标的并行（已做）

- **多标的回测**时，各标的的 `calculate_indicators(df)` 互不依赖，已在引擎内用 `ThreadPoolExecutor` 并行（线程数 ≤ 标的数与 8 的较小值）。
- Numba/NumPy 计算会释放 GIL，多线程可真实并行，墙钟时间随标的数增加而接近线性缩短（在 CPU 核数允许范围内）。

---

## 五、还有哪些可继续压榨

- **策略层**：若策略只依赖 close（或少量列），可增加「轻量模式」：引擎传入 `close[:i+1]` 等数组 + current_date，策略返回信号，从而完全不在内循环碰 DataFrame（需新接口）。
- **多标的对齐**：当前用 pandas 的 reindex/ffill/bfill 做日期对齐；若标的很多，可考虑用 NumPy 的 searchsorted + 预分配矩阵做对齐，减少临时 DataFrame。
- **结果表 results**：目前仍是 list of dict 再 `pd.DataFrame(results)`；若列固定，可预分配数组（如 date、portfolio_value、cash 等）再一次性建 DataFrame，减少中间 dict 与 append。

---

## 六、小结

- **热路径**：回测循环内用 NumPy 数组、不 copy 历史切片、预分配序列数组；指标有 Numba 用 Numba、无 Numba 用纯 NumPy，零 Pandas。
- **多标的**：指标计算按标的并行（ThreadPoolExecutor），墙钟时间可随标的数近似线性下降。
- **接口**：策略仍收 DataFrame（或切片），便于可读与兼容；约定只读即可。
- **极限**：单核上 Numba 已是 Python 内最实用的极速方案；更快可考虑 Cython/Rust 或 GPU，但收益/成本需按项目权衡。
