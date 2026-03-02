# 设计原则：极速数据处理 + AI 融合

本文档是 **Quant Framework** 的一级设计规范：框架以**极快数据处理**与**AI 融合**为并列目标，文档即规范，实现与之对齐。

---

## 一、双目标

| 目标 | 含义 | 落地方式 |
|------|------|----------|
| **极速数据处理** | 从磁盘到策略的整条数据路径延迟与吞吐最优 | 多层存储（Binary Mmap → Arrow IPC → Parquet）、内存/磁盘缓存、回测热路径 NumPy 化、指标向量化/Numba |
| **AI 融合** | 非结构化数据（新闻/研报）与行情在同一时间轴参与决策 | RagContextProvider 与 DataManager 平级、策略内 `get_rag_context(..., as_of_date=current_date)`、回测不偷看未来 |

二者同等重要：数据层同时服务「结构化极速读」与「非结构化按日检索」；策略层可只用量化、或同时用 RAG 上下文，API 统一。

---

## 二、极速数据处理原则

1. **读路径**  
   启用 `fast_io=True` 时，读顺序为 Binary Mmap → Arrow IPC → Parquet → 适配器；能走 mmap/零拷贝的优先，避免不必要的解压与拷贝。详见 [IO_FAST.md](IO_FAST.md)。

2. **回测热路径**  
   单 bar 内：OHLC/close 预提取为 NumPy 数组、O(1) 索引；策略接收 `iloc` 切片不 copy；`portfolio_values`/日收益预分配数组。详见 [PERFORMANCE.md](PERFORMANCE.md)。

3. **缓存**  
   可选 CacheManager（内存按字节 LRU + 磁盘 Parquet），与存储格式一致；`save_data` 按 symbol 前缀失效，避免脏读。

4. **指标**  
   在加载/预处理阶段集中计算一次，不在回测每 bar 计算；优先 Numba/NumPy，fallback 纯 NumPy 替代 Pandas rolling。

---

## 三、AI 融合原则

1. **时间一致**  
   回测时策略在每个 bar 只能使用「截至当前 bar 日期」的数据。RAG 侧通过 `as_of_date=current_date` 与文档 `created_at` 过滤，保证不偷看未来。

2. **平级注入**  
   DataManager（结构化）与 RagContextProvider（非结构化）由调用方或 BacktestEngine 构造并注入策略；策略通过 `self.get_rag_context(query, symbol, as_of_date=current_date)` 获取上下文。

3. **写入不阻塞**  
   RAG 文档通过 `ingest_put` 非阻塞入队，后台 worker 分块、嵌入、写入；检索读 COW 快照，短临界区，不阻塞回测主循环。

---

## 四、文档即规范

- **架构与数据流**：[ARCHITECTURE.md](ARCHITECTURE.md) 描述分层与量化/RAG 衔接，实现不得与之矛盾。
- **性能与 I/O**：[PERFORMANCE.md](PERFORMANCE.md)、[IO_FAST.md](IO_FAST.md) 描述热路径与存储层级，新增优化需同步更新文档。
- **RAG 使用**：[RAG.md](RAG.md) 描述接入、检索、与策略融合写法；`get_rag_context(as_of_date=...)` 的语义以文档为准。
- **索引**：[INDEX.md](INDEX.md) 维护文档与主题的映射，新文档需加入索引。

凡涉及「极速」或「AI 融合」的 API 行为，以本文档与上述文档为准；代码注释可引用文档章节，避免重复与偏离。

---

## 五、推荐组合（最佳实践）

- **极速回测（纯量化）**  
  `DataManager(data_dir=..., fast_io=True)`，必要时加 `CacheManager`；回测前确保目标区间已写入 binary/arrow（可先跑一遍预热）。

- **极速 + AI 融合回测**  
  同上 DataManager，再 `RagContextProvider(pipeline=...)` 注入策略；策略在 `on_bar` 内调用 `self.get_rag_context(..., as_of_date=current_date)`，与行情同时间点决策。

- **文档阅读顺序**  
  README → ARCHITECTURE → DESIGN（本文）→ IO_FAST / PERFORMANCE（按需）→ RAG.md（用 RAG 时）。

上述组合确保框架在「数据处理极快」与「AI 融合规范」上同时达标，并可由文档复现与审计。
