# RAG 模块：使用方式与结构

本文档说明 **quant_framework.rag** 的用途、配置、使用方式以及与量化框架的融合；详细优化与对比见 [OPTIMIZATION.md](OPTIMIZATION.md)、[COMPARISON.md](COMPARISON.md)。

---

## 一、设计目标

- **实时**：支持流式/队列接入非结构化数据，边进边处理边入库。  
- **高效**：内存向量索引 + 关键词 BM25、批量嵌入、可选持久化扩展。  
- **准确**：混合检索（向量 + 关键词）+ 可选 Cross-Encoder 重排，提升召准率。

---

## 二、架构简图

```
  文档源 → IngestQueue → [Worker] → ProcessingPipeline → VectorStore + KeywordIndex
                                                              ↑
  用户 Query → embed → vector_search_topk + keyword search → RRF → Reranker → 上下文
```

- **实时接入**：IngestQueue（有界队列）、IngestStream、FileWatcherIngestAdapter、DirectoryIngestAdapter。  
- **处理**：TextNormalizer → Chunker → Embedder；整批分块 + 单次批量嵌入。  
- **存储**：InMemoryVectorStore（COW、可选 Numba 的 core dot+top-k）、KeywordIndex（BM25）。  
- **检索**：HybridRetriever（RRF）、VectorOnlyRetriever；元数据过滤、批量检索、可选重排。  
- **管道**：RAGPipeline 串联接入→处理→存储与查询→检索→重排；非阻塞 ingest、后台 worker、去重、健康检查、异步 API、Prompt 模板。

---

## 三、目录结构（quant_framework/rag/）

| 文件/目录 | 说明 |
|-----------|------|
| `config.py` | 全局配置（分块、嵌入、检索、重排、队列、突发上限等）。 |
| `core.py` | 数值热路径（dot + top-k），可选 Numba；可替换为 Rust/C++ 的边界。 |
| `types.py` | Document、Chunk、SearchResult。 |
| `pipeline.py` | RAGPipeline 入口。 |
| `ingestion/` | 实时接入：BaseIngestAdapter、IngestQueue、IngestStream、FileWatcherIngestAdapter、DirectoryIngestAdapter。 |
| `processing/` | 文档处理：normalizer、chunker、embedder、ProcessingPipeline。 |
| `store/` | VectorStore/InMemoryVectorStore、KeywordIndex。 |
| `retrieval/` | HybridRetriever、VectorOnlyRetriever、Reranker。 |

---

## 四、与量化框架融合

- **数据层**：`quant_framework.data.RagContextProvider` 持有 RAG 管道，提供 `get_context(query, symbol=..., as_of_date=...)`，与 DataManager 平级。  
- **策略**：`BaseStrategy(..., rag_provider=RagContextProvider(...))`，在 `on_bar` 中调用 `self.get_rag_context(query, symbol, as_of_date=current_date)` 获取与当前 bar 时间一致的上下文，避免回测用未来信息。  
- 详见 [ARCHITECTURE.md](ARCHITECTURE.md) 与示例 `examples/example_rag_with_backtest.py`。

---

## 五、使用方式（要点）

1. **批量写入**：`pipeline.add_documents([Document(...), ...])`（整批分块 + 单次批量嵌入）。  
2. **实时写入**：`pipeline.ingest_put(Document(...))` 非阻塞入队并立即返回；构造时可选 `start_worker=True`，后台 daemon 自动聚批处理并入库。  
3. **文件监听**：`FileWatcherIngestAdapter(watch_path).run_forever(on_docs=...)`，在 `on_docs` 里调用 `pipeline.add_documents(docs)`。  
4. **检索**：`pipeline.retrieve(query, top_k=5, metadata_filter=...)`；**批量**：`pipeline.retrieve_batch(queries, top_k=5)`。  
5. **生成上下文**：`pipeline.get_context_for_generation(query, top_k=5, max_chars=4000, context_strategy="concat"|"merge_adjacent")`。  
6. **元数据过滤**：`metadata_filter={"source_contains": "news", "created_before": datetime, "key": value}`。  
7. **异步**：`await pipeline.async_retrieve(query)`、`await pipeline.async_ingest_put(doc)`。  
8. **可观测**：`pipeline.get_stats()`、`pipeline.health_check()`；可选 `on_retrieve_callback` / `on_embed_callback`。  
9. **Prompt 模板**：`from quant_framework.rag import format_prompt, list_templates`；`format_prompt("alpaca", query=..., context=...)` 等。

---

## 六、依赖

- **基础**：无额外依赖（使用 DummyEmbedder 时可跑通流程）。  
- **推荐**：`pip install sentence-transformers`，用于真实嵌入与可选 Cross-Encoder 重排。  
- **可选**：`numba` 用于 core 热路径加速；后续可替换为 Chroma/FAISS 等持久化向量库（实现 VectorStore 接口）。

---

## 七、文档索引

- 整体架构与数据流：[ARCHITECTURE.md](ARCHITECTURE.md)  
- 优化说明（易懂版、循环、极限、配置）：[OPTIMIZATION.md](OPTIMIZATION.md)  
- 与 LangChain/LlamaIndex/Haystack 对比：[COMPARISON.md](COMPARISON.md)  
- 项目总索引：[INDEX.md](INDEX.md)
