# 与主流 RAG 框架对比

本框架与 GitHub 上常见的 Python RAG 方案（LangChain、LlamaIndex、Haystack 等）在**能力、依赖、性能与适用场景**上的对比，以及本框架的差异化优势。

---

## 一、能力对比表

| 能力 | 本框架 (quant_framework.rag) | LangChain | LlamaIndex | Haystack |
|------|-----------------------------|-----------|------------|----------|
| **实时写入** | 非阻塞队列 + 后台 worker，立即返回 | 多为同步或需自建 | 同步/异步需配置 | 同步 write_documents |
| **混合检索** | 向量 + BM25 RRF，内置 | 需组合 Vector + BM25 等 | 支持 | BM25 + Embedding 可组合 |
| **检索时元数据过滤** | 支持（source/created_at/自定义） | 依赖 VectorStore 实现 | 支持 | 支持 |
| **批量检索** | 单次嵌入 + 矩阵批量查，低开销 | 多按条调用 | 支持 | 按条为主 |
| **Query 嵌入缓存** | 内置 LRU，可配置大小 | 无内置 | 无内置 | 无内置 |
| **向量库内存控制** | max_vectors FIFO 淘汰 + 关键词同步 | 依赖后端 | 依赖后端 | 依赖 DocumentStore |
| **文档去重** | 内容 hash LRU 去重，可配置 | 需自实现 | 需自实现 | 需自实现 |
| **重排序** | 可选 Cross-Encoder 重排 | 需组合 | 支持 | 可组合 |
| **配置校验** | 启动时校验，失败即报错 | 运行时易暴露 | 运行时 | 运行时 |
| **健康检查** | health_check() + get_stats() | 依赖 LangSmith 等 | 需自建 | 需自建 |
| **异步 API** | async_retrieve / async_ingest_put | 有 async 链 | 有 | 部分 |
| **Prompt 模板** | 内置 format_prompt 常用模板 | 丰富 | 丰富 | 有 PromptBuilder |
| **默认依赖** | 仅 numpy/pandas，嵌入可选 | 重（langchain-* 多包） | 重 | 中等 |
| **与量化回测融合** | 原生 RagContextProvider + as_of_date | 需自建 | 需自建 | 需自建 |

---

## 二、性能与资源（相对印象）

| 维度 | 本框架 | 典型对比 |
|------|--------|----------|
| **写入路径** | 仅入队，O(1)；处理在后台 | LangChain/LlamaIndex 常同步处理，易阻塞 |
| **检索路径** | 短临界区 COW、单矩阵点积、可选批查 | 框架 overhead 约 3–14ms；本框架无额外编排层 |
| **嵌入** | 整批文档单次批量 embed、query 可缓存 | 与通用框架一致，批处理由配置控制 |
| **内存** | 单块 numpy、float32、FIFO 可配 | 无内置上限的框架易 OOM |
| **依赖体积** | 不装 sentence-transformers 也可跑（DummyEmbedder） | 多数方案强依赖嵌入/向量库 |

*注：具体数字依赖数据规模与机器，上表为定性对比。*

---

## 三、适用场景建议

| 场景 | 更合适的选择 |
|------|----------------|
| **量化/回测 + 非结构化上下文** | 本框架（已与 DataManager/BaseStrategy 融合，as_of_date 防未来信息） |
| **强实时写入、低延迟检索** | 本框架（非阻塞入队、COW、可选批检、query 缓存） |
| **轻量部署、少依赖** | 本框架（numpy 即可跑通，嵌入/重排可选） |
| **需要复杂 Agent/多步编排** | LangChain / LangGraph |
| **复杂文档解析、多数据源 ETL** | LlamaIndex |
| **快速搭 RAG 原型、偏标准 Pipeline** | Haystack |

---

## 四、本框架的差异化优势

1. **写入与检索解耦**：写入只入队并返回，检索读 COW 快照，不与写入争锁，延迟稳定。  
2. **检索侧一次到位**：混合检索、元数据过滤、批量检索、重排、合成策略、Prompt 模板均在同一管道内可配，无需多库拼装。  
3. **资源与准确性可控**：max_vectors、去重、min_chunk_length、元数据过滤、合成策略，便于在内存与效果间做权衡。  
4. **可观测与可运维**：health_check、get_stats（含延迟）、on_retrieve/on_embed 回调，便于监控与排障。  
5. **与业务绑定**：为量化场景设计 RagContextProvider 与 as_of_date，回测时间一致、数据层统一。  
6. **零重依赖可运行**：不装嵌入模型也能跑通流程（DummyEmbedder），适合 CI/演示与渐进式接入。

结合 [OPTIMIZATION.md](OPTIMIZATION.md) 与 [RAG.md](RAG.md)，可在架构评审或选型时说明：本框架在**实时性、资源控制、准确性、可观测性与量化融合**上对标并超越常见开源 RAG 方案的默认用法。
