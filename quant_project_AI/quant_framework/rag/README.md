# RAG 模块

本目录为 **Quant Framework** 的 RAG（检索增强）实现：实时接入、分块、嵌入、向量/关键词混合检索、重排与上下文生成，并与量化策略通过 `RagContextProvider` 融合。

## 目录概览

- **config / types / pipeline**：配置、数据类型、RAGPipeline 入口。  
- **ingestion/**：队列、流、文件监听、目录扫描等接入适配器。  
- **processing/**：文本规范化、分块、嵌入、处理管道。  
- **store/**：内存向量库（COW）、BM25 关键词索引。  
- **retrieval/**：混合检索、重排。

## 文档与示例

- **使用方式、配置、与量化融合**：见项目根下 [docs/RAG.md](../../docs/RAG.md)。  
- **整体架构（量化 + RAG）**：见 [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)。  
- **优化说明与对比**：见 [docs/OPTIMIZATION.md](../../docs/OPTIMIZATION.md)、[docs/COMPARISON.md](../../docs/COMPARISON.md)。  
- **示例**：`examples/example_rag.py`、`examples/example_rag_with_backtest.py`。

本目录内原有详细优化文档（OPTIMIZATION.md、LOOP_OPTIMIZATION.md、LIMIT_OPTIMIZATION.md、COMPARISON.md）已合并或迁移至 `docs/`，便于统一查阅。
