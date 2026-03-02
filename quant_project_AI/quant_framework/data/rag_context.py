"""
RAG 上下文提供者：与 DataManager 平级，为策略提供按标的/日期/查询的非结构化上下文。
用于回测时保证“仅用 as_of_date 之前入库的文档”，与量化框架时间轴一致。
"""

from datetime import datetime
from typing import Optional

from ..rag import RAGPipeline
from ..rag.config import RAGConfig
from ..rag.types import Document


class RagContextProvider:
    """
    为策略提供 RAG 检索上下文，可绑定 as_of_date 以在回测中避免未来信息。
    与 DataManager 平级注入策略，由 BacktestEngine 或调用方在构造策略时传入。
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None, config: Optional[RAGConfig] = None):
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = RAGPipeline(config=config or RAGConfig(), start_worker=True)

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    def get_context(
        self,
        query: str,
        symbol: Optional[str] = None,
        top_k: int = 5,
        max_chars: int = 4000,
        as_of_date: Optional[datetime] = None,
        metadata_filter: Optional[dict] = None,
        context_strategy: str = "concat",
    ) -> str:
        """
        获取用于 LLM 或策略的检索上下文字符串。

        Args:
            query: 检索查询（如「苹果公司近期新闻」）。
            symbol: 可选标的，可拼进 query。
            top_k: 返回片段数。
            max_chars: 上下文最大字符数。
            as_of_date: 回测时传入当前 bar 日期，仅返回 created_at < as_of_date 的文档（与 pipeline 元数据过滤一致）。
            metadata_filter: 额外元数据过滤，与 as_of_date 合并后传给检索。
            context_strategy: 合成策略，"concat" | "merge_adjacent"。

        Returns:
            拼接好的上下文字符串。
        """
        if symbol:
            q = f"{query} {symbol}".strip()
        else:
            q = query
        mf = dict(metadata_filter) if metadata_filter else {}
        if as_of_date is not None:
            mf["created_before"] = as_of_date
        if mf:
            results = self._pipeline.retrieve(q, top_k=top_k, metadata_filter=mf)
            parts = []
            total = 0
            for r in results:
                block = f"[{r.rank}] {r.chunk.text}\n"
                if total + len(block) > max_chars:
                    break
                parts.append(block)
                total += len(block)
            return "\n".join(parts) if parts else ""
        return self._pipeline.get_context_for_generation(
            q, top_k=top_k, max_chars=max_chars, context_strategy=context_strategy
        )

    def retrieve(self, query: str, top_k: int = 5):
        """直接返回检索结果列表，供策略自行拼接或打分。"""
        return self._pipeline.retrieve(query, top_k=top_k)

    def ingest_document(self, content: str, source: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        """实时接入单条文档（非阻塞入队）。"""
        doc = Document(content=content, source=source, metadata=metadata or {})
        return self._pipeline.ingest_put(doc)

    def add_documents(self, docs: list) -> int:
        """批量写入文档（同步处理并入库）。docs 为 Document 列表或 dict 列表（需含 'content'）。"""
        from ..rag.types import Document as RAGDocument
        rag_docs: list = []
        for d in docs:
            if isinstance(d, RAGDocument):
                rag_docs.append(d)
            elif isinstance(d, dict):
                rag_docs.append(RAGDocument(
                    content=d.get("content", ""),
                    source=d.get("source"),
                    metadata=d.get("metadata", {}),
                ))
            else:
                rag_docs.append(RAGDocument(content=str(d)))
        return self._pipeline.add_documents(rag_docs)
