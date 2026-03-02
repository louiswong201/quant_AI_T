"""
RAG 端到端管道：实时接入（非阻塞 put + 后台 worker）→ 处理 → 入库；查询 → 检索 → 重排
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from .config import RAGConfig
from .ingestion import IngestQueue, IngestStream
from .processing import Chunker, ProcessingPipeline
from .processing.embedder import BaseEmbedder, SentenceTransformerEmbedder, DummyEmbedder
from .retrieval import Reranker, Retriever
from .retrieval.reranker import IdentityReranker
from .retrieval.retriever import HybridRetriever, VectorOnlyRetriever
from .store import InMemoryVectorStore, KeywordIndex, VectorStore
from .types import Document, SearchResult

logger = logging.getLogger(__name__)


def _default_stop() -> bool:
    return False


class RAGPipeline:
    """
    一站式 RAG：ingest_put 非阻塞入队，后台 daemon worker 聚批处理并入库；
    检索路径短临界区、与写入并发安全。
    """

    def __init__(self, config: Optional[RAGConfig] = None, start_worker: bool = True):
        self.config = config or RAGConfig()

        try:
            embedder_candidate = SentenceTransformerEmbedder(
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
            dim = embedder_candidate.dimension
            self.embedder: BaseEmbedder = embedder_candidate
        except Exception:
            logger.debug("SentenceTransformerEmbedder init failed, using DummyEmbedder", exc_info=True)
            self.embedder = DummyEmbedder(dimension=self.config.embedding_dim)
            dim = self.embedder.dimension

        chunker = Chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            by_sentence=self.config.chunk_by_sentence,
            min_chunk_length=self.config.min_chunk_length,
        )
        self.processing = ProcessingPipeline(chunker=chunker, embedder=self.embedder)
        ttl = getattr(self.config, "ttl_seconds", None)
        self.vector_store: VectorStore = InMemoryVectorStore(
            dimension=dim,
            max_vectors=self.config.max_vectors,
            ttl_seconds=ttl,
        )
        if self.config.vector_store_path:
            self.vector_store.load(self.config.vector_store_path)
        self.keyword_index = KeywordIndex()
        cache_size = getattr(self.config, "query_embedding_cache_size", 0) or 0

        if self.config.use_hybrid:
            vw, kw = self.config.hybrid_weights
            self.retriever: Retriever = HybridRetriever(
                vector_store=self.vector_store,
                keyword_index=self.keyword_index,
                embedder=self.embedder,
                vector_weight=vw,
                keyword_weight=kw,
                vector_top_k=self.config.vector_top_k,
                keyword_top_k=self.config.keyword_top_k,
                query_embedding_cache_size=cache_size,
            )
        else:
            self.retriever = VectorOnlyRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                vector_top_k=self.config.vector_top_k,
                query_embedding_cache_size=cache_size,
            )

        if self.config.rerank_model:
            try:
                from .retrieval.reranker import CrossEncoderReranker
                self.reranker: Reranker = CrossEncoderReranker(
                    model_name=self.config.rerank_model,
                    top_k=self.config.rerank_top_k,
                )
            except Exception:
                logger.debug("CrossEncoderReranker init failed, using IdentityReranker", exc_info=True)
                self.reranker = IdentityReranker()
        else:
            self.reranker = IdentityReranker()

        self._ingest_queue = IngestQueue(max_size=self.config.ingest_queue_max_size)
        self._worker_stop = threading.Event()
        self._worker_started = False
        self._dedup_hashes: OrderedDict = OrderedDict()
        self._dedup_max = max(0, getattr(self.config, "dedup_cache_max", 0))
        self._last_retrieve_sec: Optional[float] = None
        if start_worker:
            self._start_ingest_worker()

    def _start_ingest_worker(self) -> None:
        """启动后台 daemon：take_blocking 聚批 → add_documents，不阻塞主线程。"""
        if self._worker_started:
            return
        self._worker_started = True

        def _run() -> None:
            batch_size = self.config.process_batch_size
            cap = getattr(self.config, "max_process_batch", None)
            take_size = min(batch_size, cap) if cap is not None else batch_size
            timeout = 0.1
            while not self._worker_stop.is_set():
                batch = self._ingest_queue.take_blocking(batch_size=take_size, timeout=timeout)
                if batch:
                    self.add_documents(batch)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def add_documents(self, docs: List[Document]) -> int:
        """批量添加：可选去重 → 整批分块 + 单次批量嵌入 → 写入存储；向量库淘汰时同步删关键词索引。返回写入的 chunk 数。"""
        if not docs:
            return 0
        if getattr(self.config, "skip_duplicate_docs", True) and self._dedup_max > 0:
            hashes = [hashlib.sha256(d.content.encode()).hexdigest() for d in docs]
            filtered = []
            hashes_to_add = []
            for d, h in zip(docs, hashes):
                if h in self._dedup_hashes:
                    self._dedup_hashes.move_to_end(h)
                    continue
                hashes_to_add.append(h)
                filtered.append(d)
            for h in hashes_to_add:
                self._dedup_hashes[h] = True
                self._dedup_hashes.move_to_end(h)
            while len(self._dedup_hashes) > self._dedup_max:
                self._dedup_hashes.popitem(last=False)
            docs = filtered
        if not docs:
            return 0
        chunks = self.processing.process_documents(docs)
        if not chunks:
            return 0
        cb = getattr(self.config, "on_embed_callback", None)
        if callable(cb):
            try:
                cb(len(chunks))
            except Exception:
                logger.debug("on_embed_callback failed", exc_info=True)
        evicted = self.vector_store.add(chunks)
        self.keyword_index.add(chunks)
        if evicted:
            self.keyword_index.remove_chunk_ids(evicted)
        return len(chunks)

    def ingest_put(self, doc: Document) -> bool:
        """实时写入：非阻塞入队，立即返回；由后台 worker 异步处理。"""
        return self._ingest_queue.put(doc)

    def run_ingest_consumer(
        self,
        batch_size: Optional[int] = None,
        on_batch: Optional[Callable[[List[Document]], None]] = None,
        stop_event: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        同步消费入口（如测试或不用后台 worker 时）：从队列取批并处理；
        stop_event() 为 True 时退出。若已 start_worker，通常不需要再调用此方法。
        """
        batch_size = batch_size or self.config.process_batch_size
        stream = IngestStream(self._ingest_queue, batch_size=batch_size)
        stop = stop_event or _default_stop

        def _on_batch(batch: List[Document]) -> None:
            if on_batch:
                on_batch(batch)
            self.add_documents(batch)

        stream.stream_continuous(on_batch=_on_batch, stop_event=stop)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        t0 = time.perf_counter()
        top_k = top_k or self.config.vector_top_k
        rerank_k = rerank_k or self.config.rerank_top_k or top_k
        results = self.retriever.retrieve(query, top_k=top_k * 2, metadata_filter=metadata_filter)
        results = self.reranker.rerank(query, results, top_k=rerank_k)
        self._last_retrieve_sec = time.perf_counter() - t0
        cb = getattr(self.config, "on_retrieve_callback", None)
        if callable(cb):
            try:
                cb(query, results)
            except Exception:
                logger.debug("on_retrieve_callback failed", exc_info=True)
        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """批量检索，一次嵌入 + 矩阵检索，再逐条重排，延迟与 CPU 更优。"""
        top_k = top_k or self.config.vector_top_k
        rerank_k = rerank_k or self.config.rerank_top_k or top_k
        raw = self.retriever.retrieve_batch(queries, top_k=top_k * 2, metadata_filter=metadata_filter)
        return [
            self.reranker.rerank(q, res, top_k=rerank_k) for q, res in zip(queries, raw)
        ]

    def get_context_for_generation(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 4000,
        context_strategy: Optional[str] = None,
    ) -> str:
        results = self.retrieve(query, top_k=top_k)
        strategy = context_strategy or "concat"
        synthesis_chars = getattr(self.config, "synthesis_chunk_chars", 0) or 0
        if strategy == "merge_adjacent" and synthesis_chars > 0:
            return self._format_context_merge_adjacent(results, max_chars, synthesis_chars)
        parts = []
        total = 0
        for r in results:
            block = f"[{r.rank}] {r.chunk.text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts) if parts else ""

    def _format_context_merge_adjacent(
        self, results: List[SearchResult], max_chars: int, synthesis_chunk_chars: int
    ) -> str:
        """按 doc_id + index 相邻合并成块，再拼接至 max_chars，利于连贯性。"""
        if not results:
            return ""
        ordered = sorted(results, key=lambda r: (r.chunk.doc_id, r.chunk.index))
        blocks = []
        current_doc, current_idx, current_text = None, None, []
        current_len = 0
        for r in ordered:
            c = r.chunk
            if c.doc_id != current_doc or (current_idx is not None and c.index != current_idx + 1):
                if current_text:
                    blocks.append(" ".join(current_text))
                current_doc, current_idx = c.doc_id, c.index
                current_text = [c.text]
                current_len = len(c.text)
            else:
                current_text.append(c.text)
                current_len += len(c.text)
                current_idx = c.index
                if current_len >= synthesis_chunk_chars:
                    blocks.append(" ".join(current_text))
                    current_text = []
                    current_len = 0
        if current_text:
            blocks.append(" ".join(current_text))
        out = []
        total = 0
        for i, b in enumerate(blocks):
            block = f"[{i + 1}] {b}\n"
            if total + len(block) > max_chars:
                break
            out.append(block)
            total += len(block)
        return "\n".join(out) if out else ""

    async def async_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """异步检索，不阻塞事件循环。"""
        return await asyncio.to_thread(
            self.retrieve, query, top_k=top_k, rerank_k=rerank_k, metadata_filter=metadata_filter
        )

    async def async_ingest_put(self, doc: Document) -> bool:
        """异步写入单条文档（非阻塞入队）。"""
        return await asyncio.to_thread(self.ingest_put, doc)

    def get_stats(self) -> dict:
        """健康检查 + 最近检索延迟，便于监控。"""
        d = self.health_check()
        if self._last_retrieve_sec is not None:
            d["last_retrieve_sec"] = round(self._last_retrieve_sec, 4)
        return d

    def health_check(self) -> dict:
        """运行状态快照：队列长度、向量条数、关键词条数，便于监控与排障。"""
        return {
            "ingest_queue_size": self._ingest_queue.size(),
            "ingest_queue_max_size": self.config.ingest_queue_max_size,
            "vector_store_size": self.vector_store.size(),
            "keyword_index_size": self.keyword_index.size(),
            "worker_started": self._worker_started,
        }

    def save_store(self, directory: Optional[str] = None) -> None:
        """持久化向量存储到磁盘。"""
        path = directory or self.config.vector_store_path
        if path and hasattr(self.vector_store, "save"):
            self.vector_store.save(path)

    @property
    def ingest_queue(self) -> IngestQueue:
        return self._ingest_queue
