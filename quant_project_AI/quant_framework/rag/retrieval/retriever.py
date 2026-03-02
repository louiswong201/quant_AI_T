"""
检索器：向量 + 关键词混合，结果融合；可选 query 嵌入缓存、元数据过滤、批量检索
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..types import Chunk, SearchResult
from ..store import VectorStore, KeywordIndex


def _metadata_match(meta: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """单条 chunk 的 metadata 是否满足过滤条件，供列表推导一次通过。"""
    for key, val in flt.items():
        if key == "source_contains":
            if str(val) not in str(meta.get("source") or ""):
                return False
        elif key == "created_before":
            ct = meta.get("created_at")
            if ct is not None and val is not None and isinstance(ct, datetime) and isinstance(val, datetime) and ct >= val:
                return False
        elif meta.get(key) != val:
            return False
    return True


def _apply_metadata_filter(
    results: List[Tuple[Chunk, float]],
    metadata_filter: Optional[Dict[str, Any]],
) -> List[Tuple[Chunk, float]]:
    """后过滤：列表推导单次分配，避免循环内 append。"""
    if not metadata_filter or not results:
        return results
    return [(c, sc) for c, sc in results if _metadata_match(c.metadata, metadata_filter)]


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[SearchResult]:
        pass

    def retrieve_batch(
        self, queries: List[str], top_k: int, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """批量检索，默认实现逐条调用 retrieve。子类可重写为一次 embed_batch + search_batch。"""
        return [self.retrieve(q, top_k) for q in queries]


class HybridRetriever(Retriever):
    """向量检索 + 关键词检索融合（RRF）；可选 query embedding 缓存（LRU）"""

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        embedder,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        vector_top_k: int = 20,
        keyword_top_k: int = 20,
        query_embedding_cache_size: int = 0,
    ):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.vector_top_k = vector_top_k
        self.keyword_top_k = keyword_top_k
        self._embed_cache: Optional[OrderedDict] = OrderedDict() if query_embedding_cache_size > 0 else None
        self._embed_cache_max = max(0, query_embedding_cache_size)

    def _rrf_merge(
        self,
        vector_results: List[Tuple[Chunk, float]],
        keyword_results: List[Tuple[Chunk, float]],
        k: int = 60,
    ) -> List[Tuple[Chunk, float]]:
        """RRF：单遍构建 scores 与 chunk_by_id，避免 vector_results + keyword_results 临时大列表。"""
        chunk_by_id: Dict[str, Chunk] = {}
        scores: Dict[str, float] = {}
        for rank, (c, _) in enumerate(vector_results):
            chunk_by_id[c.chunk_id] = c
            scores[c.chunk_id] = scores.get(c.chunk_id, 0) + self.vector_weight / (k + rank + 1)
        for rank, (c, _) in enumerate(keyword_results):
            chunk_by_id[c.chunk_id] = c
            scores[c.chunk_id] = scores.get(c.chunk_id, 0) + self.keyword_weight / (k + rank + 1)
        order = sorted(scores.items(), key=itemgetter(1), reverse=True)
        return [(chunk_by_id[cid], sc) for cid, sc in order if cid in chunk_by_id]

    def _get_query_embedding(self, query: str) -> List[float]:
        if self._embed_cache is not None and self._embed_cache_max > 0:
            if query in self._embed_cache:
                self._embed_cache.move_to_end(query)
                return self._embed_cache[query]
            emb = self.embedder.embed([query])[0]
            self._embed_cache[query] = emb
            self._embed_cache.move_to_end(query)
            while len(self._embed_cache) > self._embed_cache_max:
                self._embed_cache.popitem(last=False)
            return emb
        return self.embedder.embed([query])[0]

    def retrieve(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        q_embedding = self._get_query_embedding(query)
        vec_results = self.vector_store.search(q_embedding, self.vector_top_k)
        kw_results = self.keyword_index.search(query, self.keyword_top_k)
        merged = self._rrf_merge(vec_results, kw_results)
        merged = _apply_metadata_filter(merged, metadata_filter)[:top_k]
        return [SearchResult(chunk=c, score=sc, rank=i + 1) for i, (c, sc) in enumerate(merged)]

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """一次批量嵌入 + search_batch，再逐 query 做关键词与 RRF，优于逐条 retrieve。"""
        if not queries:
            return []
        embeddings = self.embedder.embed(queries)
        vec_batch = self.vector_store.search_batch(embeddings, self.vector_top_k)
        out: List[List[SearchResult]] = []
        for q, vec_results in zip(queries, vec_batch):
            kw_results = self.keyword_index.search(q, self.keyword_top_k)
            merged = self._rrf_merge(vec_results, kw_results)
            merged = _apply_metadata_filter(merged, metadata_filter)[:top_k]
            out.append([SearchResult(chunk=c, score=sc, rank=i + 1) for i, (c, sc) in enumerate(merged)])
        return out


class VectorOnlyRetriever(Retriever):
    """仅向量检索；可选 query embedding 缓存"""

    def __init__(self, vector_store: VectorStore, embedder, vector_top_k: int = 20, query_embedding_cache_size: int = 0):
        self.vector_store = vector_store
        self.embedder = embedder
        self.vector_top_k = vector_top_k
        self._embed_cache: Optional[OrderedDict] = OrderedDict() if query_embedding_cache_size > 0 else None
        self._embed_cache_max = max(0, query_embedding_cache_size)

    def _get_query_embedding(self, query: str) -> List[float]:
        if self._embed_cache is not None and self._embed_cache_max > 0:
            if query in self._embed_cache:
                self._embed_cache.move_to_end(query)
                return self._embed_cache[query]
            emb = self.embedder.embed([query])[0]
            self._embed_cache[query] = emb
            self._embed_cache.move_to_end(query)
            while len(self._embed_cache) > self._embed_cache_max:
                self._embed_cache.popitem(last=False)
            return emb
        return self.embedder.embed([query])[0]

    def retrieve(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        q_embedding = self._get_query_embedding(query)
        results = self.vector_store.search(q_embedding, min(top_k, self.vector_top_k))
        results = _apply_metadata_filter(results, metadata_filter)[:top_k]
        return [SearchResult(chunk=c, score=sc, rank=i + 1) for i, (c, sc) in enumerate(results)]

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        if not queries:
            return []
        embeddings = self.embedder.embed(queries)
        batch = self.vector_store.search_batch(embeddings, min(top_k, self.vector_top_k))
        out: List[List[SearchResult]] = []
        for results in batch:
            results = _apply_metadata_filter(results, metadata_filter)[:top_k]
            out.append([SearchResult(chunk=c, score=sc, rank=i + 1) for i, (c, sc) in enumerate(results)])
        return out
