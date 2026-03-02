"""
RAG 配置（含校验，避免无效参数导致难排查问题）
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGConfig:
    """RAG 全局配置"""

    # 分块
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_by_sentence: bool = True
    min_chunk_length: int = 0  # 短于此次的块丢弃，提高准确性

    # 嵌入
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # 向量存储
    vector_store_path: Optional[str] = None
    vector_top_k: int = 20
    max_vectors: Optional[int] = None  # 超过则 FIFO 淘汰，控制内存
    ttl_seconds: Optional[float] = None  # 文档过期秒数，None=不过期

    # 混合检索
    use_hybrid: bool = True
    keyword_top_k: int = 20
    hybrid_weights: tuple = (0.6, 0.4)
    rerank_top_k: int = 5
    rerank_model: Optional[str] = None
    query_embedding_cache_size: int = 256  # 相同 query 复用嵌入，0 表示关闭

    # 实时接入
    ingest_queue_max_size: int = 10000
    process_batch_size: int = 16
    process_workers: int = 2
    # 突发保护：单批处理上限，避免信息量/交易量突增时单批过大导致延迟尖刺与 OOM
    max_process_batch: Optional[int] = None  # None = 不封顶，等于 process_batch_size 逻辑上限

    # 准确性：文档去重
    skip_duplicate_docs: bool = True
    dedup_cache_max: int = 100_000  # 内容 hash 缓存条数

    # 合成策略：merge_adjacent 时合并相邻块直到该字符数（0=不合并，仅 concat）
    synthesis_chunk_chars: int = 0

    # 可观测：回调（可选）
    on_retrieve_callback: Optional[Any] = None  # (query, results) -> None
    on_embed_callback: Optional[Any] = None    # (n_texts: int) -> None

    # 扩展配置
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 须满足 0 <= chunk_overlap < chunk_size")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size 必须大于 0")
        if self.ingest_queue_max_size <= 0:
            raise ValueError("ingest_queue_max_size 必须大于 0")
        if self.process_batch_size <= 0:
            raise ValueError("process_batch_size 必须大于 0")
        if self.max_process_batch is not None and self.max_process_batch <= 0:
            raise ValueError("max_process_batch 若设置则必须大于 0")
        if self.max_vectors is not None and self.max_vectors <= 0:
            raise ValueError("max_vectors 若设置则必须大于 0")
        if self.dedup_cache_max < 0:
            raise ValueError("dedup_cache_max 不能为负")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_by_sentence": self.chunk_by_sentence,
            "min_chunk_length": self.min_chunk_length,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size,
            "vector_store_path": self.vector_store_path,
            "vector_top_k": self.vector_top_k,
            "max_vectors": self.max_vectors,
            "use_hybrid": self.use_hybrid,
            "keyword_top_k": self.keyword_top_k,
            "hybrid_weights": self.hybrid_weights,
            "rerank_top_k": self.rerank_top_k,
            "rerank_model": self.rerank_model,
            "query_embedding_cache_size": self.query_embedding_cache_size,
            "ingest_queue_max_size": self.ingest_queue_max_size,
            "process_batch_size": self.process_batch_size,
            "process_workers": self.process_workers,
            "max_process_batch": self.max_process_batch,
            "skip_duplicate_docs": self.skip_duplicate_docs,
            "dedup_cache_max": self.dedup_cache_max,
            "synthesis_chunk_chars": self.synthesis_chunk_chars,
            "extra": self.extra,
        }
