"""
处理管道：整批文档 → 规范化 → 分块 → 单次批量嵌入（减少调用、提高吞吐）
"""

from itertools import chain
from typing import List, Optional

from ..types import Chunk, Document
from .chunker import Chunker
from .embedder import BaseEmbedder
from .normalizer import TextNormalizer


class ProcessingPipeline:
    """
    批量处理：先对整批文档做规范化与分块得到 all_chunks，
    再对 all_chunks 做一次 embed(texts)，避免 N 次嵌入调用。
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: BaseEmbedder,
        normalizer: Optional[TextNormalizer] = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.normalizer = normalizer or TextNormalizer()

    def _rebuild_doc(self, doc: Document, normalized_content: str) -> Document:
        """Rebuild document with normalized content while preserving created_at for as_of_date filtering."""
        return Document(
            content=normalized_content,
            doc_id=doc.doc_id,
            source=doc.source,
            metadata=doc.metadata,
            created_at=doc.created_at,
        )

    def process_document(self, doc: Document) -> List[Chunk]:
        """单文档路径（兼容旧用法）；仍会单次调用 embed。"""
        normalized = self.normalizer.normalize_and_filter(doc.content)
        if not normalized:
            return []
        chunks = self.chunker.chunk_document(self._rebuild_doc(doc, normalized))
        return self.embedder.embed_chunks(chunks)

    def process_documents(self, docs: List[Document]) -> List[Chunk]:
        """
        整批文档 → 整批分块 → 单次批量嵌入。
        用 chain.from_iterable 一次展开所有 chunk，避免多次 list.extend 与中间列表。
        """
        if not docs:
            return []
        normalized = [
            (doc, self.normalizer.normalize_and_filter(doc.content))
            for doc in docs
        ]
        normalized = [(doc, n) for doc, n in normalized if n]
        if not normalized:
            return []
        all_chunks = list(
            chain.from_iterable(
                self.chunker.chunk_document(self._rebuild_doc(doc, n))
                for doc, n in normalized
            )
        )
        if not all_chunks:
            return []
        return self.embedder.embed_chunks(all_chunks)
