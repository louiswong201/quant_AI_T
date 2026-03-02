"""
分块器：按字符/句子切分，带重叠（正则编译一次，热路径复用）
"""

import re
from typing import List

from ..types import Chunk, Document


class Chunker:
    """固定长度 + 重叠分块，可选按句子边界对齐"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        by_sentence: bool = True,
        min_chunk_length: int = 0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.by_sentence = by_sentence
        self.min_chunk_length = min_chunk_length
        self._sentence_pattern = re.compile(r"(?<=[。！？.!?\n])\s*")

    def _split_sentences(self, text: str) -> List[str]:
        parts = self._sentence_pattern.split(text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        text = doc.content.strip()
        if not text:
            return []

        if self.by_sentence:
            sentences = self._split_sentences(text)
            segments: List[str] = []
            current = []
            current_len = 0
            for s in sentences:
                s_len = len(s) + 1
                if current_len + s_len > self.chunk_size and current:
                    segments.append(" ".join(current))
                    # overlap: 保留最后几句
                    overlap_len = 0
                    overlap = []
                    for x in reversed(current):
                        if overlap_len + len(x) + 1 <= self.chunk_overlap:
                            overlap.insert(0, x)
                            overlap_len += len(x) + 1
                        else:
                            break
                    current = overlap
                    current_len = sum(len(x) + 1 for x in current)
                current.append(s)
                current_len += s_len
            if current:
                segments.append(" ".join(current))
        else:
            segments = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                segments.append(text[start:end])
                start = end - self.chunk_overlap if end < len(text) else len(text)

        meta_base = {"source": doc.source, **doc.metadata}
        if getattr(doc, "created_at", None) is not None:
            meta_base["created_at"] = doc.created_at
        return [
            Chunk(text=s, chunk_id=f"{doc.doc_id}_{i}", doc_id=doc.doc_id, index=i, metadata=dict(meta_base))
            for i, seg in enumerate(segments)
            if (s := seg.strip()) and len(s) >= self.min_chunk_length
        ]
