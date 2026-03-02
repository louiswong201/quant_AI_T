"""
RAG (Retrieval-Augmented Generation) 模块

实时获取非结构化数据、处理并支持高效检索与生成。
架构：实时接入 → 流式处理 → 向量/关键词混合检索 → 重排序 → 生成增强
"""

from .config import RAGConfig
from .pipeline import RAGPipeline
from .prompts import format_prompt, list_templates
from .types import Document, Chunk, SearchResult

__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "Document",
    "Chunk",
    "SearchResult",
    "format_prompt",
    "list_templates",
]
