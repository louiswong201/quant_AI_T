"""
重排序器：对检索结果精排，提升准确率
"""

from abc import ABC, abstractmethod
from typing import List

from ..types import SearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        pass


class CrossEncoderReranker(Reranker):
    """使用 sentence-transformers 的 Cross-Encoder 重排（可选依赖）"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers 未安装，无法使用 CrossEncoder 重排")
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not results:
            return results
        self._lazy_load()
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)
        if hasattr(scores, 'tolist'):
            score_list = scores.tolist()
        else:
            score_list = list(scores)
        if isinstance(score_list, (int, float)):
            score_list = [score_list]
        indexed = list(zip(results, score_list))
        indexed.sort(key=lambda x: -x[1])
        top = indexed[:top_k]
        return [
            SearchResult(chunk=r.chunk, score=sc, rank=i + 1)
            for i, (r, sc) in enumerate(top)
        ]


class IdentityReranker(Reranker):
    """不重排，直接截断"""

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        return results[:top_k]
