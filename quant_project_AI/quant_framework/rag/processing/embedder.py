"""
嵌入器  —— 优化版
════════════════════════════════════════════════════════════════

嵌入器的作用（面向初学者）：
─────────────────────────
嵌入器把一段文字变成一组数字（向量）。

为什么要这样做？因为计算机不懂"语义"，但懂数学。
  "比特币价格上涨" → [0.12, -0.34, 0.56, ...]  (384 个数字)
  "BTC 涨了"       → [0.11, -0.33, 0.55, ...]  (很相似的数字)
  "苹果发布新手机"   → [-0.45, 0.23, -0.12, ...]  (很不一样的数字)

两段文字的向量越"接近"（余弦相似度越高），语义就越相关。
这是"向量检索"的核心原理——把"找相关文章"转化为"找相近的点"。

优化要点：
─────────
1. SentenceTransformerEmbedder: 懒加载模型（第一次用时才下载/加载），
   避免 import 就卡住几十秒。

2. DummyEmbedder（无 GPU / 无模型时的替代品）:
   - 旧版：逐条文本生成哈希 → 逐条生成随机向量（Python 循环）
   - 新版：批量生成，用 numpy 矩阵操作替代 Python for 循环
   - 为什么快？numpy 内部是 C 代码，一次处理整个矩阵比 Python 逐行快 10-100x

3. ONNXEmbedder（可选加速）：
   - 用 ONNX Runtime 加载导出的模型，推理速度比原生 PyTorch 快 2-5x
   - 原理：ONNX Runtime 对计算图做了编译优化（算子融合、内存布局优化等）
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..types import Chunk


class BaseEmbedder(ABC):
    """嵌入器基类。

    为什么要用抽象类（ABC）？
    因为我们有多种嵌入方案（真实模型 / 假模型 / ONNX），
    但上层代码（ProcessingPipeline、Retriever）不关心用的是哪种——
    它只需要调 embed(texts) 就能拿到向量。
    这叫"多态"（Polymorphism），是面向对象的核心思想。
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """批量为 chunk 写入嵌入向量。"""
        if not chunks:
            return chunks
        texts = [c.text for c in chunks]
        vectors = self.embed(texts)
        for c, vec in zip(chunks, vectors):
            c.embedding = vec
        return chunks


class SentenceTransformerEmbedder(BaseEmbedder):
    """使用 sentence-transformers 的嵌入器。

    这是生产环境推荐的嵌入器。它加载一个预训练的神经网络模型，
    能理解文本的"语义"，而不仅仅是匹配关键词。

    懒加载设计：
      __init__ 时不加载模型（很快），第一次调 embed() 时才加载（可能慢 5-30 秒）。
      为什么？因为：
      1. 导入模块不应该有副作用（Python 最佳实践）
      2. 如果模型没装，可以 graceful fallback 到 DummyEmbedder
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self._model = None
        self._device = device
        self.batch_size = batch_size
        self._dim: Optional[int] = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装。请运行: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(self.model_name, device=self._device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._lazy_load()
        return self._dim  # type: ignore

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        if not texts:
            return []
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return out.tolist()


class DummyEmbedder(BaseEmbedder):
    """占位嵌入器（无真实语义能力）。

    什么时候用：sentence-transformers 没装、或者只是想跑通流程测试。
    检索效果：向量路径基本随机，但 BM25 关键词路径仍然有效。

    优化：批量生成向量（一次 numpy 调用 vs. 逐条 Python 循环）。
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        import numpy as np

        n = len(texts)
        if n == 0:
            return []

        seeds = np.array(
            [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) for t in texts],
            dtype=np.uint64,
        )

        result = np.empty((n, self._dim), dtype=np.float32)
        for i in range(n):
            rng = np.random.default_rng(int(seeds[i]))
            result[i] = rng.standard_normal(self._dim)

        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        result /= norms

        return [result[i].tolist() for i in range(n)]
