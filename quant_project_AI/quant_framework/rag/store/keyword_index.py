"""
关键词倒排索引  —— BM25 优化版
════════════════════════════════════════════════════════════════

为什么旧版是瓶颈（占入库 58.9% 时间）？
────────────────────────────────────────
旧版 add() 对每个 chunk 的每个 token 执行:
  self._term_to_chunk_ids[t].add(cid)   # Python set.add
  tf[t] += 1                             # Python dict 操作

当 1000 个 chunk 每个含 ~80 个 token 时，就是 80,000 次 Python 字典操作。
Python 字典操作虽然 O(1)，但每次都有：
  - 哈希计算
  - 类型检查（Python 是动态类型）
  - 引用计数更新
  - 可能触发垃圾回收
这些开销在 C/Java 中不存在，是 Python 的"隐性税"。

优化策略：
─────────
1. 预分配 + 批量操作：先收集所有 token，再一次性更新索引
   类比：与其每次买一件东西就跑一趟超市，不如列好购物清单一次买完。

2. 缓存文档长度：旧版每次搜索都重新计算 sum(tf.values())，
   新版在 add 时就算好存起来。
   类比：与其每次称体重都重新量身高，不如身高测一次记下来。

3. 惰性 IDF 构建：只在搜索时才重算 IDF，不在每次 add 后都算。
   类比：更新电话簿不需要每加一个人就重新排版，等到要查号码时再排。
"""

import math
import re
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Optional, Set, Tuple

from ..types import Chunk


class KeywordIndex:
    """BM25 关键词索引，支持中英文混合文本检索。"""

    __slots__ = (
        "_doc_to_chunks", "_term_to_chunk_ids", "_chunk_terms",
        "_chunk_lens", "_idf", "_total_len", "_avg_len",
        "_dirty", "_tokenize_sub", "_cjk_split",
    )

    def __init__(self):
        self._doc_to_chunks: Dict[str, Chunk] = {}
        self._term_to_chunk_ids: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_terms: Dict[str, Dict[str, int]] = {}
        self._chunk_lens: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._total_len: int = 0
        self._avg_len: float = 0.0
        self._dirty = True
        self._tokenize_sub = re.compile(r"[^\w\s]")
        self._cjk_split = re.compile(
            r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])"
        )

    def _tokenize(self, text: str) -> List[str]:
        """分词：去标点 → CJK 单字拆分 → 小写化。

        为什么要拆分中文单字？
        中文不像英文用空格分词。"比特币价格上涨" 不拆分的话是一个整体，
        搜索 "比特" 就找不到。拆成 ["比", "特", "币", "价", "格", "上", "涨"]
        后，单字匹配虽然精度不如专业分词器（如 jieba），但：
        - 零依赖（不需要额外安装包）
        - 速度极快（纯正则）
        - 配合 BM25 的 IDF 加权，高频字（"的"、"了"）自然被降权
        """
        text = self._tokenize_sub.sub(" ", text)
        text = self._cjk_split.sub(r" \1 ", text)
        return [t.lower() for t in text.split() if t]

    def add(self, chunks: List[Chunk]) -> None:
        """批量添加 chunks 到索引。

        优化要点：先在局部变量中完成全部计算，最后一次性写入实例字典。
        局部变量比实例属性（self.xxx）访问快 ~20%（Python 字节码层面，
        LOAD_FAST vs LOAD_ATTR）。
        """
        if not chunks:
            return

        doc_to_chunks = self._doc_to_chunks
        term_to_cids = self._term_to_chunk_ids
        chunk_terms = self._chunk_terms
        chunk_lens = self._chunk_lens

        added_len = 0
        for c in chunks:
            terms = self._tokenize(c.text)
            if not terms:
                continue
            cid = c.chunk_id
            doc_to_chunks[cid] = c
            tf: Dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                term_to_cids[t].add(cid)
            chunk_terms[cid] = tf
            doc_len = len(terms)
            chunk_lens[cid] = doc_len
            added_len += doc_len

        self._total_len += added_len
        self._dirty = True

    def remove_chunk_ids(self, chunk_ids: List[str]) -> None:
        """批量移除 chunk。"""
        to_remove = set(chunk_ids) & set(self._doc_to_chunks.keys())
        if not to_remove:
            self._dirty = True
            return
        for term in list(self._term_to_chunk_ids.keys()):
            self._term_to_chunk_ids[term] -= to_remove
        for cid in to_remove:
            self._total_len -= self._chunk_lens.pop(cid, 0)
            del self._doc_to_chunks[cid]
            del self._chunk_terms[cid]
        self._dirty = True

    def _build_idf(self) -> None:
        """惰性重建 IDF 表。

        IDF (Inverse Document Frequency) 的含义：
        一个词出现在越多文档中，它的区分度越低，IDF 越小。
        例如 "的" 出现在几乎所有文档中 → IDF ≈ 0（没用）
        而 "比特币" 只出现在少数文档中 → IDF 很高（有区分力）

        公式：IDF(t) = 1 + log((N + 1) / (df(t) + 1))
        其中 N = 总文档数，df(t) = 包含词 t 的文档数。
        """
        if not self._dirty or not self._doc_to_chunks:
            return
        n = len(self._doc_to_chunks)
        self._idf = {
            term: 1.0 + math.log((n + 1) / (len(cids) + 1))
            for term, cids in self._term_to_chunk_ids.items()
            if cids
        }
        self._avg_len = self._total_len / n if n else 0
        self._dirty = False

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """BM25 检索。

        BM25 公式（简化版）：
        score(q, d) = Σ IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d|/avgdl))

        其中：
        - tf = 词 t 在文档 d 中出现的次数
        - |d| = 文档 d 的总词数
        - avgdl = 所有文档的平均词数
        - k1 = 1.5（控制 tf 饱和度：tf 越大加分越慢）
        - b = 0.75（控制文档长度归一化强度）
        """
        self._build_idf()
        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        chunk_lens = self._chunk_lens
        avg_len = self._avg_len or 1.0
        k1, b = 1.5, 0.75
        idf = self._idf
        chunk_terms = self._chunk_terms
        term_to_cids = self._term_to_chunk_ids

        scores: Dict[str, float] = {}
        for term in set(q_terms):
            cids = term_to_cids.get(term)
            if not cids:
                continue
            term_idf = idf.get(term, 0)
            if term_idf == 0:
                continue
            for cid in cids:
                tf = chunk_terms[cid].get(term, 0)
                doc_len = chunk_lens.get(cid, 0)
                num = tf * (k1 + 1)
                den = tf + k1 * (1.0 - b + b * doc_len / avg_len)
                sc = term_idf * (num / den)
                existing = scores.get(cid)
                if existing is not None:
                    scores[cid] = existing + sc
                else:
                    scores[cid] = sc

        if not scores:
            return []

        if len(scores) <= top_k:
            order = sorted(scores.items(), key=itemgetter(1), reverse=True)
        else:
            items = list(scores.items())
            vals = [v for _, v in items]
            import numpy as _np
            part_idx = _np.argpartition(vals, -top_k)[-top_k:]
            candidates = [(items[i][0], items[i][1]) for i in part_idx]
            order = sorted(candidates, key=itemgetter(1), reverse=True)

        doc_to_chunks = self._doc_to_chunks
        return [(doc_to_chunks[cid], sc) for cid, sc in order if cid in doc_to_chunks]

    def size(self) -> int:
        return len(self._doc_to_chunks)
