"""
有界队列：基于 queue.Queue，支持非阻塞 put、超时 get，供实时接入与 worker 消费
"""

from queue import Empty, Full, Queue
from typing import List

from ..types import Document


class IngestQueue:
    """
    线程安全有界 FIFO，封装 queue.Queue。
    - put(doc): 非阻塞，满则返回 False，不阻塞调用方（实时写入路径）。
    - take(n): 非阻塞取最多 n 条；供同步批处理或测试用。
    - take_blocking(timeout, batch_size): 阻塞取一批，供 worker 使用。
    """

    def __init__(self, max_size: int = 10000):
        self._q: Queue = Queue(maxsize=max_size)
        self._max_size = max_size

    def put(self, doc: Document) -> bool:
        """非阻塞放入；队列满返回 False。"""
        try:
            self._q.put(doc, block=False)
            return True
        except Full:
            return False

    def put_many(self, docs: List[Document]) -> int:
        n = 0
        for d in docs:
            if not self.put(d):
                break
            n += 1
        return n

    def take(self, n: int = 1) -> List[Document]:
        """非阻塞取最多 n 条。"""
        out: List[Document] = []
        for _ in range(n):
            try:
                out.append(self._q.get_nowait())
            except Empty:
                break
        return out

    def get(self, block: bool = True, timeout: float = 0.1) -> Document:
        """单条 get，供 worker 循环使用。block=True 时最多等 timeout 秒。"""
        return self._q.get(block=block, timeout=timeout)

    def get_nowait(self) -> Document:
        return self._q.get_nowait()

    def take_blocking(self, batch_size: int, timeout: float = 0.1) -> List[Document]:
        """
        先阻塞取一条（timeout），再非阻塞取满 batch_size 或队列空。
        供后台 worker 聚批使用。
        """
        batch: List[Document] = []
        try:
            batch.append(self._q.get(block=True, timeout=timeout))
        except Empty:
            return []
        while len(batch) < batch_size:
            try:
                batch.append(self._q.get_nowait())
            except Empty:
                break
        return batch

    def size(self) -> int:
        return self._q.qsize()

    def is_empty(self) -> bool:
        return self._q.empty()
