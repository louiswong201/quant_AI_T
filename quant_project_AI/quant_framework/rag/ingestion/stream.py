"""
流式接入：从队列消费并 yield 给下游处理
"""

from typing import Callable, Iterator, List, Optional

from ..types import Document
from .queue import IngestQueue


class IngestStream:
    """从 IngestQueue 持续取文档并流式产出，供 pipeline 消费"""

    def __init__(
        self,
        queue: IngestQueue,
        batch_size: int = 16,
        timeout_seconds: Optional[float] = None,
    ):
        self._queue = queue
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds

    def stream(self) -> Iterator[List[Document]]:
        """按 batch_size 一批批产出文档，无文档时返回空列表并结束一批"""
        while True:
            batch = self._queue.take(self._batch_size)
            if not batch:
                break
            yield batch

    def stream_continuous(
        self,
        on_batch: Callable[[List[Document]], None],
        stop_event: Optional[Callable[[], bool]] = None,
    ):
        """
        持续运行：不断从队列取 batch，调用 on_batch；
        stop_event() 返回 True 时退出。
        """
        import time
        while True:
            if stop_event and stop_event():
                break
            batch = self._queue.take(self._batch_size)
            if batch:
                on_batch(batch)
            else:
                if self._timeout_seconds:
                    time.sleep(min(0.5, self._timeout_seconds))
                else:
                    time.sleep(0.1)
