"""
文件监听接入：监控目录/文件变化，实时产出新文档
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from ..types import Document
from .base import BaseIngestAdapter

logger = logging.getLogger(__name__)


class FileWatcherIngestAdapter(BaseIngestAdapter):
    """监听目录下文本文件的新增/变更，实时产出 Document"""

    def __init__(
        self,
        watch_path: str,
        extensions: Optional[List[str]] = None,
        encoding: str = "utf-8",
        poll_interval: float = 1.0,
        max_file_size_mb: float = 10.0,
    ):
        self.watch_path = Path(watch_path).resolve()
        self.extensions = extensions or [".txt", ".md", ".json", ".csv"]
        self.encoding = encoding
        self.poll_interval = poll_interval
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)
        self._known: Dict[str, float] = {}  # path -> mtime

    def _is_safe_path(self, path: Path) -> bool:
        """Verify *path* resolves to a location within ``self.watch_path``."""
        try:
            resolved = path.resolve()
            return str(resolved).startswith(str(self.watch_path))
        except (OSError, ValueError):
            return False

    def _read_file(self, path: Path) -> Optional[str]:
        try:
            if not self._is_safe_path(path):
                logger.warning("Blocked path outside watch directory: %s", path)
                return None
            if path.stat().st_size > self.max_file_size:
                return None
            return path.read_text(encoding=self.encoding, errors="replace")
        except Exception:
            logger.debug("_read_file failed for path=%s", path, exc_info=True)
            return None

    def _list_files(self) -> List[Path]:
        if not self.watch_path.exists():
            return []
        out: List[Path] = []
        for p in self.watch_path.rglob("*"):
            if p.is_file() and any(p.suffix.lower() == e.lower() for e in self.extensions):
                out.append(p)
        return out

    def fetch(self) -> Iterator[Document]:
        """单次轮询：只产出有变更的文件内容"""
        for path in self._list_files():
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if key not in self._known or self._known[key] != mtime:
                self._known[key] = mtime
                content = self._read_file(path)
                if content:
                    yield Document(
                        content=content,
                        source=key,
                        metadata={"path": key, "mtime": mtime},
                    )

    def run_forever(self, on_docs: Callable[[List[Document]], None], stop: Optional[Callable[[], bool]] = None):
        """持续轮询，将新文档交给 on_docs"""
        while True:
            if stop and stop():
                break
            docs = list(self.fetch())
            if docs:
                on_docs(docs)
            time.sleep(self.poll_interval)
