"""
目录接入：一次性扫描目录下文本文件并产出 Document
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional

from ..types import Document
from .base import BaseIngestAdapter

logger = logging.getLogger(__name__)


class DirectoryIngestAdapter(BaseIngestAdapter):
    """从目录递归读取指定扩展名文件，产出 Document"""

    def __init__(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        encoding: str = "utf-8",
        max_file_size_mb: float = 10.0,
    ):
        self.directory = Path(directory)
        self.extensions = extensions or [".txt", ".md", ".json", ".csv"]
        self.encoding = encoding
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)

    def _read(self, path: Path) -> Optional[str]:
        try:
            if path.stat().st_size > self.max_file_size:
                return None
            return path.read_text(encoding=self.encoding, errors="replace")
        except Exception:
            logger.debug("_read failed for path=%s", path, exc_info=True)
            return None

    def fetch(self) -> Iterator[Document]:
        if not self.directory.exists():
            return
        for p in self.directory.rglob("*"):
            if not p.is_file():
                continue
            if not any(p.suffix.lower() == e.lower() for e in self.extensions):
                continue
            content = self._read(p)
            if content:
                yield Document(
                    content=content,
                    source=str(p),
                    metadata={"path": str(p)},
                )
