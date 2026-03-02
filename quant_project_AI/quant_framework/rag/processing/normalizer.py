"""
文本规范化：清洗、去噪、统一格式
"""

import re
from typing import Optional


class TextNormalizer:
    """轻量文本规范化，便于分块与检索"""

    def __init__(
        self,
        strip_whitespace: bool = True,
        collapse_newlines: bool = True,
        min_length: int = 10,
    ):
        self.strip_whitespace = strip_whitespace
        self.collapse_newlines = collapse_newlines
        self.min_length = min_length

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        s = text
        if self.collapse_newlines:
            s = re.sub(r"\n+", "\n", s)
            s = re.sub(r"[ \t]+", " ", s)
        if self.strip_whitespace:
            s = s.strip()
        return s

    def normalize_and_filter(self, text: str) -> Optional[str]:
        """规范化并过滤过短内容"""
        s = self.normalize(text)
        return s if len(s) >= self.min_length else None
