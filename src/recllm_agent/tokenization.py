from __future__ import annotations

import re

_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def tokenize(text: str) -> list[str]:
    """Simple mixed Chinese/English tokenizer for BM25.

    It intentionally avoids external dependencies. English tokens are lowercased
    words; Chinese uses character unigrams plus adjacent bigrams to improve
    phrase matching for terms such as "语义" and "召回".
    """

    tokens: list[str] = []
    tokens.extend(match.group(0).lower() for match in _ASCII_WORD_RE.finditer(text))

    cjk_chars = [char for char in text if _is_cjk(char)]
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))
    return tokens
