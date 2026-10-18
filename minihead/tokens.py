"""Token counting.

Counts use OpenAI's o200k_base encoding (GPT-4o, GPT-4.1, o-series, GPT-5)
exactly — via ``tiktoken`` when installed, otherwise via the bundled
pure-Python implementation in ``o200k.py``. For Claude models these counts
are a close proxy, not Anthropic's own tokenizer.
"""

from __future__ import annotations

from functools import lru_cache

from . import o200k


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    return o200k.count(text) if text else 0


def tokenizer_name() -> str:
    return o200k.backend()
