"""minihead — a small, dependency-free context compressor for LLM agents.

How it works:

* a content router that picks a compressor per content type,
* type-aware compressors for JSON, logs, code, search results and prose,
* reversible compression (CCR): originals are stored locally and the model
  can fetch them back with the ``minihead_retrieve`` tool,
* cache safety: only the newest turn is compressed and results are memoised,
  so a request prefix stays byte-identical across turns.
"""

from .pipeline import CompressResult, compress, compress_text
from .ccr import Store

__all__ = ["compress", "compress_text", "CompressResult", "Store"]
__version__ = "0.2.0"
