"""The compression pipeline: route each block of tool output to a compressor.

``compress_text`` compresses one string. ``compress`` walks a chat request
(OpenAI ``messages`` or Anthropic ``messages``) and compresses tool output
blocks in place.

Cache safety (mode="cache", the default)
----------------------------------------
Providers cache the request *prefix*. If an older message changes between
turns the cache misses and you pay full price for the whole history. So in
cache mode only the *live zone* — tool results that arrived after the last
assistant message — is freshly compressed. Older blocks are swapped for
their compressed form only if they were sent compressed before (a memo
hit), which by construction reproduces the earlier bytes exactly.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .ccr import MARKER_RE, Store
from .compressors import (
    compress_code,
    compress_diff,
    compress_log,
    compress_paths,
    compress_search,
    compress_text as _compress_prose,
    crush_json,
)
from .detect import NUMBERED_LINE, Kind, detect, parse_json
from .tokens import count_tokens

MIN_TOKENS = 250  # blocks smaller than this pass through untouched
MIN_GAIN = 0.05  # keep the compressed form only if it saves at least 5%
CODE_MIN_TOKENS = 1500  # only outline code files that are genuinely large

_default_store: Store | None = None


def default_store() -> Store:
    global _default_store
    if _default_store is None:
        _default_store = Store()
    return _default_store


@dataclass
class BlockResult:
    kind: str
    tokens_before: int
    tokens_after: int
    text: str


def compress_text(
    text: str,
    store: Store | None = None,
    kind: Kind | str | None = None,
    min_tokens: int = MIN_TOKENS,
    record: bool = True,
) -> BlockResult:
    """Compress one block of text. Returns the original when compression doesn't pay."""
    store = store or default_store()
    before = count_tokens(text)
    if before < min_tokens or MARKER_RE.search(text):
        return BlockResult("passthrough", before, before, text)

    # File reads often arrive as "   12|code" lines. Compress the bare content;
    # code outlines still cite the original line numbers via ``base``.
    body, base = _strip_line_numbers(text)
    k = Kind(kind) if kind else detect(body)
    if k is Kind.JSON:
        out = crush_json(body, parse_json(body), store)
    elif k is Kind.LOG:
        out = compress_log(body, store)
    elif k is Kind.SEARCH:
        out = compress_search(body, store)
    elif k is Kind.PATHS:
        out = compress_paths(body)
    elif k is Kind.CODE:
        out = compress_code(body, store, original=text, base=base) if before >= CODE_MIN_TOKENS else text
        if out == text and before >= CODE_MIN_TOKENS:
            k, out = Kind.TEXT, _compress_prose(body, store)  # no outline possible: treat as text
    elif k is Kind.DIFF:
        out = compress_diff(body, store)
    else:
        out = _compress_prose(body, store)

    after = count_tokens(out)
    if after > before * (1 - MIN_GAIN):
        return BlockResult("passthrough", before, before, text)
    if record:
        store.record(k.value, before, after)
    return BlockResult(k.value, before, after, out)


def _strip_line_numbers(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    if len(nonblank) < 5 or sum(1 for ln in nonblank if NUMBERED_LINE.match(ln)) < 0.95 * len(nonblank):
        return text, 1
    first = NUMBERED_LINE.match(nonblank[0])
    base = int(first.group(0).strip().rstrip("|")) if first else 1
    return "\n".join(ln.split("|", 1)[1] if NUMBERED_LINE.match(ln) else ln for ln in lines), base


# ── message walking ───────────────────────────────────────────────────────


@dataclass
class CompressResult:
    messages: list[dict]
    tokens_before: int = 0
    tokens_after: int = 0
    blocks: list[BlockResult] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def compression_ratio(self) -> float:
        return self.tokens_saved / self.tokens_before if self.tokens_before else 0.0


# File-read tools: the agent usually reads a file in order to edit it, so its
# output is passed through untouched (set MINIHEAD_COMPRESS_READS=1 to override).
READ_TOOL_RE = re.compile(r"^(read|read_?file|view(_file)?|open_?file|cat|get_file_contents?)$", re.I)


def _tool_names(messages: list[dict]) -> dict[str, str]:
    """Map tool-call id -> tool name, across Anthropic and OpenAI formats."""
    names: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "function_call" and msg.get("call_id"):
            names[msg["call_id"]] = msg.get("name", "")
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                names[call["id"]] = (call.get("function") or {}).get("name", "")
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    names[block["id"]] = block.get("name", "")
    return names


def _text_slots(msg: dict) -> list[tuple[Any, Any, str | None]]:
    """Return (container, key, tool_call_id) for each tool-output string inside ``msg``."""
    slots: list[tuple[Any, Any, str | None]] = []
    role = msg.get("role")
    content = msg.get("content")

    def add_content(container: Any, key: Any, call_id: str | None) -> None:
        value = container[key]
        if isinstance(value, str):
            slots.append((container, key, call_id))
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    slots.append((part, "text", call_id))

    if role == "tool":  # OpenAI chat completions
        add_content(msg, "content", msg.get("tool_call_id"))
    elif role == "function_call_output" or msg.get("type") == "function_call_output":  # OpenAI responses
        if isinstance(msg.get("output"), str):
            slots.append((msg, "output", msg.get("call_id")))
    elif isinstance(content, list):  # Anthropic: tool_result blocks inside a user message
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and "content" in block:
                add_content(block, "content", block.get("tool_use_id"))
    return slots


def _live_zone_start(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" or messages[i].get("type") == "function_call":
            return i + 1
    return 0


def compress(
    messages: list[dict],
    model: str | None = None,  # accepted for API familiarity; tokenizer is model-agnostic
    store: Store | None = None,
    mode: str = "cache",
    min_tokens: int = MIN_TOKENS,
    protect_reads: bool | None = None,
) -> CompressResult:
    """Compress tool outputs in a chat request. Returns a new list; the input is not mutated."""
    store = store or default_store()
    if protect_reads is None:
        protect_reads = os.environ.get("MINIHEAD_COMPRESS_READS") != "1"
    msgs = copy.deepcopy(messages)
    live_from = _live_zone_start(msgs) if mode == "cache" else 0
    names = _tool_names(msgs) if protect_reads else {}
    result = CompressResult(messages=msgs)

    for idx, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            continue
        for container, key, call_id in _text_slots(msg):
            original = container[key]
            if protect_reads and call_id and READ_TOOL_RE.match(names.get(call_id, "")):
                n = count_tokens(original)
                result.blocks.append(BlockResult("protected-read", n, n, original))
                result.tokens_before += n
                result.tokens_after += n
                continue
            memo = store.memo_get(original)
            if memo is not None:
                container[key] = memo
                b, a = count_tokens(original), count_tokens(memo)
                result.blocks.append(BlockResult("memo", b, a, memo))
                result.tokens_before += b
                result.tokens_after += a
                continue
            if idx < live_from:
                n = count_tokens(original)
                result.tokens_before += n
                result.tokens_after += n
                continue
            block = compress_text(original, store, min_tokens=min_tokens)
            if block.kind != "passthrough":
                container[key] = block.text
                store.memo_put(original, block.text)
            result.blocks.append(block)
            result.tokens_before += block.tokens_before
            result.tokens_after += block.tokens_after
    return result


# ── retrieval tool schema (for proxies / SDK users) ─────────────────────────

RETRIEVE_TOOL_NAME = "minihead_retrieve"
RETRIEVE_DESCRIPTION = (
    "Fetch the full original of content that was compressed. Call this with the 12-character "
    "hash from a <<ccr:HASH ...>> marker when the omitted part is needed to answer."
)
_SCHEMA = {
    "type": "object",
    "properties": {"hash": {"type": "string", "description": "12-char hash from a <<ccr:...>> marker"}},
    "required": ["hash"],
}
ANTHROPIC_RETRIEVE_TOOL = {"name": RETRIEVE_TOOL_NAME, "description": RETRIEVE_DESCRIPTION, "input_schema": _SCHEMA}
OPENAI_RETRIEVE_TOOL = {
    "type": "function",
    "function": {"name": RETRIEVE_TOOL_NAME, "description": RETRIEVE_DESCRIPTION, "parameters": _SCHEMA},
}


def retrieve(hash_key: str, store: Store | None = None) -> str:
    store = store or default_store()
    original = store.get(hash_key.strip())
    if original is None:
        return f"No stored original for {hash_key!r} (unknown or expired)."
    return original
