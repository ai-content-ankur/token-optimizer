"""Exact o200k_base tokenizer (GPT-4o / GPT-4.1 / o-series / GPT-5) in pure Python.

This reproduces OpenAI's ``tiktoken`` o200k_base encoding with the standard
library only, so token counts in reports are the counts OpenAI bills for,
not an estimate.

* Merge ranks: ``data/o200k_base.tiktoken``, byte-identical to OpenAI's
  published file (sha256 ``446a9538...1a2d``, checked at load time).
* Pre-tokenisation: OpenAI's o200k regex, with its ``\\p{..}`` Unicode
  classes expanded into explicit code-point ranges so Python's ``re`` can
  run it (``re`` has no ``\\p{..}``). ``\\s`` is replaced by the Unicode
  White_Space set, which is what tiktoken's regex engine uses.
* BPE: the same lowest-rank-first merge loop tiktoken uses.

If ``tiktoken`` itself is installed, ``count`` uses it instead.

Ranks file and algorithm: MIT License, Copyright (c) 2022 OpenAI, Shantanu Jain.
"""

from __future__ import annotations

import base64
import hashlib
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

RANKS_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_RANKS_PATH = Path(__file__).with_name("data") / "o200k_base.tiktoken"

# Unicode White_Space property (what \s means in tiktoken's regex engine)
_WS = "\t\n\x0b\x0c\r \x85\xa0  -     　"


def _category_ranges() -> dict[str, str]:
    """Map each needed Unicode general category to a regex range string."""
    wanted = {"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No"}
    ranges: dict[str, list[list[int]]] = {c: [] for c in wanted}
    cat = unicodedata.category
    for cp in range(sys.maxunicode + 1):
        c = cat(chr(cp))
        if c in wanted:
            r = ranges[c]
            if r and r[-1][1] == cp - 1:
                r[-1][1] = cp
            else:
                r.append([cp, cp])

    def fmt(cp: int) -> str:
        return "\\U%08x" % cp

    return {
        c: "".join(fmt(a) if a == b else f"{fmt(a)}-{fmt(b)}" for a, b in rs) for c, rs in ranges.items()
    }


@lru_cache(maxsize=1)
def _pattern() -> re.Pattern:
    r = _category_ranges()
    L = r["Lu"] + r["Ll"] + r["Lt"] + r["Lm"] + r["Lo"]
    N = r["Nd"] + r["Nl"] + r["No"]
    M = r["Mn"] + r["Mc"] + r["Me"]
    upper = r["Lu"] + r["Lt"] + r["Lm"] + r["Lo"] + M
    lower = r["Ll"] + r["Lm"] + r["Lo"] + M
    contr = "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    parts = [
        rf"[^\r\n{L}{N}]?[{upper}]*[{lower}]+{contr}",
        rf"[^\r\n{L}{N}]?[{upper}]+[{lower}]*{contr}",
        rf"[{N}]{{1,3}}",
        rf" ?[^{_WS}{L}{N}]+[\r\n/]*",
        rf"[{_WS}]*[\r\n]+",
        rf"[{_WS}]+(?![^{_WS}])",
        rf"[{_WS}]+",
    ]
    return re.compile("|".join(parts))


@lru_cache(maxsize=1)
def _ranks() -> dict[bytes, int]:
    raw = _RANKS_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RANKS_SHA256:
        raw = raw.replace(b"\r\n", b"\n")  # undo a Windows line-ending conversion, if any
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RANKS_SHA256:
        raise RuntimeError(f"o200k ranks file is corrupted (sha256 {digest})")
    ranks = {}
    for line in raw.splitlines():
        if line:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    return ranks


def _bpe(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    if piece in ranks:
        return [ranks[piece]]
    parts = [piece[i : i + 1] for i in range(len(piece))]
    while len(parts) > 1:
        best_rank = None
        best_i = -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_rank is None:
            break
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return [ranks[p] for p in parts]


@lru_cache(maxsize=200_000)
def _encode_piece(piece: str) -> tuple[int, ...]:
    return tuple(_bpe(piece.encode("utf-8", "surrogatepass"), _ranks()))


def encode(text: str) -> list[int]:
    out: list[int] = []
    for m in _pattern().finditer(text):
        out.extend(_encode_piece(m.group(0)))
    return out


try:  # prefer the real thing when available
    import tiktoken as _tk  # type: ignore

    _TK = _tk.get_encoding("o200k_base")
except Exception:  # pragma: no cover
    _TK = None


def count(text: str) -> int:
    if not text:
        return 0
    if _TK is not None:
        return len(_TK.encode(text, disallowed_special=()))
    return len(encode(text))


def backend() -> str:
    return "tiktoken o200k_base" if _TK is not None else "o200k_base (pure-Python, verified ranks)"
