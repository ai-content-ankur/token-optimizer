"""Prose / unknown-text and diff compressors.

Prose is already dense, so this stays mostly lossless: normalise
whitespace, fold runs of repeated lines, and only if the text is still
over budget keep a head and tail with a CCR marker for the middle.
"""

from __future__ import annotations

import re

from ..ccr import Store, marker
from ..tokens import count_tokens

TEXT_BUDGET = 6000  # tokens; above this, the middle is offloaded
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)


def _fold_repeats(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        out.append(lines[i] if run < 3 or not lines[i].strip() else f"{lines[i]}  [repeated {run}x]")
        i = j + 1
    return out


def compress_text(text: str, store: Store) -> str:
    cleaned = _MULTI_BLANK.sub("\n\n", _TRAILING_WS.sub("", text))
    cleaned = "\n".join(_fold_repeats(cleaned.split("\n")))
    if count_tokens(cleaned) <= TEXT_BUDGET:
        return cleaned
    # keep ~60% of budget from the start and ~25% from the end, on line boundaries
    lines = cleaned.split("\n")
    head, tail, used = [], [], 0
    for ln in lines:
        t = count_tokens(ln) + 1
        if used + t > TEXT_BUDGET * 0.6:
            break
        head.append(ln)
        used += t
    used = 0
    for ln in reversed(lines[len(head):]):
        t = count_tokens(ln) + 1
        if used + t > TEXT_BUDGET * 0.25:
            break
        tail.append(ln)
        used += t
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    key = store.put(text, "text")
    return "\n".join(head + [marker(key, f"{omitted} lines omitted from the middle")] + tail)


_HUNK = re.compile(r"^@@ ")


def compress_diff(text: str, store: Store) -> str:
    """Shrink long unchanged-context runs inside diff hunks; keep every +/- line."""
    out: list[str] = []
    run: list[str] = []
    changed = False

    def flush() -> None:
        nonlocal changed
        if len(run) > 6:
            out.extend(run[:3])
            out.append(f" ... [{len(run) - 5} unchanged lines]")
            out.extend(run[-2:])
            changed = True
        else:
            out.extend(run)
        run.clear()

    for ln in text.split("\n"):
        if ln.startswith(" ") and not _HUNK.match(ln):
            run.append(ln)
        else:
            flush()
            out.append(ln)
    flush()
    if not changed:
        return text
    key = store.put(text, "diff")
    out.append(marker(key, "full diff"))
    return "\n".join(out)
