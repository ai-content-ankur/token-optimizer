"""Log / build-output compressor.

Keeps every line that matters for debugging — errors, failures, stack
traces and their surrounding context, plus the head and tail — and folds
repetitive lines into one representative with a count. A line's
"signature" masks only its trailing variable part (numbers, hex, ids,
timestamps), so distinct messages are never merged together.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict

from ..ccr import Store, marker

HEAD = 5
TAIL = 10
CONTEXT = 2
MAX_KEPT_PLAIN = 40

_ERR = re.compile(
    r"\b(FATAL|CRITICAL|ERROR|ERR!|FAIL(ED|URE)?|PANIC|panicked|Exception|Traceback|"
    r"error(\[E\d+\])?:|AssertionError|Segmentation fault|Killed|OOM|exit (code|status) [1-9])",
    re.IGNORECASE,
)
_WARN = re.compile(r"\b(WARN(ING)?|DEPRECATED)\b", re.IGNORECASE)
_TRACE_CONT = re.compile(r"^(\s+\S|\s*at \S|\s*File \"|\s*\^+\s*$|\s*\.\.\.|\s*--> )")
_TS = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}[T ][\d:.,]+Z?\]?\s*|^\s*\[?\d{2}:\d{2}:\d{2}[.,\d]*\]?\s*")
_VAR = re.compile(
    r"0x[0-9a-fA-F]+|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|\b[0-9a-f]{12,}\b|\d+(\.\d+)?"
)


def signature(line: str) -> str:
    body = _TS.sub("", line)
    # keep the message identifier intact; mask variables only after the
    # first ':' or '=' (or the whole line if it has neither)
    m = re.search(r"[:=]", body)
    if m and m.start() > 3:
        head, tail = body[: m.start() + 1], body[m.start() + 1 :]
        return head + _VAR.sub("#", tail)
    return _VAR.sub("#", body)


def compress_log(text: str, store: Store) -> str:
    lines = text.splitlines()
    n = len(lines)
    if n <= HEAD + TAIL + 5:
        return text

    keep: set[int] = set(range(min(HEAD, n))) | set(range(max(0, n - TAIL), n))

    # errors + context + full stack traces
    i = 0
    while i < n:
        if _ERR.search(lines[i]):
            keep.update(range(max(0, i - CONTEXT), min(n, i + CONTEXT + 1)))
            j = i + 1
            while j < n and (_TRACE_CONT.match(lines[j]) or (lines[j].strip() == "" and j + 1 < n and _TRACE_CONT.match(lines[j + 1]))):
                keep.add(j)
                j += 1
            if j > i + 1 and j < n:
                keep.add(j)  # the "ValueError: ..." line that closes a traceback
            i = j
            continue
        i += 1

    # one representative per warning signature
    seen_warn: set[str] = set()
    for i, ln in enumerate(lines):
        if _WARN.search(ln):
            sig = signature(ln)
            if sig not in seen_warn:
                seen_warn.add(sig)
                keep.add(i)

    # plain lines: first occurrence of each signature, counted
    sig_counts: Counter[str] = Counter(signature(ln) for ln in lines)
    first_of: "OrderedDict[str, int]" = OrderedDict()
    for i, ln in enumerate(lines):
        first_of.setdefault(signature(ln), i)
    plain_budget = MAX_KEPT_PLAIN
    for sig, idx in first_of.items():
        if idx in keep or not sig.strip():
            continue
        if plain_budget <= 0:
            break
        keep.add(idx)
        plain_budget -= 1

    out: list[str] = []
    gap = 0
    for i, ln in enumerate(lines):
        if i in keep:
            if gap:
                out.append(f"  ... [{gap} lines omitted]")
                gap = 0
            sig = signature(ln)
            c = sig_counts[sig]
            first = first_of[sig] == i
            out.append(ln + (f"  [x{c} similar]" if first and c > 1 and not _ERR.search(ln) else ""))
        else:
            gap += 1
    if gap:
        out.append(f"  ... [{gap} lines omitted]")

    dropped = n - len(keep)
    if dropped <= 0:
        return text
    key = store.put(text, "log")
    out.append(marker(key, f"full log: {n} lines, {dropped} omitted"))
    return "\n".join(out)
