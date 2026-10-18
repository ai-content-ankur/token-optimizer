"""grep / ripgrep result compressor.

Groups ``path:line:content`` hits by file, keeps the first few per file
and the files in order of first appearance, and reports what was cut.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from ..ccr import Store, marker

PER_FILE = 4
MAX_FILES = 25
_HIT = re.compile(r"^(?P<path>[^\s:]+\.[A-Za-z0-9]{1,6})[:-](?P<line>\d+)[:-](?P<text>.*)$")


def compress_search(text: str, store: Store) -> str:
    groups: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    other: list[str] = []
    for ln in text.splitlines():
        m = _HIT.match(ln)
        if m:
            groups.setdefault(m["path"], []).append((m["line"], m["text"].strip()))
        elif ln.strip():
            other.append(ln)

    total = sum(len(v) for v in groups.values())
    out: list[str] = []
    shown = 0
    for fi, (path, hits) in enumerate(groups.items()):
        if fi >= MAX_FILES:
            break
        extra = f"  (+{len(hits) - PER_FILE} more)" if len(hits) > PER_FILE else ""
        out.append(f"{path}  [{len(hits)} hits]{extra}")
        for line_no, content in sorted(hits[:PER_FILE], key=lambda h: int(h[0])):
            out.append(f"  {line_no}: {content[:200]}")
            shown += 1
    if len(groups) > MAX_FILES:
        rest = list(groups.keys())[MAX_FILES:]
        out.append(f"... +{len(rest)} more files: " + ", ".join(rest[:40]) + (" ..." if len(rest) > 40 else ""))
    out.extend(other[:10])
    if shown >= total and len(groups) <= MAX_FILES:
        return "\n".join(out)
    key = store.put(text, "search")
    out.append(marker(key, f"{total} hits in {len(groups)} files, {total - shown} hidden"))
    return "\n".join(out)
