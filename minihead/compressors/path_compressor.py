"""File-list compressor (glob results, ``grep -l``, directory listings).

Lossless: paths are grouped under their directory, so a directory prefix
repeated hundreds of times is written once. Every path can be rebuilt
exactly as ``<dir>/<name>`` (backslashes are normalised to ``/``).
"""

from __future__ import annotations

from collections import OrderedDict


def compress_paths(text: str) -> str:
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        norm = ln.replace("\\", "/")
        d, sep, name = norm.rpartition("/")
        groups.setdefault(d + "/" if sep else "./", []).append(name)
    out = [f"[{sum(len(v) for v in groups.values())} paths, grouped by directory as dir/: name, name, ...]"]
    for d, names in groups.items():
        out.append(f"{d}: " + ", ".join(names))
    return "\n".join(out)


def expand_paths(compressed: str) -> list[str]:
    """Inverse of compress_paths (used by tests to prove it is lossless)."""
    paths = []
    for ln in compressed.splitlines()[1:]:
        d, _, names = ln.partition(": ")
        for n in names.split(", "):
            paths.append(n if d == "./" else d + n)
    return paths
