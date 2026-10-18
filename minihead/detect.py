"""Content router: decide what kind of payload a block of text is."""

from __future__ import annotations

import ast
import json
import re
from enum import Enum


class Kind(str, Enum):
    JSON = "json"
    LOG = "log"
    SEARCH = "search"
    CODE = "code"
    DIFF = "diff"
    PATHS = "paths"
    TEXT = "text"


_LOG_LINE = re.compile(
    r"^\s*(\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\[?\d{2}:\d{2}:\d{2}|"
    r"\[?(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL)\b|"
    r"(PASSED|FAILED|ERROR|SKIPPED)\b|npm (WARN|ERR)|\s*Compiling |\s*at \S+ \()",
    re.IGNORECASE,
)
_SEARCH_LINE = re.compile(r"^[^\s:]+\.[A-Za-z0-9]{1,6}[:-]\d+[:-]")
_CODE_HINT = re.compile(
    r"^\s*(def |class |import |from \S+ import |function |const |let |var |export |"
    r"public |private |protected |fn |impl |struct |package |#include|using |"
    r"async def |@\w+|if __name__|return\b|\}|\{$)"
)
_DIFF_LINE = re.compile(r"^(diff --git |@@ |\+\+\+ |--- )")
_PATH_LINE = re.compile(r"^(?:[A-Za-z]:)?[\w.\-/\\@+~]+(?:[/\\][\w.\-@+~ ]+)*(?::\d+)?$")
NUMBERED_LINE = re.compile(r"^\s*\d+\|")


def parse_json(text: str):
    s = text.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except ValueError:
        return None


def _is_python_module(text: str) -> bool:
    """Valid Python with at least two defs/classes (prose and logs are not valid Python)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body) >= 2


def detect(text: str) -> Kind:
    if parse_json(text) is not None:
        return Kind.JSON

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return Kind.TEXT
    sample = lines[:400]
    n = len(sample)

    if sum(1 for ln in sample[:50] if _DIFF_LINE.match(ln)) >= 2:
        return Kind.DIFF
    if n >= 8 and sum(1 for ln in sample if _PATH_LINE.match(ln) and ("/" in ln or "\\" in ln or "." in ln)) / n > 0.9:
        return Kind.PATHS
    if sum(1 for ln in sample if _SEARCH_LINE.match(ln)) / n > 0.6:
        return Kind.SEARCH
    log_hits = sum(1 for ln in sample if _LOG_LINE.match(ln))
    code_hits = sum(1 for ln in sample if _CODE_HINT.match(ln))
    punct = sum(ln.count(";") + ln.count("{") + ln.count("}") + ln.count("(") for ln in sample)

    if log_hits / n > 0.4:
        return Kind.LOG
    md = sum(1 for ln in sample if ln.lstrip().startswith(("#", "- ", "* ", "> ", "|")))
    if md / n > 0.5:
        return Kind.TEXT  # markdown / bullet prose
    if code_hits / n > 0.15 or (code_hits / n > 0.05 and punct / n > 1.5):
        return Kind.CODE
    if code_hits and _is_python_module(text):
        return Kind.CODE
    return Kind.TEXT
