"""Code compressor.

Agents often read whole files only to find one function. This compressor
keeps a file's *outline* — imports, class/function signatures, docstring
first lines, short bodies — and elides long function bodies. It is
deliberately conservative: bodies under ``MIN_BODY`` lines are kept, and
the elided ranges are named with line numbers so the agent can re-read
exactly the part it needs (or fetch the original via CCR).

Python is handled with the ``ast`` module. Other C-like languages use a
brace-matching pass over lines that look like function definitions.
"""

from __future__ import annotations

import ast
import re

from ..ccr import Store, marker

MIN_BODY = 12  # elide only bodies longer than this many lines


def _python(text: str, base: int = 1) -> tuple[str, int] | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    lines = text.splitlines()
    spans: list[tuple[int, int, int]] = []  # (first body line, last body line, indent)

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = child.body
                start = body[0]
                # keep the docstring
                if (
                    isinstance(start, ast.Expr)
                    and isinstance(getattr(start, "value", None), ast.Constant)
                    and isinstance(start.value.value, str)
                ):
                    if len(body) == 1:
                        continue
                    start = body[1]
                first = start.lineno
                last = child.end_lineno or first
                if last - first + 1 >= MIN_BODY:
                    indent = len(lines[first - 1]) - len(lines[first - 1].lstrip())
                    spans.append((first, last, indent))
                    continue  # don't descend into an elided body
            visit(child)

    visit(tree)
    if not spans:
        return None
    out: list[str] = []
    elided = 0
    spans.sort()
    cur = 1
    for first, last, indent in spans:
        out.extend(lines[cur - 1 : first - 1])
        out.append(" " * indent + f"...  # lines {first + base - 1}-{last + base - 1} elided ({last - first + 1} lines)")
        elided += last - first + 1
        cur = last + 1
    out.extend(lines[cur - 1 :])
    return "\n".join(out), elided


_FUNC_START = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public|private|protected|static|async|override|"
    r"final|virtual|inline|pub(?:\([^)]*\))?|unsafe|extern|func|fn|function|def|\w+[\w<>\[\],\s*&:]*)"
    r"[\s\w<>\[\],*&:.]*\([^;]*\)\s*(?:->\s*[^{]+|:\s*[^{=]+|throws [\w, .]+|const|noexcept)?\s*\{\s*$"
)
_ARROW_START = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]+)?=>\s*\{\s*$")
_CONTROL = re.compile(r"^\s*(if|for|while|switch|catch|else|do|try)\b")


def _braces(text: str, base: int = 1) -> tuple[str, int] | None:
    lines = text.splitlines()
    out: list[str] = []
    elided = 0
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        out.append(ln)
        if (_FUNC_START.match(ln) or _ARROW_START.match(ln)) and not _CONTROL.match(ln):
            depth = ln.count("{") - ln.count("}")
            j = i + 1
            while j < n and depth > 0:
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1
            body_first, body_last = i + 2, j - 1  # 1-based, exclusive of closing brace line
            if depth == 0 and body_last - body_first + 1 >= MIN_BODY:
                indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                out.append(" " * indent + f"/* lines {body_first + base - 1}-{body_last + base - 1} elided ({body_last - body_first + 1} lines) */")
                out.append(lines[j - 1])
                elided += body_last - body_first + 1
                i = j
                continue
        i += 1
    if not elided:
        return None
    return "\n".join(out), elided


def compress_code(text: str, store: Store, original: str | None = None, base: int = 1) -> str:
    """``original`` is what gets stored for retrieval when ``text`` was pre-processed."""
    result = _python(text, base) or _braces(text, base)
    if result is None:
        return text
    body, elided = result
    key = store.put(original if original is not None else text, "code")
    return body + "\n" + marker(key, f"original file, {elided} body lines elided")
