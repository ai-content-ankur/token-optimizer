"""Command line: python -m minihead <command>

  compress [FILE]        compress a file (or stdin) and print the result
  retrieve HASH          print the original behind a <<ccr:HASH>> marker
  stats                  tokens saved so far
  proxy [--port 8787]    run the drop-in compression proxy
  mcp                    run the MCP server on stdio
  bench                  run the built-in benchmark
  cursor-report          measure savings on your real Cursor agent sessions
"""

from __future__ import annotations

import argparse
import json
import sys

from .ccr import Store
from .tokens import tokenizer_name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="minihead", description="Context compression for LLM agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="compress a file or stdin")
    c.add_argument("file", nargs="?", default="-")
    c.add_argument("--kind", choices=["json", "log", "search", "code", "diff", "text"])
    c.add_argument("--stats-only", action="store_true", help="print only the token counts")

    r = sub.add_parser("retrieve", help="print a stored original")
    r.add_argument("hash")

    sub.add_parser("stats", help="show savings")

    pr = sub.add_parser("proxy", help="run the HTTP proxy")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=8787)
    pr.add_argument("--quiet", action="store_true")

    sub.add_parser("mcp", help="run the MCP server on stdio")
    sub.add_parser("bench", help="run the built-in benchmark")
    sub.add_parser("cursor-report", help="measure savings on your real Cursor sessions", add_help=False)

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "cursor-report":  # has its own argument parser
        from .cursor_report import run as cursor_run

        return cursor_run(argv[1:])
    a = p.parse_args(argv)

    if a.cmd == "compress":
        from .pipeline import compress_text

        if a.file == "-":
            text = sys.stdin.read()
        else:
            with open(a.file, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        block = compress_text(text, kind=a.kind, min_tokens=0)
        if not a.stats_only:
            sys.stdout.write(block.text + "\n")
        saved = block.tokens_before - block.tokens_after
        pct = 100 * saved / block.tokens_before if block.tokens_before else 0
        print(
            f"[minihead] {block.kind}: {block.tokens_before} -> {block.tokens_after} tokens "
            f"(-{pct:.0f}%, {tokenizer_name()})",
            file=sys.stderr,
        )
        return 0

    if a.cmd == "retrieve":
        from .pipeline import retrieve

        sys.stdout.write(retrieve(a.hash) + "\n")
        return 0

    if a.cmd == "stats":
        print(json.dumps(Store().stats(), indent=2))
        return 0

    if a.cmd == "proxy":
        from .proxy import serve

        serve(a.host, a.port, verbose=not a.quiet)
        return 0

    if a.cmd == "mcp":
        from .mcp_server import serve

        serve()
        return 0

    if a.cmd == "bench":
        from .bench import run

        run()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
