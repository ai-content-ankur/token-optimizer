"""Minimal MCP server over stdio (JSON-RPC 2.0, newline-delimited).

Register it with your agent, e.g. for Claude Code:

    claude mcp add minihead -- python -m minihead mcp

Tools:
  minihead_retrieve(hash)        full original behind a <<ccr:HASH ...>> marker
  minihead_compress(text, kind)  compress a block of text on demand
  minihead_stats()               tokens saved so far
"""

from __future__ import annotations

import json
import sys

from . import __version__
from .ccr import Store
from .pipeline import RETRIEVE_DESCRIPTION, RETRIEVE_TOOL_NAME, compress_text, retrieve

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": RETRIEVE_TOOL_NAME,
        "description": RETRIEVE_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {"hash": {"type": "string", "description": "12-char hash from a <<ccr:...>> marker"}},
            "required": ["hash"],
        },
    },
    {
        "name": "minihead_compress",
        "description": "Compress a large block of text (JSON, logs, search results, code, prose) "
        "before reasoning over it. Dropped parts stay retrievable with minihead_retrieve.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "kind": {"type": "string", "enum": ["json", "log", "search", "code", "diff", "text"]},
            },
            "required": ["text"],
        },
    },
    {
        "name": "minihead_stats",
        "description": "Show how many tokens minihead has saved.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle(req: dict, store: Store):
    method = req.get("method")
    id_ = req.get("id")
    if id_ is None:  # notification
        return None
    if method == "initialize":
        client_version = (req.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
        return _result(id_, {
            "protocolVersion": client_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "minihead", "version": __version__},
        })
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": TOOLS})
    if method == "tools/call":
        params = req.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            if name == RETRIEVE_TOOL_NAME:
                text = retrieve(str(args.get("hash", "")), store)
            elif name == "minihead_compress":
                block = compress_text(str(args.get("text", "")), store, kind=args.get("kind"), min_tokens=0)
                text = block.text + f"\n\n[minihead: {block.kind}, {block.tokens_before} -> {block.tokens_after} tokens]"
            elif name == "minihead_stats":
                text = json.dumps(store.stats(), indent=2)
            else:
                return _error(id_, -32602, f"unknown tool {name!r}")
        except Exception as e:  # report tool failures to the model, don't crash the server
            return _result(id_, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        return _result(id_, {"content": [{"type": "text", "text": text}]})
    return _error(id_, -32601, f"method not found: {method}")


def serve(store: Store | None = None) -> None:
    store = store or Store()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except ValueError:
            resp = _error(None, -32700, "parse error")
        else:
            resp = handle(req, store)
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()
