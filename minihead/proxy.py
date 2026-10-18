"""Drop-in HTTP proxy: point your agent's base URL at it, zero code changes.

    python -m minihead proxy --port 8787

    # Claude Code / Anthropic SDK
    set ANTHROPIC_BASE_URL=http://127.0.0.1:8787
    # OpenAI-compatible clients
    set OPENAI_BASE_URL=http://127.0.0.1:8787/v1

Requests to ``/v1/messages`` (Anthropic), ``/v1/chat/completions`` and
``/v1/responses`` (OpenAI) get their tool outputs compressed; everything
else — and every response — is relayed byte-for-byte, streaming included.

Retrieval: for non-streaming requests that contain CCR markers the proxy
adds a ``minihead_retrieve`` tool and answers those calls itself, so the
client never sees them. Streaming clients (Claude Code, Cursor) get
retrieval through the MCP server instead: ``python -m minihead mcp``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .ccr import MARKER_RE, Store
from .pipeline import (
    ANTHROPIC_RETRIEVE_TOOL,
    OPENAI_RETRIEVE_TOOL,
    RETRIEVE_TOOL_NAME,
    compress,
    retrieve,
)

ANTHROPIC_URL = os.environ.get("MINIHEAD_ANTHROPIC_URL", "https://api.anthropic.com").rstrip("/")
OPENAI_URL = os.environ.get("MINIHEAD_OPENAI_URL", "https://api.openai.com").rstrip("/")
MAX_RETRIEVE_ROUNDS = 3
HOP_BY_HOP = {
    "host", "content-length", "connection", "keep-alive", "proxy-authorization",
    "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "accept-encoding",
}


def _upstream_for(path: str, headers) -> str:
    if path.startswith("/v1/messages") or headers.get("anthropic-version") or headers.get("x-api-key"):
        return ANTHROPIC_URL
    return OPENAI_URL


def _has_marker(obj) -> bool:
    return bool(MARKER_RE.search(json.dumps(obj, ensure_ascii=False)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    store: Store
    verbose = False

    # ── plumbing ──────────────────────────────────────────────────────
    def log_message(self, fmt, *args):  # quieter default logging
        if self.verbose:
            sys.stderr.write("[minihead] " + (fmt % args) + "\n")

    def _forward_headers(self) -> dict:
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}

    def _send(self, status: int, body: bytes, ctype: str = "application/json", headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            if k.lower() not in HOP_BY_HOP and k.lower() != "content-type":
                self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _open(self, method: str, url: str, body: bytes | None, headers: dict):
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as e:
            return e  # HTTPError is file-like: status + headers + body

    def _relay(self, resp) -> None:
        """Stream an upstream response to the client unchanged."""
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    # ── verbs ─────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, b'{"ok":true}')
        if self.path == "/minihead/stats":
            return self._send(200, json.dumps(self.store.stats(), indent=2).encode())
        if self.path.startswith("/minihead/retrieve/"):
            key = self.path.rsplit("/", 1)[-1]
            return self._send(200, retrieve(key, self.store).encode(), "text/plain; charset=utf-8")
        self._passthrough("GET", None)

    def do_DELETE(self):
        self._passthrough("DELETE", None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        if not any(path.endswith(p) for p in ("/messages", "/chat/completions", "/responses")):
            return self._passthrough("POST", raw)
        try:
            body = json.loads(raw)
        except ValueError:
            return self._passthrough("POST", raw)

        anthropic = _upstream_for(path, self.headers) == ANTHROPIC_URL
        key = "input" if path.endswith("/responses") else "messages"
        msgs = body.get(key)
        if isinstance(msgs, list):
            res = compress(msgs, store=self.store)
            body[key] = res.messages
            if res.tokens_saved:
                self.log_message(
                    "%s: %d -> %d tokens (-%d%%)", path, res.tokens_before, res.tokens_after,
                    round(100 * res.compression_ratio),
                )

        streaming = bool(body.get("stream"))
        if streaming or key == "input" or not _has_marker(body.get(key)):
            return self._passthrough("POST", json.dumps(body).encode())
        return self._with_retrieval(body, anthropic)

    # ── helpers ───────────────────────────────────────────────────────
    def _passthrough(self, method: str, body: bytes | None):
        upstream = _upstream_for(self.path, self.headers)
        resp = self._open(method, upstream + self.path, body, self._forward_headers())
        self._relay(resp)

    def _with_retrieval(self, body: dict, anthropic: bool):
        """Non-streaming request: let the model call minihead_retrieve; answer it here."""
        tools = body.setdefault("tools", [])
        tools.append(ANTHROPIC_RETRIEVE_TOOL if anthropic else OPENAI_RETRIEVE_TOOL)
        url = (ANTHROPIC_URL if anthropic else OPENAI_URL) + self.path
        headers = self._forward_headers()
        headers["Content-Type"] = "application/json"

        for _ in range(MAX_RETRIEVE_ROUNDS + 1):
            resp = self._open("POST", url, json.dumps(body).encode(), headers)
            data = resp.read()
            if resp.status != 200:
                return self._send(resp.status, data, resp.headers.get("Content-Type", "application/json"))
            out = json.loads(data)
            calls = self._retrieve_calls(out, anthropic)
            if not calls:
                break
            if not calls["only_retrieve"]:
                out = self._strip_retrieve(out, anthropic)
                break
            self._append_answers(body, out, calls["ids"], anthropic)
        out = self._strip_retrieve(out, anthropic)  # never leak our tool to the client
        return self._send(200, json.dumps(out).encode())

    @staticmethod
    def _retrieve_calls(out: dict, anthropic: bool):
        if anthropic:
            uses = [b for b in out.get("content", []) if b.get("type") == "tool_use"]
            mine = [(b["id"], b.get("input", {}).get("hash", "")) for b in uses if b.get("name") == RETRIEVE_TOOL_NAME]
        else:
            msg = (out.get("choices") or [{}])[0].get("message") or {}
            uses = msg.get("tool_calls") or []
            mine = []
            for c in uses:
                if c.get("function", {}).get("name") == RETRIEVE_TOOL_NAME:
                    try:
                        h = json.loads(c["function"].get("arguments") or "{}").get("hash", "")
                    except ValueError:
                        h = ""
                    mine.append((c["id"], h))
        if not mine:
            return None
        return {"ids": mine, "only_retrieve": len(mine) == len(uses)}

    def _append_answers(self, body: dict, out: dict, ids, anthropic: bool) -> None:
        if anthropic:
            body["messages"].append({"role": "assistant", "content": out["content"]})
            body["messages"].append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": i, "content": retrieve(h, self.store)} for i, h in ids],
            })
        else:
            body["messages"].append(out["choices"][0]["message"])
            for i, h in ids:
                body["messages"].append({"role": "tool", "tool_call_id": i, "content": retrieve(h, self.store)})

    @staticmethod
    def _strip_retrieve(out: dict, anthropic: bool) -> dict:
        if anthropic:
            out["content"] = [
                b for b in out.get("content", [])
                if not (b.get("type") == "tool_use" and b.get("name") == RETRIEVE_TOOL_NAME)
            ]
            if out.get("stop_reason") == "tool_use" and not any(b.get("type") == "tool_use" for b in out["content"]):
                out["stop_reason"] = "end_turn"
        else:
            for ch in out.get("choices", []):
                m = ch.get("message") or {}
                if m.get("tool_calls"):
                    m["tool_calls"] = [c for c in m["tool_calls"] if c.get("function", {}).get("name") != RETRIEVE_TOOL_NAME]
                    if not m["tool_calls"]:
                        m.pop("tool_calls")
                        if ch.get("finish_reason") == "tool_calls":
                            ch["finish_reason"] = "stop"
        return out


def serve(host: str = "127.0.0.1", port: int = 8787, store: Store | None = None, verbose: bool = True) -> None:
    Handler.store = store or Store()
    Handler.verbose = verbose
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"minihead proxy on http://{host}:{port}  ->  anthropic={ANTHROPIC_URL}  openai={OPENAI_URL}")
    print(f"  ANTHROPIC_BASE_URL=http://{host}:{port}    OPENAI_BASE_URL=http://{host}:{port}/v1")
    print(f"  live savings: http://{host}:{port}/minihead/stats")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
