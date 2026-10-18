import io
import json
import sys
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minihead import compress, compress_text  # noqa: E402
from minihead import mcp_server, proxy  # noqa: E402
from minihead.ccr import MARKER_RE, Store  # noqa: E402
from minihead.detect import Kind, detect  # noqa: E402


def big_json(n=200, bad=123):
    items = [{"id": i, "status": "ok", "latency_ms": 40 + i % 7, "region": "eu"} for i in range(n)]
    items[bad]["status"] = "error"
    return json.dumps(items, indent=2)


def big_log(n=800):
    lines = [f"2026-10-18T10:00:{i % 60:02d}Z INFO worker: processed job {i} in {i % 90}ms" for i in range(n)]
    lines[500] = "2026-10-18T10:08:20Z FATAL db: connection refused host=10.0.0.5"
    return "\n".join(lines)


class StoreTest(unittest.TestCase):
    def test_roundtrip_and_markers(self):
        s = Store(":memory:")
        key = s.put("hello world")
        self.assertEqual(s.get(key), "hello world")
        self.assertEqual(s.resolve_markers(f"x <<ccr:{key} note>> y"), "x hello world y")
        self.assertIsNone(s.get("000000000000"))

    def test_stats_ledger(self):
        s = Store(":memory:")
        s.record("json", 100, 20)
        st = s.stats()
        self.assertEqual(st["tokens_saved"], 80)
        self.assertEqual(st["by_kind"]["json"]["calls"], 1)


class DetectTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(detect(big_json()), Kind.JSON)
        self.assertEqual(detect(big_log()), Kind.LOG)
        self.assertEqual(detect("a.py:1:x\nb.py:2:y\nc.ts:3:z"), Kind.SEARCH)
        self.assertEqual(detect("def f(x):\n    return x\n\nclass A:\n    pass\n"), Kind.CODE)
        self.assertEqual(detect("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b"), Kind.DIFF)
        self.assertEqual(detect("Just some ordinary words in a sentence."), Kind.TEXT)


class CompressorTest(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")

    def test_json_keeps_outlier_and_is_reversible(self):
        text = big_json()
        b = compress_text(text, self.s)
        self.assertEqual(b.kind, "json")
        self.assertLess(b.tokens_after, b.tokens_before * 0.3)
        self.assertIn('"error"', b.text)
        out = json.loads(b.text)  # still valid JSON
        self.assertIn("_ccr_dropped", out)
        key = MARKER_RE.search(b.text).group(1)
        self.assertEqual(json.loads(self.s.get(key)), json.loads(text))

    def test_small_payload_passthrough(self):
        b = compress_text('{"a": 1}', self.s)
        self.assertEqual(b.kind, "passthrough")
        self.assertEqual(b.text, '{"a": 1}')

    def test_log_keeps_fatal(self):
        b = compress_text(big_log(), self.s)
        self.assertEqual(b.kind, "log")
        self.assertIn("FATAL db: connection refused", b.text)
        self.assertLess(b.tokens_after, b.tokens_before * 0.2)

    def test_log_keeps_whole_traceback(self):
        lines = [f"INFO step {i} ok" for i in range(300)]
        lines[150:150] = [
            "Traceback (most recent call last):",
            '  File "a.py", line 3, in <module>',
            "    main()",
            '  File "a.py", line 2, in main',
            "    1/0",
            "ZeroDivisionError: division by zero",
        ]
        b = compress_text("\n".join(lines), self.s, kind="log")
        for needle in ('File "a.py", line 2', "1/0", "ZeroDivisionError"):
            self.assertIn(needle, b.text)

    def test_code_outline(self):
        body = "\n".join(f"    x{i} = {i}" for i in range(40))
        src = "import os\n\n" + "\n\n".join(f"def f{k}(a, b):\n    \"\"\"Doc {k}.\"\"\"\n{body}\n    return a" for k in range(10))
        b = compress_text(src, self.s, kind="code", min_tokens=0)
        self.assertIn("def f3(a, b):", b.text)
        self.assertIn('"""Doc 3."""', b.text)
        self.assertIn("elided", b.text)
        self.assertNotIn("x20 = 20", b.text)

    def test_brace_code_outline(self):
        body = "\n".join(f"  const v{i} = compute({i});" for i in range(30))
        src = "\n\n".join(f"export function handler{k}(req, res) {{\n{body}\n  return res;\n}}" for k in range(6))
        b = compress_text(src, self.s, kind="code", min_tokens=0)
        self.assertIn("export function handler4(req, res) {", b.text)
        self.assertIn("elided", b.text)

    def test_search_grouping(self):
        hits = "\n".join(f"src/m{i % 3}.py:{i}:needle_{i}" for i in range(120))
        b = compress_text(hits, self.s)
        self.assertEqual(b.kind, "search")
        self.assertIn("src/m0.py  [40 hits]", b.text)

    def test_already_compressed_is_skipped(self):
        b1 = compress_text(big_log(), self.s)
        b2 = compress_text(b1.text, self.s, min_tokens=0)
        self.assertEqual(b2.kind, "passthrough")


class TokenizerTest(unittest.TestCase):
    def test_o200k_known_values(self):
        from minihead import o200k

        self.assertEqual(o200k.encode("hello world"), [24912, 2375])
        self.assertEqual(o200k.count(""), 0)
        self.assertGreater(o200k.count("def f(x):\n    return x\n"), 5)


class PathAndReadTest(unittest.TestCase):
    def test_path_list_is_lossless(self):
        from minihead.compressors.path_compressor import expand_paths

        paths = [f"src/pkg{i % 7}/module_{i}.py" for i in range(300)]
        b = compress_text("\n".join(paths), Store(":memory:"))
        self.assertEqual(b.kind, "paths")
        self.assertNotIn("<<ccr:", b.text)
        self.assertEqual(sorted(expand_paths(b.text)), sorted(paths))
        self.assertLess(b.tokens_after, b.tokens_before * 0.7)

    def test_numbered_read_outlines_with_original_line_numbers(self):
        body = "\n".join(f"    x{i} = {i}" for i in range(40))
        src = "\n\n".join(f"def f{k}(a):\n{body}\n    return a" for k in range(12)).split("\n")
        numbered = "\n".join(f"{n + 101:>6}|{ln}" for n, ln in enumerate(src))  # read from line 101
        s = Store(":memory:")
        b = compress_text(numbered, s, min_tokens=0)
        self.assertEqual(b.kind, "code")
        self.assertIn("lines 102-", b.text)  # first body line is source line 102
        key = MARKER_RE.search(b.text).group(1)
        self.assertEqual(s.get(key), numbered)  # the numbered original is what comes back


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")

    def anthropic_msgs(self, *payloads):
        msgs = [{"role": "user", "content": "debug the outage"}]
        for i, p in enumerate(payloads):
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "run_query", "input": {}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": p}]})
        return msgs

    def test_anthropic_tool_results(self):
        msgs = self.anthropic_msgs(big_log())
        r = compress(msgs, store=self.s)
        self.assertGreater(r.compression_ratio, 0.5)
        self.assertIn("<<ccr:", r.messages[2]["content"][0]["content"])
        self.assertNotIn("<<ccr:", msgs[2]["content"][0]["content"])  # input untouched

    def test_openai_tool_messages(self):
        msgs = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big_json()},
        ]
        r = compress(msgs, store=self.s)
        self.assertIn("_ccr_dropped", r.messages[2]["content"])

    def test_cache_mode_keeps_prefix_stable(self):
        # turn 1: one tool result in the live zone -> compressed
        t1 = compress(self.anthropic_msgs(big_log()), store=self.s)
        sent_1 = json.dumps(t1.messages)
        # turn 2: same history plus a new result; the old block must come out identical
        t2 = compress(self.anthropic_msgs(big_log(), big_json()), store=self.s)
        self.assertTrue(json.dumps(t2.messages).startswith(sent_1[:-1]))
        self.assertIn("_ccr_dropped", t2.messages[4]["content"][0]["content"])

    def test_file_reads_are_protected(self):
        msgs = [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read", "input": {"path": "big.log"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": big_log()}]},
        ]
        r = compress(msgs, store=self.s)
        self.assertEqual(r.messages, msgs)
        self.assertEqual(r.blocks[0].kind, "protected-read")
        r2 = compress(msgs, store=Store(":memory:"), protect_reads=False)
        self.assertIn("<<ccr:", r2.messages[2]["content"][0]["content"])

    def test_cache_mode_leaves_unseen_history_alone(self):
        fresh = Store(":memory:")
        msgs = self.anthropic_msgs(big_log(), "small")
        r = compress(msgs, store=fresh)
        # the big log is history the provider already cached uncompressed: don't touch it
        self.assertEqual(r.messages[2], msgs[2])
        r_all = compress(msgs, store=Store(":memory:"), mode="all")
        self.assertNotEqual(r_all.messages[2], msgs[2])


class MCPTest(unittest.TestCase):
    def test_tools(self):
        s = Store(":memory:")
        init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}, s)
        self.assertEqual(init["result"]["serverInfo"]["name"], "minihead")
        tools = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, s)
        self.assertEqual({t["name"] for t in tools["result"]["tools"]}, {"minihead_retrieve", "minihead_compress", "minihead_stats"})
        out = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "minihead_compress", "arguments": {"text": big_log()}}}, s)
        text = out["result"]["content"][0]["text"]
        key = MARKER_RE.search(text).group(1)
        got = mcp_server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "minihead_retrieve", "arguments": {"hash": key}}}, s)
        self.assertEqual(got["result"]["content"][0]["text"], big_log())
        self.assertIsNone(mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, s))

    def test_stdio_loop(self):
        s = Store(":memory:")
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
        ]
        stdin = io.TextIOWrapper(io.BytesIO(("\n".join(lines) + "\n").encode()))
        stdout = io.TextIOWrapper(io.BytesIO())
        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = stdin, stdout
        try:
            mcp_server.serve(s)
            stdout.flush()
            replies = stdout.buffer.getvalue().decode().strip().splitlines()
        finally:
            sys.stdin, sys.stdout = old_in, old_out
        self.assertEqual([json.loads(r)["id"] for r in replies], [1, 2])


# ── proxy end-to-end against a fake upstream ───────────────────────────────


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list = []
    mode = "plain"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstream.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if body.get("stream"):
            payload = b"event: message_start\ndata: {}\n\nevent: message_stop\ndata: {}\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        msgs = body["messages"]
        last = msgs[-1]
        retrieved = isinstance(last.get("content"), list) and last["content"] and last["content"][0].get("type") == "tool_result" \
            and last["content"][0]["tool_use_id"] == "r1"
        if FakeUpstream.mode == "retrieve" and not retrieved:
            text = json.dumps(msgs)
            key = MARKER_RE.search(text).group(1)
            out = {"content": [{"type": "tool_use", "id": "r1", "name": "minihead_retrieve", "input": {"hash": key}}], "stop_reason": "tool_use"}
        else:
            out = {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        up_url = f"http://127.0.0.1:{cls.up.server_address[1]}"
        proxy.ANTHROPIC_URL = proxy.OPENAI_URL = up_url
        proxy.Handler.store = Store(":memory:")
        proxy.Handler.verbose = False
        cls.px = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        threading.Thread(target=cls.px.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.px.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.px.shutdown()
        cls.up.shutdown()

    def post(self, body):
        req = urllib.request.Request(
            self.base + "/v1/messages", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def body(self, stream):
        return {
            "model": "claude-x", "max_tokens": 100, "stream": stream,
            "messages": [
                {"role": "user", "content": "why did it fail?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "logs", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big_log()}]},
            ],
        }

    def test_streaming_passthrough_compresses_request(self):
        FakeUpstream.mode, FakeUpstream.requests = "plain", []
        status, ctype, data = self.post(self.body(stream=True))
        self.assertEqual(status, 200)
        self.assertIn("event-stream", ctype)
        self.assertIn(b"message_stop", data)
        sent = FakeUpstream.requests[-1]
        self.assertEqual(sent["headers"].get("x-api-key"), "sk-test")  # auth forwarded
        self.assertIn("<<ccr:", sent["body"]["messages"][2]["content"][0]["content"])

    def test_retrieval_loop_is_invisible_to_client(self):
        FakeUpstream.mode, FakeUpstream.requests = "retrieve", []
        status, _, data = self.post(self.body(stream=False))
        out = json.loads(data)
        self.assertEqual(out["content"][0]["text"], "done")
        self.assertEqual(len(FakeUpstream.requests), 2)
        answer = FakeUpstream.requests[1]["body"]["messages"][-1]["content"][0]["content"]
        self.assertEqual(answer, big_log())  # the model got the original back
        self.assertIn("minihead_retrieve", [t["name"] for t in FakeUpstream.requests[0]["body"]["tools"]])

    def test_stats_endpoint(self):
        with urllib.request.urlopen(self.base + "/minihead/stats", timeout=10) as r:
            self.assertIn("tokens_saved", json.loads(r.read()))


if __name__ == "__main__":
    unittest.main()
