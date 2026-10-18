# token-optimizer

Tools for cutting the tokens AI coding agents spend on tool output, and for proving it.

- **`minihead`**: a small, dependency-free context compressor. It compresses tool outputs
  (JSON, logs, search results, file lists, code, diffs, prose) before they reach the LLM.
  It keeps the originals locally so nothing is lost, and runs as a library, a drop-in
  proxy or an MCP server. Pure Python standard library.
- **`minihead cursor-report`**: measures what minihead would have saved on your real
  Cursor agent sessions and writes a shareable HTML report.

## How minihead works

```
agent / app ──► minihead ──────────────────────────────────────► LLM provider
               │ router (detect.py) picks a compressor per block │
               │   json   → keep head/tail + outlier rows,       │
               │            columnar re-encode, dropped-row stats│
               │   log    → errors + context + tracebacks, fold  │
               │            repeats into "[xN similar]"          │
               │   search → group grep hits by file, cap per file│
               │   paths  → group file lists by directory        │
               │            (lossless)                           │
               │   code   → outline: signatures + docstrings,    │
               │            long bodies elided with line ranges  │
               │   diff   → keep every +/- line, trim context    │
               │   text   → whitespace/repeat folding, head+tail │
               │ CCR store (SQLite): originals behind <<ccr:HASH>>│
               │ memo: same input → same bytes (prompt cache safe)│
```

- **Reversible.** Every lossy step stores the original and leaves a marker such as
  `<<ccr:5e7bf1c4b3c7 244 of 250 items omitted>>`. The model fetches the original with
  the `minihead_retrieve` tool (from the MCP server, or handled inside the proxy for
  non-streaming requests).
- **Cache-safe.** Providers cache the request prefix, so changing an old message costs
  the whole cache. In the default `cache` mode only tool results newer than the last
  assistant message are compressed. Older blocks are replaced only by their previously
  sent compressed form, which reproduces the same bytes.
- **File reads are protected.** Output from file-read tools (`Read`, `read_file`,
  `view`…) passes through untouched by default, because the agent usually reads a
  file in order to edit it. Set `MINIHEAD_COMPRESS_READS=1` to compress reads too.
- **Exact token counts.** Counts use OpenAI's `o200k_base` tokenizer (GPT-4o, GPT-4.1,
  o-series, GPT-5). It runs from a bundled pure-Python implementation, or `tiktoken`
  if installed. The bundled one matched OpenAI's reference algorithm on 5.4 million
  tokens with zero differences. For Claude models, counts are a close proxy.

## Install (Windows, macOS, Linux)

```powershell
git clone https://github.com/<you>/token-optimizer
cd token-optimizer
python -m pip install -e .
```

Requires Python 3.10+. No other dependencies.

## Use it

```powershell
# Compress a file and see the savings
minihead compress build.log
minihead compress results.json --stats-only

# Drop-in proxy (zero code changes)
minihead proxy --port 8787
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"   # Claude Code / Anthropic SDK
$env:OPENAI_BASE_URL    = "http://127.0.0.1:8787/v1" # OpenAI-compatible clients
# live savings: http://127.0.0.1:8787/minihead/stats

# MCP server (gives agents minihead_retrieve / _compress / _stats)
claude mcp add minihead -- python -m minihead mcp

# Library
python -c "from minihead import compress; print(compress(messages).tokens_saved)"

# Other
minihead retrieve 5e7bf1c4b3c7   # print an original
minihead stats                   # tokens saved so far
minihead bench                   # built-in benchmark
python -m unittest discover -s tests -v
```

Data lives in `~/.minihead/store.db` (override with `MINIHEAD_HOME`). Upstreams default
to the official APIs; override with `MINIHEAD_ANTHROPIC_URL` / `MINIHEAD_OPENAI_URL`.

**Cursor note:** Cursor sends requests through its own servers, so the proxy only
works with your own API key plus a public tunnel to the proxy. The MCP server and the
report below work without that.

## Proving the savings on real Cursor sessions

```powershell
minihead cursor-report                    # all projects in ~/.cursor/projects
minihead cursor-report --project myapp    # only matching projects
minihead cursor-report --examples         # add 3 before/after snippets (shows file contents)
```

Cursor stores each agent chat as JSONL in
`~/.cursor/projects/<project>/agent-transcripts/`. The transcripts record every tool
call's arguments but not its result. So the report:

1. Re-runs each `Read`, `Grep` and `Glob` call **read-only** against the project files on
   disk, rebuilding what the agent saw.
2. Compresses each rebuilt result with minihead.
3. Counts tokens before and after with the exact `o200k_base` tokenizer.
4. Checks that every lossy result can be restored byte-for-byte from the local store.

It writes `minihead-report.html`, plus `.csv` and `.json` files with one row per tool
result so anyone can re-check the numbers. Terminal commands are never re-run, so their
output (often the most compressible) is left out. Nothing leaves the machine, and the
report contains only numbers unless `--examples` is passed.

When replaying on a different machine from where the sessions ran, rewrite paths
with `--map "d:\app=/home/me/app"`.

## Benchmark (`minihead bench`, seeded, exact o200k tokens)

| Scenario               | Before | After | Saved | Needle |
|------------------------|-------:|------:|------:|--------|
| Code search (100 hits) |  2,009 | 1,409 |  30%  | kept   |
| SRE incident logs      | 48,652 |   729 |  99%  | kept   |
| JSON API listing (250) | 20,077 |   353 |  98%  | kept   |
| Large Python file      | 14,391 |   999 |  93%  | kept   |
| Prose                  |  1,862 | 1,862 |   0%  | kept   |

These payloads are synthetic and repetitive, so they flatter the numbers. Each one
hides a single key fact (the "needle"), and the table shows it survives. Use
`cursor-report` for numbers on real work.

## Limitations

- Pure Python: compression takes a few milliseconds per block.
- Prose has no ML compressor: it gets lossless whitespace and repeat folding, and
  very long prose (over 6,000 tokens) keeps its start and end with the middle stored
  for retrieval. Typical prose under that size passes through unchanged.
- Code outlining uses Python's `ast` for Python and a brace-matching heuristic for
  C-like languages; other languages pass through.
- The proxy's built-in retrieval loop covers non-streaming requests; streaming
  clients retrieve through the MCP server.

## License

MIT for the code in this repository. The bundled `o200k_base.tiktoken` file is
OpenAI's, under the MIT License; see `THIRD_PARTY_NOTICES.md`.
