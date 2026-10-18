"""Measure what minihead would have saved on your real Cursor agent sessions.

    python -m minihead cursor-report                 # all projects
    python -m minihead cursor-report --project myapp # only matching projects

Cursor keeps a transcript of every agent chat in
``~/.cursor/projects/<project>/agent-transcripts/<id>/<id>.jsonl``. Each
transcript records the tool calls the agent made (file reads, searches,
file listings, terminal commands) but not their results. This module
rebuilds the results of the read-only calls — Read, Grep, Glob — by running
them again against the project on disk, then measures each result before
and after minihead with OpenAI's o200k_base tokenizer.

Terminal commands are never re-run (they could have side effects), so the
report counts them as unmeasured. Nothing leaves the machine. The report
contains numbers only — no file contents — unless ``--examples`` is passed.
"""

from __future__ import annotations

import csv
import datetime as _dt
import fnmatch
import html
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, o200k
from .ccr import MARKER_RE, Store
from .pipeline import READ_TOOL_RE, compress_text

REPLAYABLE = {"read", "grep", "glob"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
             "dist", "build", ".next", ".turbo", "target", ".idea", ".vscode", ".tox", "coverage"}
MAX_FILE_BYTES = 2_000_000


# ── path handling ───────────────────────────────────────────────────────────


class PathMapper:
    """Translate paths recorded in transcripts (often Windows paths) to local ones."""

    def __init__(self, maps: list[str]):
        self.maps = []
        for m in maps:
            src, _, dst = m.partition("=")
            self.maps.append((self._norm(src), dst))

    @staticmethod
    def _norm(p: str) -> str:
        return p.replace("\\", "/").rstrip("/").lower()

    def local(self, p: str) -> Path:
        n = self._norm(p)
        for src, dst in self.maps:
            if n == src or n.startswith(src + "/"):
                rest = p.replace("\\", "/")[len(src):].lstrip("/")
                return Path(dst) / rest if rest else Path(dst)
        return Path(p)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


_FILES_CACHE: dict[Path, list[Path]] = {}


def list_files(root: Path) -> list[Path]:
    """Non-ignored files under root (git's view when it's a repo, like ripgrep)."""
    root = root.resolve()
    if root in _FILES_CACHE:
        return _FILES_CACHE[root]
    files: list[Path] = []
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=60,
        )
        if out.returncode == 0 and out.stdout:
            files = [root / p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]
            files = [f for f in files if f.is_file()]
    except (OSError, subprocess.SubprocessError):
        files = []
    if not files:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            files.extend(Path(dirpath) / f for f in filenames)
    files.sort()
    _FILES_CACHE[root] = files
    return files


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8000]:
        return None  # binary
    return data.decode("utf-8", "replace")


# ── replaying tools ─────────────────────────────────────────────────────────


def replay_read(inp: dict, pm: PathMapper) -> str | None:
    path = pm.local(str(inp.get("path", "")))
    text = _read_text(path)
    if text is None:
        return None
    lines = text.splitlines()
    start = max(int(inp.get("offset") or 1), 1)
    limit = inp.get("limit")
    chunk = lines[start - 1 : (start - 1 + int(limit)) if limit else None]
    return "\n".join(f"{start + i:>6}|{ln}" for i, ln in enumerate(chunk))


def _glob_match(rel: str, pattern: str) -> bool:
    pattern = pattern.replace("\\", "/")
    if "/" not in pattern:
        return fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern[3:])
    return fnmatch.fnmatch(rel, pattern)


def replay_glob(inp: dict, pm: PathMapper, root: Path) -> str | None:
    base = pm.local(str(inp["target_directory"])) if inp.get("target_directory") else root
    if not base.is_dir():
        return None
    pattern = str(inp.get("glob_pattern") or "**/*")
    hits = [_rel(f, base) for f in list_files(base) if _glob_match(_rel(f, base), pattern)]
    return "\n".join(hits) if hits else "No files found"


def replay_grep(inp: dict, pm: PathMapper, root: Path) -> str | None:
    target = pm.local(str(inp["path"])) if inp.get("path") else root
    flags = re.IGNORECASE if inp.get("-i") else 0
    if inp.get("multiline"):
        flags |= re.MULTILINE | re.DOTALL
    try:
        rx = re.compile(str(inp.get("pattern", "")), flags)
    except re.error:
        return None  # ripgrep-only syntax we can't reproduce faithfully
    if target.is_file():
        files, base = [target], target.parent
    elif target.is_dir():
        files, base = list_files(target), target
    else:
        return None
    if inp.get("glob"):
        files = [f for f in files if _glob_match(_rel(f, base), str(inp["glob"]))]
    after = int(inp.get("-A") or inp.get("-C") or 0)
    before = int(inp.get("-B") or inp.get("-C") or 0)
    mode = inp.get("output_mode") or "content"

    out: list[str] = []
    for f in files:
        text = _read_text(f)
        if text is None:
            continue
        lines = text.splitlines()
        hit_idx = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if not hit_idx:
            continue
        rel = _rel(f, base)
        if mode == "files_with_matches":
            out.append(rel)
        elif mode == "count":
            out.append(f"{rel}:{len(hit_idx)}")
        else:
            shown: set[int] = set()
            hits = set(hit_idx)
            for i in hit_idx:
                for j in range(max(0, i - before), min(len(lines), i + after + 1)):
                    if j not in shown:
                        shown.add(j)
                        sep = ":" if j in hits else "-"
                        out.append(f"{rel}{sep}{j + 1}{sep}{lines[j]}")
    if inp.get("head_limit"):
        out = out[: int(inp["head_limit"])]
    return "\n".join(out) if out else "No matches found"


# ── measurement ─────────────────────────────────────────────────────────────


@dataclass
class Item:
    session: str
    project: str
    tool: str
    kind: str
    before: int
    after: int
    lossy: bool
    recoverable: bool
    protected: bool


@dataclass
class Totals:
    sessions: int = 0
    transcripts_skipped: int = 0
    conversation_tokens: int = 0  # user + assistant text and tool-call arguments
    tool_calls: Counter = field(default_factory=Counter)
    replayed: Counter = field(default_factory=Counter)
    not_replayable: Counter = field(default_factory=Counter)  # tool ran but result can't be rebuilt
    missing: Counter = field(default_factory=Counter)  # file gone / unreadable now


def _guess_root(calls: list[dict], pm: PathMapper) -> Path:
    paths = []
    for c in calls:
        inp = c.get("input") or {}
        for k in ("target_directory", "path"):
            v = inp.get(k)
            if isinstance(v, str) and (":" in v[:3] or v.startswith(("/", "\\"))):
                paths.append(v.replace("\\", "/"))
    if not paths:
        return Path.cwd()
    # most common first two path segments, e.g. "d:/finance_tracker"
    heads = Counter("/".join(p.split("/")[:2]) for p in paths)
    return pm.local(heads.most_common(1)[0][0])


def analyse(transcript: Path, project: str, pm: PathMapper, store: Store, totals: Totals,
            items: list[Item], examples: list[dict] | None) -> None:
    calls: list[dict] = []
    try:
        lines = transcript.read_text(encoding="utf-8").splitlines()
    except OSError:
        totals.transcripts_skipped += 1
        return
    for ln in lines:
        try:
            msg = json.loads(ln)
        except ValueError:
            continue
        for block in (msg.get("message") or {}).get("content") or []:
            if block.get("type") == "text":
                totals.conversation_tokens += o200k.count(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(block)
                totals.conversation_tokens += o200k.count(json.dumps(block.get("input", {}), ensure_ascii=False))
    if not calls and not lines:
        totals.transcripts_skipped += 1
        return
    totals.sessions += 1
    root = _guess_root(calls, pm)
    session = transcript.stem[:8]

    for call in calls:
        name = str(call.get("name", ""))
        key = name.lower()
        totals.tool_calls[name] += 1
        if key not in REPLAYABLE:
            totals.not_replayable[name] += 1
            continue
        inp = call.get("input") or {}
        try:
            if key == "read":
                output = replay_read(inp, pm)
            elif key == "grep":
                output = replay_grep(inp, pm, root)
            else:
                output = replay_glob(inp, pm, root)
        except Exception:
            output = None
        if output is None:
            totals.missing[name] += 1
            continue
        totals.replayed[name] += 1
        block = compress_text(output, store, record=False)
        lossy = bool(MARKER_RE.search(block.text)) and block.kind != "passthrough"
        recoverable = (not lossy) or _recoverable(block.text, block.kind, output, store)
        items.append(Item(session, project, name, block.kind, block.tokens_before, block.tokens_after,
                          lossy, recoverable, bool(READ_TOOL_RE.match(name))))
        if examples is not None and block.kind != "passthrough" and len(examples) < 3:
            examples.append({"tool": name, "kind": block.kind, "before": output[:2500], "after": block.text[:2500],
                             "tb": block.tokens_before, "ta": block.tokens_after})


def _recoverable(text: str, kind: str, original: str, store: Store) -> bool:
    """Every marker must resolve; for whole-block compressors the stored copy must equal the original."""
    keys = MARKER_RE.findall(text)
    stored = [store.get(k) for k in keys]
    if not keys or any(v is None for v in stored):
        return False
    if kind == "json":  # markers hold the dropped arrays; check each is valid JSON
        try:
            for v in stored:
                json.loads(v)
            return True
        except ValueError:
            return False
    return original in stored


def find_transcripts(cursor_dir: Path, project_filter: str | None) -> list[tuple[str, Path]]:
    found = []
    for proj in sorted(p for p in cursor_dir.iterdir() if p.is_dir()):
        if project_filter and project_filter.lower() not in proj.name.lower():
            continue
        for t in sorted((proj / "agent-transcripts").glob("*/*.jsonl")):
            found.append((proj.name, t))
    return found


# ── report ──────────────────────────────────────────────────────────────────


def _pct(a: int, b: int) -> str:
    return f"{100 * (1 - b / a):.1f}%" if a else "–"


def _summarise(items: list[Item], totals: Totals) -> dict:
    def agg(rows):
        b = sum(r.before for r in rows)
        a = sum(r.after for r in rows)
        return {"items": len(rows), "before": b, "after": a, "saved": b - a}

    shipped = [Item(**{**r.__dict__, "after": r.before}) if r.protected else r for r in items]
    by_tool = {t: agg([r for r in shipped if r.tool == t]) for t in sorted({r.tool for r in items})}
    by_kind = {k: agg([r for r in items if r.kind == k]) for k in sorted({r.kind for r in items})}
    s_all, s_ship = agg(items), agg(shipped)
    convo = totals.conversation_tokens
    return {
        "shipped": s_ship,
        "aggressive": s_all,
        "by_tool": by_tool,
        "by_kind": by_kind,
        "session_total_before": convo + s_ship["before"],
        "session_total_after": convo + s_ship["after"],
        "lossy_items": sum(r.lossy for r in items),
        "lossy_recoverable": sum(r.lossy and r.recoverable for r in items),
        "sessions": totals.sessions,
        "tool_calls": dict(totals.tool_calls),
        "replayed": dict(totals.replayed),
        "not_replayable": dict(totals.not_replayable),
        "missing": dict(totals.missing),
        "conversation_tokens": convo,
    }


def _bar(before: int, after: int) -> str:
    w = 100 * after / before if before else 100
    return (f'<div class="bar"><span style="width:{w:.1f}%"></span></div>')


def render_html(s: dict, meta: dict, examples: list[dict] | None) -> str:
    e = html.escape
    ship, agg = s["shipped"], s["aggressive"]
    rows_tool = "".join(
        f"<tr><td>{e(t)}</td><td class=n>{v['items']:,}</td><td class=n>{v['before']:,}</td>"
        f"<td class=n>{v['after']:,}</td><td class=n><b>{_pct(v['before'], v['after'])}</b></td>"
        f"<td>{_bar(v['before'], v['after'])}</td></tr>"
        for t, v in s["by_tool"].items()
    )
    rows_kind = "".join(
        f"<tr><td>{e(k)}</td><td class=n>{v['items']:,}</td><td class=n>{v['before']:,}</td>"
        f"<td class=n>{v['after']:,}</td><td class=n>{_pct(v['before'], v['after'])}</td></tr>"
        for k, v in s["by_kind"].items()
    )
    unmeasured = ", ".join(f"{e(k)} ×{v}" for k, v in sorted(s["not_replayable"].items(), key=lambda x: -x[1])) or "none"
    missing = sum(s["missing"].values())
    ex_html = ""
    if examples:
        ex_html = "<h2>Examples (before → after)</h2>" + "".join(
            f"<details><summary>{e(x['tool'])} · {e(x['kind'])} · {x['tb']:,} → {x['ta']:,} tokens</summary>"
            f"<div class=ex><pre>{e(x['before'])}</pre><pre>{e(x['after'])}</pre></div></details>"
            for x in examples
        )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Token Savings Report</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a64;--line:#e4e1d8;--acc:#1f6f5c;--acc2:#d6ebe4;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151614;--fg:#ecebe6;--mute:#a09e96;--line:#2c2d2a;--acc:#5cc3a6;--acc2:#1e3a32;--card:#1c1d1b}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:36px 0 10px}}
.sub{{color:var(--mute)}}.hero{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}}.big{{font-size:30px;font-weight:700;color:var(--acc)}}
.lbl{{color:var(--mute);font-size:13px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--mute);font-weight:600}}.n{{text-align:right;font-variant-numeric:tabular-nums}}.bar{{background:var(--acc2);height:8px;border-radius:4px;min-width:80px}}
.bar span{{display:block;height:8px;background:var(--acc);border-radius:4px}}ul{{padding-left:20px}}code,pre{{font:12.5px/1.45 ui-monospace,Consolas,monospace}}
pre{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;max-height:320px;white-space:pre-wrap;word-break:break-word}}
.ex{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}@media(max-width:640px){{.ex{{grid-template-columns:1fr}}}}
details{{margin:8px 0}}summary{{cursor:pointer}}.meta td{{border:0;padding:2px 8px 2px 0;vertical-align:top}}.meta td:first-child{{color:var(--mute);white-space:nowrap}}
</style></head><body><main>
<h1>Token savings on real Cursor sessions</h1>
<div class=sub>minihead {e(meta['version'])} · generated {e(meta['generated'])} · {s['sessions']} agent sessions · {sum(s['replayed'].values()):,} tool results replayed</div>

<div class=hero>
<div class=card><div class=lbl>Tool-output tokens saved</div><div class=big>{_pct(ship['before'], ship['after'])}</div>
<div class=lbl>{ship['before']:,} → {ship['after']:,}</div></div>
<div class=card><div class=lbl>Whole-session tokens saved</div><div class=big>{_pct(s['session_total_before'], s['session_total_after'])}</div>
<div class=lbl>{s['session_total_before']:,} → {s['session_total_after']:,}</div></div>
<div class=card><div class=lbl>Lossy items recoverable byte-for-byte</div><div class=big>{s['lossy_recoverable']} / {s['lossy_items']}</div>
<div class=lbl>checked against the stored originals</div></div>
</div>

<p>Default settings: file reads are passed through untouched (the agent may be about to edit them).
With file-read compression switched on, tool-output savings would be <b>{_pct(agg['before'], agg['after'])}</b>
({agg['before']:,} → {agg['after']:,}).</p>

<h2>By tool (default settings)</h2>
<table><tr><th>Tool</th><th class=n>Results</th><th class=n>Tokens before</th><th class=n>After</th><th class=n>Saved</th><th>After ÷ before</th></tr>{rows_tool}</table>

<h2>By content type (all results, before read protection)</h2>
<table><tr><th>Detected as</th><th class=n>Results</th><th class=n>Before</th><th class=n>After</th><th class=n>Saved</th></tr>{rows_kind}</table>

<h2>How this was measured</h2>
<ul>
<li><b>Data:</b> Cursor's own agent transcripts on this machine (<code>~/.cursor/projects/*/agent-transcripts</code>). Cursor records each tool call's arguments but not its result, so every Read, Grep and Glob call was <b>re-run read-only</b> against the project files as they are today.</li>
<li><b>Tokens:</b> OpenAI's <code>o200k_base</code> tokenizer (GPT-4o / GPT-4.1 / GPT-5) — {e(meta['tokenizer'])}; ranks file sha256 <code>{e(meta['ranks_sha'])}</code>, identical to OpenAI's published file. For Claude models the counts are a close proxy.</li>
<li><b>Whole-session</b> = user messages + assistant messages + tool-call arguments + replayed tool results.</li>
<li><b>Not measured:</b> {unmeasured}. Terminal commands are never re-run, so their output — often the most compressible (build and test logs) — is left out. {missing} replayable calls pointed at files that no longer exist.</li>
<li><b>Lossy</b> results carry a <code>&lt;&lt;ccr:…&gt;&gt;</code> marker; the original is stored locally and the model can fetch it with <code>minihead_retrieve</code>. Every such original was checked to be retrievable.</li>
<li><b>Not shown here:</b> whether answers stay identical. That needs an A/B run against the model; see the minihead README.</li>
</ul>

<h2>Reproduce</h2>
<pre>pip install -e .   # in the minihead folder
python -m minihead cursor-report{e(meta['args'])}</pre>
<table class=meta>
<tr><td>minihead</td><td>{e(meta['version'])}</td></tr><tr><td>Python</td><td>{e(meta['python'])}</td></tr>
<tr><td>Platform</td><td>{e(meta['platform'])}</td></tr><tr><td>Raw data</td><td>{e(meta['csv'])} (one row per tool result, numbers only)</td></tr>
</table>
{ex_html}
</main></body></html>"""


def run(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="minihead cursor-report", description=__doc__.split("\n\n")[0])
    p.add_argument("--cursor-dir", default=str(Path.home() / ".cursor" / "projects"))
    p.add_argument("--project", help="only projects whose folder name contains this text")
    p.add_argument("--map", action="append", default=[], metavar="FROM=TO",
                   help="rewrite recorded paths, e.g. 'd:\\app=/mnt/app' when replaying on another machine")
    p.add_argument("--out", default="minihead-report.html")
    p.add_argument("--examples", action="store_true", help="include 3 before/after snippets (shows file contents)")
    a = p.parse_args(argv)

    cursor_dir = Path(a.cursor_dir).expanduser()
    if not cursor_dir.is_dir():
        print(f"No Cursor projects folder at {cursor_dir}", file=sys.stderr)
        return 1
    transcripts = find_transcripts(cursor_dir, a.project)
    if not transcripts:
        print("No agent transcripts found.", file=sys.stderr)
        return 1

    pm = PathMapper(a.map)
    totals = Totals()
    items: list[Item] = []
    examples: list[dict] | None = [] if a.examples else None
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "report.db")
        for i, (proj, t) in enumerate(transcripts, 1):
            print(f"\r[{i}/{len(transcripts)}] {proj[:50]:50}", end="", file=sys.stderr, flush=True)
            analyse(t, proj, pm, store, totals, items, examples)
        store._conn.close()
    print(file=sys.stderr)

    summary = _summarise(items, totals)
    out = Path(a.out)
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "project", "tool", "detected_as", "tokens_before", "tokens_after",
                    "lossy", "recoverable", "file_read_protected"])
        for r in items:
            w.writerow([r.session, r.project, r.tool, r.kind, r.before, r.after, int(r.lossy),
                        int(r.recoverable), int(r.protected)])
    args_echo = (f" --project {a.project}" if a.project else "") + "".join(f' --map "{m}"' for m in a.map)
    meta = {
        "version": __version__,
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tokenizer": o200k.backend(),
        "ranks_sha": o200k.RANKS_SHA256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "csv": csv_path.name,
        "args": args_echo,
    }
    out.write_text(render_html(summary, meta, examples), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")

    sh = summary["shipped"]
    print(f"Sessions: {summary['sessions']}   tool results replayed: {sum(summary['replayed'].values())}")
    print(f"Tool-output tokens: {sh['before']:,} -> {sh['after']:,}  (saved {_pct(sh['before'], sh['after'])})")
    print(f"Whole session:      {summary['session_total_before']:,} -> {summary['session_total_after']:,}"
          f"  (saved {_pct(summary['session_total_before'], summary['session_total_after'])})")
    print(f"Report: {out.resolve()}")
    return 0
