"""Built-in benchmark: seeded, offline payloads shaped like real agent tool output.

Each scenario also checks that the "needle" — the one detail an agent
would need to answer correctly — survives compression.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from .ccr import Store
from .pipeline import compress_text
from .tokens import tokenizer_name

SEED = 20261018


def _search(rng: random.Random) -> tuple[str, str]:
    files = [f"src/{d}/{n}.py" for d in ("api", "core", "db", "auth", "utils") for n in ("models", "views", "service", "helpers")]
    lines = []
    for _ in range(100):
        f = rng.choice(files)
        lines.append(f"{f}:{rng.randint(1, 900)}:    result = fetch_user(session, user_id={rng.randint(1, 9999)})")
    lines.insert(37, "src/auth/service.py:412:    raise TokenExpired('refresh token reused')  # needle")
    return "\n".join(lines), "TokenExpired"


def _logs(rng: random.Random) -> tuple[str, str]:
    out = []
    t = 0
    for i in range(1500):
        t += rng.randint(1, 900)
        ts = f"2026-10-18T10:{(t // 60000) % 60:02d}:{(t // 1000) % 60:02d}.{t % 1000:03d}Z"
        r = rng.random()
        if i == 1067:
            out.append(f"{ts} FATAL db.pool: connection pool exhausted (max=50, waiting=213)")
            out.append("Traceback (most recent call last):")
            out.append('  File "/app/db/pool.py", line 88, in acquire')
            out.append("    conn = self._queue.get(timeout=self.timeout)")
            out.append("queue.Empty: pool exhausted")
        elif r < 0.03:
            out.append(f"{ts} WARN http.client: slow upstream response latency_ms={rng.randint(800, 4000)} host=payments")
        elif r < 0.5:
            out.append(f"{ts} INFO http.access: GET /api/v1/orders/{rng.randint(1000, 99999)} 200 {rng.randint(3, 90)}ms")
        elif r < 0.8:
            out.append(f"{ts} DEBUG cache: hit key=order:{rng.randint(1000, 99999)} ttl={rng.randint(10, 600)}")
        else:
            out.append(f"{ts} INFO worker: processed job id={rng.getrandbits(48):012x} in {rng.randint(5, 300)}ms")
    return "\n".join(out), "connection pool exhausted"


def _json_api(rng: random.Random) -> tuple[str, str]:
    labels = ["bug", "enhancement", "docs", "question"]
    items = []
    for i in range(250):
        items.append({
            "number": 4000 + i,
            "title": f"Issue about component {rng.choice(['router', 'cache', 'proxy', 'cli'])} #{i}",
            "state": "open" if rng.random() < 0.7 else "closed",
            "label": rng.choice(labels),
            "comments": rng.randint(0, 12),
            "author": f"user{rng.randint(1, 80)}",
            "created_at": f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T12:00:00Z",
            "ci": "passing",
        })
    items[173]["ci"] = "failing"  # the needle: one failing CI among 250
    items[173]["title"] = "Proxy drops SSE frames under load"
    return json.dumps({"total": len(items), "items": items}, indent=2), "Proxy drops SSE frames"


def _code(rng: random.Random) -> tuple[str, str]:
    parts = ['"""Order service."""', "import json", "import logging", "from dataclasses import dataclass", ""]
    for i in range(30):
        parts.append(f"def handler_{i}(request, ctx):")
        parts.append(f'    """Handle request type {i}."""')
        for j in range(rng.randint(14, 30)):
            parts.append(f"    value_{j} = ctx.lookup('{i}-{j}') or request.get('k{j}', {j})")
        parts.append("    return value_0")
        parts.append("")
    parts.append("def apply_discount(order, code):  # needle")
    parts.append("    return order.total * 0.9 if code == 'SAVE10' else order.total")
    return "\n".join(parts), "def apply_discount"


def _prose(rng: random.Random) -> tuple[str, str]:
    words = "the system routes requests through a local proxy that compresses tool output before it reaches the model".split()
    paras = [" ".join(rng.choice(words) for _ in range(rng.randint(40, 90))).capitalize() + "." for _ in range(25)]
    paras.insert(5, "The rollback window is exactly 72 hours.")
    return "\n\n".join(paras), "72 hours"


SCENARIOS = [
    ("Code search (100 hits)", _search),
    ("SRE incident logs", _logs),
    ("JSON API listing (250)", _json_api),
    ("Large Python file", _code),
    ("Prose", _prose),
]


def run() -> list[dict]:
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "bench.db")
        rows = []
        for name, make in SCENARIOS:
            text, needle = make(rng)
            b = compress_text(text, store, record=False)
            kept = needle in b.text
            recoverable = kept or needle in store.resolve_markers(b.text)
            rows.append({
                "scenario": name, "kind": b.kind, "before": b.tokens_before, "after": b.tokens_after,
                "saved": 1 - b.tokens_after / b.tokens_before, "needle": "kept" if kept else ("via CCR" if recoverable else "LOST"),
            })
        store._conn.close()
    print(f"tokenizer: {tokenizer_name()}   seed: {SEED}\n")
    print(f"{'Scenario':26} {'Kind':8} {'Before':>8} {'After':>8} {'Saved':>7}  Needle")
    for r in rows:
        print(f"{r['scenario']:26} {r['kind']:8} {r['before']:>8,} {r['after']:>8,} {r['saved']:>6.0%}  {r['needle']}")
    tb = sum(r["before"] for r in rows)
    ta = sum(r["after"] for r in rows)
    print(f"{'Total':26} {'':8} {tb:>8,} {ta:>8,} {1 - ta / tb:>6.0%}")
    return rows
