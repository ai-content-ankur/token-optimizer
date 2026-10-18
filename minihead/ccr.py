"""CCR — reversible compression store.

Whenever a compressor drops content it saves the original here under a
short hash and leaves a marker such as ``<<ccr:3f9a1c2b0d4e 120 rows>>`` in
the compressed text. The model (or the user) can then call
``minihead_retrieve`` with that hash to get the full original back.

The store is a single SQLite file, so it is safe to share between the
proxy, the MCP server and the library in separate processes.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_TTL_SECONDS = 24 * 3600
MARKER_RE = re.compile(r"<<ccr:([0-9a-f]{12})(?:[ ,][^>]*)?>>")


def default_home() -> Path:
    env = os.environ.get("MINIHEAD_HOME")
    return Path(env) if env else Path.home() / ".minihead"


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def marker(key: str, note: str) -> str:
    return f"<<ccr:{key} {note}>>"


class Store:
    """SQLite-backed key/value store for originals plus a savings ledger."""

    def __init__(self, path: str | os.PathLike | None = None, ttl: int = DEFAULT_TTL_SECONDS):
        if path is None:
            home = default_home()
            home.mkdir(parents=True, exist_ok=True)
            path = home / "store.db"
        self.path = str(path)
        self.ttl = ttl
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS originals ("
                " key TEXT PRIMARY KEY, kind TEXT, content TEXT, created REAL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS ledger ("
                " ts REAL, kind TEXT, tokens_before INTEGER, tokens_after INTEGER)"
            )
            # memo of input-hash -> compressed output, so a block that was sent
            # compressed once is sent byte-identically on every later turn
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, output TEXT, created REAL)"
            )

    # ── originals ──────────────────────────────────────────────────────
    def put(self, content: str, kind: str = "text") -> str:
        key = short_hash(content)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO originals(key, kind, content, created) VALUES (?,?,?,?)",
                (key, kind, content, time.time()),
            )
        return key

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content, created FROM originals WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        content, created = row
        if self.ttl and time.time() - created > self.ttl:
            return None
        return content

    def purge_expired(self) -> int:
        cutoff = time.time() - self.ttl
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM originals WHERE created < ?", (cutoff,))
        return cur.rowcount

    # ── compression memo (prefix-cache stability) ──────────────────────
    def memo_get(self, text: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT output FROM memo WHERE key = ?", (short_hash(text) + str(len(text)),)
            ).fetchone()
        return row[0] if row else None

    def memo_put(self, text: str, output: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memo(key, output, created) VALUES (?,?,?)",
                (short_hash(text) + str(len(text)), output, time.time()),
            )

    # ── savings ledger ─────────────────────────────────────────────────
    def record(self, kind: str, before: int, after: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO ledger(ts, kind, tokens_before, tokens_after) VALUES (?,?,?,?)",
                (time.time(), kind, before, after),
            )

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*), SUM(tokens_before), SUM(tokens_after)"
                " FROM ledger GROUP BY kind ORDER BY kind"
            ).fetchall()
            stored = self._conn.execute("SELECT COUNT(*) FROM originals").fetchone()[0]
        by_kind = {
            k: {"calls": n, "tokens_before": b or 0, "tokens_after": a or 0} for k, n, b, a in rows
        }
        before = sum(v["tokens_before"] for v in by_kind.values())
        after = sum(v["tokens_after"] for v in by_kind.values())
        return {
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": before - after,
            "ratio": round(1 - after / before, 3) if before else 0.0,
            "originals_stored": stored,
            "by_kind": by_kind,
        }

    def resolve_markers(self, text: str) -> str:
        """Replace every marker in ``text`` with the stored original (if still present)."""

        def sub(m: re.Match) -> str:
            original = self.get(m.group(1))
            return original if original is not None else m.group(0)

        return MARKER_RE.sub(sub, text)
