"""JSON crusher.

Large arrays dominate tool output (search hits, API listings, metrics).
For each long array we keep:

* the first and last few items (shape + boundaries),
* items that are statistical outliers — a categorical field holding a rare
  value (``"status": "error"`` among 300 ``"ok"``), or a numeric field more
  than 3 standard deviations from the mean,
* and then add a sentinel item carrying a CCR marker plus a per-field
  summary of what was dropped, so counts and ranges stay answerable.

Rows of uniform dicts are re-encoded column-wise (``_cols`` / ``_rows``),
which removes the repeated keys — usually the single biggest saving.
Everything is emitted as minified JSON.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from ..ccr import Store, marker

HEAD = 3
TAIL = 2
MAX_ITEMS = 15  # arrays at or below this length are left alone
MAX_OUTLIERS = 20
MAX_STRING = 1500
RARE_SHARE = 0.05


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _field_outliers(items: list[dict]) -> set[int]:
    """Indices of rows that carry a rare categorical value or a numeric outlier."""
    keys = Counter(k for it in items for k in it)
    n = len(items)
    flagged: set[int] = set()
    for key, present in keys.items():
        values = [it.get(key) for it in items]
        nums = [v for v in values if _is_number(v)]
        if len(nums) >= max(8, int(0.8 * present)):
            mean = sum(nums) / len(nums)
            var = sum((x - mean) ** 2 for x in nums) / len(nums)
            sd = math.sqrt(var)
            if sd > 0:
                for i, v in enumerate(values):
                    if _is_number(v) and abs(v - mean) / sd > 3:
                        flagged.add(i)
            continue
        # categorical: short scalars only (ids/free text are all unique)
        cats = [v if isinstance(v, (str, bool)) or v is None else None for v in values]
        hashable = [c for c in cats if c is None or (isinstance(c, str) and len(c) <= 40) or isinstance(c, bool)]
        if len(hashable) < 0.8 * n:
            continue
        freq = Counter(cats)
        if len(freq) < 2 or len(freq) > max(6, n // 10):
            continue  # constant field, or too many distinct values to be categorical
        dominant = freq.most_common(1)[0][1]
        if dominant / n < 0.5:
            continue
        for i, c in enumerate(cats):
            if freq[c] / n <= RARE_SHARE:
                flagged.add(i)
    # rows missing a field most others have, or carrying an extra one
    for i, it in enumerate(items):
        for key, present in keys.items():
            if present / n >= 0.95 and key not in it:
                flagged.add(i)
            elif present / n <= RARE_SHARE and key in it:
                flagged.add(i)
    return flagged


def _summary(dropped: list[Any]) -> dict:
    """Small per-field digest of dropped dict rows."""
    out: dict[str, Any] = {}
    dicts = [d for d in dropped if isinstance(d, dict)]
    if not dicts:
        return out
    keys = Counter(k for d in dicts for k in d)
    for key, _ in keys.most_common(12):
        vals = [d.get(key) for d in dicts if key in d]
        nums = [v for v in vals if _is_number(v)]
        if nums and len(nums) == len(vals):
            out[key] = {"min": min(nums), "max": max(nums), "sum": round(sum(nums), 4)}
            continue
        strs = [v for v in vals if isinstance(v, (str, bool)) or v is None]
        freq = Counter(json.dumps(v) if not isinstance(v, str) else v for v in strs)
        if 1 < len(freq) <= 8:
            out[key] = dict(freq.most_common())
        elif len(freq) == 1:
            out[key] = {"all": next(iter(freq))}
    return out


def _columnar(rows: list[Any]) -> Any:
    """Re-encode a list of same-keyed dicts as {_cols, _rows}; otherwise return as-is."""
    if len(rows) < 4 or not all(isinstance(r, dict) for r in rows):
        return rows
    cols = list(rows[0].keys())
    if not cols or any(list(r.keys()) != cols for r in rows if "_ccr_dropped" not in r):
        return rows
    body = [r for r in rows if "_ccr_dropped" not in r]
    extra = [r for r in rows if "_ccr_dropped" in r]
    encoded: dict[str, Any] = {"_cols": cols, "_rows": [[r[c] for c in cols] for r in body]}
    if extra:
        encoded.update(extra[0])
    return encoded


class JSONCrusher:
    def __init__(self, store: Store, columnar: bool = True):
        self.store = store
        self.columnar = columnar
        self.dropped_items = 0

    def crush_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.crush_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return self._crush_list(value)
        if isinstance(value, str) and len(value) > MAX_STRING:
            key = self.store.put(value, "json-string")
            keep = MAX_STRING // 2
            return value[:keep] + " " + marker(key, f"{len(value) - keep} more chars") + " " + value[-200:]
        return value

    def _crush_list(self, items: list) -> Any:
        items = [self.crush_value(v) for v in items]
        n = len(items)
        if n <= MAX_ITEMS:
            return _columnar(items) if self.columnar else items

        keep = set(range(min(HEAD, n))) | set(range(max(0, n - TAIL), n))
        if all(isinstance(it, dict) for it in items):
            outliers = sorted(_field_outliers(items) - keep)
            keep |= set(outliers[:MAX_OUTLIERS])
        kept = [items[i] for i in sorted(keep)]
        dropped = [items[i] for i in range(n) if i not in keep]
        if not dropped:
            return _columnar(kept) if self.columnar else kept

        self.dropped_items += len(dropped)
        key = self.store.put(_dumps(items), "json-array")
        sentinel: dict[str, Any] = {
            "_ccr_dropped": marker(key, f"{len(dropped)} of {n} items omitted"),
        }
        summ = _summary(dropped)
        if summ:
            sentinel["_dropped_summary"] = summ
        if self.columnar and all(isinstance(r, dict) for r in kept):
            enc = _columnar(kept)
            if isinstance(enc, dict):
                enc.update(sentinel)
                return enc
        return kept + [sentinel]


def crush_json(text: str, data: Any, store: Store) -> str:
    crusher = JSONCrusher(store)
    out = _dumps(crusher.crush_value(data))
    return out
