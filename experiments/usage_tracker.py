"""
Thread-safe token usage accumulator for amortized cost analysis.

Records every LLM / embedding / reranker call with token counts + latency,
keyed by (call_type, model). Used during a single agent run on a benchmark;
runner_locomo.py is expected to reset() before each agent and snapshot() after.

Call type taxonomy (passed by callsite, NEVER inferred):
  - "extract"        : per-turn LLM extraction (CogCanvas extract+gleaning, GraphRAG entity/relation, Summarization compress)
  - "gen"            : per-query answer generation (final answer LLM call)
  - "gen_aux"        : per-query auxiliary gen (e.g. CogCanvas query complexity classifier)
  - "retrieve_embed" : embedding API call (query embedding OR per-turn corpus embedding)
  - "rerank"         : reranker API call (no tokens; just call count + doc count + latency)
  - "judge"          : LLM-judge scoring call — EXCLUDED from agent cost at report time

If a callsite cannot determine its call_type from static context, it should pass
"unknown" explicitly. Never silently default.
"""

import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any


# ---------- module-level state (thread-safe via Lock) ----------

_LOCK = threading.Lock()

# key = (call_type, model) -> aggregate dict
_USAGE: Dict[tuple, Dict[str, float]] = {}

# enabled flag — if False, all track_* are no-ops (for production/non-cost runs)
_ENABLED = True

# thread-local override for call_type (used by the `as_call_type` context manager)
_LOCAL = threading.local()


def _empty_entry() -> Dict[str, float]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms_sum": 0.0,
        # rerank-only fields (kept here for uniform schema; 0 for non-rerank entries)
        "doc_count_sum": 0,
        "query_chars_sum": 0,
    }


# ---------- public API ----------

def enable(flag: bool = True) -> None:
    """Globally enable / disable tracking. Defaults to enabled."""
    global _ENABLED
    _ENABLED = flag


def reset() -> None:
    """Clear all accumulated state. Call once before each agent run."""
    with _LOCK:
        _USAGE.clear()


def track_llm(
    call_type: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float = 0.0,
) -> None:
    """Record a chat-completion call.

    call_type is taken from the explicit arg, OR from a `with as_call_type(...)`
    context manager if the explicit arg is None.
    """
    if not _ENABLED:
        return
    effective_type = call_type or getattr(_LOCAL, "call_type", None) or "unknown"
    key = (effective_type, model)
    with _LOCK:
        e = _USAGE.setdefault(key, _empty_entry())
        e["calls"] += 1
        e["prompt_tokens"] += int(prompt_tokens or 0)
        e["completion_tokens"] += int(completion_tokens or 0)
        e["total_tokens"] += int(prompt_tokens or 0) + int(completion_tokens or 0)
        e["latency_ms_sum"] += float(latency_ms or 0.0)


def track_embed(
    model: str,
    prompt_tokens: int,
    num_inputs: int,
    latency_ms: float = 0.0,
    call_type: str = "retrieve_embed",
) -> None:
    """Record an embedding-API call.

    For embedding APIs, prompt_tokens is the only meaningful token count
    (no completion). num_inputs = batch size (texts in this request).
    """
    if not _ENABLED:
        return
    key = (call_type, model)
    with _LOCK:
        e = _USAGE.setdefault(key, _empty_entry())
        e["calls"] += 1
        e["prompt_tokens"] += int(prompt_tokens or 0)
        e["total_tokens"] += int(prompt_tokens or 0)
        e["doc_count_sum"] += int(num_inputs or 0)
        e["latency_ms_sum"] += float(latency_ms or 0.0)


def track_rerank(
    model: str,
    num_docs: int,
    query_chars: int,
    latency_ms: float = 0.0,
    call_type: str = "rerank",
) -> None:
    """Record a reranker call. No token cost (BGE local), only call count +
    document count + latency for compute-side reporting."""
    if not _ENABLED:
        return
    key = (call_type, model)
    with _LOCK:
        e = _USAGE.setdefault(key, _empty_entry())
        e["calls"] += 1
        e["doc_count_sum"] += int(num_docs or 0)
        e["query_chars_sum"] += int(query_chars or 0)
        e["latency_ms_sum"] += float(latency_ms or 0.0)


@contextmanager
def as_call_type(call_type: str):
    """Context manager to set call_type for all track_llm() calls in this thread
    that pass call_type=None. Lets callers tag a block (e.g. 'judge') without
    threading the arg through every helper."""
    prev = getattr(_LOCAL, "call_type", None)
    _LOCAL.call_type = call_type
    try:
        yield
    finally:
        _LOCAL.call_type = prev


def snapshot() -> Dict[str, Any]:
    """Return a JSON-serializable copy of the current usage state.

    Schema:
    {
      "entries": [
        {"call_type": ..., "model": ..., "calls": N, "prompt_tokens": ...,
         "completion_tokens": ..., "total_tokens": ...,
         "latency_ms_sum": ..., "doc_count_sum": ..., "query_chars_sum": ...},
        ...
      ],
      "totals": {
         "by_call_type": {ct: {"calls", "prompt_tokens", "completion_tokens",
                               "total_tokens", "latency_ms_sum"} for ct in cts}
      }
    }
    """
    with _LOCK:
        entries: List[Dict[str, Any]] = []
        by_ct: Dict[str, Dict[str, float]] = {}
        for (ct, model), v in _USAGE.items():
            entries.append({
                "call_type": ct,
                "model": model,
                **v,
            })
            acc = by_ct.setdefault(ct, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "latency_ms_sum": 0.0,
                "doc_count_sum": 0, "query_chars_sum": 0,
            })
            for k in ("calls", "prompt_tokens", "completion_tokens",
                     "total_tokens", "doc_count_sum", "query_chars_sum"):
                acc[k] += v[k]
            acc["latency_ms_sum"] += v["latency_ms_sum"]

        return {
            "entries": sorted(entries, key=lambda e: (e["call_type"], e["model"])),
            "totals": {"by_call_type": by_ct},
        }


def get_state() -> Dict[tuple, Dict[str, float]]:
    """Direct access to internal dict (read-only intent). Used by tests."""
    with _LOCK:
        return {k: dict(v) for k, v in _USAGE.items()}


# ---------- helpers for callsites ----------

def safe_usage_from_openai_response(response: Any) -> Dict[str, int]:
    """Extract {prompt_tokens, completion_tokens} from an OpenAI-style response.
    Returns zeros if usage is missing (some proxy servers omit it)."""
    pt, ct = 0, 0
    try:
        u = getattr(response, "usage", None)
        if u is None and isinstance(response, dict):
            u = response.get("usage")
        if u is not None:
            if isinstance(u, dict):
                pt = int(u.get("prompt_tokens", 0) or 0)
                ct = int(u.get("completion_tokens", 0) or 0)
            else:
                pt = int(getattr(u, "prompt_tokens", 0) or 0)
                ct = int(getattr(u, "completion_tokens", 0) or 0)
    except Exception:
        pass
    return {"prompt_tokens": pt, "completion_tokens": ct}


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Tiktoken fallback for when API response omits usage.
    Best-effort; uses cl100k_base for unknown models."""
    if not text:
        return 0
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # last-resort char-based estimate
        return max(1, len(text) // 4)
