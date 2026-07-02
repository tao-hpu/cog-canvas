"""Shared fixed-context-budget retrieval helper for the fidelity-curve anchors.

The fidelity curve must hold the *answerer's context budget* constant across
representations, NOT top_k -- otherwise a representation whose stored items are
large (e.g. SeCom's multi-turn segments) is fed far more text than one whose
items are tiny (e.g. Mem0's atomic facts) at the same top_k, confounding
representation fidelity with retrieval budget.

When CONTEXT_BUDGET_CHARS is set, anchors retrieve a wide candidate pool and
then greedily fill the context up to that character budget instead of taking a
fixed top_k. Unset -> legacy top_k behaviour (preserves earlier results).
"""

import os


def context_budget_chars():
    v = os.getenv("CONTEXT_BUDGET_CHARS", "0")
    try:
        return int(v) or None
    except ValueError:
        return None


# When a budget is active, fetch this many ranked candidates before filling.
WIDE_K = 80


def effective_k(top_k, budget):
    return WIDE_K if budget else top_k


def retrieve_topk(default=10):
    """Final number of items fed to the answerer. Paper backbone uses top-15;
    set RETRIEVE_TOPK=15 to match it. Unset -> per-agent default."""
    v = os.getenv("RETRIEVE_TOPK", "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def fill_to_budget(items, get_text, budget):
    """Return the best-first prefix of `items` whose rendered text fits `budget`
    characters (always keep at least one item)."""
    if not budget:
        return items
    out, total = [], 0
    for it in items:
        t = get_text(it)
        if out and total + len(t) > budget:
            break
        out.append(it)
        total += len(t)
    return out
