#!/usr/bin/env python3
"""
W5 mechanism probe: store-level gold-answer survival analysis.

For questions the chunk pipeline answers correctly but the artifact pipeline
fails (the discordant n10 set), decompose the artifact failure into:

  - write_time_loss:  gold-answer content tokens appear verbatim in the
                      conversation but NOT in the artifact store -> the
                      extractor discarded the information before any query
                      arrived. Unrecoverable at read time.
  - query_time_miss:  gold-answer content tokens are fully present in the
                      artifact store, yet the pipeline still answered wrong
                      -> retrieval/generation failure, not a storage failure.
  - inferential_gold: gold-answer tokens are not fully present even in the
                      raw conversation text -> the gold is a paraphrase or
                      inference; token coverage cannot diagnose these, so
                      they are reported separately rather than force-binned.
  - trivial_gold:     gold answer has no content tokens after stopword
                      filtering (e.g. "Yes") -> not diagnosable.

Store text = concatenation of every canvas object's content + quote + context
fields (everything retrieval could possibly surface). Zero API cost: reads
cached extraction canvases.

Usage:
    python -m experiments.store_survival \
        --chunks experiments/results/locomo_chunks_C_nograph_10_cat123.json \
        --artifacts experiments/results/locomo_full_budgetmatch_k60_tok2000.json \
        --cache-dir experiments/cache/extraction/d2b44bf2 \
        -o experiments/results/store_survival_w5.json
"""

import argparse
import json
import re
from pathlib import Path

from experiments.locomo_adapter import load_locomo, convert_to_eval_format

STOPWORDS = frozenset("""
a an the and or but if then else of in on at to for from by with about as is
are was were be been being am do does did doing have has had having will would
can could should shall may might must not no nor it its it's this that these
those he she they them his her their there here when where what which who whom
why how all any both each few more most other some such only own same so than
too very s t don now i you we us our your my me him
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set:
    """Lowercased alphanumeric content tokens, stopword-filtered."""
    return {t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS}


def load_passed(path: str) -> dict:
    """(conv_id, question) -> (passed, category, ground_truth)."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for conv in data["conversations"]:
        for q in conv["questions"]:
            out[(conv["id"], q["question"])] = (
                q["passed"], q.get("category"), q.get("ground_truth"),
            )
    return out


def store_token_sets(cache_file: Path) -> list:
    """Per-object content token sets (one set per canvas object).

    The unit matters: a date like "7 May 2023" has tokens that each appear
    somewhere in almost any store, so pooled-store coverage is trivially 1.0.
    A fact only survives as a retrievable unit if a SINGLE object carries it.
    """
    with open(cache_file) as f:
        data = json.load(f)
    sets = []
    for obj in data.get("objects", []):
        parts = []
        for field in ("content", "quote", "context"):
            v = obj.get(field)
            if v:
                parts.append(str(v))
        if parts:
            # Temporal extraction embeds a raw conversation window in its
            # content ("TIME: ... [Turn k (Session: ...)] ..."). Facts that
            # survive ONLY inside such windows are accidental verbatim
            # residue, not deliberately extracted artifacts — tag the unit.
            is_window = "[Turn" in (obj.get("content") or "")
            sets.append((tokens(" ".join(parts)), is_window))
    return sets


def conversation_token_sets(conv) -> list:
    """Per-turn content token sets (turn text + its session datetime)."""
    sets = []
    for t in conv.turns:
        parts = []
        for field in ("user", "assistant", "session_datetime"):
            v = getattr(t, field, None)
            if v:
                parts.append(str(v))
        if parts:
            sets.append(tokens(" ".join(parts)))
    return sets


def max_unit_coverage(gold_tokens: set, unit_sets: list) -> float:
    """Best single-unit coverage: max over units of |gold ∩ unit| / |gold|."""
    if not gold_tokens:
        return float("nan")
    if not unit_sets:
        return 0.0
    return max(len(gold_tokens & u) / len(gold_tokens) for u in unit_sets)


def coverage(gold_tokens: set, store: set) -> float:
    if not gold_tokens:
        return float("nan")
    return len(gold_tokens & store) / len(gold_tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--chunks", required=True, help="chunk-pipeline result JSON")
    ap.add_argument("--artifacts", required=True, help="artifact-pipeline result JSON")
    ap.add_argument("--cache-dir", required=True,
                    help="extraction cache dir holding the artifact canvases")
    ap.add_argument("--dataset", default="experiments/data/locomo10.json")
    ap.add_argument("-o", "--output", default=None, help="detail JSON output path")
    args = ap.parse_args()

    chunks = load_passed(args.chunks)
    arts = load_passed(args.artifacts)
    common = sorted(set(chunks) & set(arts))
    n10 = [k for k in common if chunks[k][0] and not arts[k][0]]
    n01 = [k for k in common if not chunks[k][0] and arts[k][0]]
    print(f"Matched questions: {len(common)} | chunk-pass/artifact-fail (n10): "
          f"{len(n10)} | artifact-pass/chunk-fail (n01): {len(n01)}")

    conv_units = {
        c.id: conversation_token_sets(c)
        for c in convert_to_eval_format(load_locomo(args.dataset))
    }

    cache_dir = Path(args.cache_dir)
    store_cache = {}
    details = []
    buckets = {"write_time_loss": 0, "query_time_miss": 0,
               "inferential_gold": 0, "trivial_gold": 0}

    for conv_id, question in n10:
        _, category, gold = arts[(conv_id, question)]
        gold_toks = tokens(gold)
        if conv_id not in store_cache:
            store_cache[conv_id] = store_token_sets(cache_dir / f"{conv_id}.json")
        store_sets = store_cache[conv_id]

        # Primary criterion: single-unit co-occurrence. Pooled coverage kept
        # as the lenient sensitivity bound.
        all_sets = [s for s, _ in store_sets]
        nonwindow_sets = [s for s, is_win in store_sets if not is_win]
        cov_store = max_unit_coverage(gold_toks, all_sets)
        cov_nonwindow = max_unit_coverage(gold_toks, nonwindow_sets)
        cov_conv = max_unit_coverage(gold_toks, conv_units[conv_id])
        pooled_store = coverage(gold_toks, set().union(*all_sets) if all_sets else set())

        if not gold_toks:
            bucket = "trivial_gold"
        elif cov_conv < 1.0:
            bucket = "inferential_gold"
        elif cov_store >= 1.0:
            bucket = "query_time_miss"
        else:
            bucket = "write_time_loss"
        buckets[bucket] += 1
        details.append({
            "conversation_id": conv_id, "question": question,
            "ground_truth": gold, "category": category,
            "store_coverage": None if cov_store != cov_store else round(cov_store, 3),
            "store_coverage_nonwindow": None if cov_nonwindow != cov_nonwindow else round(cov_nonwindow, 3),
            "store_coverage_pooled": None if pooled_store != pooled_store else round(pooled_store, 3),
            "conversation_coverage": None if cov_conv != cov_conv else round(cov_conv, 3),
            "bucket": bucket,
        })

    n = len(n10)
    diagnosable = buckets["write_time_loss"] + buckets["query_time_miss"]
    print(f"\n=== Survival decomposition of {n} chunk-pass/artifact-fail questions ===")
    for b, c in buckets.items():
        print(f"  {b:18s}: {c:4d}  ({c/n:6.1%} of n10)")
    if diagnosable:
        wt = buckets["write_time_loss"]
        print(f"\n  Among the {diagnosable} diagnosable (gold verbatim in conversation):")
        print(f"    write-time loss : {wt}/{diagnosable} = {wt/diagnosable:.1%}")
        print(f"    query-time miss : {diagnosable-wt}/{diagnosable} = "
              f"{(diagnosable-wt)/diagnosable:.1%}")

    # Query-time-miss sub-split: does the fact survive in a deliberately
    # extracted artifact, or only as accidental verbatim residue inside a
    # temporal context window?
    qtm = [d for d in details if d["bucket"] == "query_time_miss"]
    if qtm:
        residue_only = sum(1 for d in qtm if d["store_coverage_nonwindow"] < 1.0)
        print(f"\n  Query-time-miss sub-split ({len(qtm)} questions):")
        print(f"    gold survives ONLY inside temporal window snippets "
              f"(accidental verbatim residue): {residue_only}/{len(qtm)} = "
              f"{residue_only/len(qtm):.1%}")
        print(f"    gold survives in a deliberate (non-window) artifact: "
              f"{len(qtm)-residue_only}/{len(qtm)} = "
              f"{(len(qtm)-residue_only)/len(qtm):.1%}")

    # Inferential sub-split: did rewriting degrade the raw material below the
    # single-turn ceiling? (chunks preserve the ceiling by construction)
    inf = [d for d in details if d["bucket"] == "inferential_gold"]
    if inf:
        degraded = sum(1 for d in inf if d["store_coverage"] < d["conversation_coverage"])
        print(f"\n  Inferential-gold sub-split ({len(inf)} questions, gold never "
              f"co-occurs verbatim in one turn):")
        print(f"    artifact store degraded below single-turn ceiling: "
              f"{degraded}/{len(inf)} = {degraded/len(inf):.1%}")
        print(f"    store preserves the ceiling (failure is query-time/reasoning): "
              f"{len(inf)-degraded}/{len(inf)} = {(len(inf)-degraded)/len(inf):.1%}")

    # Per-category breakdown
    cats = {}
    for d in details:
        cats.setdefault(d["category"], {}).setdefault(d["bucket"], 0)
        cats[d["category"]][d["bucket"]] += 1
    print("\n  Per-category buckets:", json.dumps(cats, indent=2, sort_keys=True))

    # Sensitivity bounds
    diag = [d for d in details if d["bucket"] in ("write_time_loss", "query_time_miss")]
    if diag:
        lenient_wt = sum(1 for d in diag if d["store_coverage_pooled"] < 1.0)
        print(f"\n  Sensitivity (lenient, tokens pooled across whole store):")
        print(f"    write-time loss lower bound: {lenient_wt}/{len(diag)} = "
              f"{lenient_wt/len(diag):.1%}")
        for thr in (0.5, 0.8):
            wt = sum(1 for d in diag if d["store_coverage"] < thr)
            print(f"  Sensitivity: best single-object coverage < {thr:.0%}: {wt}/{len(diag)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "chunks_file": args.chunks, "artifacts_file": args.artifacts,
                "cache_dir": str(cache_dir), "n_common": len(common),
                "n10": n, "n01": len(n01), "buckets": buckets,
                "diagnosable": diagnosable, "details": details,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetail written to {args.output}")


if __name__ == "__main__":
    main()
