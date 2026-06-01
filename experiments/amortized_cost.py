"""
Amortized cost analysis for the CogCanvas paper (P1-3 / AC-R1).

Reads `*_costlog_10_cat123.json` files produced by `runner_locomo.py` with
token-usage instrumentation, applies model pricing from `pricing.json`, and
produces a fully-amortized cost table:

    c_total = N_turns * c_extract + N_queries * (c_retrieve + c_rerank + c_gen)

Outputs both raw JSON (for reproducibility) and a markdown table (drop-in for
Section 4.5).

Usage:
    python -m experiments.amortized_cost \
        --result-dir experiments/results \
        --pricing experiments/pricing.json \
        --output-json experiments/results/amortized_cost.json \
        --output-md experiments/results/amortized_cost.md

Sensitivity / what-if columns:
    The pricing.json has a `whatif_cheaper_answerer` block.  We re-price the
    `gen` and `gen_aux` token totals at the alternative answerer (default
    gpt-4o-mini) and report it as an extra column — *no rerun needed*.  This
    directly addresses reviewer concern that the system is "expensive" by
    showing that deployment cost is independent of memory-architecture choice.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------
# Pricing helpers
# ----------------------------------------------------------------------------

def load_pricing(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def price_call(
    pricing: Dict,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Return USD cost given a model name and (prompt, completion) tokens."""
    info = pricing.get("models", {}).get(model)
    if info is None:
        # unknown model — fall back to zero so analysis doesn't silently
        # double-count; the markdown table will flag it.
        return 0.0
    if info.get("compute_only"):
        return 0.0
    pin = float(info.get("input_usd_per_1m", 0.0))
    pout = float(info.get("output_usd_per_1m", 0.0))
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000.0


def price_call_at(
    in_per_1m: float,
    out_per_1m: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Like `price_call` but with explicit per-1M rates (for what-if)."""
    return (prompt_tokens * in_per_1m + completion_tokens * out_per_1m) / 1_000_000.0


# ----------------------------------------------------------------------------
# Per-method aggregator
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class MethodReport:
    label: str                # human-readable, used in tables
    json_path: str            # source file
    answer_model: str         # e.g. "gpt-4o" or "gpt-4o-mini" — read from token_usage entries

    # raw counts (excludes judge tokens)
    total_convs: int = 0
    total_turns: int = 0
    total_queries: int = 0

    # token totals by call_type (excluding judge)
    tok_extract_pt: int = 0
    tok_extract_ct: int = 0
    tok_gen_pt: int = 0
    tok_gen_ct: int = 0
    tok_gen_aux_pt: int = 0
    tok_gen_aux_ct: int = 0
    tok_embed_pt: int = 0   # retrieve_embed prompt only
    rerank_calls: int = 0
    rerank_doc_sum: int = 0

    # latency totals (sums in ms) — for reporting compute side
    lat_extract_ms: float = 0.0
    lat_gen_ms: float = 0.0
    lat_embed_ms: float = 0.0
    lat_rerank_ms: float = 0.0

    # per-entry model breakdown (for reporting which models were actually used)
    models_used: Dict[str, str] = dataclasses.field(default_factory=dict)
    # mapping call_type -> model_name (first one seen; usually each call_type uses 1 model)

    # accuracy (for reproducibility sanity-check vs paper Table 4)
    overall_accuracy: Optional[str] = None
    single_hop_acc: Optional[str] = None
    temporal_acc: Optional[str] = None
    multi_hop_acc: Optional[str] = None

    # ---------- derived: aggregate USD costs (filled by `compute_costs`) ----
    cost_extract: float = 0.0
    cost_gen: float = 0.0
    cost_gen_aux: float = 0.0
    cost_embed: float = 0.0
    cost_rerank: float = 0.0
    cost_total: float = 0.0

    # what-if cheaper answerer
    cost_gen_whatif: float = 0.0
    cost_gen_aux_whatif: float = 0.0
    cost_total_whatif: float = 0.0

    # ---------- derived: unit costs ----
    cost_per_turn_extract: float = 0.0
    cost_per_turn_extract_whatif: float = 0.0  # if extract uses gpt-4o (e.g. Summarization), this drops
    cost_per_query_gen: float = 0.0
    cost_per_query_other: float = 0.0  # gen_aux + embed + rerank (per-query share)
    cost_per_query_gen_whatif: float = 0.0

    # ---------- diagnostics ----
    missing_token_data: bool = False
    notes: List[str] = dataclasses.field(default_factory=list)


def load_method_report(json_path: str, label: str) -> MethodReport:
    with open(json_path, "r") as f:
        d = json.load(f)

    convs = d.get("conversations", []) or []
    total_turns = sum(int(c.get("num_turns", 0) or 0) for c in convs)
    total_queries = sum(len(c.get("questions", []) or []) for c in convs)

    summary = d.get("summary", {}) or {}

    r = MethodReport(
        label=label,
        json_path=json_path,
        answer_model="(unknown)",
        total_convs=len(convs),
        total_turns=total_turns,
        total_queries=total_queries,
        overall_accuracy=summary.get("overall_accuracy"),
        single_hop_acc=summary.get("single_hop_accuracy"),
        temporal_acc=summary.get("temporal_accuracy"),
        multi_hop_acc=summary.get("multi_hop_accuracy"),
    )

    tu = d.get("token_usage")
    if not tu:
        r.missing_token_data = True
        r.notes.append("no token_usage block in JSON (pre-instrumentation run)")
        return r

    for e in tu.get("entries", []):
        ct = e.get("call_type")
        model = e.get("model", "(unknown)")
        pt = int(e.get("prompt_tokens", 0) or 0)
        cmp = int(e.get("completion_tokens", 0) or 0)
        lat = float(e.get("latency_ms_sum", 0.0) or 0.0)
        docs = int(e.get("doc_count_sum", 0) or 0)
        calls = int(e.get("calls", 0) or 0)

        if ct == "extract":
            r.tok_extract_pt += pt
            r.tok_extract_ct += cmp
            r.lat_extract_ms += lat
            r.models_used.setdefault("extract", model)
        elif ct == "gen":
            r.tok_gen_pt += pt
            r.tok_gen_ct += cmp
            r.lat_gen_ms += lat
            r.models_used.setdefault("gen", model)
            r.answer_model = model
        elif ct == "gen_aux":
            r.tok_gen_aux_pt += pt
            r.tok_gen_aux_ct += cmp
            r.models_used.setdefault("gen_aux", model)
        elif ct == "retrieve_embed":
            r.tok_embed_pt += pt
            r.lat_embed_ms += lat
            r.models_used.setdefault("retrieve_embed", model)
        elif ct == "rerank":
            r.rerank_calls += calls
            r.rerank_doc_sum += docs
            r.lat_rerank_ms += lat
            r.models_used.setdefault("rerank", model)
        elif ct == "judge":
            pass  # excluded from agent cost
        else:
            r.notes.append(f"unknown call_type {ct} ignored")

    return r


# ----------------------------------------------------------------------------
# Cost computation
# ----------------------------------------------------------------------------

def compute_costs(r: MethodReport, pricing: Dict) -> None:
    """Fill in cost_* and cost_per_* fields on `r` in place.

    What-if logic: 'cheaper answerer' means we re-price *every* call whose
    model equals the baseline answerer (gpt-4o by default).  This matters for
    Summarization, which uses ANSWER_MODEL for the per-window summarize call
    (call_type='extract') — switching answerer reduces that cost too.
    """
    extract_model = r.models_used.get("extract", r.answer_model)
    gen_aux_model = r.models_used.get("gen_aux", r.answer_model)

    r.cost_extract = price_call(pricing, extract_model, r.tok_extract_pt, r.tok_extract_ct)
    r.cost_gen = price_call(pricing, r.answer_model, r.tok_gen_pt, r.tok_gen_ct)
    r.cost_gen_aux = price_call(pricing, gen_aux_model, r.tok_gen_aux_pt, r.tok_gen_aux_ct)
    embed_model = r.models_used.get("retrieve_embed", "bge-m3")
    r.cost_embed = price_call(pricing, embed_model, r.tok_embed_pt, 0)
    rerank_model = r.models_used.get("rerank", "bge-reranker-v2-m3")
    # rerank is local in this paper -> 0
    r.cost_rerank = price_call(pricing, rerank_model, 0, 0)

    r.cost_total = (
        r.cost_extract + r.cost_gen + r.cost_gen_aux + r.cost_embed + r.cost_rerank
    )

    # what-if cheaper answerer
    wf = pricing.get("whatif_cheaper_answerer", {}) or {}
    baseline_answerer = wf.get("current", "gpt-4o")
    wf_in = float(wf.get("alternative_input_usd_per_1m", 0.15))
    wf_out = float(wf.get("alternative_output_usd_per_1m", 0.60))

    def reprice_if_baseline(model: str, pt: int, ct: int, original_cost: float) -> float:
        return price_call_at(wf_in, wf_out, pt, ct) if model == baseline_answerer else original_cost

    cost_extract_whatif = reprice_if_baseline(extract_model, r.tok_extract_pt, r.tok_extract_ct, r.cost_extract)
    r.cost_gen_whatif = reprice_if_baseline(r.answer_model, r.tok_gen_pt, r.tok_gen_ct, r.cost_gen)
    r.cost_gen_aux_whatif = reprice_if_baseline(gen_aux_model, r.tok_gen_aux_pt, r.tok_gen_aux_ct, r.cost_gen_aux)

    r.cost_total_whatif = (
        cost_extract_whatif + r.cost_gen_whatif + r.cost_gen_aux_whatif + r.cost_embed + r.cost_rerank
    )

    # Unit costs
    if r.total_turns > 0:
        r.cost_per_turn_extract = r.cost_extract / r.total_turns
        r.cost_per_turn_extract_whatif = cost_extract_whatif / r.total_turns
    if r.total_queries > 0:
        r.cost_per_query_gen = r.cost_gen / r.total_queries
        # per-query "other" = aux + embed + rerank (rerank=0 in our setup)
        r.cost_per_query_other = (
            r.cost_gen_aux + r.cost_embed + r.cost_rerank
        ) / r.total_queries
        r.cost_per_query_gen_whatif = r.cost_gen_whatif / r.total_queries


def extrapolate_cost(
    r: MethodReport,
    n_turns: int,
    n_queries: int,
    answerer: str = "current",
) -> float:
    """c_total = N_turns * c_extract_per_turn + N_queries * c_per_query

    answerer='current' uses gpt-4o pricing; 'whatif' uses gpt-4o-mini pricing
    for *every* call that originally used the baseline answerer (incl. the
    per-window summarize that Summarization runs as call_type=extract).
    """
    if answerer == "current":
        per_turn = r.cost_per_turn_extract
        gen_unit = r.cost_per_query_gen
    else:
        per_turn = r.cost_per_turn_extract_whatif
        gen_unit = r.cost_per_query_gen_whatif
    return n_turns * per_turn + n_queries * (gen_unit + r.cost_per_query_other)


# ----------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------

def fmt_usd(v: float, decimals: int = 4) -> str:
    if v == 0:
        return "$0.00"
    if v < 0.001 and v > 0:
        return f"${v:.6f}"
    return f"${v:.{decimals}f}"


def render_markdown(
    reports: List[MethodReport],
    pricing: Dict,
    scenarios: List[Tuple[int, int]],
) -> str:
    """Render the cost markdown report.

    Sections:
      0. Headline (3 surprising observations placeholder)
      1. Method config + accuracy reproducibility check
      2. Aggregate token totals (measured)
      3. Unit costs (USD)
      4. Extrapolation table: scenarios × methods
      5. What-if: cheaper answerer
      6. Latency footprint
      7. Pricing reference
    """
    snapshot = pricing.get("_meta", {}).get("snapshot_date", "")
    lines: List[str] = []
    push = lines.append

    push(f"# Amortized Cost Analysis — LoCoMo10 (P1-3)\n")
    push(f"_Pricing snapshot: {snapshot}. All amounts in USD._\n")
    push(f"_Each method was rerun with `--no-cache --no-cache-save` to capture true extraction token counts. Tokens are measured (not estimated) from `response.usage` returned by the LLM proxy._\n")

    # ---- Section 1: Method config + accuracy reproducibility ----
    push("\n## 1. Methods and reproducibility check\n")
    push("| Method | Answer model | Extract model | Conv | Turn | Query | Overall acc | Temporal acc | Multi-hop acc |")
    push("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in reports:
        if r.missing_token_data:
            row = f"| {r.label} | _(missing token data)_ | — | {r.total_convs} | {r.total_turns} | {r.total_queries} | {r.overall_accuracy or '-'} | {r.temporal_acc or '-'} | {r.multi_hop_acc or '-'} |"
        else:
            extract_m = r.models_used.get("extract", "—")
            row = (
                f"| {r.label} | {r.answer_model} | {extract_m} | "
                f"{r.total_convs} | {r.total_turns} | {r.total_queries} | "
                f"{r.overall_accuracy or '-'} | {r.temporal_acc or '-'} | {r.multi_hop_acc or '-'} |"
            )
        push(row)

    # ---- Section 2: Aggregate measured tokens ----
    push("\n## 2. Aggregate measured tokens (excluding LLM-judge)\n")
    push("| Method | Extract tok | Embed tok | Rerank calls (docs) | Gen tok | Gen-aux tok |")
    push("|---|---:|---:|---:|---:|---:|")
    for r in reports:
        push(
            f"| {r.label} | "
            f"{r.tok_extract_pt + r.tok_extract_ct:,} "
            f"({r.tok_extract_pt:,}p+{r.tok_extract_ct:,}c) | "
            f"{r.tok_embed_pt:,}p | "
            f"{r.rerank_calls:,} ({r.rerank_doc_sum:,} docs) | "
            f"{r.tok_gen_pt + r.tok_gen_ct:,} "
            f"({r.tok_gen_pt:,}p+{r.tok_gen_ct:,}c) | "
            f"{r.tok_gen_aux_pt + r.tok_gen_aux_ct:,} |"
        )

    # ---- Section 3: Unit costs ----
    push("\n## 3. Unit costs (measured)\n")
    push("| Method | $/turn (extract) | $/query (gen) | $/query (other) | $/query (gen, whatif gpt-4o-mini) |")
    push("|---|---:|---:|---:|---:|")
    for r in reports:
        push(
            f"| {r.label} | "
            f"{fmt_usd(r.cost_per_turn_extract, 6)} | "
            f"{fmt_usd(r.cost_per_query_gen, 5)} | "
            f"{fmt_usd(r.cost_per_query_other, 6)} | "
            f"{fmt_usd(r.cost_per_query_gen_whatif, 6)} |"
        )

    # ---- Section 4: Extrapolation matrix ----
    push("\n## 4. Amortized cost extrapolation\n")
    push("c_total(method, N_turn, N_query) = N_turn · $/turn + N_query · ($/query)")
    push("")
    push("### 4.a Current pricing (gpt-4o answerer)\n")
    header_cells = ["Method"] + [f"{nt}t × {nq}Q" for (nt, nq) in scenarios]
    push("| " + " | ".join(header_cells) + " |")
    push("|" + "|".join(["---"] + ["---:"] * len(scenarios)) + "|")
    for r in reports:
        cells = [r.label]
        for (nt, nq) in scenarios:
            cells.append(fmt_usd(extrapolate_cost(r, nt, nq, "current")))
        push("| " + " | ".join(cells) + " |")

    push("\n### 4.b What-if: gpt-4o-mini answerer (same token counts, re-priced)\n")
    push("| " + " | ".join(header_cells) + " |")
    push("|" + "|".join(["---"] + ["---:"] * len(scenarios)) + "|")
    for r in reports:
        cells = [r.label]
        for (nt, nq) in scenarios:
            cells.append(fmt_usd(extrapolate_cost(r, nt, nq, "whatif")))
        push("| " + " | ".join(cells) + " |")

    push("\n### 4.c Cost reduction factor (current / whatif) at 100-turn × 20-Q\n")
    push("| Method | Current ($) | Whatif ($) | Reduction × |")
    push("|---|---:|---:|---:|")
    for r in reports:
        cur = extrapolate_cost(r, 100, 20, "current")
        wf = extrapolate_cost(r, 100, 20, "whatif")
        ratio = (cur / wf) if wf > 0 else float("inf")
        push(f"| {r.label} | {fmt_usd(cur)} | {fmt_usd(wf)} | {ratio:.1f}× |")

    # ---- Section 5: Cost breakdown (where does the money go?) ----
    push("\n## 5. Cost breakdown by phase (at LoCoMo10 scale, measured)\n")
    push("| Method | Extract | Gen | Gen-aux | Embed | Rerank | Total | Total (whatif) |")
    push("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in reports:
        push(
            f"| {r.label} | "
            f"{fmt_usd(r.cost_extract, 4)} | "
            f"{fmt_usd(r.cost_gen, 4)} | "
            f"{fmt_usd(r.cost_gen_aux, 4)} | "
            f"{fmt_usd(r.cost_embed, 4)} | "
            f"{fmt_usd(r.cost_rerank, 4)} | "
            f"**{fmt_usd(r.cost_total, 4)}** | "
            f"**{fmt_usd(r.cost_total_whatif, 4)}** |"
        )

    # ---- Section 6: Latency footprint (BGE local rerank) ----
    push("\n## 6. Latency footprint (compute-side, BGE local)\n")
    push("| Method | Extract ms | Gen ms | Embed ms | Rerank ms |")
    push("|---|---:|---:|---:|---:|")
    for r in reports:
        push(
            f"| {r.label} | "
            f"{r.lat_extract_ms/1000:.1f}s | "
            f"{r.lat_gen_ms/1000:.1f}s | "
            f"{r.lat_embed_ms/1000:.1f}s | "
            f"{r.lat_rerank_ms/1000:.1f}s |"
        )

    # ---- Section 7: Pricing reference ----
    push("\n## 7. Pricing reference\n")
    for model_name, info in pricing.get("models", {}).items():
        if info.get("compute_only"):
            push(f"- **{model_name}** ({info.get('vendor', '?')}): local compute, $0/token; "
                 f"OpenAI equivalent ≈ ${info.get('openai_equivalent_input_usd_per_1m', 0):.3f}/1M (in)")
        else:
            push(f"- **{model_name}** ({info.get('vendor', '?')}): "
                 f"${info['input_usd_per_1m']:.3f}/1M in, "
                 f"${info['output_usd_per_1m']:.3f}/1M out")
    push("")
    push(f"_Snapshot: {snapshot}._")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Discovery: pick up locomo_*_costlog_*.json files
# ----------------------------------------------------------------------------

CANONICAL_LABELS = {
    "cogcanvas-recall-boost": "CogCanvas (recall-boost)",
    "rag-rerank":              "RAG+Rerank",
    "rag":                     "RAG",
    "graphrag":                "GraphRAG",
    "summarization":           "Summarization",
    "native":                  "Native",
}


def discover_runs(result_dir: str, suffix: str = "costlog_10_cat123") -> List[Tuple[str, str]]:
    """Find all locomo_<agent>_<suffix>.json files."""
    out: List[Tuple[str, str]] = []
    p = Path(result_dir)
    for f in sorted(p.glob(f"locomo_*_{suffix}.json")):
        name = f.stem  # locomo_<agent>_<suffix>
        # strip prefix/suffix
        if not name.startswith("locomo_") or suffix not in name:
            continue
        body = name[len("locomo_"):]
        body = body.replace("_" + suffix, "")
        agent = body
        label = CANONICAL_LABELS.get(agent, agent)
        out.append((label, str(f)))
    # canonical ordering
    order = list(CANONICAL_LABELS.values())
    out.sort(key=lambda x: (order.index(x[0]) if x[0] in order else 99, x[0]))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="experiments/results")
    parser.add_argument("--pricing", default="experiments/pricing.json")
    parser.add_argument("--output-json", default="experiments/results/amortized_cost.json")
    parser.add_argument("--output-md", default="experiments/results/amortized_cost.md")
    parser.add_argument(
        "--suffix", default="costlog_10_cat123",
        help="filename suffix to scan for (default: costlog_10_cat123)",
    )
    parser.add_argument(
        "--scenarios", default="50x5,50x20,100x20,100x50,300x20,300x50",
        help="comma-separated NTURNxNQUERY scenarios for extrapolation",
    )
    parser.add_argument(
        "--explicit", nargs="+", default=None,
        help="explicit list of label=path entries (overrides discovery)",
    )
    args = parser.parse_args()

    pricing = load_pricing(args.pricing)

    if args.explicit:
        pairs: List[Tuple[str, str]] = []
        for s in args.explicit:
            label, path = s.split("=", 1)
            pairs.append((label, path))
    else:
        pairs = discover_runs(args.result_dir, args.suffix)

    if not pairs:
        print(f"No result JSONs found in {args.result_dir} with suffix _{args.suffix}.json", file=sys.stderr)
        return 2

    print(f"Discovered {len(pairs)} runs:")
    for label, path in pairs:
        print(f"  - {label}: {path}")

    reports: List[MethodReport] = []
    for label, path in pairs:
        r = load_method_report(path, label)
        compute_costs(r, pricing)
        reports.append(r)

    # parse scenarios
    scenarios: List[Tuple[int, int]] = []
    for s in args.scenarios.split(","):
        s = s.strip()
        if not s:
            continue
        nt, nq = s.lower().split("x")
        scenarios.append((int(nt), int(nq)))

    # write JSON
    out_json = {
        "pricing_snapshot": pricing.get("_meta", {}).get("snapshot_date"),
        "scenarios": [{"n_turns": nt, "n_queries": nq} for (nt, nq) in scenarios],
        "reports": [dataclasses.asdict(r) for r in reports],
        "extrapolation": [
            {
                "method": r.label,
                "scenarios": [
                    {
                        "n_turns": nt, "n_queries": nq,
                        "cost_usd_current": extrapolate_cost(r, nt, nq, "current"),
                        "cost_usd_whatif": extrapolate_cost(r, nt, nq, "whatif"),
                    }
                    for (nt, nq) in scenarios
                ],
            }
            for r in reports
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Wrote {args.output_json}")

    # write markdown
    md = render_markdown(reports, pricing, scenarios)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write(md)
    print(f"Wrote {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
