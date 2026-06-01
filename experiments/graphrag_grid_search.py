"""GraphRAG grid search driver (P1-2, ARR Aug 2026 §4 + Appendix).

Reviewer 8E-W3 / uZ-W2 / AC-R3: "GraphRAG default settings (under-tuned)".
Sweep the configuration space to report a tuned-baseline number against
which CogCanvas is compared.

Knob design (graphrag 2.x CLI):
  index-time (requires reindex)
    chunk_size      : 400 / 800 / 1200
    max_gleanings   : 1 / 2 / 3
  query-time (reuses index)
    community_level : 1 / 2 / 3
    search_method   : local / global / drift
                      (drift = closest analog to "hybrid" in ROADMAP;
                       basic ignored — it's plain BM25 fallback, not GraphRAG)

Two tiers:
  Tier 1 — 1D sweeps around default (cs=800, g=1, cl=2, local). 9 configs,
           run on a small conv subset to find which knobs matter.
  Tier 2 — Top-K configs from Tier 1 on full LoCoMo (paper baseline).

Implementation: each cell subprocesses ``python -m experiments.runner_locomo``
with the matching ``--graphrag-*`` flags. Cells with an existing output file
are skipped (idempotent).

Usage examples
--------------
Smoke test a single config on 1 conv, 5 questions::

    python -m experiments.graphrag_grid_search --tier smoke

Run full Tier 1 on 3 conversations, LLM judge, categories 1-3::

    python -m experiments.graphrag_grid_search \\
        --tier 1 --num-conv 3 --max-questions 0 \\
        --categories 1,2,3 --llm-score

Re-run a specific config (still skips if output exists; pass --force to redo)::

    python -m experiments.graphrag_grid_search --configs cs=400,g=1,cl=2,m=local
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

OUT_DIR = Path("experiments/results/graphrag_grid")
DEFAULT_DATASET = "experiments/data/locomo10.json"

# 1D sweeps around the default config
DEFAULT_CFG = ("cs", 800, "g", 1, "cl", 2, "m", "local")


@dataclass(frozen=True)
class Cell:
    chunk_size: int
    max_gleanings: int
    community_level: int
    search_method: str

    @property
    def tag(self) -> str:
        return (
            f"cs{self.chunk_size}_g{self.max_gleanings}"
            f"_cl{self.community_level}_{self.search_method}"
        )

    def to_cli(self) -> list[str]:
        return [
            "--graphrag-chunk-size", str(self.chunk_size),
            "--graphrag-max-gleanings", str(self.max_gleanings),
            "--graphrag-community-level", str(self.community_level),
            "--graphrag-search-method", self.search_method,
        ]


DEFAULT = Cell(800, 1, 2, "local")


def tier1_cells() -> list[Cell]:
    """Default + 1-knob-away cells. 9 unique configs total."""
    cells: list[Cell] = [DEFAULT]
    # chunk_size sweep
    for cs in (400, 1200):
        cells.append(Cell(cs, 1, 2, "local"))
    # max_gleanings sweep
    for g in (2, 3):
        cells.append(Cell(800, g, 2, "local"))
    # community_level sweep
    for cl in (1, 3):
        cells.append(Cell(800, 1, cl, "local"))
    # search_method sweep
    for m in ("global", "drift"):
        cells.append(Cell(800, 1, 2, m))
    return cells


def smoke_cells() -> list[Cell]:
    """Single non-default cell. Tests the entire grid plumbing end-to-end."""
    return [Cell(800, 2, 2, "local")]  # gleanings=2 forces a new index build


def parse_explicit_cells(spec: str) -> list[Cell]:
    """Parse 'cs=400,g=1,cl=2,m=local;cs=800,g=2,cl=3,m=drift' style."""
    cells = []
    for tup in spec.split(";"):
        tup = tup.strip()
        if not tup:
            continue
        kv = dict(p.split("=") for p in tup.split(","))
        cells.append(Cell(
            chunk_size=int(kv["cs"]),
            max_gleanings=int(kv["g"]),
            community_level=int(kv["cl"]),
            search_method=kv["m"],
        ))
    return cells


def output_path(cell: Cell, suffix: str) -> Path:
    return OUT_DIR / f"locomo_{cell.tag}{suffix}.json"


def run_cell(cell: Cell, args: argparse.Namespace) -> dict:
    """Subprocess a single grid cell, return summary dict."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path(cell, args.suffix)

    if out.exists() and not args.force:
        print(f"[SKIP] {cell.tag} — output exists: {out}")
        with open(out) as f:
            return {"cell": asdict(cell), "skipped": True, "output": str(out),
                    "summary": _extract_summary(json.load(f))}

    cmd = [
        sys.executable, "-m", "experiments.runner_locomo",
        "--agent", "graphrag",
        "--dataset", args.dataset,
        "--output", str(out),
        "--workers", str(args.workers),
        *cell.to_cli(),
    ]
    if args.num_conv:
        cmd += ["--samples", str(args.num_conv)]
    if args.max_questions:
        cmd += ["--max-questions", str(args.max_questions)]
    if args.categories:
        cmd += ["--categories", args.categories]
    if args.llm_score:
        cmd += ["--llm-score"]
    if args.compression_turn is not None:
        cmd += ["--compression-turn", str(args.compression_turn)]

    print(f"\n[RUN ] {cell.tag}")
    print(f"       cmd: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    print(f"[DONE] {cell.tag} rc={rc} elapsed={elapsed:.1f}s")

    summary: dict = {"cell": asdict(cell), "rc": rc, "elapsed_s": elapsed,
                     "output": str(out)}
    if out.exists():
        with open(out) as f:
            summary["summary"] = _extract_summary(json.load(f))
    return summary


def _extract_summary(result_json: dict) -> dict:
    """Pull headline numbers out of a runner_locomo result file.

    The runner writes ``summary.{accuracy, exact_match_rate, ...}`` plus
    per-conv numbers under ``conversations[i].{accuracy, ...}``. We surface
    the summary block when present, otherwise average the per-conv numbers
    so single-sample smoke runs still report something useful.
    """
    summary = result_json.get("summary") or {}
    if summary:
        return {
            "overall_accuracy": summary.get("overall_accuracy") or summary.get("accuracy"),
            "exact_match_rate": summary.get("exact_match_rate"),
            "avg_f1_score": summary.get("avg_f1_score"),
            "single_hop_accuracy": summary.get("single_hop_accuracy"),
            "temporal_accuracy": summary.get("temporal_accuracy"),
            "multi_hop_accuracy": summary.get("multi_hop_accuracy"),
            "num_questions": summary.get("num_questions"),
            "num_passed": summary.get("num_passed"),
        }
    convs = result_json.get("conversations") or []
    if not convs:
        return {}
    def avg(key: str):
        vals = [c.get(key) for c in convs if c.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    return {
        "overall_accuracy": avg("accuracy"),
        "exact_match_rate": avg("exact_match_rate"),
        "avg_f1_score": avg("avg_f1_score"),
        "single_hop_accuracy": avg("single_hop_accuracy"),
        "temporal_accuracy": avg("temporal_accuracy"),
        "multi_hop_accuracy": avg("multi_hop_accuracy"),
        "num_conversations": len(convs),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--tier", choices=["smoke", "1"], default="smoke",
                   help="smoke = 1 cell on 1 conv; 1 = Tier 1 (9 cells).")
    p.add_argument("--configs", default=None,
                   help="Explicit cells, e.g. 'cs=400,g=1,cl=2,m=local;...'. "
                        "Overrides --tier.")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--num-conv", type=int, default=None,
                   help="Subset of conversations (default: tier default).")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Cap questions per conversation (for smoke).")
    p.add_argument("--categories", default="1,2,3",
                   help="LoCoMo categories filter (default 1,2,3).")
    p.add_argument("--workers", type=int, default=4,
                   help="Per-cell QA workers (default 4 — keep modest, graphrag "
                        "spawns its own subprocesses).")
    p.add_argument("--compression-turn", type=int, default=None)
    p.add_argument("--llm-score", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells whose output already exists.")
    p.add_argument("--suffix", default="",
                   help="Output filename suffix (e.g. '_pilot').")
    p.add_argument("--summary-out", default=None,
                   help="Path for aggregated grid summary JSON.")
    args = p.parse_args()

    # Pick cells
    if args.configs:
        cells = parse_explicit_cells(args.configs)
    elif args.tier == "smoke":
        cells = smoke_cells()
        if args.num_conv is None:
            args.num_conv = 1
        if args.max_questions is None:
            args.max_questions = 5
    else:
        cells = tier1_cells()
        if args.num_conv is None:
            args.num_conv = 3

    print(f"Grid: tier={args.tier} configs={args.configs!r} cells={len(cells)}")
    print(f"  dataset={args.dataset} num_conv={args.num_conv} "
          f"max_q={args.max_questions} categories={args.categories} "
          f"llm_score={args.llm_score}")
    for c in cells:
        print(f"   - {c.tag}")

    summaries = [run_cell(c, args) for c in cells]

    # Aggregate
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w") as f:
            json.dump({"cells": summaries}, f, indent=2)
        print(f"\nAggregated summary -> {args.summary_out}")

    print("\n=== GRID DONE ===")
    for s in summaries:
        cell = Cell(**s["cell"])
        summ = s.get("summary", {})
        oa = summ.get("overall_accuracy")
        print(f"  {cell.tag:40s}  acc={oa!s:>8s}  rc={s.get('rc', 'skip')!s:>4s}")


if __name__ == "__main__":
    main()
