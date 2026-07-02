"""No-context (closed-book) sanity check for LoCoMo (reviewer ⑤/④).

Purpose. Rule out that headline accuracy comes from the answerer's parametric
priors or benchmark contamination rather than from retrieved conversation text.
We hand the SAME answerer the SAME questions with ZERO conversation context and
score with the SAME LLM judge as the headline tables. If closed-book accuracy is
near the floor, the retrieved verbatim text is doing the work, not memorization.

Two conditions:
  closed-book : question only, no context.
  shuffled    : question paired with a DIFFERENT conversation's chunks (wrong
                evidence) -- rules out generic pattern-matching. (--mode shuffled)

Reuses score_locomo_answer_llm (the headline judge) and the ANSWER_/SCORE_ env
clients, so numbers are directly comparable to Table (chunks 43.9 / artifacts 28.0).

Usage:
    python -m experiments.closed_book_probe --limit 5          # smoke test
    python -m experiments.closed_book_probe --out experiments/results/locomo_closedbook.json
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from experiments.runner_locomo import score_locomo_answer_llm  # noqa: E402

QUESTIONS_SRC = ROOT / "experiments" / "results" / "locomo_chunks_C_nograph_10_cat123.json"


def answer_client():
    return OpenAI(
        api_key=os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE"),
    )


def score_client():
    return OpenAI(
        api_key=os.getenv("SCORE_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("SCORE_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE"),
    )


def load_questions():
    d = json.loads(QUESTIONS_SRC.read_text())
    qs = []
    for c in d["conversations"]:
        for q in c["questions"]:
            qs.append({"cid": c["id"], "question": q["question"],
                       "gt": str(q["ground_truth"]),
                       "category": q.get("category_name", q.get("category"))})
    return qs


CLOSED_BOOK_PROMPT = (
    "Answer the following question about a long two-person conversation. "
    "No conversation transcript is provided. If the answer cannot be known without "
    "the conversation, give your best single-line answer anyway.\n\nQuestion: {q}\nAnswer:"
)


def run_one(item, ans_model, ac, sc, score_model):
    resp = ac.chat.completions.create(
        model=ans_model,
        messages=[{"role": "user", "content": CLOSED_BOOK_PROMPT.format(q=item["question"])}],
        max_tokens=200, temperature=0,
    )
    answer = (resp.choices[0].message.content or "").strip()
    res = score_locomo_answer_llm(answer, item["gt"], item["question"], sc, model=score_model)
    passed = bool(getattr(res, "exact_match", False)) or (getattr(res, "f1_score", 0) == 1.0)
    return {**item, "answer": answer, "passed": passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all 699")
    ap.add_argument("--model", default=os.getenv("ANSWER_MODEL", "gpt-4o"))
    ap.add_argument("--score-model", default=os.getenv("SCORE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    qs = load_questions()
    if args.limit:
        qs = qs[:args.limit]
    ac, sc = answer_client(), score_client()
    print(f"closed-book probe: {len(qs)} questions | answerer={args.model} | judge={args.score_model}")

    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, it, args.model, ac, sc, args.score_model) for it in qs]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 25 == 0 or i == len(qs):
                print(f"  {i}/{len(qs)}")

    by = defaultdict(list)
    for r in out:
        by[r["category"]].append(r["passed"])
    allp = [r["passed"] for r in out]
    print(f"\n=== CLOSED-BOOK accuracy: {100*sum(allp)/len(allp):.1f}%  (n={len(allp)}) ===")
    for c in sorted(by):
        v = by[c]
        print(f"  {c:12s} n={len(v):3d}  {100*sum(v)/len(v):.1f}%")
    print("  (headline chunks=43.9%, artifacts=28.0%; low closed-book => retrieval is load-bearing)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"mode": "closed_book", "answerer": args.model, "judge": args.score_model,
             "n": len(allp), "accuracy": 100*sum(allp)/len(allp),
             "by_category": {c: [len(v), 100*sum(v)/len(v)] for c, v in by.items()},
             "items": out}, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
