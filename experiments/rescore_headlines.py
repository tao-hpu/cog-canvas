"""Re-judge the saved headline answers with an alternate, different-family LLM
judge to show the chunks>artifacts ordering is not an artifact of the GPT-4o-mini
judge (rebuts the "judge biased toward lexical grounding" reviewer concern).

Reads the SAME saved (question, ground_truth, answer) triples that produced the
main tables and re-scores them with the *identical* judge functions/prompts the
paper uses (score_locomo_answer_llm / score_longmemeval_answer), only changing
the judge model. No retrieval or answer generation is re-run.

Usage:
    python -m experiments.rescore_headlines --judges gpt-4o-mini,qwen-plus --workers 12 --limit 0
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from experiments.runner_locomo import score_locomo_answer_llm
from experiments.runner_longmemeval import score_longmemeval_answer

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LOCO = {"chunks": "locomo_chunks_C_nograph_10_cat123.json",
        "artifacts": "locomo_full_10_cat123_0609.json"}
LME = {"chunks": "lme_s_chunks_nograph_500.json",
       "artifacts": "lme_s_cogcanvas_full_500.json"}


def load_env():
    env = {}
    for line in (HERE.parent / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def make_client(env):
    return OpenAI(api_key=env.get("SCORE_API_KEY"),
                  base_url=env.get("SCORE_API_BASE"))


def loco_questions(path):
    d = json.loads(Path(path).read_text())
    return [q for c in d["conversations"] for q in c["questions"]]


def lme_questions(path):
    d = json.loads(Path(path).read_text())
    return [q for c in d["conversations"] for q in c["questions"]]


def judge_loco(q, client, model):
    r = score_locomo_answer_llm(q["answer"], q["ground_truth"], q["question"],
                                client, model=model)
    return 1 if r.f1_score == 1.0 else 0


def judge_lme(q, client, model):
    r = score_longmemeval_answer(q["answer"], q["ground_truth"], q["question"],
                                 q["question_type"], q["is_abstention"],
                                 client, model=model)
    return 1 if r.correct else 0


def score_set(name, questions, judge_fn, client, model, workers, limit):
    if limit:
        questions = questions[:limit]
    n = len(questions)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(lambda q: judge_fn(q, client, model), questions))
    acc = 100.0 * sum(verdicts) / n
    return acc, n, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="gpt-4o-mini,qwen-plus")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--out", default="results/rescore_headlines.json")
    args = ap.parse_args()

    env = load_env()
    client = make_client(env)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    benches = [
        ("LoCoMo (Cat1-3, 699Q)", LOCO, loco_questions, judge_loco,
         {"chunks": 43.9, "artifacts": 28.0}),
        ("LongMemEval-S (500Q)", LME, lme_questions, judge_lme,
         {"chunks": 67.4, "artifacts": 45.4}),
    ]

    report = {}
    for bname, files, loader, judge_fn, paper in benches:
        qs = {rep: loader(RESULTS / f) for rep, f in files.items()}
        print(f"\n{'='*72}\n{bname}\n{'='*72}")
        print(f"  paper (gpt-4o-mini headline): chunks={paper['chunks']}  "
              f"artifacts={paper['artifacts']}  gap={paper['chunks']-paper['artifacts']:.1f}pp")
        report[bname] = {"paper": paper, "judges": {}}
        for model in judges:
            row = {}
            for rep in ("chunks", "artifacts"):
                acc, n, verdicts = score_set(bname, qs[rep], judge_fn, client,
                                             model, args.workers, args.limit)
                row[rep] = round(acc, 1)
                row[f"{rep}_n"] = n
            gap = row["chunks"] - row["artifacts"]
            row["gap"] = round(gap, 1)
            report[bname]["judges"][model] = row
            print(f"  judge={model:14s} chunks={row['chunks']:5.1f}  "
                  f"artifacts={row['artifacts']:5.1f}  gap={gap:+5.1f}pp  (n={row['chunks_n']})")

    out = HERE.parent / args.out
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
