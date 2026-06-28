"""Re-judge the saved headline answers with a PANEL of different-family LLM
judges to show the chunks>artifacts ordering is not an artifact of the
GPT-4o-mini judge (rebuts the "judge biased toward lexical grounding" concern).

Reads the SAME saved (question, ground_truth, answer) triples that produced the
main tables and re-scores them with the *identical* judge functions/prompts the
paper uses (score_locomo_answer_llm / score_longmemeval_answer), changing only
the judge model. No retrieval or answer generation is re-run.

For each judge it reports chunks acc, artifacts acc, the gap, and the paired
McNemar exact test (two-sided) over the per-question verdicts -- the same test
used for the headline tables.

Usage:
    python -m experiments.rescore_headlines \
        --judges gpt-4o-mini,qwen-plus,gemini-2.5-pro --workers 12
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

import experiments.llm_utils as _llm
from experiments.runner_locomo import score_locomo_answer_llm
from experiments.runner_longmemeval import score_longmemeval_answer

# Reasoning judges (e.g. gemini-2.5-pro) emit empty content under the judge
# protocol's max_tokens=10 (the budget is consumed by hidden reasoning), which
# would silently score every answer INCORRECT. Bump the output allowance for
# such models only; the judging PROMPT/criterion is unchanged. The judge
# functions do `from experiments.llm_utils import call_llm_with_retry` locally
# on every call, so patching the source module here is sufficient.
_REASONING_PREFIXES = ("gemini-2.5", "o1", "o3", "deepseek-r", "deepseek-reasoner")
_orig_call = _llm.call_llm_with_retry


def _patched_call(*a, **k):
    m = k.get("model", "")
    if any(m.startswith(p) for p in _REASONING_PREFIXES):
        k["max_tokens"] = max(k.get("max_tokens", 0), 2048)
    return _orig_call(*a, **k)


_llm.call_llm_with_retry = _patched_call

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


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial on discordant pairs). Returns (p, log10p)."""
    n = b + c
    if n == 0:
        return 1.0, 0.0
    k = min(b, c)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             for i in range(k + 1)]
    m = max(terms)
    logsum = m + math.log(sum(math.exp(t - m) for t in terms))
    logp = logsum + n * math.log(0.5) + math.log(2.0)
    log10p = logp / math.log(10.0)
    p = math.exp(logp) if logp > -700 else 0.0
    return min(p, 1.0), log10p


def loco_pairs(chunks_path, arti_path):
    """Align chunks vs artifacts by (conversation id, question text)."""
    cd = json.loads(Path(chunks_path).read_text())
    ad = json.loads(Path(arti_path).read_text())
    cmap = {(c["id"], q["question"]): q for c in cd["conversations"] for q in c["questions"]}
    amap = {(c["id"], q["question"]): q for c in ad["conversations"] for q in c["questions"]}
    keys = [k for k in cmap if k in amap]
    return [(cmap[k], amap[k]) for k in keys]


def lme_pairs(chunks_path, arti_path):
    """Align by question_id."""
    cd = json.loads(Path(chunks_path).read_text())
    ad = json.loads(Path(arti_path).read_text())
    cmap = {q["question_id"]: q for c in cd["conversations"] for q in c["questions"]}
    amap = {q["question_id"]: q for c in ad["conversations"] for q in c["questions"]}
    keys = [k for k in cmap if k in amap]
    return [(cmap[k], amap[k]) for k in keys]


def judge_loco(q, client, model):
    r = score_locomo_answer_llm(q["answer"], q["ground_truth"], q["question"],
                                client, model=model)
    return 1 if r.f1_score == 1.0 else 0


def judge_lme(q, client, model):
    r = score_longmemeval_answer(q["answer"], q["ground_truth"], q["question"],
                                 q["question_type"], q["is_abstention"],
                                 client, model=model)
    return 1 if r.correct else 0


def score_bench(pairs, judge_fn, client, model, workers):
    """Return (chunk_acc, arti_acc, gap, b, c, p, log10p, n)."""
    def one(pair):
        cq, aq = pair
        return judge_fn(cq, client, model), judge_fn(aq, client, model)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(one, pairs))
    n = len(verdicts)
    ch = sum(v[0] for v in verdicts)
    ar = sum(v[1] for v in verdicts)
    b = sum(1 for c_, a_ in verdicts if c_ and not a_)   # chunks-only correct
    c = sum(1 for c_, a_ in verdicts if a_ and not c_)   # artifacts-only correct
    p, log10p = mcnemar_exact(b, c)
    return (100.0 * ch / n, 100.0 * ar / n, 100.0 * (ch - ar) / n, b, c, p, log10p, n)


def fmt_p(p, log10p):
    if p == 0.0 or log10p < -4:
        return f"p<1e{math.ceil(log10p)}"
    return f"p={p:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="gpt-4o-mini,qwen-plus,gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/rescore_headlines.json")
    args = ap.parse_args()

    env = load_env()
    client = make_client(env)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    benches = [
        ("LoCoMo (Cat1-3)", loco_pairs(RESULTS / LOCO["chunks"], RESULTS / LOCO["artifacts"]),
         judge_loco, {"chunks": 43.9, "artifacts": 28.0, "gap": 15.9}),
        ("LongMemEval-S", lme_pairs(RESULTS / LME["chunks"], RESULTS / LME["artifacts"]),
         judge_lme, {"chunks": 67.4, "artifacts": 45.4, "gap": 22.0}),
    ]

    report = {"judges": judges, "benchmarks": {}}
    for bname, pairs, judge_fn, paper in benches:
        print(f"\n{'='*78}\n{bname}  (n={len(pairs)} paired questions)\n{'='*78}")
        print(f"  paper headline (gpt-4o-mini): chunks={paper['chunks']} "
              f"artifacts={paper['artifacts']} gap=+{paper['gap']}pp")
        report["benchmarks"][bname] = {"n": len(pairs), "paper": paper, "judges": {}}
        for model in judges:
            ch, ar, gap, b, c, p, log10p, n = score_bench(pairs, judge_fn, client, model, args.workers)
            family = ("OpenAI" if model.startswith("gpt") else
                      "Alibaba" if model.startswith("qwen") else
                      "Google" if model.startswith("gemini") else
                      "Anthropic" if model.startswith("claude") else
                      "DeepSeek" if model.startswith("deepseek") else
                      "xAI" if model.startswith("grok") else
                      "Zhipu" if model.startswith("glm") else "?")
            report["benchmarks"][bname]["judges"][model] = {
                "family": family, "chunks": round(ch, 1), "artifacts": round(ar, 1),
                "gap": round(gap, 1), "mcnemar_b": b, "mcnemar_c": c,
                "p": p, "log10p": round(log10p, 1), "n": n}
            print(f"  {model:16s} [{family:8s}] chunks={ch:5.1f} artifacts={ar:5.1f} "
                  f"gap=+{gap:4.1f}pp  McNemar(b={b},c={c}) {fmt_p(p,log10p)}")

    out = HERE.parent / args.out
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
