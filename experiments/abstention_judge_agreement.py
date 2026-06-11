"""Agreement between the cat-5 LLM abstention judge and the lexical fallback.

For each category-5 question in a result JSON, re-applies the lexical
ABSTENTION_PATTERNS check to the stored answer and compares it against the
LLM abstention judge's verdict recorded at run time (the `passed` field).

Usage:
    python -m experiments.abstention_judge_agreement <result.json> [...]
"""

import json
import sys

from experiments.runner_locomo import ABSTENTION_PATTERNS


def agreement(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    n = agree = 0
    llm_yes_lex_no = lex_yes_llm_no = 0
    both_yes = both_no = 0
    for conv in data["conversations"]:
        for q in conv["questions"]:
            if q.get("category") != 5:
                continue
            llm = bool(q["passed"])
            lex = bool(ABSTENTION_PATTERNS.search(q["answer"] or ""))
            n += 1
            if llm == lex:
                agree += 1
                if llm:
                    both_yes += 1
                else:
                    both_no += 1
            elif llm:
                llm_yes_lex_no += 1
            else:
                lex_yes_llm_no += 1

    # Cohen's kappa
    llm_pos = both_yes + llm_yes_lex_no
    lex_pos = both_yes + lex_yes_llm_no
    po = agree / n
    pe = (llm_pos / n) * (lex_pos / n) + ((n - llm_pos) / n) * ((n - lex_pos) / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    print(f"\n{path}")
    print(f"  cat-5 questions: {n}")
    print(f"  raw agreement: {agree}/{n} = {po * 100:.1f}%   Cohen's kappa = {kappa:.3f}")
    print(f"  both abstain: {both_yes}  both answer: {both_no}")
    print(f"  LLM=abstain / lexical=answer: {llm_yes_lex_no}")
    print(f"  lexical=abstain / LLM=answer: {lex_yes_llm_no}")
    print(f"  abstention rate -- LLM judge: {llm_pos / n * 100:.1f}%  lexical: {lex_pos / n * 100:.1f}%")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        agreement(p)
