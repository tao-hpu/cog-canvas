"""Score an independent annotator's labels against (a) the LLM judge and
(b) the original author annotator, for both the main correctness study and the
category-5 abstention study.

Outputs, ready to drop into the judge-validation appendix:
  MAIN  (100 items)
    independent-human vs. LLM-judge : agreement + Cohen's kappa (overall + strata)
    independent-human vs. author    : inter-annotator agreement + kappa
  ABSTENTION (50 items)  -- first human validation of the abstention judge
    independent-human vs. LLM-judge : agreement + kappa
    independent-human vs. annotator2: inter-annotator agreement + kappa

Usage:
    python score_independent.py annotations_export.json
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
MAIN_KEY = RESULTS / "judge_agreement" / "key.json"
MAIN_SHEET = RESULTS / "judge_agreement" / "annotation_sheet.csv"
CAT5_KEY = RESULTS / "cat5_blind_labels_KEY.csv"
CAT5_A2 = RESULTS / "cat5_blind_labels_annotator2.csv"
SHORT_ANSWER_TOKENS = 30


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, k


def confusion(pairs):
    tp = sum(1 for a, b in pairs if a and b)
    fn = sum(1 for a, b in pairs if a and not b)
    fp = sum(1 for a, b in pairs if not a and b)
    tn = sum(1 for a, b in pairs if not a and not b)
    return tp, fn, fp, tn


def report(title, pairs, label_a, label_b):
    po, k = kappa(pairs)
    tp, fn, fp, tn = confusion(pairs)
    print(f"\n{title}: n={len(pairs)} agreement={po:.1%} kappa={k:.3f}")
    print(f"  confusion ({label_a} x {label_b}): "
          f"both-yes={tp} {label_a}-only={fn} {label_b}-only={fp} both-no={tn}")
    return po, k


def score_main(export):
    labels = export.get("main", {})
    key = json.loads(MAIN_KEY.read_text())
    author = {}
    benchmark = {}
    with MAIN_SHEET.open() as f:
        for r in csv.DictReader(f):
            v = r["human_verdict"].strip().upper()
            if v in ("CORRECT", "INCORRECT"):
                author[r["id"]] = (v == "CORRECT")
            benchmark[r["id"]] = r["benchmark"]

    vs_judge, vs_author, strata = [], [], {}
    missing = 0
    for qid, k in key.items():
        hv = labels.get(qid)
        if hv not in ("CORRECT", "INCORRECT"):
            missing += 1
            continue
        h = (hv == "CORRECT")
        j = bool(k["judge_verdict"])
        vs_judge.append((h, j))
        for name in (
            f"benchmark={benchmark.get(qid,'?')}",
            f"pipeline={k['pipeline']}",
            "answer=short" if k["answer_tokens"] <= SHORT_ANSWER_TOKENS else "answer=long",
        ):
            strata.setdefault(name, []).append((h, j))
        if qid in author:
            vs_author.append((h, author[qid]))

    print("=" * 64)
    print("MAIN CORRECTNESS STUDY")
    print("=" * 64)
    if missing:
        print(f"WARNING: {missing} main items missing a valid label (skipped)")
    report("independent-human vs. LLM-judge", vs_judge, "human", "judge")
    for name in sorted(strata):
        po, k = kappa(strata[name])
        print(f"    {name:20s} n={len(strata[name]):3d} agreement={po:.1%} kappa={k:.3f}")
    report("independent-human vs. author (inter-annotator)", vs_author, "indep", "author")


def score_abstention(export):
    labels = export.get("abstention", {})
    judge = {}
    with CAT5_KEY.open() as f:
        for r in csv.DictReader(f):
            judge[str(r["id"])] = (str(r["llm_judge_abstained"]).strip().lower() == "true")
    a2 = {}
    if CAT5_A2.exists():
        with CAT5_A2.open() as f:
            for r in csv.DictReader(f):
                lab = str(r.get("label", "")).strip()
                if lab in ("0", "1"):
                    a2[str(r["id"])] = (lab == "1")

    vs_judge, vs_a2 = [], []
    missing = 0
    for qid, jv in judge.items():
        hv = labels.get(qid)
        if hv not in ("ABSTAINED", "ANSWERED"):
            missing += 1
            continue
        h = (hv == "ABSTAINED")
        vs_judge.append((h, jv))
        if qid in a2:
            vs_a2.append((h, a2[qid]))

    print("\n" + "=" * 64)
    print("ABSTENTION STUDY (category 5) -- first human validation")
    print("=" * 64)
    if missing:
        print(f"WARNING: {missing} abstention items missing a valid label (skipped)")
    report("independent-human vs. LLM-judge", vs_judge, "human", "judge")
    if vs_a2:
        report("independent-human vs. annotator2 (inter-annotator)", vs_a2, "indep", "annot2")


def main():
    if len(sys.argv) != 2:
        print("usage: python score_independent.py annotations_export.json")
        sys.exit(1)
    export = json.loads(Path(sys.argv[1]).read_text())
    print(f"annotator: {export.get('annotator','?')}")
    score_main(export)
    score_abstention(export)


if __name__ == "__main__":
    main()
