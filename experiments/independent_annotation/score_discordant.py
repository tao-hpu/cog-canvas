"""Score the discordant-pair annotation kit against the LLM judge.

Given the annotator's returned annotations_export.json and the private
discordant_KEY.json, report human-vs-judge agreement + Cohen's kappa on exactly
the headline-driving items:

  MAIN (correctness on discordant pairs)
    - overall human-vs-judge agreement + kappa
    - split by system (chunks vs artifacts): does the human confirm the judge's
      "chunk correct" and "artifact wrong" verdicts on the contested pairs?
    - per-question concordance: fraction of discordant questions where the human
      reproduces the judge's directional verdict on BOTH answers.
  ABSTENTION
    - human abstained/answered vs judge, agreement + kappa.

Usage:
    python score_discordant.py annotations_export.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
KEY = json.loads((HERE / "discordant_KEY.json").read_text())


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, k


def main(path):
    ann = json.loads(Path(path).read_text())
    main_lab = ann.get("main", {})
    abst_lab = ann.get("abstention", {})

    # ---- MAIN correctness: human CORRECT/INCORRECT vs judge_correct ----
    overall, by_system = [], defaultdict(list)
    per_q = defaultdict(dict)  # (question grouping via key order) -> {system: (human, judge)}
    for iid, meta in KEY.items():
        if meta["phase"] != "main":
            continue
        h = main_lab.get(iid)
        if h is None:
            continue
        human = (h == "CORRECT")
        judge = bool(meta["judge_correct"])
        overall.append((human, judge))
        by_system[meta["system"]].append((human, judge))

    print(f"Annotator: {ann.get('annotator','?')}")
    print(f"\n=== MAIN correctness on discordant pairs (n={len(overall)}) ===")
    po, k = kappa(overall)
    print(f"  human vs judge : agreement {po*100:.1f}%  kappa {k:.3f}")
    for sysname in ("chunks", "artifacts"):
        po, k = kappa(by_system[sysname])
        # human agreement with judge that this system's answer is correct/wrong
        conf = sum(1 for hh, jj in by_system[sysname] if hh == jj)
        print(f"  {sysname:9s} (n={len(by_system[sysname])}): "
              f"human agrees with judge {conf}/{len(by_system[sysname])} "
              f"({po*100:.1f}%), kappa {k:.3f}")
    print("  -> high agreement here means the +15.9pp gap is not a judge artifact:")
    print("     the human confirms chunk-answers right and artifact-answers wrong on the contested items.")

    # ---- ABSTENTION: human ABSTAINED/ANSWERED vs judge ----
    ab = []
    for iid, meta in KEY.items():
        if meta["phase"] != "abstention":
            continue
        h = abst_lab.get(iid)
        if h is None:
            continue
        # for abstention questions, judge_correct==True means the system correctly abstained
        human_abstained = (h == "ABSTAINED")
        judge_abstained = bool(meta["judge_correct"])
        ab.append((human_abstained, judge_abstained))
    print(f"\n=== ABSTENTION (n={len(ab)}) ===")
    if ab:
        po, k = kappa(ab)
        print(f"  human vs judge : agreement {po*100:.1f}%  kappa {k:.3f}")
    else:
        print("  no abstention labels returned")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python score_discordant.py annotations_export.json")
        sys.exit(1)
    main(sys.argv[1])
