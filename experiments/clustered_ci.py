"""Cluster-bootstrap 95% CIs for the chunks-vs-artifacts headline gaps (review W9).

Pairs per-question outcomes between two result files by (conversation id,
question, ground truth), then bootstraps at the conversation level (resample
clusters with replacement) so the CI respects within-conversation correlation.

Usage:
    python -m experiments.clustered_ci
"""

import json
import random
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

COMPARISONS = [
    ("LoCoMo (10 conversation clusters)",
     "locomo_chunks_C_nograph_10_cat123.json",
     "locomo_full_10_cat123_0609.json"),
    ("LongMemEval-S (500 clusters)",
     "lme_s_chunks_nograph_500.json",
     "lme_s_cogcanvas_full_500.json"),
]

N_BOOT = 10000
SEED = 7


def load(fname):
    data = json.loads((RESULTS / fname).read_text())
    out = {}
    for conv in data["conversations"]:
        for q in conv["questions"]:
            key = (conv["id"], q["question"], str(q["ground_truth"]))
            out[key] = bool(q["passed"] if "passed" in q else q["correct"])
    return out


def cluster_ci(a, b, label):
    keys = sorted(set(a) & set(b))
    convs = {}
    for k in keys:
        convs.setdefault(k[0], []).append((a[k], b[k]))
    cids = sorted(convs)
    gap = (sum(a[k] for k in keys) - sum(b[k] for k in keys)) / len(keys)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        pairs = [p for cid in (rng.choice(cids) for _ in cids) for p in convs[cid]]
        boots.append(sum(x for x, _ in pairs) / len(pairs)
                     - sum(y for _, y in pairs) / len(pairs))
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
    print(f"{label}: paired n={len(keys)}, gap={gap*100:.1f}pp, "
          f"95% CI [{lo*100:.1f}, {hi*100:.1f}] over {len(cids)} clusters")


if __name__ == "__main__":
    for label, f_chunks, f_artifacts in COMPARISONS:
        cluster_ci(load(f_chunks), load(f_artifacts), label)
