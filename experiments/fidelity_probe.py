"""
Measure the FIDELITY of each storage representation directly.

Instead of assuming a fidelity rank order, we measure how much of each stored
representation is literally verbatim from the source conversation, using
word-bigram overlap:

    fidelity(item) = |bigrams(item) in source| / |bigrams(item)|

averaged (length-weighted) over every stored item the representation produces.
Verbatim chunks score ~1.0; abstractive/paraphrased representations score lower.
This yields a measured x-axis for the fidelity curve so "accuracy vs fidelity"
is plotted against a quantity, not an assumed ordering.

Usage:
    python -m experiments.fidelity_probe --convs 2 --out experiments/results/fidelity_locomo/fidelity_scores.json
"""

import argparse
import json
import re
from pathlib import Path

from experiments.locomo_adapter import load_locomo, convert_to_eval_format

_WORD = re.compile(r"[a-z0-9]+")


def toks(s):
    return _WORD.findall((s or "").lower())


def bigrams(ts):
    return set(zip(ts, ts[1:]))


def item_fidelity(item, src_bgs):
    bg = bigrams(toks(item))
    if not bg:
        return None, 0
    return len(bg & src_bgs) / len(bg), len(bg)


def store_texts(agent):
    """Pull the list of stored representation strings out of an agent."""
    if hasattr(agent, "_vector_store"):           # rag
        return [c.content for c in agent._vector_store]
    if hasattr(agent, "_store"):                   # secom (segments)
        return [s.content for s in agent._store]
    if hasattr(agent, "_facts"):                   # mem0
        return [f.text for f in agent._facts]
    if hasattr(agent, "_notes"):                   # amem
        return [n.render() for n in agent._notes]
    if hasattr(agent, "_artifacts"):               # artifacts-flat
        return [a.render() for a in agent._artifacts]
    if hasattr(agent, "_summary"):                 # summarization
        return [agent._summary] if agent._summary else []
    return []


def build_agent(name):
    if name == "rag":
        from experiments.agents.rag_agent import RagAgent
        return RagAgent()
    if name == "secom":
        from experiments.agents.secom_agent import SecomAgent
        return SecomAgent(compress=True)
    if name == "summarization":
        from experiments.agents.summarization_agent import SummarizationAgent
        return SummarizationAgent()
    if name == "mem0":
        from experiments.agents.mem0_agent import Mem0Agent
        return Mem0Agent()
    if name == "amem":
        from experiments.agents.amem_agent import AMemAgent
        return AMemAgent()
    if name == "artifacts-flat":
        from experiments.agents.artifacts_flat_agent import ArtifactsFlatAgent
        return ArtifactsFlatAgent()
    raise ValueError(name)


AGENTS = ["rag", "secom", "summarization", "mem0", "amem", "artifacts-flat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=2)
    ap.add_argument("--dataset", default="experiments/data/locomo10.json")
    ap.add_argument("--out", default="experiments/results/fidelity_locomo/fidelity_scores.json")
    args = ap.parse_args()

    convs = convert_to_eval_format(load_locomo(args.dataset))[: args.convs]
    scores = {}
    for name in AGENTS:
        per_conv = []
        n_items_total = 0
        for conv in convs:
            src_bgs = bigrams(toks(" ".join(
                f"{t.user} {t.assistant}" for t in conv.turns)))
            agent = build_agent(name)
            agent.reset()
            for t in conv.turns:
                agent.process_turn(t)
            agent.on_compression(conv.turns[-5:])
            items = store_texts(agent)
            num, den = 0.0, 0.0
            for it in items:
                fid, w = item_fidelity(it, src_bgs)
                if fid is not None:
                    num += fid * w
                    den += w
            if den:
                per_conv.append(num / den)
            n_items_total += len(items)
        fidelity = sum(per_conv) / len(per_conv) if per_conv else 0.0
        scores[name] = {"fidelity": round(fidelity, 4),
                        "n_items": n_items_total, "n_convs": len(per_conv)}
        print(f"{name:16} fidelity={fidelity:.3f}  (items={n_items_total} over {len(per_conv)} convs)")

    Path(args.out).write_text(json.dumps(scores, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
