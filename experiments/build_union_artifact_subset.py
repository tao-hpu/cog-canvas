#!/usr/bin/env python3
"""
Derive the artifact-only store that is the exact artifact subset of a union store.

A union store (cogcanvas-union) holds verbatim chunks AND extracted artifacts in
one canvas. Its chunk objects are byte-identical to the chunk store's objects
(verified below), so removing them leaves exactly the artifacts that the union
arm retrieves from. Running the artifact arm on this subset, the chunk arm on the
chunk store, and the union arm on the union store gives a three-way comparison in
which union = chunks + artifacts holds object for object.

Usage:
    python -m experiments.build_union_artifact_subset \
        --union experiments/cache/extraction/7c39e0a4 \
        --chunks experiments/cache/extraction/010f67bf \
        --out experiments/cache/extraction/union7c39e0a4_artifact_subset
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for uf in sorted(Path(args.union).glob("locomo_*.json")):
        union = json.load(open(uf))
        chunk_contents = {o["content"].strip()
                          for o in json.load(open(Path(args.chunks) / uf.name))["objects"]}
        is_chunk = lambda o: o["type"] == "key_fact" and o["content"].strip() in chunk_contents
        dropped = [o for o in union["objects"] if is_chunk(o)]
        kept = [o for o in union["objects"] if not is_chunk(o)]
        assert {o["content"].strip() for o in dropped} == chunk_contents, uf.name
        keep_ids = {o["id"] for o in kept}
        graph = union.get("graph") or {}
        new_graph = {
            "nodes": [n for n in graph.get("nodes", []) if n in keep_ids],
            "edges": {
                rel: {src: [d for d in dsts if d in keep_ids]
                      for src, dsts in adj.items() if src in keep_ids}
                for rel, adj in graph.get("edges", {}).items()
            },
        }
        new_graph["edges"] = {rel: {s: d for s, d in adj.items() if d}
                              for rel, adj in new_graph["edges"].items()}
        store = dict(union)
        store["objects"] = kept
        store["graph"] = new_graph
        json.dump(store, open(out / uf.name, "w"), ensure_ascii=False)
        print(f"{uf.name}: union {len(union['objects'])} = chunks {len(dropped)} + artifacts {len(kept)}")


if __name__ == "__main__":
    main()
