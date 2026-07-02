"""Hybrid (BM25 + dense) retrieval fusion for the fidelity-curve anchors.

Replicates the paper backbone's retriever (cogcanvas/canvas.py): fuse
0.7 * raw-cosine-semantic + 0.3 * max-normalized-BM25 over the stored items.
The flat anchors were dense-only, which matches the weak naive-RAG baseline
(~28-30% Cat1-3) rather than the paper's hybrid chunk pipeline (~44%); adding
BM25 fusion restores the proper verbatim anchor so the comparison is on the
same footing as the headline.
"""

import string
from rank_bm25 import BM25Okapi

_PUNCT = str.maketrans("", "", string.punctuation)


def _tok(text):
    return (text or "").translate(_PUNCT).lower().split()


def hybrid_order(query, texts, dense_sims, w_dense=0.7, w_bm25=0.3):
    """Return (sorted_indices, fused_scores) ranking items by 0.7*cosine+0.3*BM25.

    `dense_sims[i]` is the raw cosine similarity of item i to the query (same as
    the paper's semantic score); BM25 is max-normalized to [0,1] as in canvas.py.
    """
    n = len(texts)
    if n == 0:
        return [], []
    bm = BM25Okapi([_tok(t) for t in texts])
    bscores = bm.get_scores(_tok(query))
    mx = max(bscores) if len(bscores) and max(bscores) > 0 else 0.0
    bnorm = [(s / mx) if mx > 0 else 0.0 for s in bscores]
    fused = [w_dense * float(dense_sims[i]) + w_bm25 * bnorm[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: fused[i], reverse=True)
    return order, fused
