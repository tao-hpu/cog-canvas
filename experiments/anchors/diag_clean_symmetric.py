"""CLEAN SYMMETRIC weak-harness comparison: verbatim chunks vs the paper's
typed-artifact extractor, with EVERY confound removed so the ONLY difference is
the stored representation.

Identical for both reps:
  - embedder text-embedding-3-small, faiss IndexFlatIP (cosine, normalized)
  - SAME retrieval depth top_k=15
  - per-session construction (chunks chunked per session; artifacts extracted
    per session)
  - per-item [session timestamp] prefix on EVERY item (symmetric temporal info,
    matching the paper backbone that hands session timestamps to the answerer)
  - same gpt-4o-mini answerer, same LLM judge

Only difference: chunks = verbatim 512-char windows; artifacts = Canvas
content+quote. Full context included as the stable ceiling/sanity.

This fixes the two biases in the previous diagnostic (artifacts had k=30 and
were the only rep with per-item timestamps).
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas")
import experiments.anchors.mem0_official_locomo as M
import numpy as np, faiss
from experiments.locomo_adapter import load_locomo, convert_to_eval_format
from cogcanvas import Canvas
from cogcanvas.llm.openai import OpenAIBackend

TOPK = 15


def _embed_index(texts):
    client = M.oai()
    embs = []
    for i in range(0, len(texts), 64):
        r = client.embeddings.create(model=M.EMBED_MODEL, input=texts[i:i + 64])
        embs.extend([d.embedding for d in r.data])
    mat = np.array(embs, dtype="float32"); faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(M.EMBED_DIM); index.add(mat)
    return index


def build_chunks_sessioned(conv, size=512, overlap=100):
    """Per-session 512-char chunks, each tagged with its session timestamp."""
    chunks, stamps = [], []
    cur, buf = None, []

    def flush():
        if not buf:
            return
        for c in M.chunk_text("\n".join(buf), size, overlap):
            chunks.append(c); stamps.append(cur)

    for t in conv.turns:
        if t.session_datetime != cur and buf:
            flush(); buf = []
        cur = t.session_datetime
        if t.user: buf.append(f"{conv.speaker_a}: {t.user}")
        if t.assistant: buf.append(f"{conv.speaker_b}: {t.assistant}")
    flush()
    rendered = [f"[{stamps[i]}] {chunks[i]}" for i in range(len(chunks))]
    return rendered, _embed_index(chunks)  # embed raw text; render adds stamp


def build_artifacts_sessioned(conv):
    """Per-session typed-artifact extraction, each tagged with its session ts."""
    backend = OpenAIBackend(model=M.ANSWER_MODEL, embedding_model=M.EMBED_MODEL,
                            api_key=M.API_KEY, api_base=M.API_BASE)
    canvas = Canvas(llm_backend=backend, embedding_model=M.EMBED_MODEL, enable_gleaning=False)
    texts, stamps = [], []
    bu, ba, cur = [], [], None

    def flush():
        if not (bu or ba):
            return
        r = canvas.extract(user="\n".join(bu), assistant="\n".join(ba), session_datetime=cur)
        for o in r.objects:
            txt = f"{o.content} {o.quote}".strip() if getattr(o, "quote", None) else o.content
            texts.append(txt); stamps.append(cur)

    for t in conv.turns:
        if t.session_datetime != cur and (bu or ba):
            flush(); bu.clear(); ba.clear()
        cur = t.session_datetime
        if t.user: bu.append(f"{conv.speaker_a}: {t.user}")
        if t.assistant: ba.append(f"{conv.speaker_b}: {t.assistant}")
    flush()
    if not texts:
        return [], None
    rendered = [f"[{stamps[i]}] {texts[i]}" for i in range(len(texts))]
    return rendered, _embed_index(texts)


def ctx(rendered, index, question, top_k=TOPK):
    if not rendered:
        return ""
    client = M.oai()
    q = np.array([client.embeddings.create(model=M.EMBED_MODEL, input=question).data[0].embedding], dtype="float32")
    faiss.normalize_L2(q)
    _, idx = index.search(q, min(top_k, len(rendered)))
    return "\n".join(f"- {rendered[i]}" for i in idx[0] if i >= 0)


def run_conv(conv, workers):
    qas = [qa for qa in conv.qa_pairs if qa.category in (1, 2, 3)]
    t0 = time.time()
    try:
        a_rend, a_idx = build_artifacts_sessioned(conv)
        c_rend, c_idx = build_chunks_sessioned(conv)
        full = M.full_context_text(conv)
        n_a, n_c = len(a_rend), len(c_rend)

        def one(qa):
            q, gt = qa.question, qa.answer
            try:
                v_a = M.judge(M.answer(ctx(a_rend, a_idx, q), q), gt, q)
                v_c = M.judge(M.answer(ctx(c_rend, c_idx, q), q), gt, q)
                v_f = M.judge(M.answer(full, q), gt, q)
            except Exception as e:
                print(f"  conv {conv.id}: skip Q ({type(e).__name__}: {e})", flush=True)
                return None
            return {"category": qa.category, "artifacts": v_a, "chunks": v_c, "full": v_f}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = [r for r in ex.map(one, qas) if r is not None]
    except Exception as e:
        print(f"  conv {conv.id}: FAILED ({type(e).__name__}: {e})", flush=True)
        rows, n_a, n_c = [], 0, 0
    print(f"  conv {conv.id}: {len(rows)} Qs, {n_a} artifacts / {n_c} chunks in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    n_conv = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    raw = load_locomo(str(M.ROOT / "experiments/data/locomo10.json"))
    convs = convert_to_eval_format(raw)[:n_conv]
    print(f"CLEAN symmetric: chunks vs artifacts | {len(convs)} convs | both top_k={TOPK}, "
          f"per-session, per-item [session ts] | answerer={M.ANSWER_MODEL} judge={M.SCORE_MODEL}", flush=True)
    all_rows = []
    out = str(M.ROOT / "experiments/results/diag_clean_symmetric.json")
    with ThreadPoolExecutor(max_workers=5) as ex:
        for rows in ex.map(lambda c: run_conv(c, 4), convs):
            all_rows.extend(rows)
            n = len(all_rows) or 1
            acc = {k: 100.0 * sum(r[k] for r in all_rows) / n for k in ("artifacts", "chunks", "full")}
            Path(out).write_text(json.dumps({"n": len(all_rows), "accuracy": acc, "rows": all_rows}, indent=2))
    n = len(all_rows)
    acc = {k: 100.0 * sum(r[k] for r in all_rows) / n for k in ("artifacts", "chunks", "full")}
    print("\n=== CLEAN SYMMETRIC RESULT (LoCoMo Cat1-3, k=15 both, symmetric timestamps) ===")
    print(f"  Verbatim chunks : {acc['chunks']:.1f}%")
    print(f"  Paper artifacts : {acc['artifacts']:.1f}%")
    print(f"  Full context    : {acc['full']:.1f}%")
    print(f"  chunks - artifacts = {acc['chunks']-acc['artifacts']:+.1f}pp")
    for cat in (1, 2, 3):
        sub = [r for r in all_rows if r["category"] == cat]
        if sub:
            m = len(sub); a = {k: round(100*sum(r[k] for r in sub)/m, 1) for k in ("artifacts", "chunks", "full")}
            print(f"  Cat{cat} (n={m}): {a}")
    print(f"  n={n}  saved -> {out}")
    print("=== done ===")


if __name__ == "__main__":
    main()
