"""Strawman diagnostic: run the PAPER'S OWN typed-artifact extractor in the
SAME weak harness as the official-Mem0 anchor (text-embedding-3-small + faiss
cosine top-k + gpt-4o-mini answerer + shared judge, NO bge-reranker), against
verbatim chunks and full context.

Question being resolved:
  In the anchor's weak harness, official Mem0 (34.8) TIED chunks (35.1).
  Does the paper's typed-artifact extractor ALSO tie chunks here, or does it
  still lose ~16pp? If it ties -> the 16pp headline gap is a strong-pipeline
  (reranker-favors-verbatim) effect, the extractor is NOT a strawman vs Mem0.
  If it still loses while Mem0 ties -> the paper's extractor design is genuinely
  weaker than Mem0's, and the strawman concern has teeth.

Artifacts get top_k=30 (parity with Mem0's budget) -- generous to artifacts.
Chunks keep top_k=15 (anchor/paper setting). Per-session extraction (the
paper's validated-equivalent granularity control, p=0.57 vs per-turn).
"""
import json, sys, time, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas")
import experiments.anchors.mem0_official_locomo as M  # helpers + .env
import numpy as np, faiss
from experiments.locomo_adapter import load_locomo, convert_to_eval_format

from cogcanvas import Canvas
from cogcanvas.llm.openai import OpenAIBackend


def build_artifacts(conv):
    """Per-session typed-artifact extraction with the paper's Canvas extractor,
    then embed each artifact's (content+quote) text with the SAME embedder/faiss
    as the chunk store -- isolating the stored representation, not retrieval."""
    backend = OpenAIBackend(model=M.ANSWER_MODEL, embedding_model=M.EMBED_MODEL,
                            api_key=M.API_KEY, api_base=M.API_BASE)
    canvas = Canvas(llm_backend=backend, embedding_model=M.EMBED_MODEL, enable_gleaning=False)
    objs, sess = [], []
    bu, ba, cur = [], [], None

    def flush():
        if not (bu or ba):
            return
        r = canvas.extract(user="\n".join(bu), assistant="\n".join(ba), session_datetime=cur)
        for o in r.objects:
            objs.append(o); sess.append(cur)

    for t in conv.turns:
        if t.session_datetime != cur and (bu or ba):
            flush(); bu.clear(); ba.clear()
        cur = t.session_datetime
        if t.user: bu.append(f"{conv.speaker_a}: {t.user}")
        if t.assistant: ba.append(f"{conv.speaker_b}: {t.assistant}")
    flush()

    texts = [(f"{o.content} {o.quote}".strip() if getattr(o, "quote", None) else o.content) for o in objs]
    if not texts:
        return [], None, []
    client = M.oai()
    embs = []
    for i in range(0, len(texts), 64):
        r = client.embeddings.create(model=M.EMBED_MODEL, input=texts[i:i + 64])
        embs.extend([d.embedding for d in r.data])
    mat = np.array(embs, dtype="float32"); faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(M.EMBED_DIM); index.add(mat)
    # render with session timestamp for temporal parity with chunks/Mem0
    rendered = [f"[{sess[i]}] {texts[i]}" for i in range(len(texts))]
    return rendered, index, objs


def artifacts_context(rendered, index, question, top_k=30):
    client = M.oai()
    q = np.array([client.embeddings.create(model=M.EMBED_MODEL, input=question).data[0].embedding], dtype="float32")
    faiss.normalize_L2(q)
    _, idx = index.search(q, min(top_k, len(rendered)))
    return "\n".join(f"- {rendered[i]}" for i in idx[0] if i >= 0)


def run_conv(conv, workers):
    qas = [qa for qa in conv.qa_pairs if qa.category in (1, 2, 3)]
    t0 = time.time()
    try:
        rendered, aindex, objs = build_artifacts(conv)
        chunks, cindex = M.build_chunks(conv)
        full = M.full_context_text(conv)
        n_art = len(objs)

        def one(qa):
            q, gt = qa.question, qa.answer
            try:
                ctx_a = artifacts_context(rendered, aindex, q) if rendered else ""
                ctx_c = M.chunk_context(chunks, cindex, q)
                v_a = M.judge(M.answer(ctx_a, q), gt, q)
                v_c = M.judge(M.answer(ctx_c, q), gt, q)
                v_f = M.judge(M.answer(full, q), gt, q)
            except Exception as e:
                print(f"  conv {conv.id}: skip Q ({type(e).__name__}: {e})", flush=True)
                return None
            return {"category": qa.category, "artifacts": v_a, "chunks": v_c, "full": v_f}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = [r for r in ex.map(one, qas) if r is not None]
    except Exception as e:
        print(f"  conv {conv.id}: FAILED ({type(e).__name__}: {e})", flush=True)
        rows, n_art = [], 0
    print(f"  conv {conv.id}: {len(rows)} Qs, {n_art} artifacts in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    n_conv = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    raw = load_locomo(str(M.ROOT / "experiments/data/locomo10.json"))
    convs = convert_to_eval_format(raw)[:n_conv]
    print(f"Paper-artifacts in WEAK harness | {len(convs)} convs | answerer={M.ANSWER_MODEL} "
          f"judge={M.SCORE_MODEL} embed={M.EMBED_MODEL} | artifacts top_k=30, chunks top_k=15", flush=True)
    all_rows = []
    out = str(M.ROOT / "experiments/results/diag_artifacts_weakharness.json")
    with ThreadPoolExecutor(max_workers=5) as ex:
        for rows in ex.map(lambda c: run_conv(c, 4), convs):
            all_rows.extend(rows)
            n = len(all_rows) or 1
            acc = {k: 100.0 * sum(r[k] for r in all_rows) / n for k in ("artifacts", "chunks", "full")}
            Path(out).write_text(json.dumps({"n": len(all_rows), "accuracy": acc, "rows": all_rows}, indent=2))
    n = len(all_rows)
    acc = {k: 100.0 * sum(r[k] for r in all_rows) / n for k in ("artifacts", "chunks", "full")}
    print("\n=== PAPER-ARTIFACTS WEAK-HARNESS RESULT (LoCoMo Cat1-3) ===")
    print(f"  Paper artifacts (k=30): {acc['artifacts']:.1f}%")
    print(f"  Verbatim chunks (k=15): {acc['chunks']:.1f}%")
    print(f"  Full context          : {acc['full']:.1f}%")
    for cat in (1, 2, 3):
        sub = [r for r in all_rows if r["category"] == cat]
        if sub:
            m = len(sub); a = {k: round(100*sum(r[k] for r in sub)/m, 1) for k in ("artifacts", "chunks", "full")}
            print(f"  Cat{cat} (n={m}): {a}")
    print(f"  n={n}  saved -> {out}")
    print("=== done ===")


if __name__ == "__main__":
    main()
