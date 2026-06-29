"""
External-system anchor: OFFICIAL Mem0 on LoCoMo.

Reviewer-facing purpose. The main paper places Mem0/A-Mem/SeCom on a single
fidelity curve by *reproducing each system's defining mechanism* inside one
fixed pipeline (so accuracy attaches to the stored representation, not to
confounded retrieval stacks). A natural objection is that an in-pipeline
reproduction might not faithfully represent the real system. This script rules
that out for Mem0 by running the **official `mem0` package end-to-end** -- its
own extraction loop, its own vector search, its native OpenAI embedder
(text-embedding-3-small) -- and comparing three stores inside ONE harness with a
SHARED answerer and the SHARED LLM judge used for the headline tables:

    (1) Mem0  : official Mem0 extracted memory (mem0.add(infer=True) -> mem0.search)
    (2) Chunks: verbatim 512-char windows, same embedder, faiss top-k
    (3) Full  : the entire transcript handed to the answerer (reference ceiling)

The claim under test is the WITHIN-HARNESS ORDERING the paper reports: lossy
extraction (Mem0) underperforms verbatim text and full context. Mem0's own
LoCoMo report (extracted 66.9 < full-context 72.9 ceiling) predicts the same
direction; here we confirm it with the real package, the same judge, and the
same answerer, so the reproduction's ordering is not a strawman artifact.

Answerer + builder are gpt-4o-mini (ANSWER_MODEL overridable); judge is
SCORE_MODEL (gpt-4o-mini), identical to the headline protocol. LoCoMo Cat 1-3.

Usage:
    python -m experiments.anchors.mem0_official_locomo --n-conv 3 --workers 8
    python -m experiments.anchors.mem0_official_locomo --n-conv 10 --workers 8 \
        --out experiments/results/anchor_mem0_official_locomo_10.json
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # cog-canvas/


def load_env():
    env_path = ROOT / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


load_env()

from openai import OpenAI  # noqa: E402
import numpy as np  # noqa: E402
import faiss  # noqa: E402

from experiments.locomo_adapter import load_locomo, convert_to_eval_format  # noqa: E402
from experiments.runner_locomo import score_locomo_answer_llm  # noqa: E402
from experiments.llm_utils import call_llm_with_retry  # noqa: E402

ANSWER_MODEL = os.getenv("ANSWER_MODEL_ANCHOR", "gpt-4o-mini")
SCORE_MODEL = os.getenv("SCORE_MODEL", "gpt-4o-mini")
EMBED_MODEL = "text-embedding-3-small"  # Mem0 native default
EMBED_DIM = 1536

# All clients hit the proxy that serves OpenAI-compatible chat + embeddings.
API_KEY = os.getenv("ANSWER_API_KEY")
API_BASE = os.getenv("ANSWER_API_BASE")


def oai():
    return OpenAI(api_key=API_KEY, base_url=API_BASE)


# --------------------------------------------------------------------------- #
# Conversation formatting
# --------------------------------------------------------------------------- #
def turns_to_messages(conv):
    """LoCoMo turn -> chat messages for Mem0 (speaker_a=user, speaker_b=assistant)."""
    msgs = []
    for t in conv.turns:
        if t.user:
            msgs.append({"role": "user", "content": f"{conv.speaker_a}: {t.user}"})
        if t.assistant:
            msgs.append({"role": "assistant", "content": f"{conv.speaker_b}: {t.assistant}"})
    return msgs


def full_context_text(conv):
    lines = []
    cur_session = None
    for t in conv.turns:
        if t.session_datetime and t.session_datetime != cur_session:
            cur_session = t.session_datetime
            lines.append(f"\n[Session: {cur_session}]")
        if t.user:
            lines.append(f"{conv.speaker_a}: {t.user}")
        if t.assistant:
            lines.append(f"{conv.speaker_b}: {t.assistant}")
    return "\n".join(lines)


def chunk_text(text, size=512, overlap=100):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


# --------------------------------------------------------------------------- #
# Official Mem0 store
# --------------------------------------------------------------------------- #
def build_mem0(conv, faiss_dir):
    from mem0 import Memory
    cfg = {
        "llm": {"provider": "openai", "config": {
            "model": ANSWER_MODEL, "temperature": 0.0,
            "openai_base_url": API_BASE, "api_key": API_KEY}},
        "embedder": {"provider": "openai", "config": {
            "model": EMBED_MODEL,
            "openai_base_url": API_BASE, "api_key": API_KEY}},
        "vector_store": {"provider": "faiss", "config": {
            "collection_name": "anchor", "path": faiss_dir,
            "embedding_model_dims": EMBED_DIM,
            # Mem0's faiss default ("euclidean") returns a raw L2 *distance* as
            # the score, but Mem0's high-level scorer ranks DESCENDING and gates
            # on score>threshold -- i.e. it treats the field as a *similarity*.
            # With euclidean the ranking is therefore inverted and the values
            # saturate the score_and_rank min(.,1.0) cap (all scores -> 1.0,
            # relevant memories pushed to the bottom). "cosine" -> IndexFlatIP
            # returns an inner product; text-embedding-3-small is unit-norm
            # (verified |v|=1.000), so IP == cosine in [0,1], higher==closer --
            # the direction Mem0's scorer expects. This is the correct, standard
            # similarity setup for Mem0 + OpenAI embeddings.
            "distance_strategy": "cosine"}},
    }
    mem = Memory.from_config(cfg)
    uid = f"loco_{conv.id}"
    # Add per-session batches (Mem0 infers facts over each batch) to mirror
    # Mem0's own LoCoMo ingestion while bounding extraction calls. Each batch
    # carries its session timestamp as metadata -- this is Mem0's documented
    # LoCoMo protocol (per-message timestamps fed at write time, surfaced at
    # read time), and it keeps the temporal information parity with the chunk
    # store, whose text carries explicit [Session: <date>] markers.
    batch, cur = [], None
    for t in conv.turns:
        if t.session_datetime != cur and batch:
            mem.add(batch, user_id=uid, infer=True, metadata={"timestamp": cur})
            batch = []
        cur = t.session_datetime
        if t.user:
            batch.append({"role": "user", "content": f"{conv.speaker_a}: {t.user}"})
        if t.assistant:
            batch.append({"role": "assistant", "content": f"{conv.speaker_b}: {t.assistant}"})
    if batch:
        mem.add(batch, user_id=uid, infer=True, metadata={"timestamp": cur})
    return mem, uid


def mem0_context(mem, uid, question, top_k=30):
    res = mem.search(question, filters={"user_id": uid}, top_k=top_k)
    items = res["results"] if isinstance(res, dict) else res
    lines = []
    for it in items:
        m = it.get("memory", "")
        if not m:
            continue
        ts = (it.get("metadata") or {}).get("timestamp")
        lines.append(f"- [{ts}] {m}" if ts else f"- {m}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Verbatim chunk store (same embedder, faiss)
# --------------------------------------------------------------------------- #
def build_chunks(conv, size=512, overlap=100):
    """Per-session verbatim chunks, each tagged with its session timestamp so
    the chunk store carries the SAME per-item temporal information the Mem0
    store gets via metadata={"timestamp": ...}. This mirrors the paper backbone,
    which hands session timestamps to the answerer for every retrieved item;
    embedding is on the raw chunk text, the timestamp is added at render time."""
    raw, stamps = [], []
    cur, buf = None, []

    def flush():
        if not buf:
            return
        for c in chunk_text("\n".join(buf), size, overlap):
            raw.append(c); stamps.append(cur)

    for t in conv.turns:
        if t.session_datetime != cur and buf:
            flush(); buf = []
        cur = t.session_datetime
        if t.user: buf.append(f"{conv.speaker_a}: {t.user}")
        if t.assistant: buf.append(f"{conv.speaker_b}: {t.assistant}")
    flush()

    client = oai()
    embs = []
    for i in range(0, len(raw), 64):
        r = client.embeddings.create(model=EMBED_MODEL, input=raw[i:i + 64])
        embs.extend([d.embedding for d in r.data])
    mat = np.array(embs, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(mat)
    rendered = [f"[{stamps[i]}] {raw[i]}" for i in range(len(raw))]
    return rendered, index


def chunk_context(chunks, index, question, top_k=15):
    client = oai()
    q = np.array([client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding],
                 dtype="float32")
    faiss.normalize_L2(q)
    _, idx = index.search(q, min(top_k, len(chunks)))
    return "\n".join(f"- {chunks[i]}" for i in idx[0] if i >= 0)


# --------------------------------------------------------------------------- #
# Answer + judge
# --------------------------------------------------------------------------- #
def answer(context, question):
    prompt = f"""Based on the following memory context, answer the question.

## Memory Context
{context}

## Question
{question}

## Instructions
1. Identify relevant facts from the context
2. Connect facts if needed for multi-hop reasoning
3. Synthesize a complete answer

## Answer
Provide a concise, direct answer."""
    client = oai()
    a = call_llm_with_retry(client=client, model=ANSWER_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=200, temperature=0, timeout=60,
                            max_retries=5, call_type="gen", verbose=False)
    return a or ""


def judge(ans, gt, q):
    client = oai()
    r = score_locomo_answer_llm(ans, gt, q, client, model=SCORE_MODEL)
    return 1 if r.f1_score == 1.0 else 0


# --------------------------------------------------------------------------- #
def run_conv(conv, workers):
    qas = [qa for qa in conv.qa_pairs if qa.category in (1, 2, 3)]
    t0 = time.time()
    faiss_dir = tempfile.mkdtemp(prefix=f"mem0_{conv.id}_")
    try:
        mem, uid = build_mem0(conv, faiss_dir)
        chunks, index = build_chunks(conv)
        full = full_context_text(conv)

        def one(qa):
            # A single flaky call (e.g. APIConnectionError on the long
            # full-context payload) must not take down the whole run. Skip the
            # question on persistent failure rather than propagating.
            q, gt = qa.question, qa.answer
            try:
                ctx_m = mem0_context(mem, uid, q)
                ctx_c = chunk_context(chunks, index, q)
                v_m = judge(answer(ctx_m, q), gt, q)
                v_c = judge(answer(ctx_c, q), gt, q)
                v_f = judge(answer(full, q), gt, q)
            except Exception as e:  # noqa: BLE001
                print(f"  conv {conv.id}: skip Q ({type(e).__name__}: {e})", flush=True)
                return None
            return {"category": qa.category, "mem0": v_m, "chunks": v_c, "full": v_f}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = [r for r in ex.map(one, qas) if r is not None]
    except Exception as e:  # noqa: BLE001 -- isolate a dead conversation
        print(f"  conv {conv.id}: FAILED ({type(e).__name__}: {e})", flush=True)
        rows = []
    finally:
        shutil.rmtree(faiss_dir, ignore_errors=True)
    print(f"  conv {conv.id}: {len(rows)} Qs in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4, help="per-question parallelism within a conversation")
    ap.add_argument("--conv-workers", type=int, default=5, help="conversations built/answered concurrently")
    ap.add_argument("--data", default=str(ROOT / "experiments/data/locomo10.json"))
    ap.add_argument("--out", default=str(ROOT / "experiments/results/anchor_mem0_official_locomo.json"))
    args = ap.parse_args()

    raw = load_locomo(args.data)
    convs = convert_to_eval_format(raw)[: args.n_conv]
    print(f"Anchor: official Mem0 vs chunks vs full | {len(convs)} convs | "
          f"answerer={ANSWER_MODEL} judge={SCORE_MODEL} embed={EMBED_MODEL}", flush=True)

    def write_summary(rows):
        n = len(rows) or 1
        acc = {k: 100.0 * sum(r[k] for r in rows) / n for k in ("mem0", "chunks", "full")}
        out = {
            "n_questions": len(rows), "n_convs": len(convs),
            "answerer": ANSWER_MODEL, "judge": SCORE_MODEL, "embedder": EMBED_MODEL,
            "accuracy": acc,
            "ordering_holds": acc["mem0"] < acc["chunks"] and acc["mem0"] < acc["full"],
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        return acc

    all_rows = []
    with ThreadPoolExecutor(max_workers=args.conv_workers) as ex:
        for rows in ex.map(lambda c: run_conv(c, args.workers), convs):
            all_rows.extend(rows)
            write_summary(all_rows)  # checkpoint after every conversation

    n = len(all_rows)
    acc = write_summary(all_rows)
    print("\n=== ANCHOR RESULT (LoCoMo Cat1-3, one harness) ===")
    print(f"  Official Mem0 (extracted): {acc['mem0']:.1f}%")
    print(f"  Verbatim chunks         : {acc['chunks']:.1f}%")
    print(f"  Full context (ceiling)  : {acc['full']:.1f}%")
    holds = acc["mem0"] < acc["chunks"] and acc["mem0"] < acc["full"]
    print(f"  n={n}  ordering Mem0<chunks & Mem0<full holds: {holds}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
