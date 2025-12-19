"""
GraphRAG Pilot Test: Run on 5 samples to estimate time and accuracy.

Usage:
    conda activate graphrag
    python experiments/test_graphrag_pilot.py
"""

import json
import time
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Force unbuffered output for nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Constants
PYTHON_PATH = os.path.expanduser("~/opt/anaconda3/envs/graphrag/bin/python")
DATA_PATH = "experiments/data/multihop_eval.json"
NUM_SAMPLES = 50  # Full run

SETTINGS_TEMPLATE = """models:
  default_chat_model:
    type: chat
    model_provider: openai
    auth_type: api_key
    api_key: {api_key}
    api_base: {api_base}
    model: gpt-4o-mini
    model_supports_json: true
    concurrent_requests: 10
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 5
  default_embedding_model:
    type: embedding
    model_provider: openai
    auth_type: api_key
    api_key: {embedding_api_key}
    api_base: {embedding_api_base}
    model: BAAI/bge-m3
    concurrent_requests: 10
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 5

input:
  storage:
    type: file
    base_dir: "input"
  file_type: text

chunks:
  size: 1000
  overlap: 100
  group_by_columns: [id]

output:
  type: file
  base_dir: "output"

cache:
  type: file
  base_dir: "cache"

reporting:
  type: file
  base_dir: "logs"

vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default

embed_text:
  model_id: default_embedding_model
  vector_store_id: default_vector_store

extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [person, technology, decision, fact, organization, concept]
  max_gleanings: 0

summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english
  async_mode: threaded

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false

community_reports:
  model_id: default_chat_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 1500
  max_input_length: 6000

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  embeddings: false

local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  chat_model_id: default_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

basic_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
"""


def setup_graphrag_project(work_dir: str, conversation_text: str) -> bool:
    """Set up a graphrag project with conversation as input."""
    work_path = Path(work_dir)

    # Create input directory first
    (work_path / "input").mkdir(parents=True, exist_ok=True)

    # Initialize graphrag
    result = subprocess.run(
        [PYTHON_PATH, "-m", "graphrag", "init"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=60
    )

    # Write settings
    settings = SETTINGS_TEMPLATE.format(
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
        embedding_api_base=os.getenv("EMBEDDING_API_BASE"),
    )
    (work_path / "settings.yaml").write_text(settings)

    # Write input
    (work_path / "input" / "conversation.txt").write_text(conversation_text)

    return True


def run_graphrag_index(work_dir: str) -> tuple[bool, float]:
    """Run graphrag indexing. Returns (success, time_seconds)."""
    start = time.time()

    try:
        result = subprocess.run(
            [PYTHON_PATH, "-m", "graphrag", "index"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 min max
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            return True, elapsed
        else:
            print(f"  Index failed: {result.stderr[:500]}")
            return False, elapsed

    except subprocess.TimeoutExpired:
        return False, time.time() - start
    except Exception as e:
        print(f"  Index error: {e}")
        return False, time.time() - start


def run_graphrag_query(work_dir: str, question: str) -> tuple[str, float]:
    """Run graphrag query. Returns (answer, time_seconds)."""
    start = time.time()

    try:
        result = subprocess.run(
            [PYTHON_PATH, "-m", "graphrag", "query",
             "--method", "local",
             "--query", question],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            return result.stdout.strip(), elapsed
        else:
            return f"[Query failed: {result.stderr[:200]}]", elapsed

    except Exception as e:
        return f"[Error: {e}]", time.time() - start


def score_answer(answer: str, keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """Score answer by keyword coverage."""
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    missing = [kw for kw in keywords if kw.lower() not in answer_lower]
    coverage = len(found) / len(keywords) if keywords else 0
    return coverage, found, missing


def main():
    print("=" * 60)
    print("GraphRAG Pilot Test")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Python: {PYTHON_PATH}")
    print("=" * 60)

    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)

    conversations = data["conversations"][:NUM_SAMPLES]

    results = []
    total_index_time = 0
    total_query_time = 0

    for i, conv in enumerate(conversations):
        conv_id = conv["id"]
        print(f"\n[{i+1}/{NUM_SAMPLES}] Processing {conv_id}")

        # Prepare conversation text
        turns = conv["turns"]
        conv_text = "\n\n".join([
            f"Turn {t['turn_id']}:\nUser: {t['user']}\nAssistant: {t['assistant']}"
            for t in turns
        ])

        # Create temp directory
        work_dir = tempfile.mkdtemp(prefix=f"graphrag_{conv_id}_")

        try:
            # Setup
            print(f"  Setting up project...")
            setup_graphrag_project(work_dir, conv_text)

            # Index
            print(f"  Running indexing...")
            index_ok, index_time = run_graphrag_index(work_dir)
            total_index_time += index_time
            print(f"  Index: {'OK' if index_ok else 'FAILED'} ({index_time:.1f}s)")

            if not index_ok:
                results.append({
                    "conv_id": conv_id,
                    "index_ok": False,
                    "index_time": index_time,
                    "questions": []
                })
                continue

            # Query each question
            questions_results = []
            for q in conv["questions"]:
                q_id = q["id"]
                question = q["question"]
                keywords = q["ground_truth_keywords"]

                print(f"  Querying: {question[:50]}...")
                answer, query_time = run_graphrag_query(work_dir, question)
                total_query_time += query_time

                coverage, found, missing = score_answer(answer, keywords)
                passed = coverage >= 0.8

                print(f"    Coverage: {coverage:.0%} ({'PASS' if passed else 'FAIL'})")

                questions_results.append({
                    "q_id": q_id,
                    "question": question,
                    "keywords": keywords,
                    "coverage": coverage,
                    "passed": passed,
                    "found": found,
                    "missing": missing,
                    "query_time": query_time,
                    "answer": answer[:500]
                })

            results.append({
                "conv_id": conv_id,
                "index_ok": True,
                "index_time": index_time,
                "questions": questions_results
            })

        finally:
            # Cleanup
            shutil.rmtree(work_dir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_questions = sum(len(r["questions"]) for r in results)
    passed_questions = sum(
        sum(1 for q in r["questions"] if q["passed"])
        for r in results
    )

    print(f"Conversations processed: {len(results)}")
    print(f"Total index time: {total_index_time:.1f}s ({total_index_time/len(results):.1f}s avg)")
    print(f"Total query time: {total_query_time:.1f}s")
    print(f"Questions: {passed_questions}/{total_questions} passed ({100*passed_questions/total_questions:.1f}%)")

    # Estimate full run
    full_convs = len(data["conversations"])
    est_time = (total_index_time / len(results)) * full_convs
    print(f"\nEstimated full run ({full_convs} convs): {est_time/3600:.1f} hours")

    # Save results
    output_path = "experiments/results/graphrag_full.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_samples": NUM_SAMPLES,
            "total_index_time": total_index_time,
            "total_query_time": total_query_time,
            "pass_rate": passed_questions / total_questions if total_questions else 0,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
