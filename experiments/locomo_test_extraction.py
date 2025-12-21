"""
Deep diagnosis: Check what was actually extracted for a specific LoCoMo conversation.
Compares extracted artifacts against ground truth questions.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from cogcanvas import Canvas
from experiments.locomo_adapter import load_locomo, convert_to_eval_format


def load_conversation(conv_id: str = "locomo_000") -> Dict[str, Any]:
    """Load a specific conversation from LoCoMo."""
    raw_data = load_locomo("experiments/data/locomo10.json")
    conversations = convert_to_eval_format(raw_data)
    for conv in conversations:
        if conv.id == conv_id:
            return {
                "id": conv.id,
                "turns": [{"user": t.user, "assistant": t.assistant, "session_datetime": getattr(t, "session_datetime", None)} for t in conv.turns],
                "questions": [{"question": q.question, "answer": q.answer, "category": q.category_name} for q in conv.qa_pairs],
            }
    raise ValueError(f"Conversation {conv_id} not found")


def extract_all_artifacts(conversation: Dict[str, Any], max_turns: int = 50) -> List[Dict]:
    """Run extraction on turns and collect artifacts."""
    canvas = Canvas()

    turns = conversation["turns"][:max_turns]  # Limit turns for faster testing
    print(f"Processing {len(turns)} turns (limited to {max_turns})...")

    artifacts = []
    for i, turn in enumerate(turns):
        user_msg = turn.get("user", "")
        assistant_msg = turn.get("assistant", "")
        session_datetime = turn.get("session_datetime", None)

        if not user_msg and not assistant_msg:
            continue

        result = canvas.extract(
            user=user_msg,
            assistant=assistant_msg,
            session_datetime=session_datetime
        )

        for obj in result.objects:
            artifacts.append({
                "turn": i,
                "type": obj.type.value,
                "content": obj.content,
                "quote": obj.quote[:100] if obj.quote else "",
                "time_raw": obj.event_time_raw,
                "time_resolved": obj.event_time,  # Resolved absolute time
                "session_datetime": session_datetime,
            })

    return artifacts, canvas


def check_coverage(artifacts: List[Dict], questions: List[Dict]) -> Dict:
    """Check which ground truth answers are covered by extracted artifacts."""

    results = {
        "covered": [],
        "missing": [],
        "partial": [],
    }

    # Flatten all artifact content INCLUDING resolved times
    all_content = " ".join([a["content"].lower() for a in artifacts])
    all_quotes = " ".join([a["quote"].lower() for a in artifacts])
    all_times = " ".join([
        (a.get("time_resolved") or a.get("time_raw") or "").lower()
        for a in artifacts
    ])
    combined = all_content + " " + all_quotes + " " + all_times

    for q in questions[:20]:  # Check first 20 questions
        gt = q["answer"].lower()
        gt_keywords = set(gt.split())

        # Check coverage
        found_keywords = [kw for kw in gt_keywords if kw in combined]
        coverage = len(found_keywords) / len(gt_keywords) if gt_keywords else 0

        item = {
            "question": q["question"],
            "category": q.get("category", "unknown"),
            "ground_truth": q["answer"],
            "coverage": coverage,
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in gt_keywords if kw not in combined],
        }

        if coverage >= 0.8:
            results["covered"].append(item)
        elif coverage >= 0.3:
            results["partial"].append(item)
        else:
            results["missing"].append(item)

    return results


def main():
    conv_id = sys.argv[1] if len(sys.argv) > 1 else "locomo_000"

    print(f"=" * 60)
    print(f"DEEP DIAGNOSIS: {conv_id}")
    print(f"=" * 60)

    # Load conversation
    conv = load_conversation(conv_id)
    print(f"Conversation: {conv['id']}")
    print(f"Turns: {len(conv['turns'])}")
    print(f"Questions: {len(conv['questions'])}")

    # Extract artifacts
    print(f"\n{'='*60}")
    print("EXTRACTING ARTIFACTS...")
    print(f"{'='*60}")

    artifacts, canvas = extract_all_artifacts(conv)
    print(f"\nExtracted {len(artifacts)} artifacts")

    # Show artifact summary by type
    type_counts = {}
    for a in artifacts:
        type_counts[a["type"]] = type_counts.get(a["type"], 0) + 1

    print("\nArtifacts by type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # Show time-related artifacts
    time_artifacts = [a for a in artifacts if a.get("time_raw") or a.get("time_resolved")]
    print(f"\nTime-related artifacts: {len(time_artifacts)}")
    for a in time_artifacts[:10]:
        raw = a.get('time_raw', '')
        resolved = a.get('time_resolved', '')
        time_info = f"{raw}" + (f" -> {resolved}" if resolved and resolved != raw else "")
        print(f"  Turn {a['turn']}: {time_info}")
        print(f"    Content: {a['content'][:70]}...")

    # Check coverage against questions
    print(f"\n{'='*60}")
    print("COVERAGE ANALYSIS")
    print(f"{'='*60}")

    coverage = check_coverage(artifacts, conv["questions"])

    print(f"\nCovered (≥80%): {len(coverage['covered'])}")
    print(f"Partial (30-80%): {len(coverage['partial'])}")
    print(f"Missing (<30%): {len(coverage['missing'])}")

    # Show missing items
    print(f"\n{'='*60}")
    print("MISSING GROUND TRUTH (not extracted)")
    print(f"{'='*60}")

    for item in coverage["missing"][:10]:
        print(f"\n[{item['category']}] Q: {item['question']}")
        print(f"  GT: {item['ground_truth']}")
        print(f"  Missing: {item['missing_keywords'][:5]}")

    # Show partial items
    print(f"\n{'='*60}")
    print("PARTIAL COVERAGE (partially extracted)")
    print(f"{'='*60}")

    for item in coverage["partial"][:5]:
        print(f"\n[{item['category']}] Q: {item['question']}")
        print(f"  GT: {item['ground_truth']}")
        print(f"  Found: {item['found_keywords'][:5]}")
        print(f"  Missing: {item['missing_keywords'][:5]}")

    # Sample of actual artifacts
    print(f"\n{'='*60}")
    print("SAMPLE EXTRACTED ARTIFACTS (first 20)")
    print(f"{'='*60}")

    for a in artifacts[:20]:
        print(f"\n[Turn {a['turn']}] [{a['type']}]")
        print(f"  Content: {a['content'][:80]}")
        if a['quote']:
            print(f"  Quote: \"{a['quote'][:60]}...\"")
        if a.get('time_raw') or a.get('time_resolved'):
            raw = a.get('time_raw', '')
            resolved = a.get('time_resolved', '')
            if resolved and resolved != raw:
                print(f"  Time: {raw} -> {resolved}")
            elif raw:
                print(f"  Time: {raw}")


if __name__ == "__main__":
    main()
