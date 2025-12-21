"""
Diagnose LoCoMo failures: Extraction vs Retrieval vs Reasoning.

This script analyzes failed questions to identify bottlenecks.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import re


def load_results(path: str) -> dict:
    """Load experiment results."""
    with open(path) as f:
        return json.load(f)


def analyze_failure_patterns(data: dict) -> dict:
    """Analyze failure patterns across all questions."""
    stats = {
        "total_questions": 0,
        "passed": 0,
        "failed": 0,
        "by_category": defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0}),
        "failure_types": defaultdict(int),
        "missing_keyword_types": defaultdict(int),
        "answer_patterns": defaultdict(int),
        "failed_examples": [],
    }

    for conv in data["conversations"]:
        for q in conv["questions"]:
            stats["total_questions"] += 1
            cat = q["category_name"]
            stats["by_category"][cat]["total"] += 1

            if q["passed"]:
                stats["passed"] += 1
                stats["by_category"][cat]["passed"] += 1
            else:
                stats["failed"] += 1
                stats["by_category"][cat]["failed"] += 1

                # Analyze failure type
                answer = q["answer"].strip().lower()
                gt = q["ground_truth"].strip().lower()
                missing = q["missing_keywords"]

                # Categorize failure
                if not answer or answer in ["i don't have enough information.", "i don't have enough information"]:
                    failure_type = "NO_INFO"  # Agent said no info - likely extraction/retrieval issue
                elif "i don't" in answer or "no information" in answer or "not mentioned" in answer:
                    failure_type = "CLAIMED_NO_INFO"  # Similar
                elif q["keyword_overlap"] > 0.3:
                    failure_type = "PARTIAL_MATCH"  # Got some keywords, reasoning issue
                elif q["keyword_overlap"] > 0:
                    failure_type = "LOW_OVERLAP"  # Got few keywords
                else:
                    failure_type = "TOTAL_MISS"  # Zero overlap

                stats["failure_types"][failure_type] += 1

                # Analyze missing keywords
                for kw in missing:
                    if re.match(r'\d{4}', kw):  # Year
                        stats["missing_keyword_types"]["year"] += 1
                    elif re.match(r'\d+', kw):  # Number
                        stats["missing_keyword_types"]["number"] += 1
                    elif kw in ["january", "february", "march", "april", "may", "june",
                               "july", "august", "september", "october", "november", "december"]:
                        stats["missing_keyword_types"]["month"] += 1
                    else:
                        stats["missing_keyword_types"]["content_word"] += 1

                # Store example
                if len(stats["failed_examples"]) < 20:
                    stats["failed_examples"].append({
                        "conv_id": conv["id"],
                        "category": cat,
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                        "answer": q["answer"][:200],
                        "keyword_overlap": q["keyword_overlap"],
                        "missing_keywords": missing[:5],
                        "failure_type": failure_type,
                    })

    return stats


def diagnose_bottleneck(stats: dict) -> str:
    """Diagnose the main bottleneck based on failure patterns."""

    failure_types = stats["failure_types"]
    total_failed = stats["failed"]

    if total_failed == 0:
        return "No failures to analyze!"

    # Calculate percentages
    no_info_pct = (failure_types["NO_INFO"] + failure_types["CLAIMED_NO_INFO"]) / total_failed * 100
    partial_pct = failure_types["PARTIAL_MATCH"] / total_failed * 100
    total_miss_pct = (failure_types["TOTAL_MISS"] + failure_types["LOW_OVERLAP"]) / total_failed * 100

    # Analyze missing keyword types
    kw_types = stats["missing_keyword_types"]
    total_missing = sum(kw_types.values())
    temporal_pct = (kw_types.get("year", 0) + kw_types.get("month", 0)) / max(total_missing, 1) * 100

    diagnosis = []
    diagnosis.append("=" * 60)
    diagnosis.append("BOTTLENECK DIAGNOSIS")
    diagnosis.append("=" * 60)

    # Main bottleneck
    if no_info_pct > 50:
        diagnosis.append(f"\n🔴 PRIMARY BOTTLENECK: EXTRACTION/RETRIEVAL ({no_info_pct:.1f}% said 'no info')")
        diagnosis.append("   → Agent cannot find relevant information")
        diagnosis.append("   → Either facts not extracted, or retrieval fails to find them")
    elif partial_pct > 30:
        diagnosis.append(f"\n🟡 PRIMARY BOTTLENECK: REASONING ({partial_pct:.1f}% partial matches)")
        diagnosis.append("   → Agent found some info but failed to reason correctly")
        diagnosis.append("   → May need better CoT prompting or more context")
    else:
        diagnosis.append(f"\n🟠 PRIMARY BOTTLENECK: MIXED")
        diagnosis.append(f"   → No info: {no_info_pct:.1f}%")
        diagnosis.append(f"   → Partial match: {partial_pct:.1f}%")
        diagnosis.append(f"   → Total miss: {total_miss_pct:.1f}%")

    # Temporal analysis
    if temporal_pct > 20:
        diagnosis.append(f"\n⏰ TEMPORAL WEAKNESS: {temporal_pct:.1f}% of missing keywords are dates/months")
        diagnosis.append("   → Dates are not being extracted or retrieved properly")

    # Category breakdown
    diagnosis.append("\n" + "-" * 60)
    diagnosis.append("ACCURACY BY CATEGORY:")
    for cat, data in sorted(stats["by_category"].items()):
        acc = data["passed"] / data["total"] * 100 if data["total"] > 0 else 0
        diagnosis.append(f"  {cat:12s}: {acc:5.1f}% ({data['passed']}/{data['total']})")

    return "\n".join(diagnosis)


def print_failure_examples(stats: dict, n: int = 10):
    """Print example failures for manual inspection."""
    print("\n" + "=" * 60)
    print(f"FAILURE EXAMPLES (showing {n})")
    print("=" * 60)

    for i, ex in enumerate(stats["failed_examples"][:n]):
        print(f"\n[{i+1}] {ex['conv_id']} | {ex['category']} | {ex['failure_type']}")
        print(f"Q: {ex['question']}")
        print(f"GT: {ex['ground_truth']}")
        print(f"A: {ex['answer'][:150]}...")
        print(f"Overlap: {ex['keyword_overlap']:.1%} | Missing: {ex['missing_keywords']}")
        print("-" * 40)


def main():
    # Default to latest cogcanvas result
    result_path = "experiments/results/locomo_cogcanvas_4omini.json"

    if len(sys.argv) > 1:
        result_path = sys.argv[1]

    if not Path(result_path).exists():
        print(f"Error: {result_path} not found")
        sys.exit(1)

    print(f"Analyzing: {result_path}\n")

    data = load_results(result_path)
    stats = analyze_failure_patterns(data)

    # Print summary
    print("=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Passed: {stats['passed']} ({stats['passed']/stats['total_questions']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total_questions']*100:.1f}%)")

    print("\n" + "-" * 60)
    print("FAILURE TYPE BREAKDOWN:")
    for ftype, count in sorted(stats["failure_types"].items(), key=lambda x: -x[1]):
        pct = count / stats["failed"] * 100 if stats["failed"] > 0 else 0
        print(f"  {ftype:20s}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "-" * 60)
    print("MISSING KEYWORD TYPES:")
    for ktype, count in sorted(stats["missing_keyword_types"].items(), key=lambda x: -x[1]):
        print(f"  {ktype:15s}: {count:4d}")

    # Diagnosis
    print("\n" + diagnose_bottleneck(stats))

    # Examples
    print_failure_examples(stats, n=10)


if __name__ == "__main__":
    main()
