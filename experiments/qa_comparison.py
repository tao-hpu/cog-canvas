"""
QA Comparison: CogCanvas vs Summarization on Next.js RFC

This script runs a qualitative comparison for the paper's case study table.
It asks discriminative questions that require specific details from the discussion.

Questions are designed so that:
- CogCanvas can provide specific citations (Turn X)
- Summarization will give vague or incomplete answers
- Truncation will fail completely (doesn't have early turns)

Usage:
    python -m experiments.qa_comparison --output experiments/results/qa_comparison.json
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Set OpenAI-compatible API configuration
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

from cogcanvas import Canvas, ObjectType
from openai import OpenAI
from experiments.github_issue_case_study import GITHUB_DISCUSSION_TURNS

# ============================================================================
# QA Questions - Designed for Maximum Discriminative Power
# ============================================================================

QA_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What specific decision was made about minimalMode?",
        "category": "Specific Decision",
        "expected_answer": "minimalMode will be deprecated in favor of the new handler architecture",
        "source_turn": 2,
        "why_discriminative": "Requires remembering a specific technical decision from early discussion",
    },
    {
        "id": "Q2",
        "question": "How does fallbackID work in the adapter API?",
        "category": "Technical Detail",
        "expected_answer": "fallbackID always references STATIC_FILE",
        "source_turn": 8,
        "why_discriminative": "Requires precise technical knowledge that summarization would lose",
    },
    {
        "id": "Q3",
        "question": "Why won't the adapter API be backported to Next.js 14?",
        "category": "Reasoning",
        "expected_answer": "It requires big refactors that are not feasible for 14.x",
        "source_turn": 8,
        "why_discriminative": "Requires remembering the reasoning behind a decision",
    },
    {
        "id": "Q4",
        "question": "List all TODO items related to routing documentation.",
        "category": "Todo List",
        "expected_answer": "1) Document routing behavior specification, 2) Document middleware matcher behavior, 3) Complete routing specification documentation",
        "source_turn": [3, 7, 11],
        "why_discriminative": "Requires aggregating information from multiple turns",
    },
    {
        "id": "Q5",
        "question": "What architectural preference did Deno Deploy express about entrypoints?",
        "category": "Stakeholder Preference",
        "expected_answer": "Deno Deploy prefers a singular entrypoint over multiple ones, as their serverless architecture would benefit from a unified entry",
        "source_turn": 4,
        "why_discriminative": "Requires remembering a specific stakeholder's preference",
    },
]


def initialize_canvas():
    """Initialize CogCanvas with extracted artifacts from RFC."""
    model = os.getenv("EXTRACTOR_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
    canvas = Canvas(extractor_model=model)

    print("Processing RFC turns through CogCanvas...")
    for i, turn in enumerate(GITHUB_DISCUSSION_TURNS, 1):
        try:
            result = canvas.extract(user=turn["user"], assistant=turn["assistant"])
            print(f"  Turn {i}: {result.count} objects extracted")
        except Exception as e:
            print(f"  Turn {i}: Error - {e}")

    stats = canvas.stats()
    print(f"\nTotal objects: {stats['total_objects']}")
    return canvas


def generate_summary(turns: list[dict]) -> str:
    """Generate a summary of all turns using LLM."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
    )

    # Combine all turns into a single text
    full_text = ""
    for i, turn in enumerate(turns, 1):
        full_text += f"\n--- Turn {i} ---\nUser: {turn['user']}\nAssistant: {turn['assistant']}\n"

    response = client.chat.completions.create(
        model=os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Summarize the following technical discussion into a concise summary that captures the main points, decisions, and action items. Keep it under 500 words.",
            },
            {
                "role": "user",
                "content": f"Please summarize this discussion:\n{full_text}",
            },
        ],
        max_tokens=600,
        temperature=0.3,
    )

    return response.choices[0].message.content


def answer_with_cogcanvas(canvas: Canvas, question: str) -> dict:
    """Answer question using CogCanvas retrieve + inject."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
    )

    # Retrieve relevant objects
    result = canvas.retrieve(question, top_k=5, include_related=True)

    # Inject into prompt
    context = canvas.inject(result, format="compact", max_tokens=800)

    # Build citations info
    citations = []
    for obj in result.objects:
        citations.append(
            {"type": obj.type.value, "content": obj.content[:100], "turn": obj.turn_id}
        )

    # Generate answer
    response = client.chat.completions.create(
        model=os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Answer the question based on the provided context. Be specific and cite relevant turns when possible. If you find specific information, quote it.",
            },
            {
                "role": "user",
                "content": f"Context from conversation:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "citations": citations,
        "objects_retrieved": result.count,
    }


def answer_with_summary(summary: str, question: str) -> dict:
    """Answer question using only the summary."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
    )

    response = client.chat.completions.create(
        model=os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Answer the question based on the provided summary. Be specific if you can. If the information is not available in the summary, say so.",
            },
            {
                "role": "user",
                "content": f"Summary of discussion:\n{summary}\n\nQuestion: {question}",
            },
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return {"answer": response.choices[0].message.content, "context_type": "summary"}


def answer_with_truncation(
    turns: list[dict], question: str, keep_last: int = 3
) -> dict:
    """Answer question with only the last N turns (simulating truncation)."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
    )

    # Only keep last N turns
    recent_turns = turns[-keep_last:]
    context = ""
    for i, turn in enumerate(recent_turns):
        turn_num = len(turns) - keep_last + i + 1
        context += f"\n--- Turn {turn_num} ---\nUser: {turn['user']}\nAssistant: {turn['assistant']}\n"

    response = client.chat.completions.create(
        model=os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Answer the question based on the provided conversation context. If the information is not available, say so.",
            },
            {
                "role": "user",
                "content": f"Recent conversation:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "turns_available": keep_last,
        "context_type": "truncation",
    }


def run_comparison():
    """Run the full QA comparison experiment."""
    print("=" * 80)
    print("QA COMPARISON: CogCanvas vs Summarization vs Truncation")
    print("=" * 80)
    print()

    # Initialize CogCanvas
    print("Step 1: Initializing CogCanvas...")
    canvas = initialize_canvas()
    print()

    # Generate summary
    print("Step 2: Generating summary for Summarization baseline...")
    summary = generate_summary(GITHUB_DISCUSSION_TURNS)
    print(f"Summary length: {len(summary)} chars")
    print(f"Summary preview: {summary[:200]}...")
    print()

    # Run QA for each question
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_turns": len(GITHUB_DISCUSSION_TURNS),
            "num_questions": len(QA_QUESTIONS),
            "methods": ["cogcanvas", "summarization", "truncation"],
        },
        "summary_generated": summary,
        "questions": [],
    }

    print("Step 3: Running QA comparison...")
    print("-" * 80)

    for q in QA_QUESTIONS:
        print(f"\n{q['id']}: {q['question']}")
        print(f"  Category: {q['category']}")
        print(f"  Source Turn(s): {q['source_turn']}")

        # CogCanvas answer
        print("  [CogCanvas] Answering...")
        cogcanvas_result = answer_with_cogcanvas(canvas, q["question"])

        # Summarization answer
        print("  [Summarization] Answering...")
        summary_result = answer_with_summary(summary, q["question"])

        # Truncation answer
        print("  [Truncation] Answering...")
        truncation_result = answer_with_truncation(
            GITHUB_DISCUSSION_TURNS, q["question"], keep_last=3
        )

        results["questions"].append(
            {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "expected_answer": q["expected_answer"],
                "source_turn": q["source_turn"],
                "cogcanvas": cogcanvas_result,
                "summarization": summary_result,
                "truncation": truncation_result,
            }
        )

        # Print preview
        print(f"  [CogCanvas] {cogcanvas_result['answer'][:100]}...")
        print(f"  [Summarization] {summary_result['answer'][:100]}...")
        print(f"  [Truncation] {truncation_result['answer'][:100]}...")

    return results


def generate_markdown_table(results: dict) -> str:
    """Generate a markdown table for the paper."""
    table = """
## QA Comparison: Next.js RFC Case Study

| Question | Category | CogCanvas | Summarization | Truncation |
|----------|----------|-----------|---------------|------------|
"""

    for q in results["questions"]:
        # Truncate answers for table
        cog_ans = q["cogcanvas"]["answer"][:150].replace("\n", " ").replace("|", "\\|")
        sum_ans = (
            q["summarization"]["answer"][:150].replace("\n", " ").replace("|", "\\|")
        )
        trunc_ans = (
            q["truncation"]["answer"][:150].replace("\n", " ").replace("|", "\\|")
        )

        # Add ... if truncated
        if len(q["cogcanvas"]["answer"]) > 150:
            cog_ans += "..."
        if len(q["summarization"]["answer"]) > 150:
            sum_ans += "..."
        if len(q["truncation"]["answer"]) > 150:
            trunc_ans += "..."

        question_short = q["question"][:50] + ("..." if len(q["question"]) > 50 else "")

        table += f"| **{q['id']}**: {question_short} | {q['category']} | {cog_ans} | {sum_ans} | {trunc_ans} |\n"

    return table


def generate_latex_table(results: dict) -> str:
    """Generate a LaTeX table for the paper."""
    latex = r"""
\begin{table*}[t]
\centering
\caption{Qualitative QA Comparison on Next.js RFC Discussion (11 turns, 68 responses)}
\label{tab:qa-comparison}
\small
\begin{tabular}{p{3cm}p{4cm}p{4cm}p{4cm}}
\toprule
\textbf{Question} & \textbf{CogCanvas} & \textbf{Summarization} & \textbf{Truncation} \\
\midrule
"""

    for q in results["questions"]:
        # Truncate and escape for LaTeX
        def escape_latex(s):
            return (
                s.replace("_", r"\_")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("#", r"\#")
            )

        question_short = escape_latex(
            q["question"][:40] + ("..." if len(q["question"]) > 40 else "")
        )
        cog_ans = escape_latex(
            q["cogcanvas"]["answer"][:120]
            + ("..." if len(q["cogcanvas"]["answer"]) > 120 else "")
        )
        sum_ans = escape_latex(
            q["summarization"]["answer"][:120]
            + ("..." if len(q["summarization"]["answer"]) > 120 else "")
        )
        trunc_ans = escape_latex(
            q["truncation"]["answer"][:120]
            + ("..." if len(q["truncation"]["answer"]) > 120 else "")
        )

        latex += (
            f"{question_short} & {cog_ans} & {sum_ans} & {trunc_ans} \\\\\n\\midrule\n"
        )

    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description="Run QA comparison experiment")
    parser.add_argument(
        "--output",
        "-o",
        default="experiments/results/qa_comparison.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        default="experiments/results/qa_comparison_table.md",
        help="Output markdown table file path",
    )
    parser.add_argument(
        "--latex",
        "-l",
        default="experiments/results/qa_comparison_table.tex",
        help="Output LaTeX table file path",
    )

    args = parser.parse_args()

    # Run comparison
    results = run_comparison()

    # Save JSON results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_path}")

    # Generate and save markdown table
    md_table = generate_markdown_table(results)
    md_path = Path(args.markdown)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"✓ Markdown table saved to: {md_path}")

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    latex_path = Path(args.latex)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("QA COMPARISON SUMMARY")
    print("=" * 80)

    print("\nQuestions and Expected vs Actual:")
    for q in results["questions"]:
        print(f"\n{q['id']}: {q['question']}")
        print(f"  Expected: {q['expected_answer']}")
        print(f"  CogCanvas: {q['cogcanvas']['answer'][:200]}...")
        if "citations" in q["cogcanvas"]:
            turns = [c["turn"] for c in q["cogcanvas"]["citations"]]
            print(f"  (Citations from turns: {turns})")

    print("\n" + "-" * 80)
    print("MARKDOWN TABLE PREVIEW:")
    print("-" * 80)
    print(md_table)


if __name__ == "__main__":
    main()
