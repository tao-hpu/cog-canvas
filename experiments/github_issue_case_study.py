"""
Case Study: CogCanvas on Real GitHub Discussion

This script demonstrates CogCanvas's ability to extract structured artifacts
from a complex, multi-party GitHub Discussion (Next.js RFC #77740).

Purpose: Qualitative evaluation for the paper - show "messy input vs clean output"

Usage:
    python -m experiments.github_issue_case_study --output experiments/results/case_study_nextjs.json
    python -m experiments.github_issue_case_study --preview
"""

import sys
import os
import argparse
from pathlib import Path

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
from cogcanvas.llm import get_backend
import json
from datetime import datetime

# ============================================================================
# RAW DATA: Next.js RFC #77740 - Deployment Adapters API
# Source: https://github.com/vercel/next.js/discussions/77740
# 68 total responses, multiple stakeholders
# ============================================================================

GITHUB_DISCUSSION_TURNS = [
    # Turn 1: Original RFC post by @leerob
    {
        "user": """RFC: Deployment Adapters API

We're introducing a Deployment Adapters API to enable easier deployment across platforms.
Vercel will use the same adapter API as every other partner.

Key Pain Points:
1. Background Work Tracking - Currently requires reverse-engineering request lifecycles
2. Config Modification - Providers must patch next.config or use undocumented env vars
3. Manifest Reliance - Undocumented manifests create fragility
4. Full Server Dependency - Entrypoints require loading entire next-server, causing cold boot issues

Build Output Changes:
- Node.js: handler(req, res, ctx) returning Promise<void>
- Edge: handler(req, ctx) returning Promise<Response>
- waitUntil callback signals background task completion

Adapter API Interface:
NextAdapter {
  name: string
  modifyConfig?(config): NextConfig
  onBuildComplete?(data): Promise<void>
}

Target: Alpha in Next.js 16 this summer.""",
        "assistant": "This RFC proposes a standardized deployment adapter API for Next.js. Key decisions: 1) Vercel uses same adapter API as partners (parity), 2) New handler signatures for Node.js and Edge, 3) waitUntil for background tasks, 4) Target alpha in Next.js 16.",
    },
    # Turn 2: @mhart asking about deprecation
    {
        "user": """@mhart: Can you clarify what happens to minimalMode?
Also, what about Vercel-specific features like x-matched-path and x-now-route-matches?
Will these become documented or refactored as generic capabilities?""",
        "assistant": """@ijjk (Vercel): minimalMode effectively goes away with the refactored handler interfaces.
The new approach allows independent entry invocation without loading the full server.
Decision: minimalMode will be deprecated in favor of new handler architecture.""",
    },
    # Turn 3: @ascorbic questioning routing
    {
        "user": """@ascorbic: If there are separate entrypoints, does this mean adapters need to implement custom routing logic?
The routing rules have historically been very lightly documented, causing lots of reverse-engineering overhead.
This is a critical concern for platform providers like Netlify.""",
        "assistant": "This raises an important open question: Will adapters need to implement full end-to-end routing? This has historically been the biggest barrier for platform providers. TODO: Document routing behavior specification.",
    },
    # Turn 4: @lucacasonato from Deno
    {
        "user": """@lucacasonato (Deno Deploy): I'd prefer a singular entrypoint over multiple ones.
Also interested in CDN cache integration hooks - currently these are flagged by minimal mode.
Deno's serverless architecture would benefit from a unified entry.""",
        "assistant": """@ijjk: next-server remains for routing resolution before invoking the correct entrypoint.
This enables more optimized deployments while keeping routing logic centralized.
Key fact: Routing will still be handled by next-server, adapters invoke specific entrypoints.""",
    },
    # Turn 5: Community questions about OpenNext
    {
        "user": """@nilsingwersen:
1. Will Vercel develop adapters directly?
2. Will you incorporate OpenNext work?
3. What about image optimization provider flexibility?""",
        "assistant": """@serhalp (Netlify, Jul 19):
- Decision: Vercel will dogfood via their own adapter
- Key fact: OpenNext implementations must be rewritten for the new API
- Image providers are configurable via existing next.config images API
This confirms the adapter approach will be used internally at Vercel too.""",
    },
    # Turn 6: PPR discussion - critical gap
    {
        "user": """@RaeesBhatti (Tilda.net): Partial Prerendering (PPR) is a critical missing feature for non-Vercel platforms.
I propose a Progressive Rendering Format standard for CDN-friendly PPR implementation.
Currently only Vercel can properly leverage PPR. This creates platform lock-in.""",
        "assistant": "Key insight: PPR parity across platforms is identified as a critical gap. Currently only Vercel supports PPR properly. TODO: Address PPR for non-Vercel platforms to prevent platform lock-in.",
    },
    # Turn 7: Comprehensive Netlify feedback
    {
        "user": """@serhalp (Netlify, Jul 25): Comprehensive feedback on gaps:
- Missing middleware matcher documentation
- Underspecified pathname format (suggests URLPattern standard)
- Insufficient routing behavior specification
- Unclear fallbackID handling for dynamic routes
- Ambiguous IMAGE type pathname mapping

Critical Question: Will adapters need to implement full end-to-end routing?
This has historically been the biggest barrier for platform providers.""",
        "assistant": """Netlify identifies multiple documentation and specification gaps:
TODO: Document middleware matcher behavior
TODO: Specify pathname format (consider URLPattern standard)
TODO: Document routing behavior specification
TODO: Clarify fallbackID handling
TODO: Clarify IMAGE type pathname mapping
Key concern: Routing complexity remains the biggest barrier for platform providers.""",
    },
    # Turn 8: Vercel response to Netlify
    {
        "user": "What is Vercel's detailed response to these routing and documentation concerns?",
        "assistant": """@ijjk (Aug 5) detailed response with key decisions:
- Decision: maxDuration/expiration/revalidate will be documented
- Decision: fallbackID always references STATIC_FILE
- Decision: allowQuery helps generate stable ISR cache keys
- Decision: next-server remains but much slimmer; adapters route at CDN/edge level
- Decision: No backport to 14.x (requires big refactors)
- Decision: Node.js signature kept matching IncomingMessage/ServerResponse for compatibility
- Decision: Considering @next-community/adapter-* namespace for community adapters
- Key fact: Undocumented private APIs will be removed with documented alternatives and lead time
- Key fact: Assets include all filesystem reads; tracing bugs will be fixed""",
    },
    # Turn 9: Beta docs announcement
    {
        "user": """@threepointone (Oct 11): Beta documentation is now available at:
https://nextjs.org/docs/beta/app/api-reference/config/next-config-js/adapterPath
This marks a significant milestone.""",
        "assistant": "Key fact: Beta documentation published at nextjs.org/docs/beta. This marks progress toward the alpha release in Next.js 16.",
    },
    # Turn 10: Additional community feedback
    {
        "user": """Community requests summary:
- @anicolaides: Need adapter-level HTTP header customization (e.g., managing unsupported stale-while-revalidate)
- @bestickley: Request customizable image optimization caching locations
- @OlegLustenko: Advocated for Docker/environment variable friendliness
- @hamzaah3: Suggested optional lifecycle hooks (onPreBuild, onPostOutput) for CI/CD integration""",
        "assistant": """Multiple community feature requests captured:
TODO: Add adapter-level HTTP header customization capability
TODO: Allow customizable image optimization caching locations
TODO: Improve Docker/environment variable friendliness
TODO: Consider optional lifecycle hooks (onPreBuild, onPostOutput)
These represent community priorities for the adapter API.""",
    },
    # Turn 11: Status update
    {
        "user": "What is the current status of this RFC as of late 2025?",
        "assistant": """Current Status Summary (Oct 2025):
Key facts:
- Beta documentation is live
- Adapters API in alpha as of Next.js 16
- Official adapter implementations pending for Netlify, Cloudflare, AWS
- Release timeline: several months away
- Community builders exploring early adoption

Remaining TODOs:
- Complete routing specification documentation
- Finalize PPR support for non-Vercel platforms
- Ship official adapters for major platforms""",
    },
]


def run_case_study():
    """Run CogCanvas on the GitHub Discussion and analyze results."""

    print("=" * 80)
    print("COGCANVAS CASE STUDY: Next.js RFC #77740 - Deployment Adapters API")
    print("=" * 80)
    print(f"\nSource: https://github.com/vercel/next.js/discussions/77740")
    print(f"Discussion Size: 68 responses from multiple stakeholders")
    print(
        f"Stakeholders: Vercel, Netlify, Deno Deploy, Cloudflare, Tilda.net, Community"
    )
    print()

    # Check API configuration
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    print(f"API Base: {api_base}")
    print(f"API Key: {'*' * 20}...{api_key[-8:] if len(api_key) > 8 else 'NOT SET'}")

    # Initialize Canvas with OpenAI backend
    # Use gpt-4o-mini for cost efficiency
    model = os.getenv("MODEL_DEFAULT", "gpt-4o-mini")  # Use weak model for extraction
    print(f"Extraction Model: {model}")
    print()

    try:
        canvas = Canvas(extractor_model=model)
        print("✓ Canvas initialized with LLM backend")
    except Exception as e:
        print(f"✗ Failed to initialize LLM backend: {e}")
        print("  Falling back to Mock backend...")
        canvas = Canvas()

    # Process each turn
    print("\n" + "-" * 80)
    print("PHASE 1: Extracting Artifacts from Discussion")
    print("-" * 80)

    all_objects = []
    for i, turn in enumerate(GITHUB_DISCUSSION_TURNS, 1):
        try:
            result = canvas.extract(user=turn["user"], assistant=turn["assistant"])
            print(f"\nTurn {i}: Extracted {result.count} objects")
            for obj in result.objects:
                all_objects.append(obj)
                print(f"  [{obj.type.value.upper():10}] {obj.content[:70]}...")
        except Exception as e:
            print(f"\nTurn {i}: Error - {e}")

    # Show statistics
    print("\n" + "-" * 80)
    print("PHASE 2: Canvas Statistics")
    print("-" * 80)

    stats = canvas.stats()
    print(f"\nTotal Objects Extracted: {stats['total_objects']}")
    print(f"Conversation Turns: {stats['turn_count']}")
    print(f"\nObjects by Type:")
    for obj_type, count in stats["by_type"].items():
        print(f"  {obj_type.upper():12}: {count}")

    # Demonstrate retrieval capabilities
    print("\n" + "-" * 80)
    print("PHASE 3: Semantic Retrieval Demo")
    print("-" * 80)

    queries = [
        "What technical decisions were made about the API design?",
        "What are the unresolved issues or gaps?",
        "What needs to be done before release?",
        "What are the concerns from platform providers?",
    ]

    for query in queries:
        print(f'\nQuery: "{query}"')
        result = canvas.retrieve(query, top_k=3)
        print(f"  Found {result.count} relevant objects:")
        for obj, score in zip(result.objects, result.scores):
            print(
                f"    [{obj.type.value.upper():10}] (score: {score:.2f}) {obj.content[:60]}..."
            )

    # List all objects by type
    print("\n" + "-" * 80)
    print("PHASE 4: All Extracted Artifacts (Organized by Type)")
    print("-" * 80)

    for obj_type in ObjectType:
        objects = canvas.list_objects(obj_type=obj_type)
        if objects:
            print(f"\n{'=' * 60}")
            print(f"{obj_type.value.upper()} ({len(objects)} items)")
            print("=" * 60)
            for i, obj in enumerate(objects, 1):
                print(f"\n  {i}. {obj.content}")
                if obj.context:
                    ctx = (
                        obj.context[:100] + "..."
                        if len(obj.context) > 100
                        else obj.context
                    )
                    print(f"     Context: {ctx}")

    # Show graph relationships
    print("\n" + "-" * 80)
    print("PHASE 5: Knowledge Graph Relationships")
    print("-" * 80)

    # Get roots (objects without incoming edges)
    roots = canvas.get_roots()
    print(f"\nRoot Objects (no dependencies): {len(roots)}")
    for obj in roots[:5]:
        print(f"  - [{obj.type.value}] {obj.content[:60]}...")

    # Show related objects for a decision
    decisions = canvas.list_objects(obj_type=ObjectType.DECISION)
    if decisions:
        first_decision = decisions[0]
        related = canvas.get_related(first_decision.id, depth=1)
        print(f"\nObjects related to decision '{first_decision.content[:40]}...':")
        for obj in related:
            print(f"  - [{obj.type.value}] {obj.content[:60]}...")

    # Generate injection context
    print("\n" + "-" * 80)
    print("PHASE 6: Context Injection (for LLM prompt)")
    print("-" * 80)

    # Simulate a follow-up query
    query = "What are the key decisions and remaining TODOs for the deployment adapter?"
    result = canvas.retrieve(query, top_k=5)
    injected = canvas.inject(result, format="markdown")
    print(f'\nQuery: "{query}"')
    print(f"\nInjected Context for LLM:")
    print(injected)

    # Summary
    print("\n" + "=" * 80)
    print("CASE STUDY SUMMARY")
    print("=" * 80)
    print(
        f"""
INPUT:
  - 68 discussion responses
  - Multiple stakeholders (Vercel, Netlify, Deno, Cloudflare, etc.)
  - Complex technical debates about API design
  - Mixed concerns: routing, PPR, documentation, backwards compatibility

OUTPUT (via CogCanvas):
  - {stats['total_objects']} structured artifacts extracted
  - {stats['by_type'].get('decision', 0)} DECISIONS (technical choices made)
  - {stats['by_type'].get('todo', 0)} TODOs (action items)
  - {stats['by_type'].get('key_fact', 0)} KEY FACTS (important information)
  - {stats['by_type'].get('reminder', 0)} REMINDERS (constraints/preferences)
  - {stats['by_type'].get('insight', 0)} INSIGHTS (conclusions/learnings)

KEY VALUE DEMONSTRATED:
  1. Extract structured, actionable artifacts from chaotic discussions
  2. Enable semantic search across all extracted knowledge
  3. Generate relevant context for follow-up questions
  4. Build a knowledge graph of relationships between artifacts
"""
    )

    # Prepare export data
    export_data = {
        "metadata": {
            "source": "https://github.com/vercel/next.js/discussions/77740",
            "title": "RFC: Deployment Adapters API",
            "discussion_size": 68,
            "stakeholders": [
                "Vercel",
                "Netlify",
                "Deno Deploy",
                "Cloudflare",
                "Tilda.net",
                "Community",
            ],
            "extraction_model": model,
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": stats,
        "objects": [],
        "graph": {"nodes": [], "edges": []},
    }

    # Export all objects
    for obj in canvas.list_objects():
        export_data["objects"].append(
            {
                "id": obj.id,
                "type": obj.type.value,
                "content": obj.content,
                "context": obj.context,
                "quote": obj.quote,
                "turn_id": obj.turn_id,
                "source": obj.source,
                "confidence": obj.confidence,
            }
        )

    # Export graph structure using the built-in to_dict method
    if hasattr(canvas, "_graph") and canvas._graph:
        graph_data = canvas._graph.to_dict()
        export_data["graph"]["nodes"] = graph_data.get("nodes", [])
        export_data["graph"]["edges"] = graph_data.get("edges", {})

    return canvas, stats, export_data


def main():
    parser = argparse.ArgumentParser(
        description="Run CogCanvas case study on GitHub Discussion"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="experiments/results/case_study_nextjs.json",
        help="Output file path",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview extraction without saving",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override extraction model (default: from .env MODEL_DEFAULT)",
    )

    args = parser.parse_args()

    # Run the case study
    canvas, stats, export_data = run_case_study()

    if args.preview:
        print("\n[Preview mode - not saving to file]")
    else:
        # Save to JSON file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")
        print(f"  - {len(export_data['objects'])} objects")
        print(f"  - {len(export_data['graph']['nodes'])} graph nodes")
        print(f"  - {len(export_data['graph']['edges'])} graph edges")


if __name__ == "__main__":
    main()
