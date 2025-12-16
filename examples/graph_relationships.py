"""
Example demonstrating graph relationship features in CogCanvas.

This example shows:
1. Manual linking of objects
2. Automatic relationship inference
3. Graph queries (paths, roots, leaves)
4. Topological sorting for causal order
5. Subgraph retrieval
"""

from cogcanvas import Canvas, CanvasObject, ObjectType


def demo_manual_linking():
    """Demonstrate manual creation of relationships."""
    print("=" * 60)
    print("Demo 1: Manual Linking")
    print("=" * 60)

    canvas = Canvas()

    # Create a decision
    decision = CanvasObject(
        id="dec1",
        type=ObjectType.DECISION,
        content="Use PostgreSQL as our database",
        context="Database selection decision",
    )

    # Create related TODOs
    todo1 = CanvasObject(
        id="todo1",
        type=ObjectType.TODO,
        content="Setup PostgreSQL development environment",
        context="Required action",
    )

    todo2 = CanvasObject(
        id="todo2",
        type=ObjectType.TODO,
        content="Design database schema for PostgreSQL",
        context="Required action",
    )

    # Add to canvas
    canvas.add(decision)
    canvas.add(todo1)
    canvas.add(todo2)

    # Create causal relationships
    canvas.link("dec1", "todo1", "leads_to")
    canvas.link("dec1", "todo2", "leads_to")

    print(f"\nCreated decision: {decision.content}")
    print(f"Created TODO 1: {todo1.content}")
    print(f"Created TODO 2: {todo2.content}")

    # Query relationships
    related = canvas.get_related("dec1")
    print(f"\nObjects related to decision: {len(related)}")
    for obj in related:
        print(f"  - {obj.type.value}: {obj.content}")

    print()


def demo_graph_queries():
    """Demonstrate graph query capabilities."""
    print("=" * 60)
    print("Demo 2: Graph Queries")
    print("=" * 60)

    canvas = Canvas()

    # Create a workflow: Decision -> Planning -> Implementation -> Testing
    objects = [
        CanvasObject(
            id="step1",
            type=ObjectType.DECISION,
            content="Adopt microservices architecture",
        ),
        CanvasObject(
            id="step2",
            type=ObjectType.TODO,
            content="Plan service boundaries and APIs",
        ),
        CanvasObject(
            id="step3",
            type=ObjectType.TODO,
            content="Implement user service",
        ),
        CanvasObject(
            id="step4",
            type=ObjectType.TODO,
            content="Test service integration",
        ),
    ]

    for obj in objects:
        canvas.add(obj)

    # Create causal chain
    canvas.link("step1", "step2", "leads_to")
    canvas.link("step2", "step3", "leads_to")
    canvas.link("step3", "step4", "leads_to")

    # Find path
    print("\nFinding path from step1 to step4:")
    path = canvas.find_path("step1", "step4")
    for i, obj in enumerate(path, 1):
        print(f"  {i}. {obj.content}")

    # Get roots (starting points)
    print("\nRoot nodes (no incoming edges):")
    roots = canvas.get_roots()
    for obj in roots:
        print(f"  - {obj.content}")

    # Get leaves (end points)
    print("\nLeaf nodes (no outgoing edges):")
    leaves = canvas.get_leaves()
    for obj in leaves:
        print(f"  - {obj.content}")

    # Topological sort
    print("\nTopological order (causal sequence):")
    sorted_objs = canvas.topological_sort()
    for i, obj in enumerate(sorted_objs, 1):
        print(f"  {i}. {obj.content}")

    print()


def demo_subgraph_retrieval():
    """Demonstrate subgraph retrieval."""
    print("=" * 60)
    print("Demo 3: Subgraph Retrieval")
    print("=" * 60)

    canvas = Canvas()

    # Create a knowledge graph
    fact1 = CanvasObject(
        id="fact1",
        type=ObjectType.KEY_FACT,
        content="Python 3.11 is 25% faster than 3.10",
    )
    fact2 = CanvasObject(
        id="fact2",
        type=ObjectType.KEY_FACT,
        content="FastAPI requires Python 3.7+",
    )
    insight = CanvasObject(
        id="insight1",
        type=ObjectType.INSIGHT,
        content="Upgrading to Python 3.11 will boost API performance",
    )
    decision = CanvasObject(
        id="dec1",
        type=ObjectType.DECISION,
        content="Upgrade to Python 3.11 for our FastAPI services",
    )
    todo = CanvasObject(
        id="todo1",
        type=ObjectType.TODO,
        content="Update Python version in Dockerfile",
    )

    for obj in [fact1, fact2, insight, decision, todo]:
        canvas.add(obj)

    # Create relationships
    canvas.link("insight1", "fact1", "references")
    canvas.link("insight1", "fact2", "references")
    canvas.link("dec1", "insight1", "caused_by")
    canvas.link("dec1", "todo1", "leads_to")

    # Get 1-hop subgraph
    print(f"\nCenter: {decision.content}")
    print("\n1-hop subgraph:")
    subgraph_1 = canvas.get_subgraph("dec1", depth=1)
    for obj in subgraph_1:
        print(f"  - [{obj.type.value}] {obj.content}")

    # Get 2-hop subgraph
    print("\n2-hop subgraph:")
    subgraph_2 = canvas.get_subgraph("dec1", depth=2)
    for obj in subgraph_2:
        print(f"  - [{obj.type.value}] {obj.content}")

    print()


def demo_automatic_inference():
    """Demonstrate automatic relationship inference."""
    print("=" * 60)
    print("Demo 4: Automatic Relationship Inference")
    print("=" * 60)

    canvas = Canvas()

    # Simulate a conversation with extraction
    print("\nTurn 1: Discussing database choice")
    result1 = canvas.extract(
        user="I think we should use PostgreSQL for our project",
        assistant="That's a good choice! PostgreSQL is very reliable.",
    )
    print(f"Extracted {result1.count} objects")

    print("\nTurn 2: Discussing setup tasks")
    result2 = canvas.extract(
        user="We need to setup PostgreSQL and configure it properly",
        assistant="Yes, let me help you with that.",
    )
    print(f"Extracted {result2.count} objects")

    # Check for inferred relationships
    print("\nChecking for automatically inferred relationships:")
    all_objects = canvas.list_objects()

    for obj in all_objects:
        related = canvas.get_related(obj.id)
        if related:
            print(f"\n{obj.type.value}: {obj.content[:50]}...")
            print(f"  Related to {len(related)} objects:")
            for rel in related:
                print(f"    - {rel.type.value}: {rel.content[:50]}...")

    print()


def demo_retrieve_with_related():
    """Demonstrate retrieval with related objects."""
    print("=" * 60)
    print("Demo 5: Retrieval with Related Objects")
    print("=" * 60)

    canvas = Canvas()

    # Create objects with relationships
    database_decision = CanvasObject(
        id="db_dec",
        type=ObjectType.DECISION,
        content="Use PostgreSQL database for production",
    )
    cache_decision = CanvasObject(
        id="cache_dec",
        type=ObjectType.DECISION,
        content="Use Redis for caching layer",
    )
    todo1 = CanvasObject(
        id="todo1",
        type=ObjectType.TODO,
        content="Setup PostgreSQL cluster",
    )
    todo2 = CanvasObject(
        id="todo2",
        type=ObjectType.TODO,
        content="Configure Redis instance",
    )

    for obj in [database_decision, cache_decision, todo1, todo2]:
        canvas.add(obj)

    canvas.link("db_dec", "todo1", "leads_to")
    canvas.link("cache_dec", "todo2", "leads_to")

    # Retrieve without related objects
    print("\nRetrieve 'PostgreSQL' without related objects:")
    result1 = canvas.retrieve("PostgreSQL", include_related=False)
    print(f"Found {result1.count} objects:")
    for obj in result1.objects:
        print(f"  - {obj.content}")

    # Retrieve with related objects
    print("\nRetrieve 'PostgreSQL' with related objects:")
    result2 = canvas.retrieve("PostgreSQL", include_related=True)
    print(f"Found {result2.count} objects:")
    for obj in result2.objects:
        print(f"  - {obj.content}")

    print()


def demo_complex_graph():
    """Demonstrate a complex knowledge graph."""
    print("=" * 60)
    print("Demo 6: Complex Knowledge Graph")
    print("=" * 60)

    canvas = Canvas()

    # Create a project planning graph
    objects = {
        "req1": CanvasObject(
            id="req1",
            type=ObjectType.KEY_FACT,
            content="System must handle 10k requests/second",
        ),
        "req2": CanvasObject(
            id="req2",
            type=ObjectType.KEY_FACT,
            content="Response time should be under 100ms",
        ),
        "insight1": CanvasObject(
            id="insight1",
            type=ObjectType.INSIGHT,
            content="Need async processing and caching for performance",
        ),
        "dec1": CanvasObject(
            id="dec1",
            type=ObjectType.DECISION,
            content="Use FastAPI with Redis caching",
        ),
        "todo1": CanvasObject(
            id="todo1",
            type=ObjectType.TODO,
            content="Setup FastAPI application structure",
        ),
        "todo2": CanvasObject(
            id="todo2",
            type=ObjectType.TODO,
            content="Implement Redis caching layer",
        ),
        "todo3": CanvasObject(
            id="todo3",
            type=ObjectType.TODO,
            content="Load test the system",
        ),
        "remind1": CanvasObject(
            id="remind1",
            type=ObjectType.REMINDER,
            content="Must maintain <100ms response time",
        ),
    }

    for obj in objects.values():
        canvas.add(obj)

    # Create complex relationships
    canvas.link("insight1", "req1", "references")
    canvas.link("insight1", "req2", "references")
    canvas.link("dec1", "insight1", "caused_by")
    canvas.link("dec1", "todo1", "leads_to")
    canvas.link("dec1", "todo2", "leads_to")
    canvas.link("todo1", "todo3", "leads_to")
    canvas.link("todo2", "todo3", "leads_to")
    canvas.link("todo3", "remind1", "references")

    # Analyze the graph
    print("\nGraph Statistics:")
    print(f"  Total objects: {canvas.size}")

    roots = canvas.get_roots()
    print(f"  Root nodes: {len(roots)}")
    for obj in roots:
        print(f"    - {obj.content[:50]}...")

    leaves = canvas.get_leaves()
    print(f"  Leaf nodes: {len(leaves)}")
    for obj in leaves:
        print(f"    - {obj.content[:50]}...")

    # Show execution order
    print("\nTopological execution order:")
    sorted_objs = canvas.topological_sort()
    for i, obj in enumerate(sorted_objs, 1):
        print(f"  {i}. [{obj.type.value}] {obj.content[:60]}...")

    # Find critical path
    print("\nCritical path from requirements to testing:")
    path = canvas.find_path("req1", "todo3")
    if path:
        print(f"  Path length: {len(path)} steps")
        for i, obj in enumerate(path, 1):
            print(f"    {i}. {obj.content[:60]}...")

    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("CogCanvas Graph Relationship Examples")
    print("=" * 60 + "\n")

    demo_manual_linking()
    demo_graph_queries()
    demo_subgraph_retrieval()
    demo_automatic_inference()
    demo_retrieve_with_related()
    demo_complex_graph()

    print("=" * 60)
    print("All demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
