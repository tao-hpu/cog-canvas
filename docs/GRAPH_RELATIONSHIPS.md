# Graph Relationships in CogCanvas

CogCanvas now includes powerful graph relationship management capabilities to track and query connections between cognitive objects.

## Overview

The graph relationship system allows you to:

1. **Create explicit relationships** between objects (manual linking)
2. **Automatically infer relationships** based on content and timing
3. **Query the graph** for paths, subgraphs, roots, and leaves
4. **Analyze causal sequences** with topological sorting
5. **Enhance retrieval** by including related objects

## Relationship Types

CogCanvas supports three types of relationships:

### 1. References
- **Type**: `references` (unidirectional)
- **Usage**: Object A mentions or cites object B
- **Example**: An INSIGHT references a KEY_FACT

### 2. Leads To
- **Type**: `leads_to` (bidirectional with `caused_by`)
- **Usage**: Object A causally leads to object B
- **Example**: A DECISION leads to a TODO

### 3. Caused By
- **Type**: `caused_by` (bidirectional with `leads_to`)
- **Usage**: Object A was caused by object B
- **Example**: A TODO was caused by a DECISION

**Note**: `leads_to` and `caused_by` are bidirectional. When you create `A leads_to B`, the system automatically creates `B caused_by A`.

## Core API

### Manual Linking

```python
from cogcanvas import Canvas, CanvasObject, ObjectType

canvas = Canvas()

# Create objects
decision = CanvasObject(
    id="dec1",
    type=ObjectType.DECISION,
    content="Use PostgreSQL for database"
)
todo = CanvasObject(
    id="todo1",
    type=ObjectType.TODO,
    content="Setup PostgreSQL cluster"
)

canvas.add(decision)
canvas.add(todo)

# Create causal link
canvas.link("dec1", "todo1", "leads_to")

# Query relationships
related = canvas.get_related("dec1")
print(f"Related to decision: {len(related)} objects")
```

### Graph Queries

#### Get Related Objects

```python
# Get 1-hop neighbors
related = canvas.get_related(obj_id, depth=1)

# Get 2-hop neighbors, specific relation
related = canvas.get_related(obj_id, depth=2, relation="leads_to")
```

#### Find Path

```python
# Find shortest path between two objects
path = canvas.find_path(from_id, to_id)
for obj in path:
    print(f"- {obj.content}")

# Find path using specific relation
path = canvas.find_path(from_id, to_id, relation="leads_to")
```

#### Get Roots and Leaves

```python
# Get starting nodes (no incoming edges)
roots = canvas.get_roots(relation="leads_to")

# Get ending nodes (no outgoing edges)
leaves = canvas.get_leaves(relation="leads_to")
```

#### Topological Sort

```python
# Get objects in causal order
sorted_objs = canvas.topological_sort(relation="leads_to")
for i, obj in enumerate(sorted_objs, 1):
    print(f"{i}. {obj.content}")
```

### Enhanced Retrieval

```python
# Retrieve with related objects included
result = canvas.retrieve(
    "database setup",
    include_related=True  # Includes 1-hop neighbors
)

# This helps find context around matching objects
for obj in result.objects:
    print(f"- [{obj.type.value}] {obj.content}")
```

## Automatic Relationship Inference

CogCanvas automatically infers relationships during `extract()` based on heuristics:

### Inference Rules

#### 1. Content-Based References
If a new object's content mentions key terms from existing objects (2+ shared keywords), creates a `references` relation.

```python
# First turn
canvas.extract(
    user="PostgreSQL supports ACID transactions",
    assistant="That's correct!"
)

# Second turn - will reference first object
canvas.extract(
    user="PostgreSQL transactions are very reliable",
    assistant="Absolutely!"
)
```

#### 2. TODO Caused By DECISION
If a TODO appears within 5 turns of a DECISION and shares keywords, creates a `caused_by` relation.

```python
# Turn 1: Decision
canvas.extract(
    user="Let's use PostgreSQL",
    assistant="Good choice!"
)

# Turn 2: TODO - will be linked to decision
canvas.extract(
    user="Need to setup PostgreSQL",
    assistant="I'll help with that."
)
```

#### 3. INSIGHT Caused By Facts
If an INSIGHT appears within 3 turns of a KEY_FACT or DECISION with shared keywords, creates a `caused_by` relation.

### Inference Configuration

Stop words are filtered during inference:
- Common words like "the", "and", "for", "with", etc.
- Words shorter than 4 characters (configurable)

## Graph Storage and Persistence

### Serialization

The graph is automatically saved with canvas state:

```python
canvas = Canvas(storage_path="my_canvas.json")

# Add objects and relationships
canvas.add(obj1)
canvas.link(obj1.id, obj2.id, "leads_to")

# Graph is automatically saved to my_canvas.json
```

### Graph Data Structure

The graph is stored in the JSON file:

```json
{
  "turn_counter": 5,
  "objects": [...],
  "graph": {
    "nodes": ["obj1", "obj2", "obj3"],
    "edges": {
      "references": {
        "obj1": ["obj2"]
      },
      "leads_to": {
        "obj2": ["obj3"]
      },
      "caused_by": {
        "obj3": ["obj2"]
      }
    }
  }
}
```

### Backward Compatibility

Loading old canvas files without graph data is supported. The system will:
1. Create an empty graph
2. Add all objects as nodes
3. Preserve object-level relationship fields

## Implementation Details

### CanvasGraph Class

Located in `cogcanvas/graph.py`, the `CanvasGraph` class manages:

- **Nodes**: Set of object IDs
- **Edges**: Adjacency lists by relation type
- **Reverse Edges**: For efficient bidirectional queries

Key methods:
- `add_node(obj)`: Add object to graph
- `add_edge(from_id, to_id, relation)`: Create relationship
- `remove_node(obj_id)`: Remove object and its edges
- `get_neighbors(obj_id, relation, direction)`: Query neighbors
- `get_subgraph(obj_id, depth, relation)`: Get N-hop subgraph
- `find_path(from_id, to_id, relation)`: BFS shortest path
- `get_roots(relation)`: Nodes with no incoming edges
- `get_leaves(relation)`: Nodes with no outgoing edges
- `topological_sort(relation)`: Kahn's algorithm for ordering

### Canvas Integration

The `Canvas` class:
1. Maintains a `CanvasGraph` instance
2. Updates graph on `add()`, `remove()`, and `link()`
3. Calls `_infer_relations()` after `extract()`
4. Saves/loads graph with canvas state

## Best Practices

### 1. Use Specific Relations for Queries

When querying roots/leaves, specify the relation to avoid bidirectional edge issues:

```python
# Good - specific relation
roots = canvas.get_roots(relation="leads_to")

# May return empty due to bidirectional edges
roots = canvas.get_roots()  # checks ALL relations
```

### 2. Leverage Automatic Inference

Design your content to trigger automatic inference:
- Use consistent terminology
- Reference previous decisions in TODOs
- Build upon previous facts in insights

### 3. Manual Links for Critical Relationships

For important causal relationships, create explicit links:

```python
# Ensure critical dependency is tracked
canvas.link(requirement_id, feature_id, "leads_to")
```

### 4. Use Topological Sort for Planning

Get execution order for tasks:

```python
tasks = canvas.topological_sort(relation="leads_to")
for i, task in enumerate(tasks, 1):
    print(f"Step {i}: {task.content}")
```

### 5. Enhance Context with Related Retrieval

Include related objects to get full context:

```python
# Get matches AND their context
result = canvas.retrieve(
    query="database migration",
    include_related=True,
    top_k=3
)
```

## Examples

See `examples/graph_relationships.py` for comprehensive demonstrations:

1. Manual linking of decisions and TODOs
2. Graph queries (paths, roots, leaves)
3. Subgraph retrieval
4. Automatic relationship inference
5. Retrieval with related objects
6. Complex knowledge graph analysis

Run the examples:

```bash
python examples/graph_relationships.py
```

## Testing

Comprehensive test coverage in:
- `tests/test_graph.py`: Core graph functionality
- `tests/test_canvas_graph.py`: Canvas integration and inference

Run tests:

```bash
pytest tests/test_graph.py tests/test_canvas_graph.py -v
```

## Performance Considerations

### Graph Operations Complexity

- **add_node**: O(1)
- **add_edge**: O(1)
- **remove_node**: O(E) where E = number of edges for that node
- **get_neighbors**: O(1) to O(k) where k = number of neighbors
- **get_subgraph**: O(V + E) for BFS up to depth d
- **find_path**: O(V + E) for BFS
- **topological_sort**: O(V + E) using Kahn's algorithm

### Memory Usage

The graph stores:
- Forward edges: O(E)
- Reverse edges: O(E)
- Nodes: O(V)

Total: O(V + 2E)

### Inference Cost

Automatic inference in `_infer_relations()`:
- Iterates over new objects × existing objects
- Complexity: O(N × M) where N = new objects, M = existing objects
- Runs after each `extract()` call

For large canvases (>1000 objects), consider:
- Reducing inference window (currently 3-5 turns)
- Manual linking for critical relationships
- Batch extraction to reduce inference calls

## Future Enhancements

Potential improvements:

1. **Weighted Edges**: Add confidence scores to relationships
2. **Edge Attributes**: Store metadata on relationships
3. **Graph Algorithms**: Community detection, centrality measures
4. **Relationship Types**: Custom user-defined relations
5. **Query Language**: DSL for complex graph queries
6. **Visualization**: Generate graph diagrams
7. **Relationship Templates**: Pre-defined patterns for common workflows

## Summary

CogCanvas graph relationships provide:

- ✅ Manual and automatic relationship management
- ✅ Three relationship types (references, leads_to, caused_by)
- ✅ Powerful graph queries (paths, subgraphs, roots, leaves)
- ✅ Topological sorting for causal analysis
- ✅ Enhanced retrieval with related objects
- ✅ Persistent graph storage
- ✅ Automatic inference from content and timing
- ✅ Comprehensive test coverage
- ✅ Clean Python API

This enables CogCanvas to maintain not just isolated facts, but a rich web of interconnected knowledge that mirrors how concepts relate in real conversations.
