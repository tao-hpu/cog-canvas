"""Graph relationship management for CogCanvas objects."""

from typing import List, Optional, Set, Dict, Tuple
from collections import defaultdict, deque

from cogcanvas.models import CanvasObject


class CanvasGraph:
    """
    Manages graph relationships between canvas objects.

    Supports three types of relationships:
    - references: Object A references object B (unidirectional)
    - leads_to: Object A leads to object B (bidirectional with caused_by)
    - caused_by: Object A is caused by object B (bidirectional with leads_to)

    The graph maintains consistency by automatically updating reverse edges
    for bidirectional relationships.
    """

    def __init__(self):
        """Initialize an empty graph."""
        # Store nodes (object IDs)
        self._nodes: Set[str] = set()

        # Store edges as adjacency lists by relation type
        # Format: {relation_type: {from_id: [to_id1, to_id2, ...]}}
        self._edges: Dict[str, Dict[str, List[str]]] = {
            "references": defaultdict(list),
            "leads_to": defaultdict(list),
            "caused_by": defaultdict(list),
        }

        # Reverse edges for efficient querying
        # Format: {relation_type: {to_id: [from_id1, from_id2, ...]}}
        self._reverse_edges: Dict[str, Dict[str, List[str]]] = {
            "references": defaultdict(list),
            "leads_to": defaultdict(list),
            "caused_by": defaultdict(list),
        }

    def add_node(self, obj: CanvasObject) -> None:
        """
        Add a node (object) to the graph.

        Args:
            obj: The CanvasObject to add
        """
        self._nodes.add(obj.id)

    def add_edge(self, from_id: str, to_id: str, relation: str) -> bool:
        """
        Add an edge between two objects.

        Args:
            from_id: Source object ID
            to_id: Target object ID
            relation: Type of relation ("references", "leads_to", "caused_by")

        Returns:
            True if edge was added, False if relation type is invalid
        """
        if relation not in self._edges:
            return False

        # Avoid duplicate edges
        if to_id in self._edges[relation][from_id]:
            return True

        # Add forward edge
        self._edges[relation][from_id].append(to_id)

        # Add reverse edge
        self._reverse_edges[relation][to_id].append(from_id)

        # For bidirectional relations, add reverse relation
        if relation == "leads_to":
            # A leads_to B implies B caused_by A
            if from_id not in self._edges["caused_by"][to_id]:
                self._edges["caused_by"][to_id].append(from_id)
                self._reverse_edges["caused_by"][from_id].append(to_id)
        elif relation == "caused_by":
            # A caused_by B implies B leads_to A
            if from_id not in self._edges["leads_to"][to_id]:
                self._edges["leads_to"][to_id].append(from_id)
                self._reverse_edges["leads_to"][from_id].append(to_id)

        return True

    def remove_node(self, obj_id: str) -> None:
        """
        Remove a node and all its edges from the graph.

        Args:
            obj_id: The object ID to remove
        """
        if obj_id not in self._nodes:
            return

        self._nodes.discard(obj_id)

        # Remove all outgoing edges
        for relation in self._edges:
            # Get targets before deletion
            targets = self._edges[relation].get(obj_id, []).copy()

            # Remove from forward edges
            if obj_id in self._edges[relation]:
                del self._edges[relation][obj_id]

            # Remove from reverse edges
            for target in targets:
                if target in self._reverse_edges[relation]:
                    self._reverse_edges[relation][target] = [
                        x for x in self._reverse_edges[relation][target] if x != obj_id
                    ]

        # Remove all incoming edges
        for relation in self._reverse_edges:
            # Get sources before deletion
            sources = self._reverse_edges[relation].get(obj_id, []).copy()

            # Remove from reverse edges
            if obj_id in self._reverse_edges[relation]:
                del self._reverse_edges[relation][obj_id]

            # Remove from forward edges
            for source in sources:
                if source in self._edges[relation]:
                    self._edges[relation][source] = [
                        x for x in self._edges[relation][source] if x != obj_id
                    ]

    def get_neighbors(
        self,
        obj_id: str,
        relation: Optional[str] = None,
        direction: str = "outgoing"
    ) -> List[str]:
        """
        Get neighboring nodes connected by the specified relation.

        Args:
            obj_id: The object ID to query
            relation: Filter by relation type (None = all relations)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of neighbor object IDs
        """
        neighbors = []

        # Determine which relations to check
        relations = [relation] if relation else list(self._edges.keys())

        for rel in relations:
            if direction in ("outgoing", "both"):
                neighbors.extend(self._edges[rel].get(obj_id, []))
            if direction in ("incoming", "both"):
                neighbors.extend(self._reverse_edges[rel].get(obj_id, []))

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                result.append(neighbor)

        return result

    def get_subgraph(
        self,
        obj_id: str,
        depth: int = 1,
        relation: Optional[str] = None
    ) -> List[str]:
        """
        Get all objects within N hops of the given object.

        Args:
            obj_id: Starting object ID
            depth: Maximum number of hops (1 = immediate neighbors)
            relation: Filter by relation type (None = all relations)

        Returns:
            List of object IDs in the subgraph (excluding the starting object)
        """
        if obj_id not in self._nodes or depth < 1:
            return []

        visited = {obj_id}
        current_level = [obj_id]

        for _ in range(depth):
            next_level = []
            for node in current_level:
                neighbors = self.get_neighbors(node, relation, direction="both")
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level

        # Return all visited nodes except the starting object
        visited.discard(obj_id)
        return list(visited)

    def find_path(
        self,
        from_id: str,
        to_id: str,
        relation: Optional[str] = None
    ) -> List[str]:
        """
        Find shortest path between two objects using BFS.

        Args:
            from_id: Starting object ID
            to_id: Target object ID
            relation: Filter by relation type (None = all relations)

        Returns:
            List of object IDs representing the path (empty if no path exists)
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return []

        if from_id == to_id:
            return [from_id]

        # BFS to find shortest path
        queue = deque([(from_id, [from_id])])
        visited = {from_id}

        while queue:
            current, path = queue.popleft()

            # Get neighbors (outgoing only for directed path finding)
            neighbors = self.get_neighbors(current, relation, direction="outgoing")

            for neighbor in neighbors:
                if neighbor == to_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def get_roots(self, relation: Optional[str] = None) -> List[str]:
        """
        Get nodes with no incoming edges.

        Args:
            relation: Filter by relation type (None = consider all relations)

        Returns:
            List of root object IDs
        """
        # If no relation specified, check all relations
        relations = [relation] if relation else list(self._reverse_edges.keys())

        # Find nodes with no incoming edges for the specified relations
        roots = []
        for node in self._nodes:
            has_incoming = False
            for rel in relations:
                if node in self._reverse_edges[rel] and self._reverse_edges[rel][node]:
                    has_incoming = True
                    break
            if not has_incoming:
                roots.append(node)

        return roots

    def get_leaves(self, relation: Optional[str] = None) -> List[str]:
        """
        Get nodes with no outgoing edges.

        Args:
            relation: Filter by relation type (None = consider all relations)

        Returns:
            List of leaf object IDs
        """
        # If no relation specified, check all relations
        relations = [relation] if relation else list(self._edges.keys())

        # Find nodes with no outgoing edges for the specified relations
        leaves = []
        for node in self._nodes:
            has_outgoing = False
            for rel in relations:
                if node in self._edges[rel] and self._edges[rel][node]:
                    has_outgoing = True
                    break
            if not has_outgoing:
                leaves.append(node)

        return leaves

    def topological_sort(self, relation: str = "leads_to") -> List[str]:
        """
        Return nodes in topological order based on causal relationships.

        Uses Kahn's algorithm for topological sorting. Only considers the
        specified relation type (default: "leads_to" for causal ordering).

        Args:
            relation: Relation type to use for ordering

        Returns:
            List of object IDs in topological order (empty if cycle detected)
        """
        if relation not in self._edges:
            return []

        # Calculate in-degree for each node
        in_degree = {node: 0 for node in self._nodes}
        for node in self._nodes:
            incoming = self._reverse_edges[relation].get(node, [])
            in_degree[node] = len(incoming)

        # Start with nodes that have no incoming edges
        queue = deque([node for node in self._nodes if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Reduce in-degree for neighbors
            for neighbor in self._edges[relation].get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If result doesn't contain all nodes, there's a cycle
        if len(result) != len(self._nodes):
            return []

        return result

    def to_dict(self) -> Dict[str, any]:
        """
        Serialize graph to dictionary for persistence.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": list(self._nodes),
            "edges": {
                relation: dict(edges)
                for relation, edges in self._edges.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "CanvasGraph":
        """
        Deserialize graph from dictionary.

        Args:
            data: Dictionary representation of the graph

        Returns:
            CanvasGraph instance
        """
        graph = cls()

        # Restore nodes
        graph._nodes = set(data.get("nodes", []))

        # Restore edges
        edges_data = data.get("edges", {})
        for relation, edges in edges_data.items():
            if relation in graph._edges:
                for from_id, to_ids in edges.items():
                    for to_id in to_ids:
                        graph.add_edge(from_id, to_id, relation)

        return graph

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self._nodes)

    def __repr__(self) -> str:
        """String representation of the graph."""
        edge_count = sum(
            len(targets)
            for edges in self._edges.values()
            for targets in edges.values()
        )
        return f"CanvasGraph(nodes={len(self._nodes)}, edges={edge_count})"
