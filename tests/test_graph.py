"""Tests for the CanvasGraph relationship management."""

import pytest
from cogcanvas.graph import CanvasGraph
from cogcanvas.models import CanvasObject, ObjectType


class TestCanvasGraph:
    """Test suite for CanvasGraph class."""

    def test_init(self):
        """Test graph initialization."""
        graph = CanvasGraph()
        assert len(graph) == 0
        assert str(graph) == "CanvasGraph(nodes=0, edges=0)"

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1", content="Test object 1")
        obj2 = CanvasObject(id="obj2", content="Test object 2")

        graph.add_node(obj1)
        graph.add_node(obj2)

        assert len(graph) == 2

    def test_add_edge(self):
        """Test adding edges between nodes."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1", content="Decision")
        obj2 = CanvasObject(id="obj2", content="TODO")

        graph.add_node(obj1)
        graph.add_node(obj2)

        # Add a "leads_to" edge
        success = graph.add_edge("obj1", "obj2", "leads_to")
        assert success

        # Verify edge exists
        neighbors = graph.get_neighbors("obj1", relation="leads_to")
        assert "obj2" in neighbors

    def test_bidirectional_relations(self):
        """Test that leads_to creates reverse caused_by relation."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")

        graph.add_node(obj1)
        graph.add_node(obj2)

        # Add leads_to edge
        graph.add_edge("obj1", "obj2", "leads_to")

        # Check forward relation
        assert "obj2" in graph.get_neighbors("obj1", "leads_to", "outgoing")

        # Check reverse relation (caused_by)
        assert "obj1" in graph.get_neighbors("obj2", "caused_by", "outgoing")

    def test_remove_node(self):
        """Test removing a node and its edges."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")
        obj3 = CanvasObject(id="obj3")

        graph.add_node(obj1)
        graph.add_node(obj2)
        graph.add_node(obj3)

        graph.add_edge("obj1", "obj2", "references")
        graph.add_edge("obj2", "obj3", "leads_to")

        # Remove middle node
        graph.remove_node("obj2")

        assert len(graph) == 2
        assert graph.get_neighbors("obj1") == []
        assert graph.get_neighbors("obj3") == []

    def test_get_neighbors_direction(self):
        """Test getting neighbors with different direction filters."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")
        obj3 = CanvasObject(id="obj3")

        graph.add_node(obj1)
        graph.add_node(obj2)
        graph.add_node(obj3)

        graph.add_edge("obj1", "obj2", "references")
        graph.add_edge("obj3", "obj2", "references")

        # Outgoing from obj1
        outgoing = graph.get_neighbors("obj1", direction="outgoing")
        assert outgoing == ["obj2"]

        # Incoming to obj2
        incoming = graph.get_neighbors("obj2", direction="incoming")
        assert set(incoming) == {"obj1", "obj3"}

        # Both directions for obj2
        both = graph.get_neighbors("obj2", direction="both")
        assert set(both) == {"obj1", "obj3"}

    def test_get_subgraph(self):
        """Test getting subgraph within N hops."""
        graph = CanvasGraph()
        objects = [CanvasObject(id=f"obj{i}") for i in range(5)]

        for obj in objects:
            graph.add_node(obj)

        # Create a chain: obj0 -> obj1 -> obj2 -> obj3 -> obj4
        for i in range(4):
            graph.add_edge(f"obj{i}", f"obj{i+1}", "leads_to")

        # Get 1-hop neighbors from obj2
        subgraph_1 = graph.get_subgraph("obj2", depth=1)
        assert set(subgraph_1) == {"obj1", "obj3"}  # obj2 excluded

        # Get 2-hop neighbors from obj2
        subgraph_2 = graph.get_subgraph("obj2", depth=2)
        assert set(subgraph_2) == {"obj0", "obj1", "obj3", "obj4"}

    def test_find_path(self):
        """Test finding shortest path between nodes."""
        graph = CanvasGraph()
        objects = [CanvasObject(id=f"obj{i}") for i in range(5)]

        for obj in objects:
            graph.add_node(obj)

        # Create paths
        graph.add_edge("obj0", "obj1", "leads_to")
        graph.add_edge("obj1", "obj2", "leads_to")
        graph.add_edge("obj0", "obj3", "leads_to")
        graph.add_edge("obj3", "obj2", "leads_to")

        # Find path (should take shorter route through obj3)
        path = graph.find_path("obj0", "obj2")
        assert len(path) == 3  # obj0 -> obj3 -> obj2 or obj0 -> obj1 -> obj2

        # Reverse path exists via caused_by (bidirectional relation)
        reverse_path = graph.find_path("obj2", "obj0")
        assert len(reverse_path) == 3  # obj2 -> obj1 -> obj0 (via caused_by)

    def test_get_roots(self):
        """Test finding nodes with no incoming edges."""
        graph = CanvasGraph()
        objects = [CanvasObject(id=f"obj{i}") for i in range(4)]

        for obj in objects:
            graph.add_node(obj)

        # Create tree structure
        graph.add_edge("obj0", "obj2", "leads_to")
        graph.add_edge("obj1", "obj2", "leads_to")
        graph.add_edge("obj2", "obj3", "leads_to")

        # Get roots for specific relation to avoid bidirectional edges
        roots_leads = graph.get_roots(relation="leads_to")
        assert set(roots_leads) == {"obj0", "obj1"}

    def test_get_leaves(self):
        """Test finding nodes with no outgoing edges."""
        graph = CanvasGraph()
        objects = [CanvasObject(id=f"obj{i}") for i in range(4)]

        for obj in objects:
            graph.add_node(obj)

        # Create tree structure
        graph.add_edge("obj0", "obj1", "leads_to")
        graph.add_edge("obj0", "obj2", "leads_to")
        graph.add_edge("obj1", "obj3", "leads_to")

        # Get leaves for specific relation to avoid bidirectional edges
        leaves_leads = graph.get_leaves(relation="leads_to")
        assert set(leaves_leads) == {"obj2", "obj3"}

    def test_topological_sort(self):
        """Test topological sorting of the graph."""
        graph = CanvasGraph()
        objects = [CanvasObject(id=f"obj{i}") for i in range(4)]

        for obj in objects:
            graph.add_node(obj)

        # Create DAG: obj0 -> obj1 -> obj3
        #             obj0 -> obj2 -> obj3
        graph.add_edge("obj0", "obj1", "leads_to")
        graph.add_edge("obj0", "obj2", "leads_to")
        graph.add_edge("obj1", "obj3", "leads_to")
        graph.add_edge("obj2", "obj3", "leads_to")

        sorted_nodes = graph.topological_sort()

        # Check that obj0 comes before obj1, obj2
        # and obj1, obj2 come before obj3
        obj0_idx = sorted_nodes.index("obj0")
        obj1_idx = sorted_nodes.index("obj1")
        obj2_idx = sorted_nodes.index("obj2")
        obj3_idx = sorted_nodes.index("obj3")

        assert obj0_idx < obj1_idx
        assert obj0_idx < obj2_idx
        assert obj1_idx < obj3_idx
        assert obj2_idx < obj3_idx

    def test_topological_sort_cycle(self):
        """Test topological sort with cycle (should return empty)."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")

        graph.add_node(obj1)
        graph.add_node(obj2)

        # Create cycle
        graph.add_edge("obj1", "obj2", "leads_to")
        graph.add_edge("obj2", "obj1", "leads_to")

        # Should return empty list due to cycle
        sorted_nodes = graph.topological_sort()
        assert sorted_nodes == []

    def test_serialization(self):
        """Test graph serialization and deserialization."""
        graph = CanvasGraph()
        objects = [CanvasObject(id=f"obj{i}") for i in range(3)]

        for obj in objects:
            graph.add_node(obj)

        graph.add_edge("obj0", "obj1", "references")
        graph.add_edge("obj1", "obj2", "leads_to")

        # Serialize
        data = graph.to_dict()

        # Deserialize
        restored_graph = CanvasGraph.from_dict(data)

        assert len(restored_graph) == 3
        assert "obj1" in restored_graph.get_neighbors("obj0", "references")
        assert "obj2" in restored_graph.get_neighbors("obj1", "leads_to")

    def test_duplicate_edges(self):
        """Test that duplicate edges are not created."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")

        graph.add_node(obj1)
        graph.add_node(obj2)

        # Add same edge twice
        graph.add_edge("obj1", "obj2", "references")
        graph.add_edge("obj1", "obj2", "references")

        neighbors = graph.get_neighbors("obj1", "references")
        assert neighbors.count("obj2") == 1  # Should only appear once

    def test_invalid_relation_type(self):
        """Test that invalid relation types are rejected."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")

        graph.add_node(obj1)
        graph.add_node(obj2)

        # Try to add edge with invalid relation
        success = graph.add_edge("obj1", "obj2", "invalid_relation")
        assert not success

    def test_relation_filter(self):
        """Test filtering neighbors by relation type."""
        graph = CanvasGraph()
        obj1 = CanvasObject(id="obj1")
        obj2 = CanvasObject(id="obj2")
        obj3 = CanvasObject(id="obj3")

        graph.add_node(obj1)
        graph.add_node(obj2)
        graph.add_node(obj3)

        graph.add_edge("obj1", "obj2", "references")
        graph.add_edge("obj1", "obj3", "leads_to")

        # Get only references
        refs = graph.get_neighbors("obj1", relation="references")
        assert refs == ["obj2"]

        # Get only leads_to
        leads = graph.get_neighbors("obj1", relation="leads_to")
        assert leads == ["obj3"]

        # Get all
        all_neighbors = graph.get_neighbors("obj1")
        assert set(all_neighbors) == {"obj2", "obj3"}
