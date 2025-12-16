#!/usr/bin/env python3
"""
CogCanvas Knowledge Graph Visualization

Generates publication-quality visualizations of extracted knowledge graphs.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Color scheme for different node types (academic/professional style)
NODE_COLORS = {
    "decision": "#E74C3C",      # Red - important decisions
    "todo": "#3498DB",          # Blue - action items
    "key_fact": "#27AE60",      # Green - factual information
    "insight": "#9B59B6",       # Purple - derived insights
    "reminder": "#F39C12",      # Orange - reminders/preferences
}

# Edge colors for different relationship types
EDGE_COLORS = {
    "references": "#95A5A6",    # Gray - general references
    "leads_to": "#2ECC71",      # Green - causal forward
    "caused_by": "#E67E22",     # Orange - causal backward
}

EDGE_STYLES = {
    "references": "dotted",
    "leads_to": "solid",
    "caused_by": "dashed",
}


def load_case_study(filepath: str) -> dict:
    """Load case study JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def build_graph(data: dict, include_edge_types: list = None) -> nx.DiGraph:
    """Build NetworkX graph from case study data."""
    G = nx.DiGraph()

    # Build ID -> object mapping
    id_to_obj = {obj["id"]: obj for obj in data["objects"]}

    # Add nodes with attributes
    for obj in data["objects"]:
        G.add_node(
            obj["id"],
            type=obj["type"],
            content=obj["content"],
            turn_id=obj.get("turn_id", 0),
            confidence=obj.get("confidence", 0.0),
        )

    # Add edges
    edges = data.get("graph", {}).get("edges", {})
    edge_types = include_edge_types or ["references", "leads_to", "caused_by"]

    for edge_type in edge_types:
        if edge_type in edges:
            for source, targets in edges[edge_type].items():
                for target in targets:
                    if source in G.nodes and target in G.nodes:
                        G.add_edge(source, target, type=edge_type)

    return G


def get_subgraph_around_node(G: nx.DiGraph, center_id: str, depth: int = 2) -> nx.DiGraph:
    """Extract subgraph around a center node up to given depth."""
    nodes_to_include = {center_id}

    # BFS outward
    current_layer = {center_id}
    for _ in range(depth):
        next_layer = set()
        for node in current_layer:
            next_layer.update(G.successors(node))
            next_layer.update(G.predecessors(node))
        nodes_to_include.update(next_layer)
        current_layer = next_layer

    return G.subgraph(nodes_to_include).copy()


def get_decision_subgraph(G: nx.DiGraph, max_nodes: int = 20) -> nx.DiGraph:
    """Get subgraph centered on decisions with highest connectivity."""
    # Find decision nodes
    decision_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "decision"]

    if not decision_nodes:
        return G

    # Rank by connectivity
    connectivity = {n: G.degree(n) for n in decision_nodes}
    top_decisions = sorted(connectivity, key=connectivity.get, reverse=True)[:3]

    # Build subgraph around top decisions
    nodes_to_include = set(top_decisions)
    for decision in top_decisions:
        nodes_to_include.update(G.successors(decision))
        nodes_to_include.update(G.predecessors(decision))

    # Limit size
    if len(nodes_to_include) > max_nodes:
        # Keep only nodes with highest degree
        node_degrees = {n: G.degree(n) for n in nodes_to_include}
        nodes_to_include = set(sorted(node_degrees, key=node_degrees.get, reverse=True)[:max_nodes])
        # Always include the top decisions
        nodes_to_include.update(top_decisions)

    return G.subgraph(nodes_to_include).copy()


def get_minimal_subgraph(G: nx.DiGraph, max_nodes: int = 12) -> nx.DiGraph:
    """Get a minimal, readable subgraph with diverse node types."""
    # Get top nodes by type (ensure diversity)
    nodes_by_type = {}
    for n, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        if t not in nodes_by_type:
            nodes_by_type[t] = []
        nodes_by_type[t].append((n, G.degree(n)))

    # Select top nodes from each type
    selected = set()
    type_quota = {"decision": 3, "todo": 3, "key_fact": 3, "insight": 2, "reminder": 1}

    for node_type, quota in type_quota.items():
        if node_type in nodes_by_type:
            sorted_nodes = sorted(nodes_by_type[node_type], key=lambda x: x[1], reverse=True)
            for n, _ in sorted_nodes[:quota]:
                selected.add(n)
                if len(selected) >= max_nodes:
                    break

    return G.subgraph(selected).copy()


def truncate_label(text: str, max_len: int = 30) -> str:
    """Truncate text for node labels."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def visualize_graph(
    G: nx.DiGraph,
    title: str = "CogCanvas Knowledge Graph",
    output_path: str = None,
    figsize: tuple = (16, 12),
    show_labels: bool = True,
    label_fontsize: int = 8,
    node_size: int = 800,
    minimal: bool = False,
):
    """Generate visualization of the knowledge graph."""

    if len(G.nodes) == 0:
        print("Warning: Empty graph, nothing to visualize")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Adjust sizes for minimal mode
    if minimal:
        node_size = 1800
        label_fontsize = 10

    # Layout - use spring for minimal (more spread out)
    if minimal:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes by type
    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=color,
                node_size=node_size,
                alpha=0.9,
                ax=ax,
            )

    # Draw edges by type
    for edge_type in ["references", "leads_to", "caused_by"]:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == edge_type]
        if edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=EDGE_COLORS[edge_type],
                style=EDGE_STYLES[edge_type],
                alpha=0.5 if minimal else 0.6,
                arrows=True,
                arrowsize=20 if minimal else 15,
                connectionstyle="arc3,rad=0.1",
                ax=ax,
            )

    # Draw labels
    if show_labels:
        labels = {}
        max_label_len = 18 if minimal else 25
        for node, data in G.nodes(data=True):
            content = data.get("content", node)
            labels[node] = truncate_label(content, max_label_len)

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=label_fontsize,
            font_weight="bold",
            ax=ax,
        )

    # Legend for node types
    node_legend = [
        mpatches.Patch(color=color, label=node_type.replace("_", " ").title())
        for node_type, color in NODE_COLORS.items()
    ]

    # Legend for edge types
    edge_legend = [
        Line2D([0], [0], color=EDGE_COLORS[et], linestyle=EDGE_STYLES[et],
               label=et.replace("_", " ").title(), linewidth=2)
        for et in ["references", "leads_to", "caused_by"]
    ]

    # Add legends
    legend1 = ax.legend(handles=node_legend, loc="upper left", title="Node Types", fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=edge_legend, loc="upper right", title="Edge Types", fontsize=9)

    # Title and styling
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_statistics(data: dict, G: nx.DiGraph):
    """Print graph statistics."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Source: {data['metadata'].get('source', 'N/A')}")
    print(f"Title: {data['metadata'].get('title', 'N/A')}")
    print(f"Discussion Size: {data['metadata'].get('discussion_size', 'N/A')} comments")
    print()
    print("Object Counts:")
    for obj_type, count in data["statistics"]["by_type"].items():
        print(f"  - {obj_type.replace('_', ' ').title()}: {count}")
    print()
    print(f"Graph Nodes: {G.number_of_nodes()}")
    print(f"Graph Edges: {G.number_of_edges()}")
    print()

    # Node type distribution in subgraph
    type_counts = defaultdict(int)
    for _, d in G.nodes(data=True):
        type_counts[d.get("type", "unknown")] += 1
    print("Subgraph Node Distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  - {t}: {c}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize CogCanvas knowledge graph")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("-o", "--output", help="Output image path (PNG/PDF)")
    parser.add_argument("--full", action="store_true", help="Show full graph (not subgraph)")
    parser.add_argument("--minimal", action="store_true", help="Minimal readable graph (~12 nodes)")
    parser.add_argument("--max-nodes", type=int, default=25, help="Max nodes in subgraph")
    parser.add_argument("--no-labels", action="store_true", help="Hide node labels")
    parser.add_argument("--figsize", type=str, default="16,12", help="Figure size (w,h)")
    args = parser.parse_args()

    # Load data
    data = load_case_study(args.input)

    # Build full graph
    G = build_graph(data)

    # Get subgraph if needed
    minimal_mode = False
    if args.minimal:
        G = get_minimal_subgraph(G, max_nodes=12)
        title = f"CogCanvas: {data['metadata'].get('title', 'Knowledge Graph')}"
        minimal_mode = True
    elif not args.full and G.number_of_nodes() > args.max_nodes:
        G = get_decision_subgraph(G, max_nodes=args.max_nodes)
        title = f"CogCanvas: {data['metadata'].get('title', 'Knowledge Graph')} (Subgraph)"
    else:
        title = f"CogCanvas: {data['metadata'].get('title', 'Knowledge Graph')}"

    # Print statistics
    print_statistics(data, G)

    # Parse figsize
    figsize = tuple(map(int, args.figsize.split(",")))

    # Generate visualization
    visualize_graph(
        G,
        title=title,
        output_path=args.output,
        figsize=figsize,
        show_labels=not args.no_labels,
        minimal=minimal_mode,
    )


if __name__ == "__main__":
    main()
