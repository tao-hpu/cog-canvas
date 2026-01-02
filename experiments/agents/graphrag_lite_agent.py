"""
GraphRAG-lite Agent: Simplified GraphRAG baseline with entity-relation graph.

This agent implements a simplified version of Microsoft's GraphRAG:
1. Entity Extraction: LLM-based extraction from conversation turns
2. Relation Linking: Co-occurrence relations (entities in same turn)
3. Graph Storage: NetworkX graph
4. Retrieval: Entity matching from question + 1-2 hop neighbor expansion

Key differences from RAG baseline:
- Extracts structured entities instead of raw text chunks
- Builds entity-relation graph for multi-hop retrieval
- Uses graph traversal instead of pure vector search

Expected performance:
- Better than RAG for questions requiring connecting multiple entities
- May struggle with exact verbatim recall (no grounding quotes)
- Different trade-offs compared to CogCanvas (fine-grained entities vs typed objects)

Reference: Microsoft GraphRAG (2024)
- Paper: https://arxiv.org/abs/2404.16130
- Implementation: https://github.com/microsoft/graphrag
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import time
import os
import json
import re
from dataclasses import dataclass, field
import networkx as nx

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry
from cogcanvas.embeddings import (
    APIEmbeddingBackend,
    MockEmbeddingBackend,
    batch_cosine_similarity,
)


@dataclass
class Entity:
    """Extracted entity from conversation."""

    name: str
    type: str  # PERSON, TECHNOLOGY, CONCEPT, DECISION, FACT, etc.
    description: str
    source_turns: List[int] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class Relation:
    """Relation between entities."""

    source: str
    target: str
    type: str  # CO_OCCUR, DEPENDS_ON, DECIDES, etc.
    source_turn: int
    description: str = ""


class GraphRAGLiteAgent(Agent):
    """
    GraphRAG-lite baseline: Entity extraction + Graph traversal retrieval.

    Workflow:
    1. On process_turn: Extract entities and relations -> Update graph
    2. On compression: Keep graph, discard raw turns
    3. On answer_question:
       - Extract entities from question
       - Match entities in graph (fuzzy + embedding similarity)
       - Expand to neighbors (1-2 hops)
       - Build context from subgraph
    """

    ENTITY_EXTRACTION_PROMPT = """Extract all important entities from the following conversation turn.
For each entity, provide:
1. name: The exact name or phrase
2. type: One of [PERSON, TECHNOLOGY, ORGANIZATION, CONCEPT, DECISION, FACT, NUMBER, DATE]
3. description: Brief description of the entity in context

Conversation:
{text}

Output as JSON array:
[
  {{"name": "PostgreSQL", "type": "TECHNOLOGY", "description": "The chosen database system"}},
  {{"name": "5 engineers", "type": "NUMBER", "description": "Team size for the project"}}
]

Only output the JSON array, no other text."""

    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        retain_recent: int = 5,
        max_hops: int = 1,  # Max hops for neighbor expansion
        top_k_entities: int = 10,  # Max entities to retrieve
        use_llm_extraction: bool = True,  # Use LLM for entity extraction
    ):
        """
        Initialize GraphRAG-lite agent.

        Args:
            model: LLM for extraction and answering
            embedding_model: Embedding model name
            retain_recent: Number of recent turns to keep in context
            max_hops: Max hops for graph traversal
            top_k_entities: Max entities to retrieve from graph
            use_llm_extraction: Whether to use LLM for entity extraction
        """
        from dotenv import load_dotenv

        load_dotenv()

        self.retain_recent = retain_recent
        self.max_hops = max_hops
        self.top_k_entities = top_k_entities
        self.use_llm_extraction = use_llm_extraction

        # Models
        self.model = model or os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
        embed_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "bge-large-zh-v1.5"
        )

        # Initialize LLM client
        self._client = None
        self._init_client()

        # Initialize Embedding backend
        try:
            embed_api_key = (
                os.getenv("EMBEDDING_API_KEY")
                or os.getenv("API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            embed_api_base = (
                os.getenv("EMBEDDING_API_BASE")
                or os.getenv("API_BASE")
                or os.getenv("OPENAI_API_BASE")
            )
            if embed_api_key:
                self.embedder = APIEmbeddingBackend(
                    model=embed_model_name,
                    api_key=embed_api_key,
                    api_base=embed_api_base,
                )
            else:
                print("Warning: EMBEDDING_API_KEY not set, using mock embeddings")
                self.embedder = MockEmbeddingBackend()
        except Exception as e:
            print(f"Failed to init embedding backend: {e}. Using mock.")
            self.embedder = MockEmbeddingBackend()

        # Graph state
        self.graph = nx.Graph()
        self._entity_index: Dict[str, Entity] = {}  # normalized_name -> Entity
        self._relations: List[Relation] = []

        # History
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []

    def _init_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if api_key:
                self._client = OpenAI(api_key=api_key, base_url=api_base)
        except ImportError:
            pass

    @property
    def name(self) -> str:
        return f"GraphRAG-lite(hops={self.max_hops}, k={self.top_k_entities})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self.graph.clear()
        self._entity_index.clear()
        self._relations.clear()
        self._history = []
        self._retained_history = []

    def process_turn(self, turn: ConversationTurn) -> None:
        """Store turn in history (extraction happens at compression time)."""
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression: Extract entities from old turns and build graph.
        """
        # Identify turns to process into graph
        turns_to_process = [t for t in self._history if t not in retained_turns]

        if turns_to_process:
            self._process_turns_to_graph(turns_to_process)

        # Update retained history
        self._retained_history = list(retained_turns)
        self._history = list(retained_turns)

    def _process_turns_to_graph(self, turns: List[ConversationTurn]) -> None:
        """Extract entities and relations from turns, build graph."""
        for turn in turns:
            turn_text = f"User: {turn.user}\nAssistant: {turn.assistant}"

            # Extract entities
            entities = self._extract_entities(turn_text, turn.turn_id)

            # Add entities to graph
            for entity in entities:
                self._add_entity_to_graph(entity)

            # Create co-occurrence relations
            if len(entities) > 1:
                for i, e1 in enumerate(entities):
                    for e2 in entities[i + 1 :]:
                        self._add_relation(
                            Relation(
                                source=self._normalize_name(e1.name),
                                target=self._normalize_name(e2.name),
                                type="CO_OCCUR",
                                source_turn=turn.turn_id,
                                description=f"Co-occurred in turn {turn.turn_id}",
                            )
                        )

    def _extract_entities(self, text: str, turn_id: int) -> List[Entity]:
        """Extract entities using LLM or fallback to regex."""
        if self.use_llm_extraction and self._client:
            return self._extract_entities_llm(text, turn_id)
        else:
            return self._extract_entities_regex(text, turn_id)

    def _extract_entities_llm(self, text: str, turn_id: int) -> List[Entity]:
        """Extract entities using LLM."""
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text)

        try:
            content = call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )

            # Parse JSON
            # Find JSON array in response
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
            else:
                entities_data = json.loads(content)

            entities = []
            for item in entities_data:
                entity = Entity(
                    name=item.get("name", ""),
                    type=item.get("type", "CONCEPT"),
                    description=item.get("description", ""),
                    source_turns=[turn_id],
                )
                if entity.name:
                    entities.append(entity)

            return entities

        except Exception as e:
            # Fallback to regex
            return self._extract_entities_regex(text, turn_id)

    def _extract_entities_regex(self, text: str, turn_id: int) -> List[Entity]:
        """Fallback: Extract entities using regex patterns."""
        entities = []
        text_lower = text.lower()

        # Technology names
        tech_patterns = [
            (r"\b(postgresql|mysql|mongodb|sqlite|redis|memcached)\b", "TECHNOLOGY"),
            (r"\b(fastapi|flask|django|express|react|vue|angular)\b", "TECHNOLOGY"),
            (r"\b(aws|azure|gcp|google cloud|digitalocean)\b", "TECHNOLOGY"),
            (r"\b(docker|kubernetes|terraform)\b", "TECHNOLOGY"),
            (r"\b(oauth|jwt|api key)\b", "TECHNOLOGY"),
        ]

        for pattern, entity_type in tech_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                entities.append(
                    Entity(
                        name=match,
                        type=entity_type,
                        description=f"Technology mentioned in conversation",
                        source_turns=[turn_id],
                    )
                )

        # Numbers with context
        number_patterns = re.findall(
            r"(\d+(?:\.\d+)?)\s*(gb|tb|mb|kb|%|requests?|hours?|minutes?|days?|weeks?|engineers?|developers?|dollars?|\$)",
            text_lower,
        )
        for num, unit in number_patterns:
            entities.append(
                Entity(
                    name=f"{num} {unit}",
                    type="NUMBER",
                    description=f"Numeric value mentioned",
                    source_turns=[turn_id],
                )
            )

        # Dollar amounts
        dollar_matches = re.findall(r"\$[\d,]+(?:\.\d{2})?", text)
        for match in dollar_matches:
            entities.append(
                Entity(
                    name=match,
                    type="NUMBER",
                    description="Dollar amount",
                    source_turns=[turn_id],
                )
            )

        # Quoted strings (often important)
        quoted = re.findall(r'"([^"]+)"', text)
        for q in quoted:
            if len(q) < 50 and len(q) > 2:
                entities.append(
                    Entity(
                        name=q,
                        type="CONCEPT",
                        description="Quoted term",
                        source_turns=[turn_id],
                    )
                )

        return entities

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        return name.lower().strip()

    def _add_entity_to_graph(self, entity: Entity) -> None:
        """Add entity to graph, merging if exists."""
        normalized = self._normalize_name(entity.name)

        if normalized in self._entity_index:
            # Merge: update description and source_turns
            existing = self._entity_index[normalized]
            existing.source_turns.extend(entity.source_turns)
            existing.source_turns = list(set(existing.source_turns))
            if entity.description and not existing.description:
                existing.description = entity.description
        else:
            # Add new entity
            self._entity_index[normalized] = entity
            self.graph.add_node(
                normalized,
                type=entity.type,
                description=entity.description,
                name=entity.name,  # Original name
                source_turns=entity.source_turns,
            )

    def _add_relation(self, relation: Relation) -> None:
        """Add relation to graph."""
        self._relations.append(relation)

        if relation.source in self.graph and relation.target in self.graph:
            # Add or update edge
            if self.graph.has_edge(relation.source, relation.target):
                # Update existing edge
                edge_data = self.graph[relation.source][relation.target]
                edge_data["count"] = edge_data.get("count", 1) + 1
            else:
                self.graph.add_edge(
                    relation.source, relation.target, type=relation.type, count=1
                )

    def answer_question(self, question: str) -> AgentResponse:
        """Answer question using graph-based retrieval."""
        start_time = time.time()

        # 1. Extract entities from question
        question_entities = self._extract_entities(question, turn_id=-1)

        # 2. Match entities in graph
        matched_nodes = self._match_entities_in_graph(question_entities, question)

        # 3. Expand to neighbors
        context_nodes = self._expand_neighbors(matched_nodes)

        # 4. Build context from subgraph
        graph_context = self._build_graph_context(context_nodes)

        # 5. Add recent history
        context_parts = []
        if graph_context:
            context_parts.append(
                "## Knowledge Graph Context (from earlier conversation)"
            )
            context_parts.append(graph_context)
            context_parts.append("")

        # Use _history (includes post-compression turns) instead of _retained_history
        if self._history:
            context_parts.append("## Recent Conversation")
            for turn in self._history:
                context_parts.append(f"User: {turn.user}")
                context_parts.append(f"Assistant: {turn.assistant}")
                context_parts.append("")

        context = (
            "\n".join(context_parts) if context_parts else "[No context available]"
        )

        # 6. Generate answer
        answer = self._generate_answer(context, question)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "matched_entities": len(matched_nodes),
                "context_entities": len(context_nodes),
            },
        )

    def _match_entities_in_graph(
        self, question_entities: List[Entity], question: str
    ) -> Set[str]:
        """Match question entities to graph nodes using fuzzy + embedding."""
        matched = set()

        # 1. Exact/fuzzy string matching
        for qe in question_entities:
            normalized = self._normalize_name(qe.name)
            if normalized in self.graph:
                matched.add(normalized)
            else:
                # Fuzzy match: check if question entity is substring of graph node or vice versa
                for node in self.graph.nodes():
                    if normalized in node or node in normalized:
                        matched.add(node)

        # 2. If no matches, try embedding similarity on question itself
        if not matched and self.graph.number_of_nodes() > 0:
            try:
                # Embed the question
                question_embedding = self.embedder.embed(question)

                # Embed all node names + descriptions
                node_texts = []
                node_names = []
                for node in self.graph.nodes():
                    data = self.graph.nodes[node]
                    text = f"{data.get('name', node)}: {data.get('description', '')}"
                    node_texts.append(text)
                    node_names.append(node)

                if node_texts:
                    node_embeddings = self.embedder.embed_batch(node_texts)
                    similarities = batch_cosine_similarity(
                        question_embedding, node_embeddings
                    )

                    # Get top-k similar nodes
                    scored = sorted(
                        zip(node_names, similarities), key=lambda x: x[1], reverse=True
                    )
                    for node, score in scored[: self.top_k_entities]:
                        if score > 0.3:  # Threshold
                            matched.add(node)
            except Exception:
                pass

        return matched

    def _expand_neighbors(self, seed_nodes: Set[str]) -> Set[str]:
        """Expand seed nodes to include neighbors up to max_hops."""
        expanded = set(seed_nodes)

        current_frontier = set(seed_nodes)
        for hop in range(self.max_hops):
            next_frontier = set()
            for node in current_frontier:
                if node in self.graph:
                    neighbors = set(self.graph.neighbors(node))
                    next_frontier.update(neighbors - expanded)

            expanded.update(next_frontier)
            current_frontier = next_frontier

            if not next_frontier:
                break

        # Limit total nodes
        if len(expanded) > self.top_k_entities * 3:
            # Prioritize seed nodes and their direct neighbors
            limited = set(seed_nodes)
            for node in seed_nodes:
                if node in self.graph:
                    limited.update(list(self.graph.neighbors(node))[:3])
            expanded = limited

        return expanded

    def _build_graph_context(self, nodes: Set[str]) -> str:
        """Build context string from graph subgraph."""
        if not nodes:
            return ""

        context_parts = []

        # Group by entity type
        by_type: Dict[str, List[str]] = {}
        for node in nodes:
            if node in self.graph:
                data = self.graph.nodes[node]
                entity_type = data.get("type", "UNKNOWN")
                if entity_type not in by_type:
                    by_type[entity_type] = []
                by_type[entity_type].append(node)

        # Format entities
        for entity_type, type_nodes in sorted(by_type.items()):
            context_parts.append(f"### {entity_type}")
            for node in sorted(type_nodes):
                data = self.graph.nodes[node]
                name = data.get("name", node)
                desc = data.get("description", "")
                turns = data.get("source_turns", [])
                turns_str = f" (from turns: {turns})" if turns else ""
                context_parts.append(f"- **{name}**: {desc}{turns_str}")
            context_parts.append("")

        # Add relationships
        subgraph = self.graph.subgraph(nodes)
        if subgraph.number_of_edges() > 0:
            context_parts.append("### Relationships")
            for u, v, data in subgraph.edges(data=True):
                u_name = self.graph.nodes[u].get("name", u)
                v_name = self.graph.nodes[v].get("name", v)
                rel_type = data.get("type", "related_to")
                context_parts.append(f"- {u_name} --[{rel_type}]--> {v_name}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer via LLM."""
        prompt = f"""You are an expert reasoning agent. Your goal is to answer the user's question by connecting discrete facts from the retrieved information.

## Retrieved Context
{context}

## Instructions
1. Analyze the retrieved information carefully
2. Even if pieces of information are not explicitly linked, use your reasoning to infer relationships
3. Synthesize a complete answer that explains the reasoning process

## Question
{question}

## Answer
"""

        if self._client is None:
            return "I don't have enough information."

        try:
            return call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
        except Exception as e:
            return f"Error: {e}"
