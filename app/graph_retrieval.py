"""Graph-Augmented Retrieval: Combine vector search with knowledge graph traversal."""

import logging
from typing import Optional
from dataclasses import dataclass

from .query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Result from graph-augmented search."""
    chunks: list[dict]
    graph_entities: list[dict]
    traversal_path: list[str]
    vector_results: int
    graph_results: int


class GraphRetriever:
    """Retriever that combines vector search with knowledge graph traversal."""

    def __init__(
        self,
        database,
        search_engine,
        query_analyzer: Optional[QueryAnalyzer] = None,
        graph_weight: float = 0.3,
        max_hops: int = 1
    ):
        """
        Initialize the graph retriever.

        Args:
            database: Database instance with entity/relationship methods
            search_engine: SearchEngine instance for vector search
            query_analyzer: Optional QueryAnalyzer for entity extraction
            graph_weight: Weight given to graph results (0-1)
            max_hops: Maximum graph traversal depth
        """
        self.db = database
        self.search_engine = search_engine
        self.analyzer = query_analyzer or QueryAnalyzer(use_llm=False)
        self.graph_weight = graph_weight
        self.max_hops = max_hops

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_graph: bool = True,
        rerank: bool = True
    ) -> GraphSearchResult:
        """
        Perform graph-augmented search.

        1. Extract entities from query
        2. Standard vector search
        3. Match query entities to graph entities
        4. Traverse graph to find related entities
        5. Retrieve chunks for related entities
        6. Merge and re-rank results

        Args:
            query: The search query
            top_k: Number of results to return
            use_graph: Whether to use graph augmentation
            rerank: Whether to rerank combined results

        Returns:
            GraphSearchResult with combined results
        """
        # 1. Extract entities from query
        query_entities = self.analyzer.extract_entities(query)
        logger.info(f"Query entities: {query_entities}")

        # 2. Standard vector search (get extra for merging)
        vector_results = self.search_engine.search(
            query,
            top_k=top_k * 2 if use_graph else top_k,
            rerank=rerank
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

        if not use_graph:
            return GraphSearchResult(
                chunks=vector_results[:top_k],
                graph_entities=[],
                traversal_path=[],
                vector_results=len(vector_results),
                graph_results=0,
            )

        # 3. Match query entities to graph entities
        matched_entities = []
        for entity_name in query_entities:
            entity = self.db.get_entity_by_name(entity_name)
            if entity:
                matched_entities.append(entity)
                logger.info(f"Matched entity: {entity_name} -> {entity['id']}")

        # 4. Traverse graph to find related entities
        related_entities = []
        traversal_path = []

        for entity in matched_entities:
            related = self.db.get_related_entities(
                entity['id'],
                direction="both"
            )
            for rel in related:
                if rel['id'] not in [e['id'] for e in related_entities]:
                    related_entities.append(rel)
                    traversal_path.append(
                        f"{entity['name']} --[{rel['relationship_type']}]--> {rel['name']}"
                    )

        logger.info(f"Found {len(related_entities)} related entities via graph")

        # 5. Retrieve chunks for related entities
        graph_chunks = []
        for entity in related_entities[:5]:  # Limit to top 5 related
            chunks = self.db.get_chunks_for_entity(entity['name'])
            for chunk in chunks[:3]:  # Max 3 chunks per entity
                if chunk['id'] not in [c['id'] for c in graph_chunks]:
                    chunk['_from_graph'] = True
                    chunk['_related_entity'] = entity['name']
                    graph_chunks.append(chunk)

        logger.info(f"Retrieved {len(graph_chunks)} chunks via graph")

        # 6. Merge results
        merged = self._merge_results(
            vector_results,
            graph_chunks,
            query,
            top_k
        )

        return GraphSearchResult(
            chunks=merged,
            graph_entities=related_entities,
            traversal_path=traversal_path,
            vector_results=len(vector_results),
            graph_results=len(graph_chunks),
        )

    def _merge_results(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        query: str,
        top_k: int
    ) -> list[dict]:
        """
        Merge vector search results with graph-retrieved results.

        Strategy:
        - Score vector results based on position (higher = better)
        - Score graph results with a base score + entity relevance bonus
        - Combine scores with weights
        - Deduplicate by chunk ID
        - Sort and return top_k
        """
        merged = {}

        # Score vector results (1.0 for first, decreasing)
        for i, chunk in enumerate(vector_results):
            chunk_id = chunk.get('id')
            vector_score = 1.0 / (i + 1)  # RRF-style scoring
            chunk['_vector_score'] = vector_score
            chunk['_combined_score'] = vector_score * (1 - self.graph_weight)
            merged[chunk_id] = chunk

        # Add graph results with graph weight
        for chunk in graph_results:
            chunk_id = chunk.get('id')

            if chunk_id in merged:
                # Boost existing result found via graph
                merged[chunk_id]['_graph_boost'] = 0.2
                merged[chunk_id]['_combined_score'] += 0.2 * self.graph_weight
                merged[chunk_id]['_from_graph'] = True
            else:
                # New result from graph
                base_score = 0.5  # Base score for graph-only results
                chunk['_vector_score'] = 0.0
                chunk['_graph_score'] = base_score
                chunk['_combined_score'] = base_score * self.graph_weight
                merged[chunk_id] = chunk

        # Sort by combined score and return top_k
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.get('_combined_score', 0),
            reverse=True
        )

        return sorted_results[:top_k]

    def find_entity_context(
        self,
        entity_name: str,
        max_chunks: int = 5
    ) -> dict:
        """
        Get comprehensive context for a specific entity.

        Returns:
            Dict with entity info, related entities, and relevant chunks
        """
        # Find the entity
        entity = self.db.get_entity_by_name(entity_name)
        if not entity:
            return {
                "entity": None,
                "related": [],
                "chunks": [],
                "error": f"Entity '{entity_name}' not found"
            }

        # Get related entities
        related = self.db.get_related_entities(entity['id'])

        # Get chunks mentioning this entity
        chunks = self.db.get_chunks_for_entity(entity_name)[:max_chunks]

        return {
            "entity": entity,
            "related": related,
            "chunks": chunks,
        }

    def build_subgraph(
        self,
        entity_names: list[str],
        max_hops: int = 1
    ) -> dict:
        """
        Build a subgraph around specified entities.

        Args:
            entity_names: Starting entities
            max_hops: How many relationship hops to traverse

        Returns:
            Dict with nodes and edges for visualization
        """
        nodes = {}
        edges = []

        # Start with specified entities
        for name in entity_names:
            entity = self.db.get_entity_by_name(name)
            if entity:
                nodes[entity['id']] = {
                    "id": entity['id'],
                    "name": entity['name'],
                    "type": entity['entity_type'],
                    "depth": 0,
                }

        # Traverse relationships
        current_ids = list(nodes.keys())

        for hop in range(max_hops):
            next_ids = []

            for entity_id in current_ids:
                related = self.db.get_related_entities(entity_id)

                for rel in related:
                    # Add node if new
                    if rel['id'] not in nodes:
                        nodes[rel['id']] = {
                            "id": rel['id'],
                            "name": rel['name'],
                            "type": rel['entity_type'],
                            "depth": hop + 1,
                        }
                        next_ids.append(rel['id'])

                    # Add edge
                    edge_key = (entity_id, rel['id'], rel['relationship_type'])
                    edges.append({
                        "source": entity_id,
                        "target": rel['id'],
                        "type": rel['relationship_type'],
                        "direction": rel['direction'],
                    })

            current_ids = next_ids

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
        }


def augment_search_with_graph(
    search_fn,
    database,
    query: str,
    top_k: int = 10,
    query_analyzer: Optional[QueryAnalyzer] = None
) -> list[dict]:
    """
    Standalone function to augment search results with graph information.

    This can be used when you don't want to use the full GraphRetriever class.

    Args:
        search_fn: Function(query, top_k) -> list[dict]
        database: Database with entity/relationship methods
        query: The search query
        top_k: Number of results
        query_analyzer: Optional QueryAnalyzer

    Returns:
        Augmented search results
    """
    analyzer = query_analyzer or QueryAnalyzer(use_llm=False)

    # Get base results
    results = search_fn(query, top_k * 2)

    # Extract query entities
    query_entities = analyzer.extract_entities(query)
    if not query_entities:
        return results[:top_k]

    # Find related entities via graph
    additional_chunks = []
    for entity_name in query_entities[:3]:
        entity = database.get_entity_by_name(entity_name)
        if entity:
            # Get 1-hop related entities
            related = database.get_related_entities(entity['id'])[:5]

            for rel in related:
                # Get chunks for related entity
                chunks = database.get_chunks_for_entity(rel['name'])[:2]
                for chunk in chunks:
                    chunk['_from_graph'] = True
                    chunk['_via_entity'] = rel['name']
                    additional_chunks.append(chunk)

    # Merge (deduplicate by ID)
    seen_ids = {r['id'] for r in results}
    for chunk in additional_chunks:
        if chunk['id'] not in seen_ids:
            results.append(chunk)
            seen_ids.add(chunk['id'])

    return results[:top_k]
