"""Search pipeline: semantic search, keyword search, RRF fusion, reranking."""
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
import logging

from .retrieval import (
    diversify_results,
    expand_query,
    merge_search_results,
    boost_multi_match_results,
    apply_recency_weight
)

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rrf_fuse(
    results_a: list[dict],
    results_b: list[dict],
    weights: tuple[float, float] = (0.65, 0.35),
    k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion combining two result sets.

    Args:
        results_a: First result set (e.g., semantic search)
        results_b: Second result set (e.g., keyword search)
        weights: Weights for each result set
        k: RRF constant (default 60)

    Returns:
        Combined and sorted results
    """
    scores = {}
    all_results = {}

    for rank, result in enumerate(results_a):
        doc_id = result['id']
        scores[doc_id] = scores.get(doc_id, 0) + weights[0] / (k + rank + 1)
        all_results[doc_id] = result

    for rank, result in enumerate(results_b):
        doc_id = result['id']
        scores[doc_id] = scores.get(doc_id, 0) + weights[1] / (k + rank + 1)
        if doc_id not in all_results:
            all_results[doc_id] = result

    # Sort by combined score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [all_results[doc_id] for doc_id in sorted_ids if doc_id in all_results]


class Reranker:
    """Cross-encoder reranker using FlagEmbedding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        use_fp16: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
            logger.info("Reranker model loaded")
        return self._model

    def rerank(self, query: str, results: list[dict], batch_size: int = 32) -> list[dict]:
        """
        Rerank results using cross-encoder model.

        Args:
            query: The search query
            results: List of result dicts with 'content' field
            batch_size: Number of pairs to score at once

        Returns:
            Results sorted by reranker score (highest first)
        """
        if not results:
            return results

        # Create query-document pairs
        pairs = [(query, r["content"]) for r in results]

        # Score all pairs
        scores = self.model.compute_score(pairs, normalize=True)

        # Handle single result case (compute_score returns float instead of list)
        if isinstance(scores, float):
            scores = [scores]

        # Sort by score (higher = more relevant)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        return [r for r, s in ranked]


class SearchEngine:
    """Hybrid search engine combining semantic and keyword search."""

    def __init__(
        self,
        database,
        embed_client,
        reranker: Optional[Reranker] = None,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
        rrf_k: int = 60
    ):
        self.db = database
        self.embed_client = embed_client
        self.reranker = reranker
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

    def semantic_search(
        self,
        query_embedding: list[float],
        limit: int = 30,
        doc_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Perform semantic (vector) search.

        Note: Using brute-force cosine similarity since sqlite-vec
        doesn't have HNSW indexing. Fine for <100k chunks.
        """
        all_chunks = self.db.get_all_chunks_for_search()

        # Filter by document name if specified
        if doc_filter:
            all_chunks = [c for c in all_chunks if doc_filter.lower() in c['filename'].lower()]

        # Calculate similarities
        for chunk in all_chunks:
            chunk['similarity'] = cosine_similarity(query_embedding, chunk['embedding'])

        # Sort by similarity (descending)
        all_chunks.sort(key=lambda x: x['similarity'], reverse=True)

        # Remove embedding from results to save memory
        for chunk in all_chunks[:limit]:
            chunk.pop('embedding', None)

        return all_chunks[:limit]

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        doc_filter: Optional[str] = None,
        use_query_expansion: bool = True,
        use_diversity: bool = True,
        max_per_doc: int = 3,
        use_recency_weight: bool = False,
        recency_weight: float = 0.10
    ) -> list[dict]:
        """
        Perform hybrid search with optional reranking and retrieval enhancements.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            rerank: Whether to use cross-encoder reranking
            doc_filter: Optional filename pattern to filter results
            use_query_expansion: Expand query with synonyms/acronyms
            use_diversity: Apply document diversity (soft constraint)
            max_per_doc: Max chunks per doc in first pass (when diversity enabled)
            use_recency_weight: Boost newer documents slightly
            recency_weight: How much to weight recency (0.10 = 10% max boost)

        Returns:
            List of search results with content, filename, etc.
        """
        logger.info(f"Searching for: {query[:50]}...")

        # 1. Query Expansion (Spec 3)
        if use_query_expansion:
            queries = expand_query(query)
            logger.info(f"Expanded query to {len(queries)} variants: {queries}")
        else:
            queries = [query]

        all_result_sets = []

        for q in queries:
            # 2. Embed query
            query_embedding = self.embed_client.embed_single(q)

            # 3. Semantic search
            semantic_results = self.semantic_search(
                query_embedding,
                limit=top_k * 3,
                doc_filter=doc_filter
            )

            # 4. Keyword search
            keyword_results = self.db.keyword_search(
                q,
                limit=top_k * 3,
                doc_filter=doc_filter
            )

            # 5. RRF Fusion
            combined = rrf_fuse(
                semantic_results,
                keyword_results,
                weights=(self.semantic_weight, self.keyword_weight),
                k=self.rrf_k
            )

            # Add score field for merging
            for i, r in enumerate(combined):
                r['score'] = 1.0 / (i + 1)  # Convert rank to score

            all_result_sets.append(combined)

        logger.info(f"Searched {len(queries)} query variants")

        # 6. Merge results from all queries (Spec 3)
        if len(all_result_sets) > 1:
            merged = merge_search_results(all_result_sets, top_k=top_k * 3)
            merged = boost_multi_match_results(merged)
            logger.info(f"Merged to {len(merged)} results")
        else:
            merged = all_result_sets[0] if all_result_sets else []

        # 7. Rerank top candidates
        if rerank and self.reranker and len(merged) > top_k:
            merged = self.reranker.rerank(query, merged[:50])
            logger.info("Reranked results")

        # 8. Apply document diversity (Spec 1)
        if use_diversity:
            merged = diversify_results(merged, top_k=top_k, max_per_doc=max_per_doc)
            logger.info(f"Applied diversity filter (max {max_per_doc} per doc)")
        else:
            merged = merged[:top_k]

        # 9. Apply recency weighting (Spec 5) - optional
        if use_recency_weight and merged:
            merged = apply_recency_weight(
                merged,
                reference_date=datetime.now(),
                recency_weight=recency_weight
            )
            logger.info("Applied recency weighting")

        return merged
