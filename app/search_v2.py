"""HERMES v2 Advanced Search Pipeline.

Integrates all Phase 1-5 components:
- Query Intelligence (classification, decomposition, entity extraction)
- Iterative Gap-Fill (SEAL-RAG style fixed-budget retrieval)
- Knowledge Graph (entity/relationship augmentation)
- Contradiction Detection (claim extraction, conflict surfacing)
- Adaptive Retrieval (confidence-based depth adjustment)
"""

import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any

from .query_analyzer import QueryAnalyzer, QueryType
from .gap_detector import GapDetector
from .chunk_scorer import ChunkScorer
from .retrieval import iterative_retrieve
from .confidence_estimator import ConfidenceEstimator, AdaptiveRetriever
from .contradiction_detector import ContradictionDetector

logger = logging.getLogger(__name__)


@dataclass
class SearchV2Config:
    """Configuration for v2 search pipeline."""
    # Query analysis
    use_query_analysis: bool = True
    decompose_multi_hop: bool = True

    # Iterative retrieval
    use_iterative: bool = True
    max_iterations: int = 3
    budget: int = 10

    # Adaptive depth
    use_adaptive: bool = True
    min_k: int = 5
    max_k: int = 30
    confidence_threshold: float = 0.6

    # Contradiction detection
    detect_contradictions: bool = True
    surface_contradictions: bool = True

    # Reranking
    use_rerank: bool = True

    # Timeouts (milliseconds)
    simple_query_timeout: int = 500
    multi_hop_timeout: int = 10000
    exploratory_timeout: int = 300000  # 5 minutes


@dataclass
class SearchV2Result:
    """Result from v2 search pipeline."""
    query: str
    query_type: str
    sub_queries: list[str]
    entities: list[str]
    results: list[dict]
    confidence: float
    confidence_explanation: str
    contradictions: Optional[dict]
    iterations: int
    timing_ms: float
    status: str


class SearchV2Pipeline:
    """Advanced search pipeline with all v2 enhancements."""

    def __init__(
        self,
        search_fn: Callable,
        config: Optional[SearchV2Config] = None,
        use_llm: bool = False,
    ):
        """
        Initialize v2 search pipeline.

        Args:
            search_fn: Base search function(query, top_k) -> list[dict]
            config: Pipeline configuration
            use_llm: Whether to use LLM for enhanced analysis
        """
        self.search_fn = search_fn
        self.config = config or SearchV2Config()
        self.use_llm = use_llm

        # Initialize components
        self.query_analyzer = QueryAnalyzer(use_llm=use_llm)
        self.gap_detector = GapDetector(query_analyzer=self.query_analyzer)
        self.chunk_scorer = ChunkScorer()
        self.confidence_estimator = ConfidenceEstimator()
        self.contradiction_detector = ContradictionDetector(use_llm=use_llm)

        logger.info("SearchV2Pipeline initialized")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = None,
        detect_contradictions: bool = None,
        **kwargs
    ) -> SearchV2Result:
        """
        Execute v2 search pipeline.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            rerank: Override config rerank setting
            detect_contradictions: Override config contradiction setting
            **kwargs: Additional args passed to base search

        Returns:
            SearchV2Result with full analysis
        """
        start_time = time.time()

        # Use config defaults if not overridden
        rerank = rerank if rerank is not None else self.config.use_rerank
        detect_contradictions = (
            detect_contradictions if detect_contradictions is not None
            else self.config.detect_contradictions
        )

        # Phase 1: Query Analysis
        query_analysis = None
        query_type = QueryType.SIMPLE
        sub_queries = []
        entities = []

        if self.config.use_query_analysis:
            query_analysis = self.query_analyzer.analyze(query)
            query_type = query_analysis.query_type
            # Extract query strings from SubQuery objects
            sub_queries = [sq.query for sq in query_analysis.sub_queries]
            entities = query_analysis.entities

            logger.debug(f"Query type: {query_type.value}, entities: {entities}")

        # Determine retrieval strategy based on query type
        if query_type == QueryType.SIMPLE:
            results, confidence, iterations = self._simple_search(
                query, top_k, entities, rerank, kwargs
            )
        elif query_type == QueryType.MULTI_HOP:
            results, confidence, iterations = self._multi_hop_search(
                query, sub_queries, top_k, entities, rerank, kwargs
            )
        elif query_type == QueryType.COMPARATIVE:
            results, confidence, iterations = self._comparative_search(
                query, sub_queries, top_k, entities, rerank, kwargs
            )
        else:  # EXPLORATORY
            results, confidence, iterations = self._exploratory_search(
                query, sub_queries, top_k, entities, rerank, kwargs
            )

        # Phase 4: Contradiction Detection
        contradictions = None
        if detect_contradictions and len(results) >= 2:
            chunks_for_detection = [
                {
                    "id": r.get("id"),
                    "content": r.get("content", ""),
                    "filename": r.get("filename", ""),
                }
                for r in results
            ]
            detected = self.contradiction_detector.detect_contradictions(
                chunks_for_detection
            )
            if detected:
                contradictions = self.contradiction_detector.surface_contradictions(
                    detected
                )

        # Calculate timing
        elapsed_ms = (time.time() - start_time) * 1000

        # Determine status
        status = "success"
        if not results:
            status = "no_results"
        elif confidence.overall < 0.3:
            status = "low_confidence"

        return SearchV2Result(
            query=query,
            query_type=query_type.value,
            sub_queries=sub_queries,
            entities=entities,
            results=results,
            confidence=confidence.overall,
            confidence_explanation=confidence.explanation,
            contradictions=contradictions,
            iterations=iterations,
            timing_ms=round(elapsed_ms, 2),
            status=status,
        )

    def _simple_search(
        self,
        query: str,
        top_k: int,
        entities: list[str],
        rerank: bool,
        kwargs: dict
    ) -> tuple[list[dict], Any, int]:
        """Simple single-pass search for basic queries."""
        if self.config.use_adaptive:
            # Use adaptive retrieval
            retriever = AdaptiveRetriever(
                search_fn=lambda q, **kw: self.search_fn(q, rerank=rerank, **kwargs, **kw),
                min_k=min(self.config.min_k, top_k),
                max_k=top_k,
                confidence_threshold=self.config.confidence_threshold,
            )
            result = retriever.search(query, query_entities=entities)
            return (
                result["results"][:top_k],
                result["confidence"],
                result["iterations"],
            )
        else:
            # Standard search
            results = self.search_fn(query, top_k=top_k, rerank=rerank, **kwargs)
            confidence = self.confidence_estimator.estimate(
                query, results, query_entities=entities
            )
            return results, confidence, 1

    def _multi_hop_search(
        self,
        query: str,
        sub_queries: list[str],
        top_k: int,
        entities: list[str],
        rerank: bool,
        kwargs: dict
    ) -> tuple[list[dict], Any, int]:
        """Iterative search for multi-hop queries (SEAL-RAG style)."""
        if not self.config.use_iterative:
            return self._simple_search(query, top_k, entities, rerank, kwargs)

        # Use iterative retrieval with gap detection
        result = iterative_retrieve(
            search_fn=lambda q, **kw: self.search_fn(q, rerank=rerank, **kwargs, **kw),
            query=query,
            budget=min(self.config.budget, top_k),
            max_iterations=self.config.max_iterations,
            query_analyzer=self.query_analyzer,
            gap_detector=self.gap_detector,
            chunk_scorer=self.chunk_scorer,
            verbose=False,
        )

        # Calculate confidence
        confidence = self.confidence_estimator.estimate(
            query, result["results"], query_entities=entities
        )

        return result["results"], confidence, result["iterations"]

    def _comparative_search(
        self,
        query: str,
        sub_queries: list[str],
        top_k: int,
        entities: list[str],
        rerank: bool,
        kwargs: dict
    ) -> tuple[list[dict], Any, int]:
        """Search for comparative queries (multiple entities to compare)."""
        if not sub_queries:
            # Fall back to simple search
            return self._simple_search(query, top_k, entities, rerank, kwargs)

        # Search for each sub-query and merge
        all_results = []
        seen_ids = set()
        per_query_k = max(3, top_k // len(sub_queries))

        for sq in sub_queries:
            results = self.search_fn(sq, top_k=per_query_k, rerank=rerank, **kwargs)
            for r in results:
                if r.get("id") not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.get("id"))

        # Also search original query for context
        original_results = self.search_fn(query, top_k=per_query_k, rerank=rerank, **kwargs)
        for r in original_results:
            if r.get("id") not in seen_ids:
                all_results.append(r)
                seen_ids.add(r.get("id"))

        # Score and rank
        scored_results = []
        for r in all_results:
            score = self.chunk_scorer.score(
                chunk=r,
                query=query,
                query_entities=entities,
            )
            r["v2_score"] = score.combined
            scored_results.append(r)

        # Sort by combined score
        scored_results.sort(key=lambda x: x.get("v2_score", 0), reverse=True)
        final_results = scored_results[:top_k]

        # Calculate confidence
        confidence = self.confidence_estimator.estimate(
            query, final_results, query_entities=entities
        )

        return final_results, confidence, len(sub_queries)

    def _exploratory_search(
        self,
        query: str,
        sub_queries: list[str],
        top_k: int,
        entities: list[str],
        rerank: bool,
        kwargs: dict
    ) -> tuple[list[dict], Any, int]:
        """Broad search for exploratory queries."""
        # For exploratory, we want diversity and coverage
        # Use higher k and iterate if needed

        expanded_k = min(top_k * 2, self.config.max_k)

        if self.config.use_iterative:
            result = iterative_retrieve(
                search_fn=lambda q, **kw: self.search_fn(q, rerank=rerank, **kwargs, **kw),
                query=query,
                budget=expanded_k,
                max_iterations=self.config.max_iterations,
                query_analyzer=self.query_analyzer,
                gap_detector=self.gap_detector,
                chunk_scorer=self.chunk_scorer,
                verbose=False,
            )

            # Apply diversity filter - max 2 chunks per document
            final_results = self._apply_diversity(result["results"], max_per_doc=2)
            final_results = final_results[:top_k]

            confidence = self.confidence_estimator.estimate(
                query, final_results, query_entities=entities
            )

            return final_results, confidence, result["iterations"]
        else:
            results = self.search_fn(query, top_k=expanded_k, rerank=rerank, **kwargs)
            final_results = self._apply_diversity(results, max_per_doc=2)[:top_k]

            confidence = self.confidence_estimator.estimate(
                query, final_results, query_entities=entities
            )

            return final_results, confidence, 1

    def _apply_diversity(
        self,
        results: list[dict],
        max_per_doc: int = 2
    ) -> list[dict]:
        """Apply document diversity filter."""
        doc_counts = {}
        filtered = []

        for r in results:
            doc_id = r.get("doc_id") or r.get("filename", "")
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            if doc_counts[doc_id] <= max_per_doc:
                filtered.append(r)

        return filtered


def create_search_v2_fn(
    search_engine,
    config: Optional[SearchV2Config] = None,
    use_llm: bool = False,
) -> SearchV2Pipeline:
    """
    Create a v2 search pipeline from an existing search engine.

    Args:
        search_engine: SearchEngine instance with .search() method
        config: Pipeline configuration
        use_llm: Whether to use LLM for enhanced analysis

    Returns:
        Configured SearchV2Pipeline
    """
    def search_fn(query: str, top_k: int = 10, **kwargs) -> list[dict]:
        return search_engine.search(query=query, top_k=top_k, **kwargs)

    return SearchV2Pipeline(
        search_fn=search_fn,
        config=config,
        use_llm=use_llm,
    )
