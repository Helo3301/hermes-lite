"""Confidence Estimation and Adaptive Retrieval Depth."""

import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Confidence assessment of retrieval results."""
    overall: float          # 0-1 overall confidence
    score_distribution: float   # How tight is the score cluster
    entity_coverage: float      # Do results cover query entities
    source_agreement: float     # Do multiple sources agree
    source_diversity: float     # Are results from diverse sources
    explanation: str


class ConfidenceEstimator:
    """Estimate confidence in retrieved results."""

    def __init__(
        self,
        min_results: int = 3,
        score_threshold: float = 0.5,
        coverage_weight: float = 0.3,
        agreement_weight: float = 0.3,
        distribution_weight: float = 0.2,
        diversity_weight: float = 0.2,
    ):
        self.min_results = min_results
        self.score_threshold = score_threshold
        self.weights = {
            "coverage": coverage_weight,
            "agreement": agreement_weight,
            "distribution": distribution_weight,
            "diversity": diversity_weight,
        }

    def estimate(
        self,
        query: str,
        chunks: list[dict],
        query_entities: Optional[list[str]] = None,
        scores: Optional[list[float]] = None
    ) -> ConfidenceScore:
        """
        Estimate confidence in retrieval results.

        Factors considered:
        - Score distribution (tight cluster = more confident)
        - Query entity coverage (entities found in results)
        - Source agreement (multiple chunks say similar things)
        - Source diversity (results from multiple documents)

        Args:
            query: The search query
            chunks: Retrieved chunks with 'content', 'score', 'doc_id' fields
            query_entities: Pre-extracted entities from query
            scores: Pre-computed relevance scores (optional)

        Returns:
            ConfidenceScore with breakdown of confidence factors
        """
        if not chunks:
            return ConfidenceScore(
                overall=0.0,
                score_distribution=0.0,
                entity_coverage=0.0,
                source_agreement=0.0,
                source_diversity=0.0,
                explanation="No results retrieved"
            )

        # Get scores
        if scores is None:
            scores = [c.get("score", 0.5) for c in chunks]

        # 1. Score distribution (tight cluster = more confident)
        score_dist = self._score_distribution_confidence(scores)

        # 2. Entity coverage
        entity_cov = self._entity_coverage_confidence(chunks, query_entities or [])

        # 3. Source agreement (do chunks agree on key facts)
        agreement = self._source_agreement_confidence(chunks)

        # 4. Source diversity (are results from multiple documents)
        diversity = self._source_diversity_confidence(chunks)

        # Calculate weighted overall score
        overall = (
            self.weights["distribution"] * score_dist +
            self.weights["coverage"] * entity_cov +
            self.weights["agreement"] * agreement +
            self.weights["diversity"] * diversity
        )

        # Build explanation
        explanation = self._build_explanation(
            score_dist, entity_cov, agreement, diversity, len(chunks)
        )

        return ConfidenceScore(
            overall=overall,
            score_distribution=score_dist,
            entity_coverage=entity_cov,
            source_agreement=agreement,
            source_diversity=diversity,
            explanation=explanation,
        )

    def _score_distribution_confidence(self, scores: list[float]) -> float:
        """Calculate confidence from score distribution."""
        if len(scores) < 2:
            return 0.5

        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)

        # High mean + low variance = high confidence
        # Low variance means results are consistently good (or bad)
        # We want high mean AND low variance

        mean_factor = mean_score  # Higher mean = better
        consistency_factor = 1 - min(std_dev * 2, 1)  # Lower std = better

        return (mean_factor * 0.6 + consistency_factor * 0.4)

    def _entity_coverage_confidence(
        self,
        chunks: list[dict],
        query_entities: list[str]
    ) -> float:
        """Calculate confidence from entity coverage."""
        if not query_entities:
            return 0.7  # No entities to check = moderate confidence

        all_text = " ".join(c.get("content", "") for c in chunks).lower()

        found = 0
        for entity in query_entities:
            if entity.lower() in all_text:
                found += 1

        return found / len(query_entities)

    def _source_agreement_confidence(self, chunks: list[dict]) -> float:
        """Calculate confidence from source agreement."""
        if len(chunks) < 2:
            return 0.5

        # Simple heuristic: look for overlapping key terms across chunks
        # More sophisticated: could use semantic similarity

        # Extract key terms from each chunk (simple approach)
        chunk_terms = []
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            # Extract capitalized terms (likely entities/methods)
            terms = set()
            import re
            for match in re.finditer(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b', chunk.get("content", "")):
                terms.add(match.group().lower())
            chunk_terms.append(terms)

        if not chunk_terms or all(len(t) == 0 for t in chunk_terms):
            return 0.5

        # Calculate pairwise overlap
        overlaps = []
        for i, terms_a in enumerate(chunk_terms):
            for terms_b in chunk_terms[i+1:]:
                if terms_a and terms_b:
                    intersection = len(terms_a & terms_b)
                    union = len(terms_a | terms_b)
                    if union > 0:
                        overlaps.append(intersection / union)

        if overlaps:
            return sum(overlaps) / len(overlaps)
        return 0.5

    def _source_diversity_confidence(self, chunks: list[dict]) -> float:
        """Calculate confidence from source diversity."""
        if len(chunks) < 2:
            return 0.3  # Single source = low diversity

        # Count unique documents
        doc_ids = set()
        for chunk in chunks:
            doc_id = chunk.get("doc_id", chunk.get("filename", id(chunk)))
            doc_ids.add(doc_id)

        # More diverse sources = higher confidence (up to a point)
        unique_ratio = len(doc_ids) / len(chunks)

        # We want diversity but not too much (3-5 sources is good)
        ideal_sources = min(5, len(chunks))
        if len(doc_ids) >= ideal_sources:
            return 1.0
        else:
            return len(doc_ids) / ideal_sources

    def _build_explanation(
        self,
        score_dist: float,
        entity_cov: float,
        agreement: float,
        diversity: float,
        num_chunks: int
    ) -> str:
        """Build human-readable explanation of confidence."""
        parts = []

        if score_dist > 0.7:
            parts.append("results have high relevance scores")
        elif score_dist < 0.3:
            parts.append("results have low relevance scores")

        if entity_cov > 0.8:
            parts.append("covers all query entities")
        elif entity_cov < 0.5:
            parts.append("missing some query entities")

        if agreement > 0.7:
            parts.append("sources agree on key points")
        elif agreement < 0.3:
            parts.append("sources have limited agreement")

        if diversity > 0.7:
            parts.append("information from diverse sources")
        elif diversity < 0.3:
            parts.append("limited source diversity")

        if not parts:
            return f"Based on {num_chunks} results"

        return f"Based on {num_chunks} results: " + "; ".join(parts)


class AdaptiveRetriever:
    """Dynamically adjust retrieval depth based on confidence."""

    def __init__(
        self,
        search_fn,
        min_k: int = 5,
        max_k: int = 30,
        confidence_threshold: float = 0.7,
        step_size: int = 5,
    ):
        """
        Initialize adaptive retriever.

        Args:
            search_fn: Function(query, top_k) -> list[dict]
            min_k: Minimum number of results
            max_k: Maximum number of results
            confidence_threshold: Target confidence level
            step_size: How many more results to fetch per iteration
        """
        self.search_fn = search_fn
        self.min_k = min_k
        self.max_k = max_k
        self.confidence_threshold = confidence_threshold
        self.step_size = step_size
        self.estimator = ConfidenceEstimator()

    def search(
        self,
        query: str,
        query_entities: Optional[list[str]] = None,
        initial_k: Optional[int] = None
    ) -> dict:
        """
        Perform adaptive search that adjusts depth based on confidence.

        Args:
            query: The search query
            query_entities: Pre-extracted entities
            initial_k: Starting number of results (default: min_k)

        Returns:
            Dict with:
            - results: Final chunks
            - confidence: ConfidenceScore
            - final_k: Actual retrieval depth
            - iterations: Number of retrieval rounds
        """
        current_k = initial_k or self.min_k
        iterations = 0

        while current_k <= self.max_k:
            iterations += 1

            # Retrieve results
            results = self.search_fn(query, top_k=current_k)

            # Estimate confidence
            confidence = self.estimator.estimate(
                query,
                results,
                query_entities=query_entities
            )

            logger.debug(
                f"Adaptive search: k={current_k}, confidence={confidence.overall:.2f}"
            )

            # Check if confidence is good enough
            if confidence.overall >= self.confidence_threshold:
                return {
                    "results": results,
                    "confidence": confidence,
                    "final_k": current_k,
                    "iterations": iterations,
                    "status": "confident",
                }

            # Check if we've hit max
            if current_k >= self.max_k:
                return {
                    "results": results,
                    "confidence": confidence,
                    "final_k": current_k,
                    "iterations": iterations,
                    "status": "max_reached",
                }

            # Increase k for next iteration
            current_k = min(current_k + self.step_size, self.max_k)

        # Shouldn't reach here, but just in case
        return {
            "results": results if 'results' in dir() else [],
            "confidence": confidence if 'confidence' in dir() else None,
            "final_k": current_k,
            "iterations": iterations,
            "status": "unknown",
        }


def adaptive_search(
    search_fn,
    query: str,
    query_entities: Optional[list[str]] = None,
    min_k: int = 5,
    max_k: int = 30,
    confidence_threshold: float = 0.7,
) -> dict:
    """
    Convenience function for adaptive search.

    Args:
        search_fn: Function(query, top_k) -> list[dict]
        query: The search query
        query_entities: Pre-extracted entities
        min_k: Minimum results
        max_k: Maximum results
        confidence_threshold: Target confidence

    Returns:
        Same as AdaptiveRetriever.search()
    """
    retriever = AdaptiveRetriever(
        search_fn=search_fn,
        min_k=min_k,
        max_k=max_k,
        confidence_threshold=confidence_threshold,
    )
    return retriever.search(query, query_entities)
