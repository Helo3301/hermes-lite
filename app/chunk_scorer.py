"""Chunk Scoring: Multi-dimensional quality assessment for retrieved chunks."""

import re
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ChunkScore:
    """Multi-dimensional score for a chunk."""
    relevance: float      # 0-1: Semantic relevance to query
    specificity: float    # 0-1: Concrete vs vague content
    factual_density: float  # 0-1: Density of entities, numbers, citations
    source_quality: float   # 0-1: Recency, authoritativeness
    confidence: float     # 0-1: Hedging language vs confident claims
    combined: float       # Weighted combination


class ChunkScorer:
    """Score chunks on multiple quality dimensions."""

    def __init__(
        self,
        relevance_weight: float = 0.40,
        specificity_weight: float = 0.20,
        factual_weight: float = 0.20,
        source_weight: float = 0.10,
        confidence_weight: float = 0.10,
    ):
        self.weights = {
            "relevance": relevance_weight,
            "specificity": specificity_weight,
            "factual_density": factual_weight,
            "source_quality": source_weight,
            "confidence": confidence_weight,
        }

        # Hedging words that reduce confidence score
        self.hedging_words = {
            "might", "may", "could", "possibly", "perhaps", "likely",
            "unlikely", "seems", "appears", "suggests", "arguably",
            "somewhat", "potentially", "presumably", "tentatively",
        }

        # Confident assertion words
        self.confident_words = {
            "demonstrates", "shows", "proves", "establishes", "confirms",
            "achieves", "outperforms", "significantly", "clearly",
        }

    def score(
        self,
        chunk: dict,
        query: str,
        query_entities: Optional[list[str]] = None
    ) -> ChunkScore:
        """
        Score a chunk on multiple quality dimensions.

        Args:
            chunk: Chunk dict with 'content', 'filename', and optional 'score'
            query: The search query
            query_entities: Pre-extracted query entities (optional)

        Returns:
            ChunkScore with individual and combined scores
        """
        content = chunk.get("content", "")
        content_lower = content.lower()

        # 1. Relevance score (from retrieval or entity overlap)
        relevance = self._score_relevance(chunk, query, query_entities)

        # 2. Specificity score (concrete details vs vague)
        specificity = self._score_specificity(content)

        # 3. Factual density (entities, numbers, citations)
        factual_density = self._score_factual_density(content)

        # 4. Source quality (recency, paper quality signals)
        source_quality = self._score_source_quality(chunk)

        # 5. Confidence (hedging vs confident claims)
        confidence = self._score_confidence(content_lower)

        # Calculate weighted combination
        combined = (
            relevance * self.weights["relevance"] +
            specificity * self.weights["specificity"] +
            factual_density * self.weights["factual_density"] +
            source_quality * self.weights["source_quality"] +
            confidence * self.weights["confidence"]
        )

        return ChunkScore(
            relevance=relevance,
            specificity=specificity,
            factual_density=factual_density,
            source_quality=source_quality,
            confidence=confidence,
            combined=combined,
        )

    def _score_relevance(
        self,
        chunk: dict,
        query: str,
        query_entities: Optional[list[str]] = None
    ) -> float:
        """Score relevance based on retrieval score and entity overlap."""
        # Start with retrieval score if available
        base_score = chunk.get("score", 0.5)
        if isinstance(base_score, (int, float)):
            base_score = min(1.0, max(0.0, float(base_score)))
        else:
            base_score = 0.5

        # Boost for entity overlap
        if query_entities:
            content_lower = chunk.get("content", "").lower()
            entity_matches = sum(
                1 for e in query_entities if e.lower() in content_lower
            )
            entity_boost = min(0.2, entity_matches * 0.1)
            base_score = min(1.0, base_score + entity_boost)

        # Boost for query terms in content
        query_terms = set(query.lower().split())
        content_terms = set(chunk.get("content", "").lower().split())
        term_overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0
        term_boost = term_overlap * 0.1

        return min(1.0, base_score + term_boost)

    def _score_specificity(self, content: str) -> float:
        """Score how specific/concrete the content is."""
        if not content:
            return 0.0

        # Count specific indicators
        specificity_signals = 0

        # Numbers and percentages (concrete data)
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', content))
        specificity_signals += min(numbers, 5)  # Cap at 5

        # Proper nouns (specific entities)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content))
        specificity_signals += min(proper_nouns, 5)

        # Technical terms (hyphenated, acronyms)
        technical = len(re.findall(r'\b[A-Z]{2,}\b|\b\w+-\w+\b', content))
        specificity_signals += min(technical, 3)

        # Specific references
        references = len(re.findall(r'\([^)]*\d{4}[^)]*\)|\[\d+\]', content))
        specificity_signals += min(references, 3)

        # Normalize to 0-1 (max ~16 signals)
        return min(1.0, specificity_signals / 10)

    def _score_factual_density(self, content: str) -> float:
        """Score density of factual information."""
        if not content:
            return 0.0

        word_count = len(content.split())
        if word_count < 10:
            return 0.0

        # Count factual indicators per word
        factual_count = 0

        # Numbers and statistics
        factual_count += len(re.findall(r'\b\d+(?:\.\d+)?%?\b', content))

        # Named entities (capitalized words)
        factual_count += len(re.findall(r'\b[A-Z][a-z]+\b', content))

        # Technical terms
        factual_count += len(re.findall(r'\b[A-Z]{2,}\b', content))

        # Mathematical expressions
        factual_count += len(re.findall(r'[=<>≤≥±]', content))

        # Calculate density (facts per 100 words)
        density = (factual_count / word_count) * 100

        # Normalize: 0-20 facts per 100 words -> 0-1
        return min(1.0, density / 20)

    def _score_source_quality(self, chunk: dict) -> float:
        """Score quality based on source characteristics."""
        score = 0.5  # Default middle score

        filename = chunk.get("filename", "")

        # arXiv papers are generally reliable
        if "arxiv" in filename.lower() or re.search(r'\d{4}\.\d+', filename):
            score += 0.2

        # Recent papers (2024-2025) get a small boost
        year_match = re.search(r'20(2[3-5])', filename)
        if year_match:
            year = int("20" + year_match.group(1))
            if year >= 2024:
                score += 0.15
            elif year >= 2023:
                score += 0.1

        # PDF format suggests formal publication
        if filename.endswith('.pdf'):
            score += 0.05

        return min(1.0, score)

    def _score_confidence(self, content_lower: str) -> float:
        """Score based on hedging vs confident language."""
        if not content_lower:
            return 0.5

        word_count = len(content_lower.split())
        if word_count < 10:
            return 0.5

        # Count hedging words
        hedge_count = sum(
            1 for word in self.hedging_words
            if re.search(rf'\b{word}\b', content_lower)
        )

        # Count confident words
        confident_count = sum(
            1 for word in self.confident_words
            if re.search(rf'\b{word}\b', content_lower)
        )

        # Calculate confidence score
        # More confident words -> higher score
        # More hedging words -> lower score
        hedge_ratio = hedge_count / (word_count / 100)  # hedges per 100 words
        confident_ratio = confident_count / (word_count / 100)

        # Base score of 0.5, adjusted by hedging/confidence
        score = 0.5 + (confident_ratio * 0.1) - (hedge_ratio * 0.1)

        return max(0.0, min(1.0, score))

    def rank_chunks(
        self,
        chunks: list[dict],
        query: str,
        query_entities: Optional[list[str]] = None,
        top_k: Optional[int] = None
    ) -> list[tuple[dict, ChunkScore]]:
        """
        Rank chunks by combined score.

        Args:
            chunks: List of chunk dicts
            query: The search query
            query_entities: Pre-extracted query entities
            top_k: Return only top K chunks

        Returns:
            List of (chunk, score) tuples sorted by combined score
        """
        scored = []
        for chunk in chunks:
            score = self.score(chunk, query, query_entities)
            scored.append((chunk, score))

        # Sort by combined score (descending)
        scored.sort(key=lambda x: x[1].combined, reverse=True)

        if top_k:
            scored = scored[:top_k]

        return scored

    def select_best_chunks(
        self,
        all_chunks: list[dict],
        query: str,
        budget: int = 10,
        query_entities: Optional[list[str]] = None,
        diversity_penalty: float = 0.1
    ) -> list[dict]:
        """
        Select the best chunks within a fixed budget.

        Implements the "replace, don't expand" strategy from SEAL-RAG.

        Args:
            all_chunks: All candidate chunks (from multiple retrieval rounds)
            query: The search query
            budget: Maximum number of chunks to return
            query_entities: Pre-extracted query entities
            diversity_penalty: Penalty for chunks from same document

        Returns:
            Selected chunks, best first
        """
        if not all_chunks:
            return []

        # Score all chunks
        scored = self.rank_chunks(all_chunks, query, query_entities)

        # Select with diversity consideration
        selected = []
        seen_docs = {}

        for chunk, score in scored:
            if len(selected) >= budget:
                break

            doc_id = chunk.get("doc_id", chunk.get("filename", "unknown"))

            # Apply diversity penalty for same document
            adjusted_score = score.combined
            if doc_id in seen_docs:
                doc_count = seen_docs[doc_id]
                adjusted_score *= (1.0 - diversity_penalty * doc_count)

            # Only add if score is still good enough
            if adjusted_score > 0.1 or len(selected) < budget // 2:
                selected.append(chunk)
                seen_docs[doc_id] = seen_docs.get(doc_id, 0) + 1

        return selected
