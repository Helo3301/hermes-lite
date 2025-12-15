"""Gap Detection: Identify missing information after initial retrieval."""

import re
import json
import logging
from dataclasses import dataclass
from typing import Optional
import urllib.request

from .query_analyzer import QueryAnalyzer, QueryType

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class GapAnalysis:
    """Analysis of coverage gaps in retrieved results."""
    missing_entities: list[str]       # Entities from query not found in chunks
    unanswered_aspects: list[str]     # Parts of query not addressed
    coverage_score: float             # 0-1, how well query is covered
    suggested_subqueries: list[str]   # Targeted queries to fill gaps
    should_iterate: bool              # Whether another retrieval round is needed


def _call_llm(prompt: str, model: str = "llama3.2") -> str:
    """Make a call to local Ollama LLM."""
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 300}
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return ""


class GapDetector:
    """Detect gaps in retrieval results to guide iterative search."""

    def __init__(self, query_analyzer: Optional[QueryAnalyzer] = None):
        self.analyzer = query_analyzer or QueryAnalyzer(use_llm=False)
        self.coverage_threshold = 0.7  # Minimum coverage to skip iteration

    def analyze_coverage(
        self,
        query: str,
        retrieved_chunks: list[dict],
        query_entities: Optional[list[str]] = None,
    ) -> GapAnalysis:
        """
        Analyze how well retrieved chunks cover the query.

        Args:
            query: The original search query
            retrieved_chunks: List of retrieved chunk dicts with 'content' field
            query_entities: Pre-extracted entities (optional, will extract if not provided)

        Returns:
            GapAnalysis with coverage assessment and suggestions
        """
        # Extract entities from query if not provided
        if query_entities is None:
            query_entities = self.analyzer.extract_entities(query)

        # Combine all retrieved content
        all_content = " ".join(
            chunk.get("content", "") for chunk in retrieved_chunks
        ).lower()

        # Check entity coverage
        missing_entities = []
        found_entities = []
        for entity in query_entities:
            if entity.lower() in all_content:
                found_entities.append(entity)
            else:
                missing_entities.append(entity)

        entity_coverage = len(found_entities) / len(query_entities) if query_entities else 1.0

        # Analyze query aspects
        query_type = self.analyzer.classify(query)
        unanswered = self._find_unanswered_aspects(query, query_type, all_content)

        # Calculate overall coverage score
        aspect_coverage = 1.0 - (len(unanswered) * 0.2)  # -20% per unanswered aspect
        coverage_score = (entity_coverage * 0.6 + max(0, aspect_coverage) * 0.4)

        # Generate sub-queries to fill gaps
        suggested_subqueries = self._generate_gap_queries(
            query, missing_entities, unanswered
        )

        # Determine if iteration is needed
        should_iterate = (
            coverage_score < self.coverage_threshold or
            len(missing_entities) > 0 or
            (query_type in [QueryType.MULTI_HOP, QueryType.COMPARATIVE] and len(unanswered) > 0)
        )

        return GapAnalysis(
            missing_entities=missing_entities,
            unanswered_aspects=unanswered,
            coverage_score=coverage_score,
            suggested_subqueries=suggested_subqueries,
            should_iterate=should_iterate,
        )

    def _find_unanswered_aspects(
        self,
        query: str,
        query_type: QueryType,
        content: str
    ) -> list[str]:
        """Identify aspects of the query not addressed in content."""
        unanswered = []
        query_lower = query.lower()

        # For comparative queries, check if both sides are addressed
        if query_type == QueryType.COMPARATIVE:
            # Look for comparison indicators in content
            comparison_words = ["difference", "similar", "better", "worse", "compared", "vs"]
            has_comparison = any(word in content for word in comparison_words)
            if not has_comparison:
                unanswered.append("comparison between entities")

        # For multi-hop, check if relationship is explained
        if query_type == QueryType.MULTI_HOP:
            # Pattern: "How does X address Y"
            patterns = [
                (r'how does (.+) (?:address|solve|improve)', "mechanism/approach"),
                (r'what (?:role|part) does (.+) play', "role/contribution"),
                (r'what evidence .+ for (.+)', "supporting evidence"),
            ]
            for pattern, aspect in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    # Check if the answer discusses this aspect
                    aspect_words = aspect.split("/")
                    if not any(word in content for word in aspect_words):
                        unanswered.append(f"explanation of {aspect}")

        # For exploratory, check if multiple perspectives are covered
        if query_type == QueryType.EXPLORATORY:
            # Check for diversity of content
            if len(content) < 500:  # Very short combined content
                unanswered.append("broader coverage of topic")

        return unanswered

    def _generate_gap_queries(
        self,
        original_query: str,
        missing_entities: list[str],
        unanswered_aspects: list[str]
    ) -> list[str]:
        """Generate targeted sub-queries to fill identified gaps."""
        sub_queries = []

        # Create lookup queries for missing entities
        for entity in missing_entities[:2]:  # Limit to 2
            sub_queries.append(f"What is {entity}?")

        # Create queries for unanswered aspects
        for aspect in unanswered_aspects[:2]:  # Limit to 2
            if "comparison" in aspect:
                sub_queries.append(f"differences {' '.join(missing_entities[:2])}")
            elif "mechanism" in aspect or "approach" in aspect:
                sub_queries.append(f"how {original_query.split()[2:5]} works")
            elif "evidence" in aspect:
                sub_queries.append(f"results {' '.join(missing_entities[:1])}")
            else:
                sub_queries.append(original_query)  # Retry original

        # If no specific gaps, try a broader query
        if not sub_queries and len(original_query.split()) > 5:
            # Extract key terms
            entities = missing_entities or self.analyzer.extract_entities(original_query)
            if entities:
                sub_queries.append(f"{entities[0]} overview")

        return sub_queries

    def should_continue_iteration(
        self,
        iteration: int,
        max_iterations: int,
        gap_analysis: GapAnalysis,
        previous_coverage: Optional[float] = None
    ) -> bool:
        """
        Decide whether to continue iterating.

        Args:
            iteration: Current iteration number (1-indexed)
            max_iterations: Maximum allowed iterations
            gap_analysis: Current gap analysis
            previous_coverage: Coverage score from previous iteration

        Returns:
            True if should continue, False if should stop
        """
        # Hard stop at max iterations
        if iteration >= max_iterations:
            return False

        # Stop if coverage is good enough
        if gap_analysis.coverage_score >= self.coverage_threshold:
            return False

        # Stop if no improvement from previous iteration
        if previous_coverage is not None:
            improvement = gap_analysis.coverage_score - previous_coverage
            if improvement < 0.05:  # Less than 5% improvement
                logger.info(f"Stopping iteration: minimal improvement ({improvement:.2%})")
                return False

        # Continue if gaps exist and we have suggestions
        return gap_analysis.should_iterate and len(gap_analysis.suggested_subqueries) > 0


class LLMGapDetector(GapDetector):
    """Gap detector that uses LLM for more sophisticated analysis."""

    def analyze_coverage_with_llm(
        self,
        query: str,
        retrieved_chunks: list[dict],
    ) -> GapAnalysis:
        """Use LLM to analyze coverage gaps (slower but more accurate)."""
        # First do quick pattern-based analysis
        base_analysis = self.analyze_coverage(query, retrieved_chunks)

        # If coverage seems good, don't bother with LLM
        if base_analysis.coverage_score > 0.85:
            return base_analysis

        # Use LLM for more sophisticated analysis
        chunks_text = "\n---\n".join(
            chunk.get("content", "")[:500] for chunk in retrieved_chunks[:5]
        )

        prompt = f"""Analyze if these retrieved passages answer the query.

Query: {query}

Retrieved passages:
{chunks_text}

What information is MISSING to fully answer the query? List up to 3 missing aspects.
If the passages fully answer the query, say "COMPLETE".

Missing information:"""

        response = _call_llm(prompt)

        # Parse LLM response
        if "COMPLETE" in response.upper():
            return GapAnalysis(
                missing_entities=base_analysis.missing_entities,
                unanswered_aspects=[],
                coverage_score=0.95,
                suggested_subqueries=[],
                should_iterate=False,
            )

        # Extract missing aspects from LLM response
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        missing_aspects = [
            l.lstrip("0123456789.-) ")
            for l in lines
            if len(l) > 10 and not l.upper().startswith("COMPLETE")
        ][:3]

        # Generate queries for LLM-identified gaps
        suggested_queries = []
        for aspect in missing_aspects:
            if len(aspect) < 100:
                suggested_queries.append(aspect)

        return GapAnalysis(
            missing_entities=base_analysis.missing_entities,
            unanswered_aspects=missing_aspects,
            coverage_score=max(0.3, base_analysis.coverage_score - 0.2),
            suggested_subqueries=suggested_queries or base_analysis.suggested_subqueries,
            should_iterate=True,
        )
