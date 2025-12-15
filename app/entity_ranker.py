"""
Entity-First Ranker: SEAL-RAG Style Chunk Ranking and Replacement

Key Insight from SEAL-RAG:
- Chunks are valuable if they cover MISSING ENTITIES, not just if they're generally relevant
- "Replace, Don't Expand" - fixed budget prevents context dilution
- 96% precision vs 22% baseline by focusing on entity coverage

This module implements entity-first ranking to improve precision from ~50% to >80%.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EntityFirstRanker:
    """
    SEAL-RAG style entity-first chunk ranking and replacement.

    Key principle: Chunks are valuable if they cover MISSING entities,
    not just if they're generally relevant.

    Business Rules:
    1. A chunk's value = fraction of required entities it covers
    2. Replacement happens ONLY when new chunk covers more missing entities
    3. Budget is FIXED - we replace, never expand (context dilution prevention)
    4. Entity matching is case-insensitive
    """

    def __init__(self, replacement_threshold: float = 0.1):
        """
        Initialize the entity-first ranker.

        Args:
            replacement_threshold: Minimum improvement required to suggest replacement.
                                   Default 0.1 means new chunk must cover 10% more
                                   missing entities than current chunk.
        """
        self.replacement_threshold = replacement_threshold

    def compute_entity_coverage(
        self,
        chunk: dict,
        target_entities: list[str]
    ) -> tuple[float, list[str]]:
        """
        Score chunk by how many target entities it contains.

        Business Rule: A chunk covering 2 of 3 required entities scores 0.67

        Args:
            chunk: Dict with 'content' key containing the chunk text
            target_entities: List of entity names to look for

        Returns:
            Tuple of (coverage_score, list_of_found_entities)
            - coverage_score: 0.0 to 1.0, fraction of entities found
            - found_entities: List of entity names that were found
        """
        if not target_entities:
            return 0.0, []

        content = chunk.get("content", "")
        content_lower = content.lower()

        found_entities = []
        for entity in target_entities:
            # Case-insensitive entity matching
            entity_lower = entity.lower()
            if entity_lower in content_lower:
                found_entities.append(entity)

        # Score is fraction of entities covered
        score = len(found_entities) / len(target_entities)

        return round(score, 2), found_entities

    def find_replacement_candidates(
        self,
        current_chunks: list[dict],
        new_chunks: list[dict],
        missing_entities: list[str]
    ) -> list[tuple[int, dict, float]]:
        """
        Find (current_idx, new_chunk, improvement) tuples for beneficial replacements.

        Business Rule: Only suggest replacement if new chunk covers MORE missing
        entities than the current chunk it would replace.

        Args:
            current_chunks: Currently selected chunks (the budget we're maintaining)
            new_chunks: Candidate chunks to potentially swap in
            missing_entities: Entities we still need to find

        Returns:
            List of (current_chunk_index, new_chunk, improvement_score) tuples.
            Only includes replacements where improvement > threshold.
        """
        if not missing_entities or not new_chunks:
            return []

        candidates = []

        # Score each new chunk by how many MISSING entities it covers
        new_chunk_scores = []
        for new_chunk in new_chunks:
            score, found = self.compute_entity_coverage(new_chunk, missing_entities)
            new_chunk_scores.append((new_chunk, score, found))

        # Score each current chunk by how many MISSING entities it covers
        current_chunk_scores = []
        for idx, current_chunk in enumerate(current_chunks):
            score, found = self.compute_entity_coverage(current_chunk, missing_entities)
            current_chunk_scores.append((idx, score, found))

        # Sort current chunks by score (ascending) - worst first for replacement
        current_chunk_scores.sort(key=lambda x: x[1])

        # Sort new chunks by score (descending) - best first
        new_chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Match best new chunks with worst current chunks
        used_new_chunks = set()
        for current_idx, current_score, _ in current_chunk_scores:
            for new_chunk, new_score, _ in new_chunk_scores:
                new_id = id(new_chunk)
                if new_id in used_new_chunks:
                    continue

                improvement = new_score - current_score

                if improvement > self.replacement_threshold:
                    candidates.append((current_idx, new_chunk, improvement))
                    used_new_chunks.add(new_id)
                    break

        return candidates

    def replace_chunks(
        self,
        current_chunks: list[dict],
        new_chunks: list[dict],
        missing_entities: list[str],
        budget: int
    ) -> tuple[list[dict], list[str]]:
        """
        Execute SEAL-RAG style replacement with fixed budget.

        Business Rules:
        1. NEVER exceed budget (this is SEAL-RAG's core "replace don't expand")
        2. Replace lowest-coverage chunks with highest-coverage new chunks
        3. Track which entities are still missing after replacement

        Args:
            current_chunks: Currently selected chunks
            new_chunks: Candidate chunks to potentially add
            missing_entities: Entities we need to find
            budget: Maximum number of chunks allowed (HARD LIMIT)

        Returns:
            Tuple of (final_chunks, still_missing_entities)
            - final_chunks: List of chunks, len <= budget
            - still_missing_entities: Entities not covered by final chunks
        """
        # Start with current chunks (copy to avoid mutation)
        result = list(current_chunks)

        # If we have room in budget, add new chunks first
        chunks_to_add = min(budget - len(result), len(new_chunks))
        if chunks_to_add > 0:
            # Score new chunks by entity coverage
            scored_new = []
            for chunk in new_chunks:
                score, _ = self.compute_entity_coverage(chunk, missing_entities)
                scored_new.append((chunk, score))
            scored_new.sort(key=lambda x: x[1], reverse=True)

            # Add the best new chunks within budget
            for chunk, _ in scored_new[:chunks_to_add]:
                result.append(chunk)

        # Now find replacement candidates for any remaining new chunks
        remaining_new = [c for c in new_chunks if c not in result]
        if remaining_new and len(result) > 0:
            candidates = self.find_replacement_candidates(
                result, remaining_new, missing_entities
            )

            # Execute replacements
            for current_idx, new_chunk, improvement in candidates:
                if improvement > 0:
                    result[current_idx] = new_chunk

        # Ensure we don't exceed budget (critical constraint)
        result = result[:budget]

        # Calculate still-missing entities
        all_found = set()
        for chunk in result:
            _, found = self.compute_entity_coverage(chunk, missing_entities)
            all_found.update(e.lower() for e in found)

        still_missing = [
            entity for entity in missing_entities
            if entity.lower() not in all_found
        ]

        return result, still_missing

    def rank_by_entity_coverage(
        self,
        chunks: list[dict],
        query_entities: list[str],
        top_k: int
    ) -> list[dict]:
        """
        Rank chunks by entity coverage and return top-k.

        This is the core ranking function that prioritizes chunks
        mentioning query entities over "distractor" chunks.

        Args:
            chunks: List of chunk dicts to rank
            query_entities: Entities from the query to match
            top_k: Number of chunks to return

        Returns:
            Top-k chunks sorted by entity coverage (descending)
        """
        if not chunks:
            return []

        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            score, found = self.compute_entity_coverage(chunk, query_entities)

            # Secondary scoring: position of first entity mention (earlier = better)
            content_lower = chunk.get("content", "").lower()
            position_bonus = 0.0
            for entity in found:
                pos = content_lower.find(entity.lower())
                if pos >= 0:
                    # Bonus for early mention (0.1 max)
                    position_bonus = max(position_bonus, 0.1 * (1 - pos / len(content_lower)))

            final_score = score + position_bonus
            scored_chunks.append((chunk, final_score, score))

        # Sort by final score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top-k chunks
        return [chunk for chunk, _, _ in scored_chunks[:top_k]]

    def score_chunk_for_missing_entities(
        self,
        chunk: dict,
        missing_entities: list[str],
        all_query_entities: Optional[list[str]] = None
    ) -> dict:
        """
        Comprehensive scoring of a chunk for entity-first ranking.

        Returns detailed scoring breakdown for debugging and analysis.

        Args:
            chunk: Chunk dict with 'content'
            missing_entities: Entities we still need
            all_query_entities: All entities from original query (optional)

        Returns:
            Dict with scoring breakdown:
            - missing_coverage: Score for missing entities (0-1)
            - found_missing: List of missing entities found
            - total_coverage: Score for all query entities (if provided)
            - is_valuable: Boolean - does this chunk help fill gaps?
        """
        missing_score, found_missing = self.compute_entity_coverage(
            chunk, missing_entities
        )

        result = {
            "missing_coverage": missing_score,
            "found_missing": found_missing,
            "is_valuable": len(found_missing) > 0
        }

        if all_query_entities:
            total_score, found_all = self.compute_entity_coverage(
                chunk, all_query_entities
            )
            result["total_coverage"] = total_score
            result["found_all"] = found_all

        return result
