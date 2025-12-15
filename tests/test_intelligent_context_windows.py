"""
Test Suite: Intelligent Chunk Context Windows (Spec 2)

When we find a matching chunk, we look at its neighbors.
But we only include neighbors that are ACTUALLY RELATED,
not just physically adjacent in the document.

The key insight: A chunk about "entity extraction" might be followed by
"Table 3: Hyperparameters" which is NOT related and would waste context.
"""

import pytest
import math

# Import from the implementation module
from app.retrieval import cosine_similarity, get_contextual_window


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def base_embedding():
    """A normalized base embedding for testing."""
    # Simple 10-dimensional embedding, normalized
    raw = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    norm = math.sqrt(sum(x*x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def related_embedding():
    """An embedding similar to base (similarity ~0.85)."""
    # Points mostly in same direction with some variation
    raw = [0.9, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0]
    norm = math.sqrt(sum(x*x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def unrelated_embedding():
    """An embedding orthogonal to base (similarity ~0.0)."""
    # Points in completely different direction
    raw = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    norm = math.sqrt(sum(x*x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def document_chunks(base_embedding, related_embedding, unrelated_embedding):
    """
    Simulate a document with chunks of varying relatedness.

    Layout:
    - Chunk 0: Related to chunk 2 (the match)
    - Chunk 1: Related to chunk 2 (the match)
    - Chunk 2: THE MATCHED CHUNK
    - Chunk 3: Related to chunk 2 (continues the thought)
    - Chunk 4: UNRELATED (it's a table or different section)
    - Chunk 5: UNRELATED
    """
    return [
        {
            'id': 100,
            'doc_id': 'paper-1',
            'chunk_index': 0,
            'content': 'Introduction to the method...',
            'embedding': related_embedding  # Similar to matched
        },
        {
            'id': 101,
            'doc_id': 'paper-1',
            'chunk_index': 1,
            'content': 'The core insight is...',
            'embedding': related_embedding  # Similar to matched
        },
        {
            'id': 102,
            'doc_id': 'paper-1',
            'chunk_index': 2,
            'content': 'SEAL-RAG introduces entity-first ranking...',
            'embedding': base_embedding  # THIS IS THE MATCHED CHUNK
        },
        {
            'id': 103,
            'doc_id': 'paper-1',
            'chunk_index': 3,
            'content': '...which prioritizes gap-closing evidence.',
            'embedding': related_embedding  # Continues the thought
        },
        {
            'id': 104,
            'doc_id': 'paper-1',
            'chunk_index': 4,
            'content': 'Table 3: Hyperparameters used in experiments.',
            'embedding': unrelated_embedding  # DIFFERENT TOPIC
        },
        {
            'id': 105,
            'doc_id': 'paper-1',
            'chunk_index': 5,
            'content': 'Figure 2: Architecture diagram.',
            'embedding': unrelated_embedding  # DIFFERENT TOPIC
        },
    ]


# ============================================================================
# TESTS: Cosine Similarity
# ============================================================================

class TestCosineSimilarity:
    """Test the similarity calculation itself."""

    def test_identical_embeddings_have_similarity_one(self):
        """Two identical embeddings should have similarity of 1.0."""
        emb = [1, 0, 0, 0]
        assert cosine_similarity(emb, emb) == pytest.approx(1.0)

    def test_orthogonal_embeddings_have_similarity_zero(self):
        """Perpendicular embeddings should have similarity of 0.0."""
        emb1 = [1, 0, 0, 0]
        emb2 = [0, 1, 0, 0]
        assert cosine_similarity(emb1, emb2) == pytest.approx(0.0)

    def test_opposite_embeddings_have_similarity_negative_one(self):
        """Opposite embeddings should have similarity of -1.0."""
        emb1 = [1, 0, 0, 0]
        emb2 = [-1, 0, 0, 0]
        assert cosine_similarity(emb1, emb2) == pytest.approx(-1.0)

    def test_empty_embeddings_return_zero(self):
        """Empty embeddings should return 0 (safe default)."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1, 2], []) == 0.0

    def test_similar_embeddings_have_high_similarity(self, base_embedding, related_embedding):
        """Similar embeddings should have similarity > 0.7."""
        sim = cosine_similarity(base_embedding, related_embedding)
        assert sim > 0.7, f"Expected high similarity, got {sim}"


# ============================================================================
# TESTS: Smart Context Windows
# ============================================================================

class TestContextWindows:
    """Test the intelligent context window selection."""

    def test_includes_related_neighbors(self, document_chunks, base_embedding):
        """Should include neighbors that are semantically related."""
        matched_chunk = document_chunks[2]  # The chunk with base_embedding

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=2,
            similarity_threshold=0.7
        )

        # Should include the matched chunk
        chunk_indices = [c['chunk_index'] for c in result]
        assert 2 in chunk_indices, "Must include the matched chunk"

        # Should include at least one related neighbor
        assert len(result) > 1, "Should include related neighbors"

    def test_excludes_unrelated_neighbors(self, document_chunks, base_embedding):
        """Should NOT include neighbors that are semantically unrelated."""
        matched_chunk = document_chunks[2]

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=3,  # Large enough to reach unrelated chunks
            similarity_threshold=0.7
        )

        chunk_indices = [c['chunk_index'] for c in result]

        # Should NOT include the unrelated chunks (4 and 5)
        assert 4 not in chunk_indices, "Should not include unrelated Table chunk"
        assert 5 not in chunk_indices, "Should not include unrelated Figure chunk"

    def test_stops_at_first_unrelated_chunk(self, document_chunks, base_embedding):
        """Should stop expanding when hitting unrelated content."""
        matched_chunk = document_chunks[2]

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=5,  # Would reach chunks 4, 5 if not stopped
            similarity_threshold=0.7
        )

        # Even with large window, should stop at chunk 3
        chunk_indices = [c['chunk_index'] for c in result]
        max_idx = max(chunk_indices)
        assert max_idx <= 3, f"Should stop at chunk 3, got up to {max_idx}"

    def test_maintains_document_order(self, document_chunks, base_embedding):
        """Returned chunks should be in document order (by chunk_index)."""
        matched_chunk = document_chunks[2]

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=2,
            similarity_threshold=0.5  # Lower threshold to include more
        )

        indices = [c['chunk_index'] for c in result]
        assert indices == sorted(indices), f"Should be sorted, got {indices}"


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestContextWindowsEdgeCases:
    """Test edge cases for context window selection."""

    def test_first_chunk_in_document(self, document_chunks, base_embedding):
        """Should handle case where matched chunk is first in document."""
        # Modify first chunk to be the "matched" one
        first_chunk = document_chunks[0].copy()
        first_chunk['embedding'] = base_embedding

        result = get_contextual_window(
            matched_chunk=first_chunk,
            all_chunks=document_chunks,
            window_size=2,
            similarity_threshold=0.7
        )

        # Should still work, just no "before" chunks
        assert len(result) >= 1
        assert result[0]['chunk_index'] == 0

    def test_last_chunk_in_document(self, document_chunks, base_embedding):
        """Should handle case where matched chunk is last in document."""
        last_chunk = document_chunks[-1].copy()
        last_chunk['embedding'] = base_embedding

        result = get_contextual_window(
            matched_chunk=last_chunk,
            all_chunks=document_chunks,
            window_size=2,
            similarity_threshold=0.7
        )

        # Should still work, just no "after" chunks
        assert len(result) >= 1

    def test_single_chunk_document(self, base_embedding):
        """Should handle document with only one chunk."""
        single_chunk = {
            'id': 1,
            'doc_id': 'tiny-paper',
            'chunk_index': 0,
            'content': 'The entire paper.',
            'embedding': base_embedding
        }

        result = get_contextual_window(
            matched_chunk=single_chunk,
            all_chunks=[single_chunk],
            window_size=2,
            similarity_threshold=0.7
        )

        assert len(result) == 1
        assert result[0]['id'] == 1

    def test_window_size_zero(self, document_chunks, base_embedding):
        """Window size of 0 should return only the matched chunk."""
        matched_chunk = document_chunks[2]

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=0,  # Don't look at neighbors at all
            similarity_threshold=0.7
        )

        assert len(result) == 1
        assert result[0]['chunk_index'] == 2


# ============================================================================
# TESTS: Threshold Behavior
# ============================================================================

class TestSimilarityThreshold:
    """Test how different thresholds affect inclusion."""

    def test_high_threshold_is_strict(self, document_chunks, base_embedding):
        """High threshold (0.9) should include fewer neighbors."""
        matched_chunk = document_chunks[2]

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=2,
            similarity_threshold=0.95  # Very strict
        )

        # Might only include the matched chunk itself
        assert len(result) <= 2

    def test_low_threshold_is_permissive(self, document_chunks, base_embedding):
        """Low threshold (0.3) should include more neighbors."""
        matched_chunk = document_chunks[2]

        result = get_contextual_window(
            matched_chunk=matched_chunk,
            all_chunks=document_chunks,
            window_size=2,
            similarity_threshold=0.3  # Very permissive
        )

        # Should include more chunks
        assert len(result) >= 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
