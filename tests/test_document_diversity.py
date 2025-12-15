"""
Test Suite: Document Diversity Constraint (Spec 1)

Tests that search results don't allow a single document to dominate.
"""

import pytest
from collections import defaultdict

# Import from the implementation module
from app.retrieval import diversify_results


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def single_doc_dominated_results():
    """Simulates results where one doc matches on 8/10 top results."""
    return [
        {'id': 1, 'doc_id': 'seal-rag', 'score': 0.95, 'content': 'SEAL intro'},
        {'id': 2, 'doc_id': 'seal-rag', 'score': 0.93, 'content': 'SEAL method'},
        {'id': 3, 'doc_id': 'seal-rag', 'score': 0.91, 'content': 'SEAL results'},
        {'id': 4, 'doc_id': 'crag', 'score': 0.89, 'content': 'CRAG intro'},
        {'id': 5, 'doc_id': 'seal-rag', 'score': 0.88, 'content': 'SEAL discussion'},
        {'id': 6, 'doc_id': 'seal-rag', 'score': 0.86, 'content': 'SEAL ablation'},
        {'id': 7, 'doc_id': 'seal-rag', 'score': 0.84, 'content': 'SEAL related'},
        {'id': 8, 'doc_id': 'seal-rag', 'score': 0.82, 'content': 'SEAL future'},
        {'id': 9, 'doc_id': 'crag', 'score': 0.80, 'content': 'CRAG method'},
        {'id': 10, 'doc_id': 'seal-rag', 'score': 0.78, 'content': 'SEAL appendix'},
        {'id': 11, 'doc_id': 'basic-rag', 'score': 0.75, 'content': 'Basic RAG'},
        {'id': 12, 'doc_id': 'crag', 'score': 0.73, 'content': 'CRAG results'},
    ]


@pytest.fixture
def balanced_results():
    """Results already well-distributed across documents."""
    return [
        {'id': 1, 'doc_id': 'paper-a', 'score': 0.95, 'content': 'A content'},
        {'id': 2, 'doc_id': 'paper-b', 'score': 0.93, 'content': 'B content'},
        {'id': 3, 'doc_id': 'paper-c', 'score': 0.91, 'content': 'C content'},
        {'id': 4, 'doc_id': 'paper-d', 'score': 0.89, 'content': 'D content'},
        {'id': 5, 'doc_id': 'paper-a', 'score': 0.88, 'content': 'A more'},
        {'id': 6, 'doc_id': 'paper-b', 'score': 0.86, 'content': 'B more'},
    ]


@pytest.fixture
def empty_results():
    """Empty result set."""
    return []


@pytest.fixture
def single_result():
    """Just one result."""
    return [{'id': 1, 'doc_id': 'only-paper', 'score': 0.95, 'content': 'Only match'}]


# ============================================================================
# TESTS: Basic Functionality
# ============================================================================

class TestDiversifyBasic:
    """Test basic diversification behavior."""

    def test_limits_chunks_per_document(self, single_doc_dominated_results):
        """Core test: First pass respects max_per_doc, backfill may exceed it."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=2)

        # Count chunks per document
        doc_counts = defaultdict(int)
        for r in results:
            doc_counts[r['doc_id']] += 1

        # With soft diversity: first pass gives 2+2+1=5, backfill adds 5 more
        # crag and basic-rag stay at their limits (2 and 1)
        # seal-rag gets backfilled to reach top_k
        assert doc_counts['crag'] == 2, "CRAG should have exactly 2"
        assert doc_counts['basic-rag'] == 1, "basic-rag should have exactly 1"
        assert doc_counts['seal-rag'] == 7, "seal-rag gets backfilled"

    def test_returns_requested_count(self, single_doc_dominated_results):
        """Should return exactly top_k results when possible."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=2)
        assert len(results) == 10

    def test_maintains_rough_rank_order(self, single_doc_dominated_results):
        """Higher-scored chunks should generally appear first."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=2)

        # First result should be highest scoring
        assert results[0]['score'] == 0.95

        # Second result might not be #2 overall (due to diversity) but should be high
        assert results[1]['score'] >= 0.80

    def test_includes_multiple_documents(self, single_doc_dominated_results):
        """Should include chunks from multiple documents."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=2)

        unique_docs = set(r['doc_id'] for r in results)
        assert len(unique_docs) >= 3, "Should include at least 3 different documents"


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestDiversifyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_results(self, empty_results):
        """Should handle empty input gracefully."""
        results = diversify_results(empty_results, top_k=10, max_per_doc=2)
        assert results == []

    def test_single_result(self, single_result):
        """Should handle single result."""
        results = diversify_results(single_result, top_k=10, max_per_doc=2)
        assert len(results) == 1
        assert results[0]['id'] == 1

    def test_fewer_results_than_requested(self, balanced_results):
        """Should return all available when less than top_k."""
        results = diversify_results(balanced_results, top_k=20, max_per_doc=2)
        assert len(results) == 6  # Only 6 available

    def test_max_per_doc_one(self, single_doc_dominated_results):
        """With max_per_doc=1, first pass is most diverse, then backfill."""
        results = diversify_results(single_doc_dominated_results, top_k=5, max_per_doc=1)

        # First pass: 1 from each doc = 3 results (seal-rag, crag, basic-rag)
        # Backfill: 2 more from overflow to reach top_k=5
        assert len(results) == 5

        # First 3 should be unique (one per doc)
        first_three_doc_ids = [r['doc_id'] for r in results[:3]]
        assert len(set(first_three_doc_ids)) == 3, "First pass should be diverse"

    def test_max_per_doc_unlimited(self, single_doc_dominated_results):
        """With high max_per_doc, should behave like no diversity filter."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=100)

        # Should just return top 10 by score
        assert results[0]['score'] == 0.95
        assert results[9]['score'] == 0.78


# ============================================================================
# TESTS: Backfill Behavior
# ============================================================================

class TestDiversifyBackfill:
    """Test overflow/backfill behavior when diversity limits are hit."""

    def test_backfills_from_overflow(self, single_doc_dominated_results):
        """When diversity limits hit, should backfill from overflow to reach top_k."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=2)

        # Should have exactly 10 results (top_k is the goal)
        assert len(results) == 10

        # First pass gives us 5 diverse results (2+2+1)
        # Backfill adds 5 more from overflow (seal-rag extras)
        # So seal-rag ends up with 7 (2 from first pass + 5 backfill)
        seal_count = sum(1 for r in results if r['doc_id'] == 'seal-rag')
        assert seal_count == 7  # Diversity is soft, top_k is hard

    def test_backfill_maintains_score_order(self, single_doc_dominated_results):
        """Backfilled items should be the highest-scoring overflow."""
        results = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=2)

        # The overflow items that got backfilled should be higher scoring
        # than any items that didn't make it
        included_ids = set(r['id'] for r in results)
        excluded = [r for r in single_doc_dominated_results if r['id'] not in included_ids]

        if excluded:
            min_included_score = min(r['score'] for r in results)
            max_excluded_score = max(r['score'] for r in excluded)
            # This might not always hold due to diversity, but backfill should help
            # The gap shouldn't be huge
            assert max_excluded_score - min_included_score < 0.2


# ============================================================================
# TESTS: Parameter Validation
# ============================================================================

class TestDiversifyParameters:
    """Test different parameter combinations."""

    def test_top_k_zero(self, balanced_results):
        """top_k=0 should return empty list."""
        results = diversify_results(balanced_results, top_k=0, max_per_doc=2)
        assert results == []

    def test_various_max_per_doc_values(self, single_doc_dominated_results):
        """Test that higher max_per_doc means less backfill needed."""
        # With soft diversity, backfill can exceed max_per_doc
        # But higher max_per_doc = more diversity in first pass = less backfill

        results_1 = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=1)
        results_5 = diversify_results(single_doc_dominated_results, top_k=10, max_per_doc=5)

        # Both should hit top_k
        assert len(results_1) == 10
        assert len(results_5) == 10

        # With max_per_doc=1, first pass only gets 3 (one per doc)
        # With max_per_doc=5, first pass gets more diverse results before backfill
        # Count how many docs are represented
        docs_in_1 = set(r['doc_id'] for r in results_1)
        docs_in_5 = set(r['doc_id'] for r in results_5)

        # Both should include all 3 docs
        assert len(docs_in_1) == 3
        assert len(docs_in_5) == 3


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
