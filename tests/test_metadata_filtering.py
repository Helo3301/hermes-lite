"""
Test Suite: Metadata Filtering (Spec 6)

Allow filtering search results by metadata BEFORE ranking.
This is more efficient than retrieving everything and filtering after.

Use cases:
- "Papers from last 3 months only"
- "Only papers in the 'medical' collection"
- "Exclude papers with fewer than 10 chunks (probably too short)"
"""

import pytest
from typing import List, Dict, Optional

# Import from the implementation module
from app.retrieval import build_filter_clause, filter_results_by_metadata


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def diverse_results():
    """
    Results from various dates, collections, and document sizes.
    Simulates a realistic mixed result set.
    """
    return [
        {
            'id': 1,
            'doc_id': 101,
            'title': 'Recent medical paper',
            'score': 0.95,
            'ingested_at': '2025-11-15',
            'collection': 'medical',
            'chunk_count': 45
        },
        {
            'id': 2,
            'doc_id': 102,
            'title': 'Old AI paper (foundational)',
            'score': 0.92,
            'ingested_at': '2024-03-20',
            'collection': 'ai-papers',
            'chunk_count': 120
        },
        {
            'id': 3,
            'doc_id': 103,
            'title': 'Recent short blog post',
            'score': 0.88,
            'ingested_at': '2025-12-01',
            'collection': 'ai-papers',
            'chunk_count': 5  # Very short
        },
        {
            'id': 4,
            'doc_id': 104,
            'title': 'Medium-age medical paper',
            'score': 0.85,
            'ingested_at': '2025-06-10',
            'collection': 'medical',
            'chunk_count': 60
        },
        {
            'id': 5,
            'doc_id': 105,
            'title': 'Very old paper',
            'score': 0.80,
            'ingested_at': '2023-01-15',
            'collection': 'ai-papers',
            'chunk_count': 30
        },
        {
            'id': 6,
            'doc_id': 106,
            'title': 'Recent AI paper',
            'score': 0.78,
            'ingested_at': '2025-10-22',
            'collection': 'ai-papers',
            'chunk_count': 55
        },
    ]


# ============================================================================
# TESTS: Date Filtering
# ============================================================================

class TestDateFiltering:
    """Test filtering by date range."""

    def test_filter_by_date_from(self, diverse_results):
        """Should only include documents from specified date onwards."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            date_from='2025-10-01'
        )

        # Should include: doc 1 (Nov), doc 3 (Dec), doc 6 (Oct)
        # Should exclude: doc 2 (2024), doc 4 (June), doc 5 (2023)
        doc_ids = [r['doc_id'] for r in filtered]

        assert 101 in doc_ids, "November paper should be included"
        assert 103 in doc_ids, "December paper should be included"
        assert 106 in doc_ids, "October paper should be included"
        assert 102 not in doc_ids, "2024 paper should be excluded"
        assert 105 not in doc_ids, "2023 paper should be excluded"

    def test_filter_by_date_to(self, diverse_results):
        """Should only include documents up to specified date."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            date_to='2025-06-30'
        )

        # Should include: doc 2 (2024), doc 4 (June), doc 5 (2023)
        doc_ids = [r['doc_id'] for r in filtered]

        assert 104 in doc_ids, "June paper should be included"
        assert 105 in doc_ids, "2023 paper should be included"
        assert 101 not in doc_ids, "November paper should be excluded"
        assert 103 not in doc_ids, "December paper should be excluded"

    def test_filter_by_date_range(self, diverse_results):
        """Should filter to documents within date range."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            date_from='2025-06-01',
            date_to='2025-11-01'
        )

        # Should include: doc 4 (June), doc 6 (Oct)
        # Should exclude: everything else
        doc_ids = [r['doc_id'] for r in filtered]

        assert 104 in doc_ids, "June paper in range"
        assert 106 in doc_ids, "October paper in range"
        assert len(filtered) == 2


# ============================================================================
# TESTS: Collection Filtering
# ============================================================================

class TestCollectionFiltering:
    """Test filtering by collection/category."""

    def test_filter_by_collection(self, diverse_results):
        """Should only include documents from specified collection."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            collection='medical'
        )

        # Should only have medical papers
        for result in filtered:
            assert result['collection'] == 'medical'

        # Should have exactly 2 medical papers
        assert len(filtered) == 2

    def test_filter_nonexistent_collection(self, diverse_results):
        """Filtering by non-existent collection should return empty."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            collection='nonexistent-collection'
        )

        assert len(filtered) == 0


# ============================================================================
# TESTS: Chunk Count Filtering
# ============================================================================

class TestChunkCountFiltering:
    """Test filtering by minimum document size."""

    def test_filter_by_min_chunks(self, diverse_results):
        """Should exclude documents with too few chunks."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            min_chunks=30
        )

        # Should exclude doc 3 (only 5 chunks)
        doc_ids = [r['doc_id'] for r in filtered]

        assert 103 not in doc_ids, "Short blog post should be excluded"
        assert 102 in doc_ids, "120-chunk paper should be included"
        assert 105 in doc_ids, "30-chunk paper should be included (exactly at threshold)"

    def test_high_min_chunks_filters_most(self, diverse_results):
        """High minimum should filter most documents."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            min_chunks=100
        )

        # Only doc 2 has 120 chunks
        assert len(filtered) == 1
        assert filtered[0]['doc_id'] == 102


# ============================================================================
# TESTS: Exclusion Filtering
# ============================================================================

class TestExclusionFiltering:
    """Test excluding specific documents."""

    def test_exclude_specific_docs(self, diverse_results):
        """Should exclude documents by ID."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            exclude_doc_ids=[101, 103]
        )

        doc_ids = [r['doc_id'] for r in filtered]

        assert 101 not in doc_ids
        assert 103 not in doc_ids
        assert len(filtered) == 4  # 6 - 2 excluded

    def test_exclude_nonexistent_docs(self, diverse_results):
        """Excluding non-existent IDs should not affect results."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            exclude_doc_ids=[999, 1000]  # Don't exist
        )

        assert len(filtered) == 6  # All still present


# ============================================================================
# TESTS: Combined Filters
# ============================================================================

class TestCombinedFilters:
    """Test using multiple filters together."""

    def test_date_and_collection(self, diverse_results):
        """Should apply both date and collection filters."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            date_from='2025-01-01',
            collection='medical'
        )

        # Medical papers from 2025: doc 1 (Nov), doc 4 (June)
        doc_ids = [r['doc_id'] for r in filtered]

        assert len(filtered) == 2
        assert all(r['collection'] == 'medical' for r in filtered)
        assert all(r['ingested_at'] >= '2025-01-01' for r in filtered)

    def test_all_filters_together(self, diverse_results):
        """Should apply all filters at once."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            date_from='2025-01-01',
            date_to='2025-12-31',
            collection='ai-papers',
            min_chunks=20,
            exclude_doc_ids=[106]
        )

        # AI papers from 2025, with 20+ chunks, excluding doc 106
        # That's just doc 3 (but it has only 5 chunks, so excluded)
        # Actually no AI papers from 2025 meet all criteria
        # Wait: doc 3 (Dec 2025, ai-papers, 5 chunks) - fails min_chunks
        # doc 6 (Oct 2025, ai-papers, 55 chunks) - excluded explicitly

        assert len(filtered) == 0


# ============================================================================
# TESTS: SQL Clause Generation
# ============================================================================

class TestSQLClauseGeneration:
    """Test the SQL WHERE clause builder."""

    def test_no_filters_returns_truthy_clause(self):
        """With no filters, should return clause that matches everything."""
        clause, params = build_filter_clause()

        assert clause == "1=1"
        assert params == []

    def test_single_filter_clause(self):
        """Single filter should produce simple clause."""
        clause, params = build_filter_clause(date_from='2025-01-01')

        assert "d.ingested_at >= ?" in clause
        assert params == ['2025-01-01']

    def test_multiple_filters_joined_with_and(self):
        """Multiple filters should be joined with AND."""
        clause, params = build_filter_clause(
            date_from='2025-01-01',
            collection='medical'
        )

        assert " AND " in clause
        assert "d.ingested_at >= ?" in clause
        assert "c.name = ?" in clause
        assert len(params) == 2

    def test_exclude_doc_ids_creates_not_in(self):
        """Exclusion should create NOT IN clause."""
        clause, params = build_filter_clause(
            exclude_doc_ids=[1, 2, 3]
        )

        assert "NOT IN" in clause
        assert "?,?,?" in clause
        assert params == [1, 2, 3]


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestFilteringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_results(self):
        """Should handle empty input."""
        filtered = filter_results_by_metadata(
            [],
            date_from='2025-01-01'
        )
        assert filtered == []

    def test_missing_metadata_fields(self):
        """Should handle results with missing metadata gracefully."""
        results = [
            {'id': 1, 'doc_id': 101, 'score': 0.95},  # Missing all metadata
        ]

        # Should not crash
        filtered = filter_results_by_metadata(
            results,
            date_from='2025-01-01'
        )

        # Missing date should probably be excluded (fails date check)
        assert len(filtered) == 0

    def test_preserves_result_order(self, diverse_results):
        """Filtering should preserve the original ranking order."""
        filtered = filter_results_by_metadata(
            diverse_results.copy(),
            collection='ai-papers'
        )

        # Verify scores are still in descending order
        scores = [r['score'] for r in filtered]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
