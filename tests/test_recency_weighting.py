"""
Test Suite: Recency Weighting (Spec 5)

Newer papers should get a slight score boost.
Why? Recent papers often cite and build on older ones,
so they include that context. Also, methods evolve.

But the boost should be SLIGHT - quality still matters more than date.
"""

import pytest
import math
from datetime import datetime, timedelta

# Import from the implementation module
from app.retrieval import apply_recency_weight


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def reference_date():
    """Fixed reference date for consistent testing."""
    return datetime(2025, 12, 1)


@pytest.fixture
def papers_of_different_ages(reference_date):
    """
    Papers with same base score but different ages.

    This tests whether recency affects ranking when quality is equal.
    """
    return [
        {
            'id': 1,
            'title': 'Paper from 2 years ago',
            'score': 0.80,
            'ingested_at': (reference_date - timedelta(days=730)).isoformat()
        },
        {
            'id': 2,
            'title': 'Paper from 1 year ago',
            'score': 0.80,
            'ingested_at': (reference_date - timedelta(days=365)).isoformat()
        },
        {
            'id': 3,
            'title': 'Paper from 1 month ago',
            'score': 0.80,
            'ingested_at': (reference_date - timedelta(days=30)).isoformat()
        },
        {
            'id': 4,
            'title': 'Paper from today',
            'score': 0.80,
            'ingested_at': reference_date.isoformat()
        },
    ]


@pytest.fixture
def mixed_quality_and_age(reference_date):
    """
    Papers with different quality AND different ages.

    This tests whether quality still dominates over recency.
    """
    return [
        {
            'id': 1,
            'title': 'Excellent old paper (foundational)',
            'score': 0.95,
            'ingested_at': (reference_date - timedelta(days=730)).isoformat()
        },
        {
            'id': 2,
            'title': 'Good recent paper',
            'score': 0.85,
            'ingested_at': (reference_date - timedelta(days=30)).isoformat()
        },
        {
            'id': 3,
            'title': 'Mediocre very recent paper',
            'score': 0.70,
            'ingested_at': reference_date.isoformat()
        },
    ]


# ============================================================================
# TESTS: Basic Recency Behavior
# ============================================================================

class TestRecencyBasic:
    """Test that recency weighting works as expected."""

    def test_newer_papers_get_higher_boost(self, papers_of_different_ages, reference_date):
        """Newer papers should get a larger recency boost."""
        results = apply_recency_weight(
            papers_of_different_ages.copy(),
            reference_date=reference_date
        )

        # Extract boosts
        boosts = {r['id']: r['_recency_boost'] for r in results}

        # Today's paper should have highest boost
        assert boosts[4] > boosts[3], "Today should beat 1 month ago"
        assert boosts[3] > boosts[2], "1 month ago should beat 1 year ago"
        assert boosts[2] > boosts[1], "1 year ago should beat 2 years ago"

    def test_equal_scores_ranked_by_recency(self, papers_of_different_ages, reference_date):
        """When base scores are equal, recency should determine order."""
        results = apply_recency_weight(
            papers_of_different_ages.copy(),
            reference_date=reference_date
        )

        # Newest should be first
        assert results[0]['id'] == 4, "Today's paper should rank first"
        assert results[1]['id'] == 3, "1-month paper should rank second"
        assert results[2]['id'] == 2, "1-year paper should rank third"
        assert results[3]['id'] == 1, "2-year paper should rank last"

    def test_boost_is_multiplicative(self, reference_date):
        """Boost should multiply the score, not add to it."""
        results = [
            {'id': 1, 'score': 0.50, 'ingested_at': reference_date.isoformat()},
            {'id': 2, 'score': 1.00, 'ingested_at': reference_date.isoformat()},
        ]

        weighted = apply_recency_weight(results.copy(), reference_date=reference_date)

        # Both should get same percentage boost
        # score * (1 + 0.15 * 1.0) = score * 1.15
        # After sorting: id=2 (1.15) is first, id=1 (0.575) is second
        assert weighted[0]['score'] == pytest.approx(1.00 * 1.15, rel=0.01)
        assert weighted[1]['score'] == pytest.approx(0.50 * 1.15, rel=0.01)


# ============================================================================
# TESTS: Quality Still Dominates
# ============================================================================

class TestQualityDominates:
    """Verify that quality/relevance still matters more than recency."""

    def test_excellent_old_paper_beats_mediocre_new(self, mixed_quality_and_age, reference_date):
        """
        A highly relevant old paper should still beat a less relevant new paper.

        This is critical - we don't want recency to override quality.
        """
        results = apply_recency_weight(
            mixed_quality_and_age.copy(),
            reference_date=reference_date
        )

        # The excellent old paper (0.95) should still beat mediocre new (0.70)
        # Even with 15% boost: 0.70 * 1.15 = 0.805
        # Old paper with minimal boost: 0.95 * 1.04 = 0.988
        assert results[0]['id'] == 1, "Excellent old paper should still win"

    def test_small_quality_gap_can_flip(self, reference_date):
        """
        With small quality differences, recency CAN change the order.
        This is the intended behavior for similar-quality papers.
        """
        results = [
            {'id': 1, 'score': 0.82, 'ingested_at': (reference_date - timedelta(days=365)).isoformat()},
            {'id': 2, 'score': 0.80, 'ingested_at': reference_date.isoformat()},
        ]

        weighted = apply_recency_weight(results.copy(), reference_date=reference_date)

        # New paper: 0.80 * 1.15 = 0.92
        # Old paper: 0.82 * 1.10 = 0.90
        # New paper should win now
        assert weighted[0]['id'] == 2, "Slightly worse but much newer should win"


# ============================================================================
# TESTS: Decay Rate Behavior
# ============================================================================

class TestDecayRate:
    """Test how decay_rate parameter affects the boost curve."""

    def test_higher_decay_favors_recent_more(self, papers_of_different_ages, reference_date):
        """Higher decay rate should create bigger gap between new and old."""
        # Low decay
        low_decay = apply_recency_weight(
            [p.copy() for p in papers_of_different_ages],
            reference_date=reference_date,
            decay_rate=0.01
        )

        # High decay
        high_decay = apply_recency_weight(
            [p.copy() for p in papers_of_different_ages],
            reference_date=reference_date,
            decay_rate=0.1
        )

        # With high decay, the gap between new and old should be larger
        low_gap = low_decay[0]['score'] - low_decay[-1]['score']
        high_gap = high_decay[0]['score'] - high_decay[-1]['score']

        assert high_gap > low_gap, "Higher decay should create bigger gap"

    def test_zero_decay_gives_uniform_boost(self, papers_of_different_ages, reference_date):
        """With decay_rate=0, all papers get the same boost regardless of age."""
        results = apply_recency_weight(
            [p.copy() for p in papers_of_different_ages],
            reference_date=reference_date,
            decay_rate=0.0
        )

        # All should have boost of 1.0 (no decay)
        boosts = [r['_recency_boost'] for r in results]
        assert all(b == pytest.approx(1.0) for b in boosts)


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestRecencyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_results(self, reference_date):
        """Should handle empty input."""
        results = apply_recency_weight([], reference_date=reference_date)
        assert results == []

    def test_missing_date_field(self, reference_date):
        """Should handle results without ingested_at field."""
        results = [
            {'id': 1, 'score': 0.80}  # No date
        ]

        # Should not crash, should assume old
        weighted = apply_recency_weight(results.copy(), reference_date=reference_date)
        assert len(weighted) == 1
        assert weighted[0]['_recency_boost'] < 0.5  # Should have low boost (assumed old)

    def test_future_date(self, reference_date):
        """Should handle dates in the future gracefully."""
        results = [
            {
                'id': 1,
                'score': 0.80,
                'ingested_at': (reference_date + timedelta(days=30)).isoformat()
            }
        ]

        # Should not crash, boost should be ~1.0 or slightly higher
        weighted = apply_recency_weight(results.copy(), reference_date=reference_date)
        assert weighted[0]['_recency_boost'] >= 1.0

    def test_very_old_paper(self, reference_date):
        """Very old papers should still get some (small) boost."""
        results = [
            {
                'id': 1,
                'score': 0.80,
                'ingested_at': (reference_date - timedelta(days=365*10)).isoformat()  # 10 years
            }
        ]

        weighted = apply_recency_weight(results.copy(), reference_date=reference_date)

        # Should still have positive boost, just small
        assert weighted[0]['_recency_boost'] > 0
        assert weighted[0]['_recency_boost'] < 0.1  # But very small


# ============================================================================
# TESTS: Integration with Ranking
# ============================================================================

class TestRecencyRankingIntegration:
    """Test that re-sorting after weighting works correctly."""

    def test_results_are_resorted(self, reference_date):
        """Results should be re-sorted after applying weights."""
        # Intentionally out of order by final score
        results = [
            {'id': 1, 'score': 0.70, 'ingested_at': reference_date.isoformat()},  # Will become ~0.805
            {'id': 2, 'score': 0.90, 'ingested_at': (reference_date - timedelta(days=730)).isoformat()},  # Will become ~0.94
            {'id': 3, 'score': 0.75, 'ingested_at': (reference_date - timedelta(days=30)).isoformat()},  # Will become ~0.86
        ]

        weighted = apply_recency_weight(results.copy(), reference_date=reference_date)

        # Should be sorted by final score
        scores = [r['score'] for r in weighted]
        assert scores == sorted(scores, reverse=True), "Should be sorted descending"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
