"""
Test Suite: Two-Phase Retrieval (Spec 4)

Fast initial retrieval + accurate re-ranking = best of both worlds.

Phase 1 (BM25): Lightning fast keyword search, casts wide net
Phase 2 (Vector): Accurate semantic ranking of the candidates

Why two phases?
- BM25 alone: Fast but misses semantic similarity ("car" vs "automobile")
- Vector alone: Accurate but slow on large collections
- Combined: Fast candidate generation + accurate final ranking

This is called "hybrid search" or "two-stage retrieval".
"""

import pytest
import math

# Import from the implementation module
from app.retrieval import (
    bm25_search,
    vector_rerank,
    reciprocal_rank_fusion,
    two_phase_search,
    cosine_similarity
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_embedding():
    """A normalized embedding for the query."""
    raw = [1, 0, 0, 0, 0, 0, 0, 0]
    norm = math.sqrt(sum(x*x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def similar_embedding():
    """Embedding similar to sample_embedding (~0.9 similarity)."""
    raw = [0.9, 0.3, 0.1, 0, 0, 0, 0, 0]
    norm = math.sqrt(sum(x*x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def different_embedding():
    """Embedding different from sample_embedding (~0.0 similarity)."""
    raw = [0, 1, 0, 0, 0, 0, 0, 0]
    norm = math.sqrt(sum(x*x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def document_corpus(sample_embedding, similar_embedding, different_embedding):
    """
    A corpus with documents that have different BM25 and vector relevance.

    This tests the core value of two-phase retrieval:
    - Doc 1: Good BM25 match, good vector match (should rank high)
    - Doc 2: Good BM25 match, bad vector match (keyword match but wrong meaning)
    - Doc 3: Bad BM25 match, good vector match (semantic match, different words)
    - Doc 4: Bad BM25 match, bad vector match (irrelevant)
    """
    return [
        {
            'id': 1,
            'content': 'retrieval augmented generation improves LLM accuracy',
            'embedding': sample_embedding,  # Perfect match
            '_description': 'Good BM25 + Good Vector'
        },
        {
            'id': 2,
            'content': 'retrieval of lost artifacts from generation ships',
            'embedding': different_embedding,  # Wrong meaning
            '_description': 'Good BM25 + Bad Vector (false positive)'
        },
        {
            'id': 3,
            'content': 'combining search with language models for Q&A',
            'embedding': similar_embedding,  # Right meaning, different words
            '_description': 'Bad BM25 + Good Vector (semantic match)'
        },
        {
            'id': 4,
            'content': 'cooking recipes and kitchen tips',
            'embedding': different_embedding,  # Completely irrelevant
            '_description': 'Bad BM25 + Bad Vector'
        },
        {
            'id': 5,
            'content': 'retrieval methods for information systems',
            'embedding': similar_embedding,
            '_description': 'Partial BM25 + Good Vector'
        },
    ]


# ============================================================================
# TESTS: BM25 Phase
# ============================================================================

class TestBM25Phase:
    """Test the BM25 keyword search phase."""

    def test_finds_keyword_matches(self, document_corpus):
        """Should find documents containing query terms."""
        results = bm25_search('retrieval augmented generation', document_corpus)

        # Should find docs with those keywords
        result_ids = [r['id'] for r in results]
        assert 1 in result_ids, "Should find exact match"
        assert 2 in result_ids, "Should find keyword match (even if wrong meaning)"

    def test_ranks_by_term_overlap(self, document_corpus):
        """Documents with more matching terms should rank higher."""
        results = bm25_search('retrieval augmented generation', document_corpus)

        # Doc 1 has all three terms, should rank highest
        assert results[0]['id'] == 1

    def test_excludes_non_matches(self, document_corpus):
        """Should not include documents without query terms."""
        results = bm25_search('retrieval augmented generation', document_corpus)

        result_ids = [r['id'] for r in results]
        assert 4 not in result_ids, "Cooking doc should not appear"

    def test_respects_top_k(self, document_corpus):
        """Should limit results to top_k."""
        results = bm25_search('retrieval', document_corpus, top_k=2)
        assert len(results) <= 2


# ============================================================================
# TESTS: Vector Reranking Phase
# ============================================================================

class TestVectorRerank:
    """Test the vector similarity reranking phase."""

    def test_reranks_by_semantic_similarity(self, document_corpus, sample_embedding):
        """Should rerank candidates by vector similarity."""
        # Give it all docs as candidates
        results = vector_rerank(sample_embedding, document_corpus.copy())

        # Doc 1 has exact embedding match, should be first
        assert results[0]['id'] == 1

    def test_demotes_false_positives(self, document_corpus, sample_embedding):
        """Keyword matches with wrong meaning should be demoted."""
        results = vector_rerank(sample_embedding, document_corpus.copy())

        # Doc 2 is a false positive (keyword match, wrong meaning)
        # Should rank lower than semantic matches
        doc2_rank = next(i for i, r in enumerate(results) if r['id'] == 2)
        doc1_rank = next(i for i, r in enumerate(results) if r['id'] == 1)

        assert doc2_rank > doc1_rank, "False positive should rank lower"

    def test_promotes_semantic_matches(self, document_corpus, sample_embedding):
        """Semantic matches (different words, same meaning) should rank well."""
        results = vector_rerank(sample_embedding, document_corpus.copy())

        # Doc 3 and 5 have similar embeddings (semantic matches)
        top_3_ids = [r['id'] for r in results[:3]]
        assert 3 in top_3_ids or 5 in top_3_ids, "Semantic matches should rank high"


# ============================================================================
# TESTS: Reciprocal Rank Fusion
# ============================================================================

class TestRRFFusion:
    """Test the RRF score combination."""

    def test_combines_rankings(self):
        """RRF should combine BM25 and vector rankings."""
        bm25_results = [
            {'id': 1, 'bm25_score': 0.9},
            {'id': 2, 'bm25_score': 0.7},
            {'id': 3, 'bm25_score': 0.5},
        ]

        vector_results = [
            {'id': 3, 'vector_score': 0.95},  # Different top pick
            {'id': 1, 'vector_score': 0.85},
            {'id': 2, 'vector_score': 0.75},
        ]

        fused = reciprocal_rank_fusion(bm25_results, vector_results)

        # All three should be present
        assert len(fused) == 3

        # Each should have RRF score
        for r in fused:
            assert 'rrf_score' in r

    def test_doc_in_both_lists_ranks_higher(self):
        """Documents appearing in both rankings should get higher RRF."""
        bm25_results = [
            {'id': 1, 'bm25_score': 0.9},
            {'id': 2, 'bm25_score': 0.7},
        ]

        vector_results = [
            {'id': 1, 'vector_score': 0.95},  # Also in BM25
            {'id': 3, 'vector_score': 0.85},  # Only in vector
        ]

        fused = reciprocal_rank_fusion(bm25_results, vector_results)

        # Doc 1 is in both lists (rank 1 in each)
        # Doc 2 only in BM25 (rank 2)
        # Doc 3 only in vector (rank 2)
        doc1_score = next(r['rrf_score'] for r in fused if r['id'] == 1)
        doc2_score = next(r['rrf_score'] for r in fused if r['id'] == 2)
        doc3_score = next(r['rrf_score'] for r in fused if r['id'] == 3)

        # Doc 1 should have highest score (appears in both at rank 1)
        assert doc1_score > doc2_score
        assert doc1_score > doc3_score

    def test_handles_disjoint_rankings(self):
        """Should handle case where rankings have no overlap."""
        bm25_results = [{'id': 1}, {'id': 2}]
        vector_results = [{'id': 3}, {'id': 4}]

        fused = reciprocal_rank_fusion(bm25_results, vector_results)

        # Should have all 4
        ids = [r['id'] for r in fused]
        assert set(ids) == {1, 2, 3, 4}


# ============================================================================
# TESTS: Full Two-Phase Pipeline
# ============================================================================

class TestTwoPhasePipeline:
    """Test the complete two-phase retrieval pipeline."""

    def test_full_pipeline_rerank_mode(self, document_corpus, sample_embedding):
        """Test rerank mode: BM25 candidates -> vector rerank."""
        results = two_phase_search(
            query='retrieval augmented generation',
            query_embedding=sample_embedding,
            documents=document_corpus,
            phase1_k=10,
            phase2_k=3,
            fusion_method='rerank'
        )

        # Should have at most 3 results
        assert len(results) <= 3

        # Best result should have good vector score
        assert results[0]['vector_score'] > 0.7

    def test_full_pipeline_rrf_mode(self, document_corpus, sample_embedding):
        """Test RRF mode: combine BM25 and vector rankings."""
        results = two_phase_search(
            query='retrieval augmented generation',
            query_embedding=sample_embedding,
            documents=document_corpus,
            phase1_k=10,
            phase2_k=3,
            fusion_method='rrf'
        )

        # Should have RRF scores
        assert all('rrf_score' in r for r in results)

        # Should be sorted by RRF score
        scores = [r['rrf_score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_false_positive_demoted(self, document_corpus, sample_embedding):
        """
        KEY TEST: False positives from BM25 should be demoted after reranking.

        Doc 2 matches keywords but has wrong meaning (different embedding).
        Two-phase should catch this and rank it lower.
        """
        results = two_phase_search(
            query='retrieval augmented generation',
            query_embedding=sample_embedding,
            documents=document_corpus,
            phase1_k=10,
            phase2_k=5,
            fusion_method='rerank'
        )

        # Find Doc 2's rank
        result_ids = [r['id'] for r in results]

        if 2 in result_ids:
            doc2_rank = result_ids.index(2)
            doc1_rank = result_ids.index(1) if 1 in result_ids else len(results)

            # Doc 2 (false positive) should rank below Doc 1 (true positive)
            assert doc2_rank > doc1_rank, \
                "False positive (keyword match, wrong meaning) should be demoted"


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestTwoPhaseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_corpus(self, sample_embedding):
        """Should handle empty corpus."""
        results = two_phase_search(
            query='anything',
            query_embedding=sample_embedding,
            documents=[],
            fusion_method='rerank'
        )
        assert results == []

    def test_no_keyword_matches(self, document_corpus, sample_embedding):
        """Should handle query with no keyword matches."""
        results = two_phase_search(
            query='quantum physics',  # Not in any doc
            query_embedding=sample_embedding,
            documents=document_corpus,
            fusion_method='rerank'
        )
        assert results == []

    def test_single_candidate(self, sample_embedding):
        """Should handle single document corpus."""
        corpus = [{'id': 1, 'content': 'test document', 'embedding': sample_embedding}]

        results = two_phase_search(
            query='test',
            query_embedding=sample_embedding,
            documents=corpus,
            fusion_method='rerank'
        )

        assert len(results) == 1
        assert results[0]['id'] == 1


# ============================================================================
# TESTS: Performance Characteristics
# ============================================================================

class TestTwoPhasePerformance:
    """
    Test that two-phase actually provides the expected benefits.

    These tests verify the theoretical advantages of two-phase retrieval.
    """

    def test_phase1_casts_wide_net(self, document_corpus):
        """Phase 1 (BM25) should retrieve many candidates quickly."""
        # With a generous phase1_k, BM25 should find all keyword matches
        results = bm25_search('retrieval', document_corpus, top_k=50)

        # Should find multiple docs with "retrieval" keyword
        assert len(results) >= 2

    def test_phase2_narrows_to_best(self, document_corpus, sample_embedding):
        """Phase 2 should narrow down to semantically best matches."""
        # Get BM25 candidates
        candidates = bm25_search('retrieval', document_corpus, top_k=50)

        # Rerank should put semantic matches first
        reranked = vector_rerank(sample_embedding, candidates, top_k=2)

        # Top results should have high vector scores
        assert all(r['vector_score'] > 0.5 for r in reranked)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
