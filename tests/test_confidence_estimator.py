"""Tests for Confidence Estimation and Adaptive Retrieval."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.confidence_estimator import (
    ConfidenceEstimator,
    ConfidenceScore,
    AdaptiveRetriever,
    adaptive_search,
)


# Test chunks with varying quality
HIGH_QUALITY_CHUNKS = [
    {
        "id": 1,
        "content": "SEAL-RAG achieves 96% precision on HotpotQA benchmark. The method uses entity extraction and iterative retrieval.",
        "doc_id": "seal_rag.pdf",
        "score": 0.92,
    },
    {
        "id": 2,
        "content": "SEAL-RAG implements a fixed-budget approach where low-quality chunks are replaced rather than accumulated.",
        "doc_id": "seal_rag.pdf",
        "score": 0.88,
    },
    {
        "id": 3,
        "content": "The SEAL-RAG method outperforms baseline RAG by addressing context dilution in multi-hop queries.",
        "doc_id": "comparison.pdf",
        "score": 0.85,
    },
]

LOW_QUALITY_CHUNKS = [
    {
        "id": 10,
        "content": "Machine learning is a field of artificial intelligence.",
        "doc_id": "intro.pdf",
        "score": 0.35,
    },
    {
        "id": 11,
        "content": "Neural networks can learn patterns from data.",
        "doc_id": "basics.pdf",
        "score": 0.30,
    },
]

MIXED_QUALITY_CHUNKS = [
    {
        "id": 20,
        "content": "SEAL-RAG uses entity extraction for query analysis.",
        "doc_id": "methods.pdf",
        "score": 0.75,
    },
    {
        "id": 21,
        "content": "The results show improvements over baseline methods.",
        "doc_id": "results.pdf",
        "score": 0.45,
    },
    {
        "id": 22,
        "content": "Retrieval augmented generation combines search with language models.",
        "doc_id": "overview.pdf",
        "score": 0.60,
    },
]


def test_confidence_empty_results():
    """Test confidence estimation with no results."""
    estimator = ConfidenceEstimator()

    score = estimator.estimate(
        query="What is SEAL-RAG?",
        chunks=[],
        query_entities=["SEAL-RAG"]
    )

    print(f"\nEmpty Results Test:")
    print(f"  Overall confidence: {score.overall:.2f}")
    print(f"  Explanation: {score.explanation}")

    assert score.overall == 0.0, "Empty results should have 0 confidence"
    assert "No results" in score.explanation
    print("PASS: Empty results handled correctly")


def test_confidence_high_quality():
    """Test confidence estimation with high quality results."""
    estimator = ConfidenceEstimator()

    score = estimator.estimate(
        query="How does SEAL-RAG work?",
        chunks=HIGH_QUALITY_CHUNKS,
        query_entities=["SEAL-RAG"]
    )

    print(f"\nHigh Quality Test:")
    print(f"  Overall confidence: {score.overall:.2f}")
    print(f"  Score distribution: {score.score_distribution:.2f}")
    print(f"  Entity coverage: {score.entity_coverage:.2f}")
    print(f"  Source agreement: {score.source_agreement:.2f}")
    print(f"  Source diversity: {score.source_diversity:.2f}")
    print(f"  Explanation: {score.explanation}")

    assert score.overall > 0.6, f"High quality results should have confidence > 0.6, got {score.overall:.2f}"
    assert score.entity_coverage > 0.7, "Entity coverage should be high"
    print("PASS: High quality results have high confidence")


def test_confidence_low_quality():
    """Test confidence estimation with low quality results."""
    estimator = ConfidenceEstimator()

    score = estimator.estimate(
        query="How does SEAL-RAG work?",
        chunks=LOW_QUALITY_CHUNKS,
        query_entities=["SEAL-RAG"]
    )

    print(f"\nLow Quality Test:")
    print(f"  Overall confidence: {score.overall:.2f}")
    print(f"  Entity coverage: {score.entity_coverage:.2f}")
    print(f"  Explanation: {score.explanation}")

    # Entity not found in low quality chunks
    assert score.entity_coverage < 0.5, "Should have low entity coverage"
    print("PASS: Low quality results have low confidence")


def test_confidence_score_distribution():
    """Test score distribution confidence calculation."""
    estimator = ConfidenceEstimator()

    # High variance scores
    high_variance = [0.9, 0.5, 0.2, 0.8, 0.3]
    conf_high_var = estimator._score_distribution_confidence(high_variance)

    # Low variance scores (consistently good)
    low_variance = [0.85, 0.88, 0.82, 0.87, 0.84]
    conf_low_var = estimator._score_distribution_confidence(low_variance)

    print(f"\nScore Distribution Test:")
    print(f"  High variance scores: {high_variance}")
    print(f"  High variance confidence: {conf_high_var:.2f}")
    print(f"  Low variance scores: {low_variance}")
    print(f"  Low variance confidence: {conf_low_var:.2f}")

    assert conf_low_var > conf_high_var, "Consistent high scores should have higher confidence"
    print("PASS: Score distribution confidence works correctly")


def test_confidence_source_diversity():
    """Test source diversity confidence calculation."""
    estimator = ConfidenceEstimator()

    # Single source
    single_source = [
        {"doc_id": "paper1.pdf", "content": "test"},
        {"doc_id": "paper1.pdf", "content": "test"},
        {"doc_id": "paper1.pdf", "content": "test"},
    ]
    conf_single = estimator._source_diversity_confidence(single_source)

    # Multiple sources
    multi_source = [
        {"doc_id": "paper1.pdf", "content": "test"},
        {"doc_id": "paper2.pdf", "content": "test"},
        {"doc_id": "paper3.pdf", "content": "test"},
    ]
    conf_multi = estimator._source_diversity_confidence(multi_source)

    print(f"\nSource Diversity Test:")
    print(f"  Single source confidence: {conf_single:.2f}")
    print(f"  Multi source confidence: {conf_multi:.2f}")

    assert conf_multi > conf_single, "Multiple sources should have higher diversity confidence"
    print("PASS: Source diversity confidence works correctly")


def test_adaptive_retrieval_confident():
    """Test adaptive retrieval that reaches confidence threshold."""
    # Mock search function that returns good results
    def mock_search(query, top_k=5):
        return HIGH_QUALITY_CHUNKS[:top_k]

    retriever = AdaptiveRetriever(
        search_fn=mock_search,
        min_k=3,
        max_k=10,
        confidence_threshold=0.5,  # Lower threshold for testing
    )

    result = retriever.search(
        query="How does SEAL-RAG work?",
        query_entities=["SEAL-RAG"]
    )

    print(f"\nAdaptive Retrieval (Confident) Test:")
    print(f"  Final k: {result['final_k']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Status: {result['status']}")
    print(f"  Confidence: {result['confidence'].overall:.2f}")

    assert result['status'] == 'confident', f"Should reach confidence, got {result['status']}"
    assert result['iterations'] <= 3, "Should not need many iterations"
    print("PASS: Adaptive retrieval reaches confidence threshold")


def test_adaptive_retrieval_max_reached():
    """Test adaptive retrieval that hits max k."""
    # Mock search function that returns low quality results
    def mock_search(query, top_k=5):
        return LOW_QUALITY_CHUNKS[:min(top_k, len(LOW_QUALITY_CHUNKS))]

    retriever = AdaptiveRetriever(
        search_fn=mock_search,
        min_k=2,
        max_k=5,
        confidence_threshold=0.9,  # High threshold
        step_size=2,
    )

    result = retriever.search(
        query="How does SEAL-RAG work?",
        query_entities=["SEAL-RAG"]
    )

    print(f"\nAdaptive Retrieval (Max Reached) Test:")
    print(f"  Final k: {result['final_k']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Status: {result['status']}")
    print(f"  Confidence: {result['confidence'].overall:.2f}")

    assert result['status'] == 'max_reached', f"Should hit max, got {result['status']}"
    print("PASS: Adaptive retrieval handles max k correctly")


def test_adaptive_search_convenience():
    """Test the convenience function for adaptive search."""
    call_count = [0]

    def mock_search(query, top_k=5):
        call_count[0] += 1
        return MIXED_QUALITY_CHUNKS[:top_k]

    result = adaptive_search(
        search_fn=mock_search,
        query="What is retrieval augmented generation?",
        query_entities=["RAG"],
        min_k=3,
        max_k=10,
        confidence_threshold=0.5,
    )

    print(f"\nConvenience Function Test:")
    print(f"  Search calls: {call_count[0]}")
    print(f"  Results: {len(result['results'])}")
    print(f"  Status: {result['status']}")

    assert 'results' in result, "Should return results"
    assert 'confidence' in result, "Should return confidence"
    assert call_count[0] >= 1, "Should call search at least once"
    print("PASS: Convenience function works correctly")


def test_confidence_explanation():
    """Test that confidence explanations are informative."""
    estimator = ConfidenceEstimator()

    # High confidence case
    high_score = estimator.estimate(
        query="What is SEAL-RAG?",
        chunks=HIGH_QUALITY_CHUNKS,
        query_entities=["SEAL-RAG"]
    )

    # Low confidence case
    low_score = estimator.estimate(
        query="What is XYZ method?",
        chunks=LOW_QUALITY_CHUNKS,
        query_entities=["XYZ"]
    )

    print(f"\nExplanation Test:")
    print(f"  High confidence explanation: {high_score.explanation}")
    print(f"  Low confidence explanation: {low_score.explanation}")

    assert len(high_score.explanation) > 10, "Should have meaningful explanation"
    assert len(low_score.explanation) > 10, "Should have meaningful explanation"
    print("PASS: Confidence explanations are informative")


def run_all_tests():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("Phase 5: Confidence Estimation Tests")
    print("=" * 60)

    tests = [
        ("Empty Results", test_confidence_empty_results),
        ("High Quality Results", test_confidence_high_quality),
        ("Low Quality Results", test_confidence_low_quality),
        ("Score Distribution", test_confidence_score_distribution),
        ("Source Diversity", test_confidence_source_diversity),
        ("Adaptive Retrieval (Confident)", test_adaptive_retrieval_confident),
        ("Adaptive Retrieval (Max Reached)", test_adaptive_retrieval_max_reached),
        ("Convenience Function", test_adaptive_search_convenience),
        ("Confidence Explanations", test_confidence_explanation),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n--- {name} ---")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
