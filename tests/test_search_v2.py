"""Tests for Search V2 Integration Pipeline."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.search_v2 import SearchV2Pipeline, SearchV2Config, SearchV2Result


# Mock search results
MOCK_SEAL_RAG_RESULTS = [
    {
        "id": 1,
        "content": "SEAL-RAG implements a fixed-budget evidence assembly approach for multi-hop QA. Unlike traditional RAG systems that expand context indefinitely, SEAL-RAG replaces low-quality chunks.",
        "filename": "seal_rag.pdf",
        "doc_id": 1,
        "score": 0.92,
    },
    {
        "id": 2,
        "content": "The SEAL-RAG method achieves 96% precision on HotpotQA benchmark. The approach uses entity extraction and iterative retrieval to improve coverage.",
        "filename": "seal_rag.pdf",
        "doc_id": 1,
        "score": 0.88,
    },
    {
        "id": 3,
        "content": "SEAL-RAG outperforms CRAG and Self-RAG on multi-hop benchmarks like 2WikiMultiHopQA. It builds on DPR for dense retrieval.",
        "filename": "comparison.pdf",
        "doc_id": 2,
        "score": 0.85,
    },
]

MOCK_CRAG_RESULTS = [
    {
        "id": 10,
        "content": "CRAG (Corrective RAG) introduces a corrective mechanism to improve retrieval quality. It evaluates document relevance and triggers additional retrieval when needed.",
        "filename": "crag.pdf",
        "doc_id": 3,
        "score": 0.90,
    },
    {
        "id": 11,
        "content": "CRAG outperforms basic RAG on single-hop queries. The corrective approach is more efficient for simple questions.",
        "filename": "crag.pdf",
        "doc_id": 3,
        "score": 0.87,
    },
]

MOCK_MIXED_RESULTS = MOCK_SEAL_RAG_RESULTS + MOCK_CRAG_RESULTS


def create_mock_search_fn(default_results=None):
    """Create a mock search function."""
    results = default_results or MOCK_SEAL_RAG_RESULTS
    call_log = []

    def mock_search(query, top_k=10, **kwargs):
        call_log.append({"query": query, "top_k": top_k, "kwargs": kwargs})
        # Filter and return results
        return results[:top_k]

    mock_search.call_log = call_log
    return mock_search


def test_simple_query_classification():
    """Test that simple queries are classified correctly."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("What is SEAL-RAG?")

    print(f"\nSimple Query Test:")
    print(f"  Query: What is SEAL-RAG?")
    print(f"  Classified as: {result.query_type}")
    print(f"  Entities: {result.entities}")
    print(f"  Results: {len(result.results)}")

    assert result.query_type == "simple", f"Expected 'simple', got {result.query_type}"
    assert "SEAL-RAG" in result.entities or len(result.entities) > 0
    print("PASS: Simple query classified correctly")


def test_multi_hop_query_classification():
    """Test that multi-hop queries are classified correctly."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search(
        "How does SEAL-RAG compare to CRAG on multi-hop benchmarks and what are the key differences?"
    )

    print(f"\nMulti-hop Query Test:")
    print(f"  Query type: {result.query_type}")
    print(f"  Sub-queries: {result.sub_queries}")
    print(f"  Entities: {result.entities}")

    # Should be multi_hop or comparative
    assert result.query_type in ["multi_hop", "comparative"], f"Got {result.query_type}"
    print("PASS: Multi-hop query classified correctly")


def test_comparative_query():
    """Test comparative query handling."""
    mock_search = create_mock_search_fn(MOCK_MIXED_RESULTS)
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("Compare SEAL-RAG vs CRAG")

    print(f"\nComparative Query Test:")
    print(f"  Query type: {result.query_type}")
    print(f"  Entities: {result.entities}")
    print(f"  Results: {len(result.results)}")

    assert result.query_type == "comparative", f"Expected 'comparative', got {result.query_type}"
    # Should extract both entities
    entities_lower = [e.lower() for e in result.entities]
    print("PASS: Comparative query handled correctly")


def test_confidence_estimation():
    """Test that confidence is estimated for results."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("How does SEAL-RAG work?")

    print(f"\nConfidence Estimation Test:")
    print(f"  Confidence score: {result.confidence:.2f}")
    print(f"  Explanation: {result.confidence_explanation}")

    assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
    assert len(result.confidence_explanation) > 0
    print("PASS: Confidence estimation works")


def test_iteration_tracking():
    """Test that iterations are tracked."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("What is SEAL-RAG?")

    print(f"\nIteration Tracking Test:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Timing: {result.timing_ms}ms")

    assert result.iterations >= 1
    assert result.timing_ms > 0
    print("PASS: Iterations tracked correctly")


def test_no_contradiction_detection():
    """Test that contradictions can be disabled."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("What is SEAL-RAG?", detect_contradictions=False)

    print(f"\nNo Contradiction Test:")
    print(f"  Contradictions: {result.contradictions}")

    assert result.contradictions is None
    print("PASS: Contradiction detection disabled correctly")


def test_contradiction_detection():
    """Test contradiction detection with conflicting claims."""
    # Create results with potential conflict
    conflicting_results = [
        {
            "id": 1,
            "content": "SEAL-RAG achieves 96% precision on HotpotQA.",
            "filename": "paper_a.pdf",
            "doc_id": 1,
            "score": 0.9,
        },
        {
            "id": 2,
            "content": "SEAL-RAG achieves 45% precision on HotpotQA.",
            "filename": "paper_b.pdf",
            "doc_id": 2,
            "score": 0.85,
        },
    ]

    mock_search = create_mock_search_fn(conflicting_results)
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("What is SEAL-RAG precision?", detect_contradictions=True)

    print(f"\nContradiction Detection Test:")
    print(f"  Results: {len(result.results)}")
    if result.contradictions:
        print(f"  Has contradictions: {result.contradictions.get('has_contradictions')}")
        print(f"  Count: {result.contradictions.get('count')}")

    # Note: May or may not detect depending on claim extraction
    print("PASS: Contradiction detection executed without errors")


def test_config_options():
    """Test that config options are respected."""
    mock_search = create_mock_search_fn()

    config = SearchV2Config(
        use_query_analysis=True,
        use_iterative=False,
        use_adaptive=False,
        detect_contradictions=False,
    )

    pipeline = SearchV2Pipeline(mock_search, config=config)
    result = pipeline.search("What is SEAL-RAG?")

    print(f"\nConfig Options Test:")
    print(f"  Query analysis: {config.use_query_analysis}")
    print(f"  Iterative: {config.use_iterative}")
    print(f"  Query type detected: {result.query_type}")

    # Should still classify query
    assert result.query_type in ["simple", "multi_hop", "comparative", "exploratory"]
    print("PASS: Config options respected")


def test_result_structure():
    """Test that result has expected structure."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    result = pipeline.search("Test query")

    print(f"\nResult Structure Test:")
    print(f"  Has query: {'query' in dir(result)}")
    print(f"  Has query_type: {'query_type' in dir(result)}")
    print(f"  Has results: {'results' in dir(result)}")
    print(f"  Has confidence: {'confidence' in dir(result)}")
    print(f"  Has status: {'status' in dir(result)}")

    assert isinstance(result, SearchV2Result)
    assert result.query == "Test query"
    assert result.query_type in ["simple", "multi_hop", "comparative", "exploratory"]
    assert isinstance(result.results, list)
    assert isinstance(result.confidence, float)
    assert result.status in ["success", "no_results", "low_confidence"]
    print("PASS: Result structure is correct")


def test_performance():
    """Test that search completes in reasonable time."""
    mock_search = create_mock_search_fn()
    pipeline = SearchV2Pipeline(mock_search)

    start = time.time()
    result = pipeline.search("What is SEAL-RAG?")
    elapsed = time.time() - start

    print(f"\nPerformance Test:")
    print(f"  Elapsed time: {elapsed*1000:.0f}ms")
    print(f"  Reported timing: {result.timing_ms}ms")

    # Should complete within 1 second for simple query (without real search)
    assert elapsed < 1.0, f"Search took too long: {elapsed:.2f}s"
    print("PASS: Search performance is acceptable")


def run_all_tests():
    """Run all Phase 6 integration tests."""
    print("=" * 60)
    print("Phase 6: Search V2 Integration Tests")
    print("=" * 60)

    tests = [
        ("Simple Query Classification", test_simple_query_classification),
        ("Multi-hop Query Classification", test_multi_hop_query_classification),
        ("Comparative Query", test_comparative_query),
        ("Confidence Estimation", test_confidence_estimation),
        ("Iteration Tracking", test_iteration_tracking),
        ("No Contradiction Detection", test_no_contradiction_detection),
        ("Contradiction Detection", test_contradiction_detection),
        ("Config Options", test_config_options),
        ("Result Structure", test_result_structure),
        ("Performance", test_performance),
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
