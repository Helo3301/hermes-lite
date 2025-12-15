"""Tests for iterative gap-fill retrieval."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.query_analyzer import QueryAnalyzer, QueryType
from app.gap_detector import GapDetector, GapAnalysis
from app.chunk_scorer import ChunkScorer, ChunkScore
from app.retrieval import iterative_retrieve


# Mock chunks for testing
MOCK_CHUNKS = {
    "seal_rag": {
        "id": 1,
        "content": "SEAL-RAG implements a fixed-budget evidence assembly approach. It uses entity extraction and gap detection to iteratively improve retrieval. The method achieves 96% precision on multi-hop QA.",
        "filename": "2512.10787v1.pdf",
        "doc_id": 1,
        "score": 0.85,
    },
    "crag": {
        "id": 2,
        "content": "CRAG (Corrective Retrieval Augmented Generation) evaluates retrieved documents and corrects retrieval when confidence is low. It uses a lightweight evaluator to assess document relevance.",
        "filename": "crag_paper.pdf",
        "doc_id": 2,
        "score": 0.80,
    },
    "context_dilution": {
        "id": 3,
        "content": "Context dilution occurs when too many irrelevant chunks are included, drowning out useful information. As retrieval depth k increases, precision typically decreases.",
        "filename": "2512.10787v1.pdf",
        "doc_id": 1,
        "score": 0.75,
    },
    "rag_basics": {
        "id": 4,
        "content": "Retrieval Augmented Generation (RAG) combines retrieval with generation. The system first retrieves relevant documents, then uses them as context for the language model.",
        "filename": "rag_survey.pdf",
        "doc_id": 3,
        "score": 0.70,
    },
    "multi_hop": {
        "id": 5,
        "content": "Multi-hop question answering requires combining information from multiple documents. Benchmarks like HotpotQA and 2WikiMultiHopQA test this capability.",
        "filename": "multi_hop_survey.pdf",
        "doc_id": 4,
        "score": 0.65,
    },
}


def mock_search(query: str, top_k: int = 10):
    """Mock search function that returns relevant chunks based on keywords."""
    query_lower = query.lower()
    results = []

    for key, chunk in MOCK_CHUNKS.items():
        content_lower = chunk["content"].lower()
        # Simple keyword matching
        if any(term in content_lower for term in query_lower.split()):
            results.append(chunk.copy())

    # Sort by score and limit
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


def test_gap_detector_coverage():
    """Test that gap detector correctly identifies missing entities."""
    detector = GapDetector()

    # Query with entities that should be in chunks
    query = "How does SEAL-RAG address context dilution?"
    chunks = [MOCK_CHUNKS["seal_rag"], MOCK_CHUNKS["context_dilution"]]

    analysis = detector.analyze_coverage(query, chunks, ["SEAL-RAG", "context dilution"])

    print(f"\nGap Detection Test:")
    print(f"  Query: {query}")
    print(f"  Coverage score: {analysis.coverage_score:.2%}")
    print(f"  Missing entities: {analysis.missing_entities}")
    print(f"  Should iterate: {analysis.should_iterate}")

    # Both entities should be found
    assert analysis.coverage_score >= 0.7, f"Coverage {analysis.coverage_score:.2%} should be >= 70%"
    assert len(analysis.missing_entities) == 0, f"Should find all entities, missing: {analysis.missing_entities}"
    print("PASS: Gap detector correctly identifies coverage")


def test_gap_detector_missing_entity():
    """Test that gap detector identifies missing entities."""
    detector = GapDetector()

    # Query with entity not in chunks
    query = "Compare SEAL-RAG and CRAG approaches"
    chunks = [MOCK_CHUNKS["seal_rag"]]  # Missing CRAG

    analysis = detector.analyze_coverage(query, chunks, ["SEAL-RAG", "CRAG"])

    print(f"\nMissing Entity Test:")
    print(f"  Query: {query}")
    print(f"  Coverage score: {analysis.coverage_score:.2%}")
    print(f"  Missing entities: {analysis.missing_entities}")

    assert "CRAG" in analysis.missing_entities, "Should identify CRAG as missing"
    assert analysis.should_iterate, "Should recommend iteration when entity missing"
    assert len(analysis.suggested_subqueries) > 0, "Should suggest sub-queries"
    print("PASS: Gap detector identifies missing entities")


def test_chunk_scorer_ranking():
    """Test that chunk scorer ranks high-quality chunks above low-quality."""
    scorer = ChunkScorer()

    query = "SEAL-RAG multi-hop retrieval"
    chunks = list(MOCK_CHUNKS.values())

    ranked = scorer.rank_chunks(chunks, query, ["SEAL-RAG", "multi-hop"])

    print(f"\nChunk Scoring Test:")
    print(f"  Query: {query}")
    for chunk, score in ranked[:3]:
        print(f"  {chunk['id']}: combined={score.combined:.3f} "
              f"(rel={score.relevance:.2f}, spec={score.specificity:.2f})")

    # SEAL-RAG chunk should rank highly
    top_chunk = ranked[0][0]
    assert "SEAL-RAG" in top_chunk["content"], "Top chunk should contain SEAL-RAG"
    print("PASS: Chunk scorer ranks relevant chunks highly")


def test_chunk_scorer_specificity():
    """Test that specific chunks score higher than vague ones."""
    scorer = ChunkScorer()

    specific_chunk = {
        "id": 100,
        "content": "SEAL-RAG achieves 96% precision on HotpotQA, compared to 22% for baseline RAG. The method uses 3 iterations with k=5 retrieved documents.",
        "score": 0.8,
    }
    vague_chunk = {
        "id": 101,
        "content": "This method might improve results. It could potentially help with some tasks. The approach seems to work in certain situations.",
        "score": 0.8,
    }

    specific_score = scorer.score(specific_chunk, "SEAL-RAG performance")
    vague_score = scorer.score(vague_chunk, "SEAL-RAG performance")

    print(f"\nSpecificity Test:")
    print(f"  Specific chunk: specificity={specific_score.specificity:.2f}, confidence={specific_score.confidence:.2f}")
    print(f"  Vague chunk: specificity={vague_score.specificity:.2f}, confidence={vague_score.confidence:.2f}")

    assert specific_score.specificity > vague_score.specificity, "Specific chunk should score higher on specificity"
    assert specific_score.confidence > vague_score.confidence, "Specific chunk should score higher on confidence"
    print("PASS: Specific chunks score higher than vague ones")


def test_iterative_simple_query():
    """Test that simple queries don't iterate unnecessarily."""
    result = iterative_retrieve(
        mock_search,
        "What is RAG?",
        budget=5,
        max_iterations=3
    )

    print(f"\nSimple Query Test:")
    print(f"  Query: What is RAG?")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Query type: {result['query_type']}")
    print(f"  Results: {len(result['results'])}")

    assert result["iterations"] == 1, f"Simple query should use 1 iteration, got {result['iterations']}"
    assert result["query_type"] == "simple", f"Should be simple query, got {result['query_type']}"
    print("PASS: Simple queries don't iterate unnecessarily")


def test_iterative_complex_query():
    """Test that complex queries iterate to improve coverage."""
    result = iterative_retrieve(
        mock_search,
        "Compare SEAL-RAG and CRAG approaches for multi-hop retrieval",
        budget=5,
        max_iterations=3,
        verbose=True
    )

    print(f"\nComplex Query Test:")
    print(f"  Query: Compare SEAL-RAG and CRAG...")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Query type: {result['query_type']}")
    print(f"  Queries used: {result['queries_used']}")
    print(f"  Final coverage: {result.get('final_coverage', 0):.2%}")

    # Complex query should use more than one query
    assert len(result["queries_used"]) >= 2, "Complex query should use multiple sub-queries"
    assert result["query_type"] in ["comparative", "multi_hop"], f"Should be complex type, got {result['query_type']}"
    print("PASS: Complex queries iterate appropriately")


def test_iterative_fixed_budget():
    """Test that iterative retrieval respects budget."""
    budget = 3

    result = iterative_retrieve(
        mock_search,
        "How does SEAL-RAG address context dilution in multi-hop retrieval?",
        budget=budget,
        max_iterations=3
    )

    print(f"\nFixed Budget Test:")
    print(f"  Budget: {budget}")
    print(f"  Results returned: {len(result['results'])}")

    assert len(result["results"]) <= budget, f"Should return <= {budget} results, got {len(result['results'])}"
    print("PASS: Iterative retrieval respects budget")


def test_iteration_latency():
    """Test that iterative retrieval completes within time limit."""
    query = "How does SEAL-RAG compare to CRAG for multi-hop question answering?"

    start = time.time()
    result = iterative_retrieve(
        mock_search,
        query,
        budget=10,
        max_iterations=3
    )
    elapsed = time.time() - start

    print(f"\nLatency Test:")
    print(f"  Query: {query[:50]}...")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Time: {elapsed*1000:.0f}ms")

    # Should complete quickly with mock search (no actual API calls)
    # Real system target is <5s
    assert elapsed < 1.0, f"Mock iterative search took {elapsed:.1f}s > 1s"
    print("PASS: Iterative retrieval completes quickly")


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2: Iterative Gap-Fill Retrieval Tests")
    print("=" * 60)

    tests = [
        ("Gap Detector Coverage", test_gap_detector_coverage),
        ("Gap Detector Missing Entity", test_gap_detector_missing_entity),
        ("Chunk Scorer Ranking", test_chunk_scorer_ranking),
        ("Chunk Scorer Specificity", test_chunk_scorer_specificity),
        ("Simple Query No Iteration", test_iterative_simple_query),
        ("Complex Query Iteration", test_iterative_complex_query),
        ("Fixed Budget Respect", test_iterative_fixed_budget),
        ("Iteration Latency", test_iteration_latency),
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
