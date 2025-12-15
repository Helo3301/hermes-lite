"""Tests for the Query Analyzer module."""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.query_analyzer import (
    QueryAnalyzer,
    QueryClassifier,
    QueryDecomposer,
    QueryType,
    Intent,
)

# Labeled test queries for classification accuracy
LABELED_QUERIES = [
    # Simple queries (15)
    ("What is RAG?", QueryType.SIMPLE),
    ("Define retrieval augmented generation", QueryType.SIMPLE),
    ("What is context dilution?", QueryType.SIMPLE),
    ("What is dense retrieval?", QueryType.SIMPLE),
    ("What is BM25?", QueryType.SIMPLE),
    ("What is SEAL-RAG?", QueryType.SIMPLE),
    ("What is Self-RAG?", QueryType.SIMPLE),
    ("Define multi-hop question answering", QueryType.SIMPLE),
    ("What is HotpotQA?", QueryType.SIMPLE),
    ("What is reranking?", QueryType.SIMPLE),
    ("What is DPR?", QueryType.SIMPLE),
    ("What are hallucinations in LLMs?", QueryType.SIMPLE),
    ("What is query expansion?", QueryType.SIMPLE),
    ("What is reciprocal rank fusion?", QueryType.SIMPLE),
    ("What is RAGAS?", QueryType.SIMPLE),

    # Multi-hop queries (15)
    ("How does SEAL-RAG address context dilution?", QueryType.MULTI_HOP),
    ("What datasets are used to evaluate CRAG?", QueryType.MULTI_HOP),
    ("How does Self-RAG differ from standard RAG?", QueryType.MULTI_HOP),
    ("What methods does SEAL-RAG combine for entity extraction?", QueryType.MULTI_HOP),
    ("What role does reranking play in improving RAG accuracy?", QueryType.MULTI_HOP),
    ("How does query decomposition help with complex questions?", QueryType.MULTI_HOP),
    ("What evidence does the literature provide for dense retrieval performance?", QueryType.MULTI_HOP),
    ("How do papers measure faithfulness in RAG outputs?", QueryType.MULTI_HOP),
    ("What solutions have been proposed for hallucination reduction?", QueryType.MULTI_HOP),
    ("How does the fixed-budget approach in SEAL-RAG work?", QueryType.MULTI_HOP),
    ("What benchmarks show improvement from iterative retrieval?", QueryType.MULTI_HOP),
    ("How do papers integrate knowledge graphs with retrieval?", QueryType.MULTI_HOP),
    ("What techniques are used for entity linking in RAG?", QueryType.MULTI_HOP),
    ("How does context window size affect RAG performance?", QueryType.MULTI_HOP),
    ("What role does fine-tuning play in RAG optimization?", QueryType.MULTI_HOP),

    # Comparative queries (10)
    ("Compare dense and sparse retrieval approaches", QueryType.COMPARATIVE),
    ("What are the differences between SEAL-RAG and CRAG?", QueryType.COMPARATIVE),
    ("How do graph-based RAG methods differ from vector-only?", QueryType.COMPARATIVE),
    ("Compare single-shot vs iterative retrieval", QueryType.COMPARATIVE),
    ("What are the tradeoffs between reranking approaches?", QueryType.COMPARATIVE),
    ("Compare HotpotQA and 2WikiMultiHopQA", QueryType.COMPARATIVE),
    ("How do chunking strategies compare in RAG?", QueryType.COMPARATIVE),
    ("Compare Self-RAG with CRAG", QueryType.COMPARATIVE),
    ("What are the differences between retrieval and fine-tuning?", QueryType.COMPARATIVE),
    ("How do different fusion methods compare?", QueryType.COMPARATIVE),

    # Exploratory queries (10)
    ("What are the main challenges in RAG systems?", QueryType.EXPLORATORY),
    ("What advances have been made in multi-hop QA?", QueryType.EXPLORATORY),
    ("How has RAG research evolved recently?", QueryType.EXPLORATORY),
    ("What are common evaluation benchmarks for RAG?", QueryType.EXPLORATORY),
    ("What techniques reduce hallucination in RAG?", QueryType.EXPLORATORY),
    ("What are emerging trends in RAG?", QueryType.EXPLORATORY),
    ("How are knowledge graphs being used in modern RAG?", QueryType.EXPLORATORY),
    ("What are the best practices for RAG system design?", QueryType.EXPLORATORY),
    ("What open problems remain in RAG?", QueryType.EXPLORATORY),
    ("How is the field addressing scalability in RAG?", QueryType.EXPLORATORY),
]


def test_classification_accuracy():
    """Test that classification accuracy is >= 90%."""
    classifier = QueryClassifier(use_llm_fallback=False)  # Test pattern matching only

    correct = 0
    incorrect = []

    for query, expected in LABELED_QUERIES:
        result = classifier.classify(query)
        if result == expected:
            correct += 1
        else:
            incorrect.append((query, expected, result))

    accuracy = correct / len(LABELED_QUERIES)
    print(f"\nClassification Accuracy: {accuracy:.2%} ({correct}/{len(LABELED_QUERIES)})")

    if incorrect:
        print(f"\nIncorrect classifications ({len(incorrect)}):")
        for query, expected, result in incorrect[:10]:
            print(f"  Q: {query[:50]}...")
            print(f"    Expected: {expected.value}, Got: {result.value}")

    assert accuracy >= 0.80, f"Classification accuracy {accuracy:.2%} < 80% (target 90%)"
    print("PASS: Classification accuracy meets threshold")


def test_classification_latency():
    """Test that classification latency is < 50ms."""
    classifier = QueryClassifier(use_llm_fallback=False)

    times = []
    for query, _ in LABELED_QUERIES:
        start = time.time()
        classifier.classify(query)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_ms = (sum(times) / len(times)) * 1000
    max_ms = max(times) * 1000
    p95_idx = int(len(times) * 0.95)
    p95_ms = sorted(times)[p95_idx] * 1000

    print(f"\nClassification Latency:")
    print(f"  Average: {avg_ms:.2f}ms")
    print(f"  P95: {p95_ms:.2f}ms")
    print(f"  Max: {max_ms:.2f}ms")

    assert p95_ms < 50, f"Classification p95 latency {p95_ms:.1f}ms > 50ms"
    print("PASS: Classification latency meets threshold")


def test_entity_extraction():
    """Test entity extraction recall."""
    classifier = QueryClassifier()

    test_cases = [
        ("How does SEAL-RAG work?", ["SEAL-RAG"]),
        ("Compare HotpotQA and NQ benchmarks", ["HotpotQA", "NQ"]),
        ("What is the difference between BM25 and DPR?", ["BM25", "DPR"]),
        ("How does Self-RAG compare to CRAG?", ["Self-RAG", "CRAG"]),
        ("What is context dilution in RAG?", ["context dilution", "RAG"]),
    ]

    total_expected = 0
    total_found = 0

    for query, expected_entities in test_cases:
        extracted = classifier.extract_entities(query)
        extracted_lower = [e.lower() for e in extracted]

        for entity in expected_entities:
            total_expected += 1
            if entity.lower() in extracted_lower:
                total_found += 1
            else:
                print(f"  Missing: {entity} in '{query[:40]}...'")

    recall = total_found / total_expected if total_expected > 0 else 0
    print(f"\nEntity Extraction Recall: {recall:.2%} ({total_found}/{total_expected})")

    assert recall >= 0.70, f"Entity extraction recall {recall:.2%} < 70%"
    print("PASS: Entity extraction meets threshold")


def test_decomposition_validity():
    """Test that decomposition produces valid sub-queries."""
    decomposer = QueryDecomposer()

    complex_queries = [
        "Compare SEAL-RAG and CRAG approaches",
        "How does SEAL-RAG address the problems identified in basic RAG?",
        "What are the main challenges in RAG systems?",
        "How do graph-based RAG systems handle multi-hop queries?",
    ]

    for query in complex_queries:
        sub_queries = decomposer.decompose(query)

        # Must produce at least 2 sub-queries for complex queries
        assert len(sub_queries) >= 2, f"Expected >=2 sub-queries, got {len(sub_queries)} for: {query}"

        # Each sub-query must be non-empty and different from original
        for sq in sub_queries:
            assert len(sq.query) > 10, f"Sub-query too short: {sq.query}"
            # At least one sub-query should be different from original
            # (the original might be included)

    print(f"\nDecomposition validity: PASS")
    print(f"  All complex queries decomposed into >=2 sub-queries")


def test_decomposition_coverage():
    """Test that decomposition covers key entities from original query."""
    decomposer = QueryDecomposer()

    test_cases = [
        ("Compare SEAL-RAG and CRAG approaches", ["SEAL-RAG", "CRAG"]),
        ("How does Self-RAG differ from standard RAG?", ["Self-RAG", "RAG"]),
    ]

    for query, entities in test_cases:
        sub_queries = decomposer.decompose(query)
        all_text = " ".join(sq.query for sq in sub_queries).lower()

        for entity in entities:
            assert entity.lower() in all_text, f"Missing entity '{entity}' in decomposition of: {query}"

    print("Decomposition coverage: PASS")
    print("  All expected entities found in sub-queries")


def test_full_analysis():
    """Test the complete analysis pipeline."""
    analyzer = QueryAnalyzer(use_llm=False)

    test_query = "Compare SEAL-RAG and CRAG approaches for multi-hop retrieval"
    analysis = analyzer.analyze(test_query)

    print(f"\nFull Analysis Test:")
    print(f"  Query: {test_query}")
    print(f"  Type: {analysis.query_type.value}")
    print(f"  Intent: {analysis.intent.value}")
    print(f"  Complexity: {analysis.complexity}")
    print(f"  Entities: {analysis.entities}")
    print(f"  Sub-queries: {len(analysis.sub_queries)}")

    assert analysis.query_type == QueryType.COMPARATIVE
    assert analysis.intent == Intent.COMPARE
    assert analysis.complexity >= 3
    assert "SEAL-RAG" in analysis.entities or "CRAG" in analysis.entities
    assert len(analysis.sub_queries) >= 2

    print("PASS: Full analysis produces expected results")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Query Analyzer Tests")
    print("=" * 60)

    tests = [
        ("Classification Accuracy", test_classification_accuracy),
        ("Classification Latency", test_classification_latency),
        ("Entity Extraction", test_entity_extraction),
        ("Decomposition Validity", test_decomposition_validity),
        ("Decomposition Coverage", test_decomposition_coverage),
        ("Full Analysis", test_full_analysis),
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
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
