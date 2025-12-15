"""Tests for Knowledge Graph components."""

import sys
import os
import time
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.entity_extractor import (
    EntityExtractor,
    RelationshipExtractor,
    DocumentEntityExtractor,
    ExtractedEntity,
)

# Database tests require sqlite_vec which is only in Docker
try:
    from app.database import Database, init_database
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Note: sqlite_vec not available, skipping database tests")


# Test texts
SEAL_RAG_TEXT = """
SEAL-RAG implements a fixed-budget evidence assembly approach for multi-hop QA.
Unlike traditional RAG systems that expand context indefinitely, SEAL-RAG
replaces low-quality chunks with better ones from targeted sub-queries.
The method achieves 96% precision on HotpotQA, compared to 22% for basic RAG.
SEAL-RAG uses entity extraction and iterative retrieval to improve coverage.
It outperforms CRAG and Self-RAG on multi-hop benchmarks like 2WikiMultiHopQA.
The approach builds on DPR for dense retrieval and uses BM25 for keyword matching.
"""

COMPARISON_TEXT = """
We compare SEAL-RAG with several baselines including CRAG, Self-RAG, and basic RAG.
On HotpotQA, SEAL-RAG achieves 61% accuracy while CRAG achieves 45%.
The method uses ColBERT for dense retrieval and reranking.
SEAL-RAG extends the ideas from Self-RAG by adding gap detection.
Both SEAL-RAG and CRAG use the BEIR benchmark for evaluation.
"""


def test_entity_extraction_methods():
    """Test extraction of method entities."""
    extractor = EntityExtractor(use_llm=False)

    entities = extractor.extract(SEAL_RAG_TEXT)
    method_names = [e.name for e in entities if e.entity_type == "method"]

    print(f"\nMethod Extraction Test:")
    print(f"  Found methods: {method_names}")

    expected = ["SEAL-RAG", "RAG", "CRAG", "Self-RAG", "DPR", "BM25"]
    found = 0
    for expected_method in expected:
        if any(expected_method.lower() == m.lower() for m in method_names):
            found += 1
        else:
            print(f"  Missing: {expected_method}")

    recall = found / len(expected)
    print(f"  Recall: {recall:.2%} ({found}/{len(expected)})")

    assert recall >= 0.70, f"Method extraction recall {recall:.2%} < 70%"
    print("PASS: Method entity extraction meets threshold")


def test_entity_extraction_datasets():
    """Test extraction of dataset entities."""
    extractor = EntityExtractor(use_llm=False)

    entities = extractor.extract(SEAL_RAG_TEXT)
    dataset_names = [e.name for e in entities if e.entity_type == "dataset"]

    print(f"\nDataset Extraction Test:")
    print(f"  Found datasets: {dataset_names}")

    expected = ["HotpotQA", "2WikiMultiHopQA"]
    found = sum(1 for d in expected if any(d.lower() == ds.lower() for ds in dataset_names))
    recall = found / len(expected)

    print(f"  Recall: {recall:.2%} ({found}/{len(expected)})")

    assert recall >= 0.70, f"Dataset extraction recall {recall:.2%} < 70%"
    print("PASS: Dataset entity extraction meets threshold")


def test_entity_extraction_speed():
    """Test that entity extraction is fast enough."""
    extractor = EntityExtractor(use_llm=False)

    # Larger text for timing
    long_text = SEAL_RAG_TEXT * 10

    start = time.time()
    entities = extractor.extract(long_text)
    elapsed = time.time() - start

    print(f"\nExtraction Speed Test:")
    print(f"  Text length: {len(long_text)} chars")
    print(f"  Entities found: {len(entities)}")
    print(f"  Time: {elapsed*1000:.0f}ms")

    # Should be < 2 seconds per document
    assert elapsed < 2.0, f"Extraction took {elapsed:.1f}s > 2s"
    print("PASS: Entity extraction speed meets threshold")


def test_relationship_extraction():
    """Test extraction of relationships between entities."""
    entity_extractor = EntityExtractor(use_llm=False)
    rel_extractor = RelationshipExtractor(use_llm=False)

    entities = entity_extractor.extract(COMPARISON_TEXT)
    relationships = rel_extractor.extract(COMPARISON_TEXT, entities)

    print(f"\nRelationship Extraction Test:")
    print(f"  Entities: {[e.name for e in entities]}")
    print(f"  Relationships found: {len(relationships)}")
    for rel in relationships:
        print(f"    {rel.source} --[{rel.relationship_type}]--> {rel.target}")

    # Should find at least one relationship
    assert len(relationships) >= 1, "Should extract at least one relationship"
    print("PASS: Relationship extraction works")


def test_database_entity_methods():
    """Test database entity storage and retrieval."""
    if not HAS_DATABASE:
        print("  Skipped: sqlite_vec not available")
        return

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        db = Database(db_path)

        # Insert test document
        doc_id = db.insert_document(
            filename="test.pdf",
            source_path="/test/path",
            doc_type="pdf",
            clean_md="Test content",
            content_hash="test123"
        )

        # Insert entities
        entity_id1 = db.insert_entity(
            name="SEAL-RAG",
            entity_type="method",
            doc_id=doc_id,
            confidence=1.0
        )
        entity_id2 = db.insert_entity(
            name="CRAG",
            entity_type="method",
            doc_id=doc_id,
            confidence=1.0
        )

        # Test retrieval
        entity = db.get_entity_by_name("seal-rag")  # Case-insensitive
        assert entity is not None, "Should find entity by name"
        assert entity['name'] == "SEAL-RAG"

        # Test relationship
        rel_id = db.insert_relationship(
            source_entity_id=entity_id1,
            target_entity_id=entity_id2,
            relationship_type="outperforms"
        )
        assert rel_id is not None

        # Test related entities
        related = db.get_related_entities(entity_id1, direction="outgoing")
        assert len(related) >= 1
        assert related[0]['name'] == "CRAG"

        # Test graph stats
        stats = db.get_graph_stats()
        assert stats['entities'] == 2
        assert stats['relationships'] == 1

        print(f"\nDatabase Entity Test:")
        print(f"  Entities stored: {stats['entities']}")
        print(f"  Relationships: {stats['relationships']}")
        print("PASS: Database entity methods work correctly")

    finally:
        os.unlink(db_path)


def test_document_entity_extractor():
    """Test the high-level document entity extractor."""
    extractor = DocumentEntityExtractor(use_llm=False)

    entities, relationships = extractor.process_document(
        SEAL_RAG_TEXT + "\n\n" + COMPARISON_TEXT,
        doc_id=1
    )

    print(f"\nDocument Entity Extractor Test:")
    print(f"  Total entities: {len(entities)}")
    print(f"  Total relationships: {len(relationships)}")

    # Group by type
    by_type = {}
    for e in entities:
        by_type[e.entity_type] = by_type.get(e.entity_type, 0) + 1
    print(f"  By type: {by_type}")

    assert len(entities) >= 5, f"Should extract at least 5 entities, got {len(entities)}"
    assert "method" in by_type, "Should have method entities"
    print("PASS: Document entity extractor works correctly")


def run_all_tests():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("Phase 3: Knowledge Graph Tests")
    print("=" * 60)

    tests = [
        ("Method Extraction", test_entity_extraction_methods),
        ("Dataset Extraction", test_entity_extraction_datasets),
        ("Extraction Speed", test_entity_extraction_speed),
        ("Relationship Extraction", test_relationship_extraction),
        ("Database Entity Methods", test_database_entity_methods),
        ("Document Entity Extractor", test_document_entity_extractor),
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
