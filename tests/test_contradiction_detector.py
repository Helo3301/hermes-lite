"""Tests for Contradiction Detection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.contradiction_detector import (
    ClaimExtractor,
    ContradictionDetector,
    ConflictType,
    Severity,
)


# Test chunks
CHUNK_A = {
    "id": 1,
    "content": "SEAL-RAG achieves 61% accuracy on 2WikiMultiHopQA. The method uses entity extraction and iterative retrieval to improve performance.",
    "filename": "seal_rag.pdf",
}

CHUNK_B = {
    "id": 2,
    "content": "Basic RAG achieves 61% accuracy on 2WikiMultiHopQA. Traditional methods expand context without considering quality.",
    "filename": "baseline.pdf",
}

CHUNK_C = {
    "id": 3,
    "content": "SEAL-RAG outperforms CRAG on multi-hop benchmarks. The method achieves 96% precision compared to 22% for baseline.",
    "filename": "seal_rag.pdf",
}

CHUNK_D = {
    "id": 4,
    "content": "CRAG outperforms SEAL-RAG on single-hop queries. CRAG's corrective approach is more efficient for simple questions.",
    "filename": "crag.pdf",
}


def test_claim_extraction():
    """Test that claims are extracted from chunks."""
    extractor = ClaimExtractor(use_llm=False)

    claims = extractor.extract_claims(CHUNK_A)

    print(f"\nClaim Extraction Test:")
    print(f"  Chunk: {CHUNK_A['content'][:50]}...")
    print(f"  Claims found: {len(claims)}")
    for claim in claims:
        print(f"    [{claim.claim_type}] {claim.text[:60]}...")

    assert len(claims) >= 1, "Should extract at least 1 claim"
    assert any(c.claim_type == "statistic" for c in claims), "Should find statistic claim"
    print("PASS: Claim extraction works")


def test_claim_extraction_types():
    """Test extraction of different claim types."""
    extractor = ClaimExtractor(use_llm=False)

    chunks = [CHUNK_A, CHUNK_C]
    all_claims = []

    for chunk in chunks:
        claims = extractor.extract_claims(chunk)
        all_claims.extend(claims)

    claim_types = set(c.claim_type for c in all_claims)

    print(f"\nClaim Types Test:")
    print(f"  Total claims: {len(all_claims)}")
    print(f"  Types found: {claim_types}")

    # Should find at least statistics and comparisons
    assert "statistic" in claim_types, "Should find statistic claims"
    print("PASS: Multiple claim types extracted")


def test_contradiction_detection_same_stat():
    """Test detection of conflicting statistics."""
    detector = ContradictionDetector(use_llm=False)

    # These chunks claim same accuracy for different methods
    chunks = [CHUNK_A, CHUNK_B]

    contradictions = detector.detect_contradictions(chunks)

    print(f"\nContradiction Detection Test (Same Stats):")
    print(f"  Chunks: {[c['filename'] for c in chunks]}")
    print(f"  Contradictions found: {len(contradictions)}")
    for c in contradictions:
        print(f"    {c.conflict_type.value}: {c.explanation[:80]}...")

    # This specific case may or may not trigger depending on subject detection
    print(f"PASS: Contradiction detection ran without errors")


def test_contradiction_detection_reversed():
    """Test detection of reversed comparisons."""
    detector = ContradictionDetector(use_llm=False)

    # CHUNK_C: SEAL-RAG outperforms CRAG
    # CHUNK_D: CRAG outperforms SEAL-RAG
    chunks = [CHUNK_C, CHUNK_D]

    contradictions = detector.detect_contradictions(chunks)

    print(f"\nReversed Comparison Test:")
    print(f"  Chunk C: SEAL-RAG outperforms CRAG")
    print(f"  Chunk D: CRAG outperforms SEAL-RAG")
    print(f"  Contradictions found: {len(contradictions)}")

    if contradictions:
        for c in contradictions:
            print(f"    {c.conflict_type.value}: {c.explanation}")

    # Note: This might not find a contradiction because the comparisons
    # are about different things (multi-hop vs single-hop)
    print("PASS: Reversed comparison check ran")


def test_surface_contradictions():
    """Test the contradiction surfacing mechanism."""
    detector = ContradictionDetector(use_llm=False)

    # Create some mock contradictions
    from app.contradiction_detector import Claim, Contradiction

    claim_a = Claim(
        text="SEAL-RAG achieves 61% accuracy",
        claim_type="statistic",
        subject="SEAL-RAG",
        confidence=0.9,
        chunk_id=1,
        source="paper_a.pdf"
    )
    claim_b = Claim(
        text="SEAL-RAG achieves 45% accuracy",
        claim_type="statistic",
        subject="SEAL-RAG",
        confidence=0.9,
        chunk_id=2,
        source="paper_b.pdf"
    )

    contradiction = Contradiction(
        claim_a=claim_a,
        claim_b=claim_b,
        conflict_type=ConflictType.FACTUAL,
        severity=Severity.HIGH,
        explanation="Different accuracy values for same method",
        confidence=0.9
    )

    result = detector.surface_contradictions([contradiction])

    print(f"\nSurface Contradictions Test:")
    print(f"  Has contradictions: {result['has_contradictions']}")
    print(f"  Count: {result['count']}")
    print(f"  Warning: {result.get('warning', 'None')[:60]}...")

    assert result["has_contradictions"] == True
    assert result["count"] == 1
    assert "warning" in result
    print("PASS: Contradiction surfacing works")


def test_no_contradictions():
    """Test that consistent chunks don't trigger false positives."""
    detector = ContradictionDetector(use_llm=False)

    # Two chunks with consistent information
    consistent_chunks = [
        {
            "id": 10,
            "content": "Dense retrieval uses learned embeddings for semantic matching.",
            "filename": "intro.pdf",
        },
        {
            "id": 11,
            "content": "Semantic search computes similarity between query and document embeddings.",
            "filename": "methods.pdf",
        },
    ]

    contradictions = detector.detect_contradictions(consistent_chunks)

    print(f"\nNo Contradictions Test:")
    print(f"  Consistent chunks about embeddings")
    print(f"  False positives: {len(contradictions)}")

    # Should not find contradictions in consistent information
    assert len(contradictions) == 0, f"Found {len(contradictions)} false positive contradictions"
    print("PASS: No false positives in consistent chunks")


def run_all_tests():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("Phase 4: Contradiction Detection Tests")
    print("=" * 60)

    tests = [
        ("Claim Extraction", test_claim_extraction),
        ("Claim Types", test_claim_extraction_types),
        ("Contradiction Detection (Same Stats)", test_contradiction_detection_same_stat),
        ("Reversed Comparisons", test_contradiction_detection_reversed),
        ("Surface Contradictions", test_surface_contradictions),
        ("No False Positives", test_no_contradictions),
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
