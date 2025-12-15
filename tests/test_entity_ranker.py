"""
Entity-First Ranker Unit Tests

Business Context:
-----------------
SEAL-RAG achieves 96% precision by ranking chunks based on whether they
cover SPECIFIC MISSING ENTITIES, not just general relevance.

Our current v2 scores ~50% precision because we rank by semantic similarity.
These tests verify that entity-first ranking improves precision for multi-hop queries.

Key Business Rules:
1. A chunk's value is determined by how many MISSING entities it covers
2. Replacement happens ONLY when a new chunk covers more missing entities
3. Budget is FIXED - we replace, never expand (SEAL-RAG's core insight)
4. Entity matching must be case-insensitive and handle variants
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Will import once implemented:
# from app.entity_ranker import EntityFirstRanker


# =============================================================================
# TEST 1: Basic Entity Coverage Scoring
# =============================================================================
# Business Rule: A chunk covering 2 of 3 required entities scores 0.67
# Why: We need to quantify how well a chunk addresses the query's needs

def test_entity_coverage_scores_by_fraction_of_entities_found():
    """
    GIVEN a chunk that mentions SEAL-RAG and DPR
    AND the query requires SEAL-RAG, DPR, and BM25
    WHEN we compute entity coverage
    THEN the score should be 2/3 = 0.67
    AND the found entities should be [SEAL-RAG, DPR]
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    chunk = {
        "content": "SEAL-RAG uses DPR for dense retrieval in its pipeline."
    }
    required_entities = ["SEAL-RAG", "DPR", "BM25"]

    score, found = ranker.compute_entity_coverage(chunk, required_entities)

    assert abs(score - 0.67) < 0.01, \
        f"Expected 0.67 (2/3 entities), got {score}"
    assert set(found) == {"SEAL-RAG", "DPR"}, \
        f"Expected SEAL-RAG and DPR, got {found}"

    print("PASS: Entity coverage correctly scores 2/3 = 0.67")


# =============================================================================
# TEST 2: Case-Insensitive Entity Matching
# =============================================================================
# Business Rule: Entity matching ignores case (seal-rag == SEAL-RAG)
# Why: Papers use inconsistent casing; we shouldn't miss matches

def test_entity_matching_is_case_insensitive():
    """
    GIVEN a chunk with lowercase "seal-rag" and "crag"
    AND the query requires uppercase "SEAL-RAG" and "CRAG"
    WHEN we compute entity coverage
    THEN both entities should be found (score = 1.0)
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    chunk = {
        "content": "seal-rag outperforms crag on multi-hop benchmarks"
    }
    required_entities = ["SEAL-RAG", "CRAG"]

    score, found = ranker.compute_entity_coverage(chunk, required_entities)

    assert score == 1.0, \
        f"Expected 1.0 (both entities found), got {score}"
    assert len(found) == 2, \
        f"Expected 2 entities found, got {len(found)}"

    print("PASS: Entity matching is case-insensitive")


# =============================================================================
# TEST 3: Replacement Candidate Detection
# =============================================================================
# Business Rule: Suggest replacement ONLY when new chunk covers MORE missing entities
# Why: This is SEAL-RAG's key insight - replace distractors with useful chunks

def test_identifies_replacement_when_new_chunk_covers_more_entities():
    """
    GIVEN current chunks with generic RAG content (0 missing entities covered)
    AND a new chunk that mentions SEAL-RAG and entity extraction
    AND the query is missing [SEAL-RAG, entity extraction]
    WHEN we find replacement candidates
    THEN the new chunk should be suggested as a replacement
    AND the improvement score should be positive
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    current_chunks = [
        {"content": "Retrieval systems use vector databases.", "id": 1}
    ]
    new_chunks = [
        {"content": "SEAL-RAG uses entity extraction to identify gaps.", "id": 2}
    ]
    missing_entities = ["SEAL-RAG", "entity extraction"]

    candidates = ranker.find_replacement_candidates(
        current_chunks, new_chunks, missing_entities
    )

    assert len(candidates) >= 1, \
        "Should identify at least one replacement candidate"

    # candidates format: (current_idx, new_chunk, improvement_score)
    _, new_chunk, improvement = candidates[0]
    assert improvement > 0, \
        f"Improvement should be positive, got {improvement}"
    assert new_chunk["id"] == 2, \
        "Should select the chunk with entity coverage"

    print("PASS: Correctly identifies replacement candidate")


# =============================================================================
# TEST 4: No Replacement When Not Beneficial
# =============================================================================
# Business Rule: Do NOT replace if new chunk doesn't improve entity coverage
# Why: Avoid churning chunks without benefit; stability matters

def test_does_not_replace_when_new_chunk_is_not_better():
    """
    GIVEN current chunks already covering SEAL-RAG and entity extraction
    AND a new chunk about generic machine learning (no target entities)
    WHEN we find replacement candidates
    THEN NO replacement should be suggested
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    current_chunks = [
        {"content": "SEAL-RAG uses entity extraction for gap detection.", "id": 1}
    ]
    new_chunks = [
        {"content": "Machine learning models require training data.", "id": 2}
    ]
    missing_entities = ["SEAL-RAG", "entity extraction"]

    candidates = ranker.find_replacement_candidates(
        current_chunks, new_chunks, missing_entities
    )

    assert len(candidates) == 0, \
        f"Should NOT suggest replacement, but found {len(candidates)} candidates"

    print("PASS: Does not replace when new chunk is not better")


# =============================================================================
# TEST 5: Budget Constraint (Fixed-k)
# =============================================================================
# Business Rule: NEVER exceed budget - this is SEAL-RAG's core "replace don't expand"
# Why: Context dilution occurs when we add too many chunks

def test_never_exceeds_budget_constraint():
    """
    GIVEN 5 current chunks and budget of 5
    AND a new chunk that covers missing entities
    WHEN we execute replacement
    THEN result should have EXACTLY 5 chunks (not 6)

    This is THE key constraint from SEAL-RAG:
    "Replace, Don't Expand" - fixed budget prevents context dilution
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    current_chunks = [
        {"content": f"Generic content chunk {i}", "id": i}
        for i in range(5)
    ]
    new_chunks = [
        {"content": "SEAL-RAG achieves 96% precision", "id": 100}
    ]
    missing_entities = ["SEAL-RAG"]
    budget = 5

    result, _ = ranker.replace_chunks(
        current_chunks, new_chunks, missing_entities, budget
    )

    assert len(result) == budget, \
        f"Budget is {budget}, but got {len(result)} chunks. MUST NOT EXPAND!"

    print("PASS: Maintains fixed budget (replace, don't expand)")


# =============================================================================
# TEST 6: Progressive Multi-Entity Gap Filling
# =============================================================================
# Business Rule: Should progressively fill gaps for multiple missing entities
# Why: Multi-hop queries often need 2-3 bridge entities to complete reasoning

def test_progressively_fills_multiple_entity_gaps():
    """
    GIVEN missing entities [SEAL-RAG, CRAG, Self-RAG]
    AND new chunks covering SEAL-RAG and CRAG (but not Self-RAG)
    WHEN we execute replacement
    THEN SEAL-RAG and CRAG should no longer be missing
    AND Self-RAG should still be missing

    This tests the iterative gap-fill behavior for multi-hop queries.
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    current_chunks = [
        {"content": "Generic retrieval information", "id": 1}
    ]
    new_chunks = [
        {"content": "SEAL-RAG is a retrieval method", "id": 2},
        {"content": "CRAG uses corrective mechanisms", "id": 3},
    ]
    missing_entities = ["SEAL-RAG", "CRAG", "Self-RAG"]
    budget = 3

    result, still_missing = ranker.replace_chunks(
        current_chunks, new_chunks, missing_entities, budget
    )

    # Verify we filled the gaps we could
    assert "Self-RAG" in still_missing, \
        "Self-RAG should still be missing (no chunk covers it)"
    assert "SEAL-RAG" not in still_missing, \
        "SEAL-RAG should be filled (chunk 2 covers it)"
    assert "CRAG" not in still_missing, \
        "CRAG should be filled (chunk 3 covers it)"

    print("PASS: Progressively fills multiple entity gaps")


# =============================================================================
# TEST 7: Precision Improvement (The Key Metric)
# =============================================================================
# Business Rule: Entity-first ranking should achieve >80% precision
# Why: SEAL-RAG achieves 96%; our target is to close the gap from ~50%

def test_achieves_target_precision_on_mixed_chunks():
    """
    GIVEN a mix of relevant and distractor chunks
    AND query entities [SEAL-RAG, HotpotQA, precision]
    WHEN we rank by entity coverage and take top-3
    THEN precision should be >= 80%

    Precision = (relevant chunks returned) / (total chunks returned)

    SEAL-RAG benchmark: 96% precision
    Our target: >80% precision (up from ~50% baseline)
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    # Simulate retrieval results: mix of relevant and distractors
    chunks = [
        {"content": "SEAL-RAG achieves 96% precision on HotpotQA", "id": 1, "_relevant": True},
        {"content": "Machine learning is a popular field", "id": 2, "_relevant": False},
        {"content": "HotpotQA is a multi-hop QA benchmark", "id": 3, "_relevant": True},
        {"content": "Python programming is widely used", "id": 4, "_relevant": False},
        {"content": "Precision measures retrieval accuracy", "id": 5, "_relevant": True},
        {"content": "Databases store information", "id": 6, "_relevant": False},
    ]
    query_entities = ["SEAL-RAG", "HotpotQA", "precision"]
    top_k = 3

    result = ranker.rank_by_entity_coverage(chunks, query_entities, top_k)

    relevant_count = sum(1 for c in result if c.get("_relevant", False))
    precision = relevant_count / len(result)

    assert precision >= 0.80, \
        f"Precision {precision:.0%} is below 80% target. " \
        f"Got {relevant_count}/{len(result)} relevant chunks."

    print(f"PASS: Achieved {precision:.0%} precision (target: 80%)")


# =============================================================================
# TEST 8: Latency Overhead Must Be Minimal
# =============================================================================
# Business Rule: Entity ranking overhead should be <50ms per call
# Why: We're already at 13s latency; can't add more

def test_ranking_latency_is_under_budget():
    """
    GIVEN 100 chunks to rank
    AND 5 target entities
    WHEN we run entity-first ranking 100 times
    THEN average latency should be < 50ms

    Performance budget: Entity ranking is O(chunks * entities)
    With 100 chunks and 5 entities = 500 string searches
    Should complete in <50ms on modern hardware.
    """
    from app.entity_ranker import EntityFirstRanker
    ranker = EntityFirstRanker()

    # Create test data
    chunks = [
        {"content": f"This is test content for chunk number {i} with some text", "id": i}
        for i in range(100)
    ]
    entities = ["SEAL-RAG", "CRAG", "Self-RAG", "DPR", "BM25"]

    # Warm up
    ranker.rank_by_entity_coverage(chunks, entities, top_k=10)

    # Measure
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        ranker.rank_by_entity_coverage(chunks, entities, top_k=10)
    elapsed_ms = (time.time() - start) / iterations * 1000

    assert elapsed_ms < 50, \
        f"Ranking took {elapsed_ms:.1f}ms, exceeds 50ms budget"

    print(f"PASS: Ranking latency {elapsed_ms:.1f}ms (budget: 50ms)")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all entity ranker tests."""
    print("=" * 70)
    print("Entity-First Ranker Tests")
    print("=" * 70)
    print()
    print("Business Context: SEAL-RAG achieves 96% precision by ranking")
    print("chunks based on MISSING ENTITY coverage, not general relevance.")
    print("Our target: >80% precision (up from ~50% baseline)")
    print()
    print("=" * 70)

    tests = [
        ("Entity Coverage Scoring", test_entity_coverage_scores_by_fraction_of_entities_found),
        ("Case-Insensitive Matching", test_entity_matching_is_case_insensitive),
        ("Replacement Detection", test_identifies_replacement_when_new_chunk_covers_more_entities),
        ("No Useless Replacement", test_does_not_replace_when_new_chunk_is_not_better),
        ("Budget Constraint", test_never_exceeds_budget_constraint),
        ("Multi-Entity Gap Fill", test_progressively_fills_multiple_entity_gaps),
        ("Precision Target", test_achieves_target_precision_on_mixed_chunks),
        ("Latency Budget", test_ranking_latency_is_under_budget),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except ImportError as e:
            print(f"SKIP: EntityFirstRanker not yet implemented ({e})")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
