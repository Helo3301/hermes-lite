# Spec: Entity-First Ranking for SEAL-RAG Style Retrieval

## Problem Statement

Our v2 retrieval uses general relevance scoring but lacks **entity-targeted replacement** - the key innovation from SEAL-RAG that achieves 96% precision vs 22% baseline.

Current flow:
```
Query → Score chunks by relevance → Return top-k
```

SEAL-RAG flow:
```
Query → Extract entities → Find gaps → Micro-query for missing entities →
Replace low-entity-coverage chunks with high-coverage ones → Return fixed-k
```

## Comparison: Our v2 vs Literature

| System | Approach | HotpotQA Precision | Multi-hop Handling |
|--------|----------|-------------------|-------------------|
| **Basic RAG** | Retrieve top-k | 22% | Single retrieval |
| **Self-RAG** | Self-reflection tokens | ~45% | Adaptive retrieval |
| **CRAG** | Corrective evaluation | ~50% | Quality filtering |
| **SEAL-RAG** | Entity-first replacement | **96%** | Micro-queries + swap |
| **Our v2** | Iterative + confidence | **?** (need to measure) | Gap-fill queries |

## Proposed Solution: EntityFirstRanker

### Core Algorithm

```python
class EntityFirstRanker:
    """
    SEAL-RAG style entity-first chunk ranking and replacement.

    Key principle: Chunks are valuable if they cover MISSING entities,
    not just if they're generally relevant.
    """

    def __init__(self, replacement_threshold: float = 0.1):
        self.replacement_threshold = replacement_threshold

    def compute_entity_coverage(
        self,
        chunk: dict,
        target_entities: list[str]
    ) -> tuple[float, list[str]]:
        """
        Score chunk by how many target entities it contains.
        Returns (score, list of found entities).
        """
        pass

    def find_replacement_candidates(
        self,
        current_chunks: list[dict],
        new_chunks: list[dict],
        missing_entities: list[str]
    ) -> list[tuple[int, dict, float]]:
        """
        Find (current_idx, new_chunk, improvement) tuples.
        Only suggest replacement if improvement > threshold.
        """
        pass

    def replace_chunks(
        self,
        current_chunks: list[dict],
        new_chunks: list[dict],
        missing_entities: list[str],
        budget: int
    ) -> tuple[list[dict], list[str]]:
        """
        Execute SEAL-RAG style replacement.
        Returns (new_chunk_list, still_missing_entities).
        """
        pass
```

### Integration Points

1. **gap_detector.py**: Use `EntityFirstRanker.compute_entity_coverage()` for gap analysis
2. **retrieval.py**: Use `replace_chunks()` in iterative loop instead of concatenating
3. **chunk_scorer.py**: Add entity coverage as a scoring dimension

## Unit Test Criteria

### Test 1: Entity Coverage Computation
```python
def test_entity_coverage_basic():
    """Chunk covering 2/3 entities should score 0.67"""
    chunk = {"content": "SEAL-RAG uses DPR for retrieval"}
    entities = ["SEAL-RAG", "DPR", "BM25"]
    score, found = ranker.compute_entity_coverage(chunk, entities)
    assert abs(score - 0.67) < 0.01
    assert set(found) == {"SEAL-RAG", "DPR"}
```

### Test 2: Entity Coverage - Case Insensitive
```python
def test_entity_coverage_case_insensitive():
    """Should match entities regardless of case"""
    chunk = {"content": "seal-rag outperforms crag on benchmarks"}
    entities = ["SEAL-RAG", "CRAG"]
    score, found = ranker.compute_entity_coverage(chunk, entities)
    assert score == 1.0
```

### Test 3: Replacement Detection
```python
def test_finds_replacement_candidate():
    """Should identify when new chunk covers more missing entities"""
    current = [{"content": "Generic RAG info", "id": 1}]
    new = [{"content": "SEAL-RAG specifically uses entity extraction", "id": 2}]
    missing = ["SEAL-RAG", "entity extraction"]

    candidates = ranker.find_replacement_candidates(current, new, missing)
    assert len(candidates) == 1
    assert candidates[0][2] > 0  # positive improvement
```

### Test 4: No Replacement When Not Beneficial
```python
def test_no_replacement_when_not_better():
    """Should NOT replace if new chunk doesn't improve entity coverage"""
    current = [{"content": "SEAL-RAG uses entity extraction for gap detection", "id": 1}]
    new = [{"content": "Machine learning is useful", "id": 2}]
    missing = ["SEAL-RAG", "entity extraction"]

    candidates = ranker.find_replacement_candidates(current, new, missing)
    assert len(candidates) == 0
```

### Test 5: Budget Constraint (Fixed-k)
```python
def test_maintains_budget():
    """Should never exceed budget (SEAL-RAG's key constraint)"""
    current = [{"content": f"chunk {i}", "id": i} for i in range(5)]
    new = [{"content": "SEAL-RAG info", "id": 100}]
    missing = ["SEAL-RAG"]

    result, _ = ranker.replace_chunks(current, new, missing, budget=5)
    assert len(result) == 5  # Never more than budget
```

### Test 6: Multi-Entity Gap Fill
```python
def test_multi_entity_gap_fill():
    """Should progressively fill multiple entity gaps"""
    current = [{"content": "Generic content", "id": 1}]
    new = [
        {"content": "SEAL-RAG is a method", "id": 2},
        {"content": "CRAG uses correction", "id": 3},
    ]
    missing = ["SEAL-RAG", "CRAG", "Self-RAG"]

    result, still_missing = ranker.replace_chunks(current, new, missing, budget=3)

    # Should have added both new chunks
    assert len(result) == 3
    # Should still be missing Self-RAG
    assert "Self-RAG" in still_missing
    assert "SEAL-RAG" not in still_missing
    assert "CRAG" not in still_missing
```

### Test 7: Precision Metric (vs SEAL-RAG's 96%)
```python
def test_precision_on_multi_hop():
    """
    Key benchmark: What % of returned chunks are actually relevant?
    SEAL-RAG achieves 96% precision.
    Target: > 80% precision with entity-first ranking.
    """
    # Simulate multi-hop query
    query_entities = ["SEAL-RAG", "HotpotQA", "precision"]

    # Mix of relevant and distractor chunks
    chunks = [
        {"content": "SEAL-RAG achieves 96% precision on HotpotQA", "relevant": True},
        {"content": "Machine learning is popular", "relevant": False},
        {"content": "HotpotQA is a multi-hop benchmark", "relevant": True},
        {"content": "Python is a programming language", "relevant": False},
        {"content": "Precision measures retrieval quality", "relevant": True},
    ]

    result = ranker.rank_by_entity_coverage(chunks, query_entities, top_k=3)

    precision = sum(1 for c in result if c["relevant"]) / len(result)
    assert precision >= 0.8, f"Precision {precision} below 80% target"
```

### Test 8: Latency Impact
```python
def test_latency_overhead():
    """Entity-first ranking should add < 50ms overhead"""
    import time

    chunks = [{"content": f"content {i}", "id": i} for i in range(100)]
    entities = ["SEAL-RAG", "CRAG", "DPR"]

    start = time.time()
    for _ in range(100):
        ranker.rank_by_entity_coverage(chunks, entities, top_k=10)
    elapsed = (time.time() - start) / 100 * 1000

    assert elapsed < 50, f"Ranking took {elapsed}ms, exceeds 50ms budget"
```

## Success Criteria

| Metric | Current v2 | Target | SEAL-RAG Reference |
|--------|-----------|--------|-------------------|
| Precision@k | ~50% (est) | >80% | 96% |
| Entity Coverage | 89% | >95% | ~95% |
| Latency Overhead | - | <100ms | N/A |
| Multi-hop Accuracy | ~36% | >50% | 61% |

## Implementation Order

1. `app/entity_ranker.py` - Core EntityFirstRanker class
2. `tests/test_entity_ranker.py` - All unit tests above
3. Integrate into `retrieval.py` iterative loop
4. Re-run benchmarks to measure improvement

## Files to Create/Modify

| File | Action |
|------|--------|
| `app/entity_ranker.py` | **CREATE** - EntityFirstRanker class |
| `tests/test_entity_ranker.py` | **CREATE** - Unit tests |
| `app/retrieval.py` | **MODIFY** - Use entity-first replacement |
| `app/gap_detector.py` | **MODIFY** - Use entity coverage in analysis |
