# Hermes v2 Test Criteria

## Ground Rules

1. **v1 must not regress** — existing functionality stays working
2. **All tests automated** — no manual "looks good to me"
3. **Measurable thresholds** — pass/fail, not subjective
4. **Test before integrate** — each phase tested independently

---

## Baseline Measurements (v1 Current State)

Before implementing anything, establish baselines:

```python
BASELINE_QUERIES = {
    "simple": [
        "What is RAG?",
        "Define retrieval augmented generation",
        "What is context dilution?",
        "What is dense retrieval?",
        "What is BM25?",
    ],
    "multi_hop": [
        "How does SEAL-RAG address context dilution?",
        "What datasets are used to evaluate CRAG?",
        "Which papers cite the original RAG paper?",
        "What methods does NeuroPath combine?",
        "How does FVA-RAG detect sycophantic responses?",
    ],
    "comparative": [
        "Compare dense and sparse retrieval approaches",
        "What are the differences between SEAL-RAG and CRAG?",
        "How do graph-based RAG methods differ from vector-only?",
        "Compare single-shot vs iterative retrieval",
        "What are tradeoffs between reranking approaches?",
    ],
    "exploratory": [
        "What are the main challenges in RAG systems?",
        "What advances have been made in multi-hop QA?",
        "How has RAG research evolved recently?",
        "What are common evaluation benchmarks for RAG?",
        "What techniques reduce hallucination in RAG?",
    ]
}
```

### Baseline Metrics to Capture
- [ ] Latency (p50, p95, p99) per query type
- [ ] Result count per query
- [ ] Unique documents retrieved per query
- [ ] Top-3 chunk relevance (manual annotation of 5 queries per type)

---

## Phase 1: Query Intelligence

### 1.1 Query Classification

**Test**: Correctly classify query type

| Query | Expected Type |
|-------|---------------|
| "What is RAG?" | simple |
| "How does X improve on Y?" | multi_hop |
| "Compare A and B" | comparative |
| "What are the main challenges in X?" | exploratory |

**Pass criteria**:
- [ ] **≥90% accuracy** on 50 labeled test queries
- [ ] Classification latency **<50ms**

**Test file**: `tests/test_query_classifier.py`

```python
def test_classification_accuracy():
    correct = 0
    for query, expected in LABELED_QUERIES:
        result = classifier.classify(query)
        if result == expected:
            correct += 1
    accuracy = correct / len(LABELED_QUERIES)
    assert accuracy >= 0.90, f"Classification accuracy {accuracy:.2%} < 90%"

def test_classification_latency():
    times = []
    for query, _ in LABELED_QUERIES:
        start = time.time()
        classifier.classify(query)
        times.append(time.time() - start)
    p95 = np.percentile(times, 95)
    assert p95 < 0.050, f"Classification p95 latency {p95*1000:.1f}ms > 50ms"
```

### 1.2 Query Decomposition

**Test**: Complex queries decomposed into valid sub-queries

| Input | Expected Sub-queries |
|-------|---------------------|
| "Compare SEAL-RAG and CRAG" | ["What is SEAL-RAG?", "What is CRAG?", comparison query] |
| "How does X address the problems in Y?" | [query about Y's problems, query about X's approach] |

**Pass criteria**:
- [ ] Decomposition produces **≥2 sub-queries** for comparative/multi-hop
- [ ] Each sub-query is **valid** (not empty, not identical to original)
- [ ] Sub-queries **cover key entities** from original query
- [ ] Decomposition latency **<500ms** (allows LLM call)

**Test file**: `tests/test_query_decomposer.py`

```python
def test_decomposition_coverage():
    query = "Compare SEAL-RAG and CRAG approaches"
    entities = ["SEAL-RAG", "CRAG"]
    sub_queries = decomposer.decompose(query)

    assert len(sub_queries) >= 2, "Should produce at least 2 sub-queries"

    # Check entity coverage
    all_text = " ".join(sub_queries)
    for entity in entities:
        assert entity.lower() in all_text.lower(), f"Missing entity: {entity}"

def test_decomposition_validity():
    for query in COMPLEX_QUERIES:
        sub_queries = decomposer.decompose(query)
        for sq in sub_queries:
            assert len(sq) > 10, "Sub-query too short"
            assert sq != query, "Sub-query identical to original"
```

### 1.3 Entity Extraction

**Test**: Extract key entities from queries

| Query | Expected Entities |
|-------|-------------------|
| "How does SEAL-RAG work?" | ["SEAL-RAG"] |
| "Compare HotpotQA and NQ benchmarks" | ["HotpotQA", "NQ"] |

**Pass criteria**:
- [ ] **≥80% recall** on known entities in test queries
- [ ] **≥70% precision** (not too many false positives)
- [ ] Extraction latency **<100ms**

---

## Phase 2: Iterative Gap-Fill Retrieval

### 2.1 Gap Detection

**Test**: Identify missing information after initial retrieval

**Setup**: Use queries where we KNOW initial retrieval misses something

```python
GAP_TEST_CASES = [
    {
        "query": "How does SEAL-RAG's entity extraction compare to CRAG's approach?",
        "initial_retrieval_covers": ["SEAL-RAG entity extraction"],
        "expected_gap": ["CRAG approach", "CRAG entity handling"],
    },
    # ... more cases
]
```

**Pass criteria**:
- [ ] Gap detector identifies **≥70% of known gaps**
- [ ] Gap detector suggests **relevant sub-queries** for gaps
- [ ] False positive rate **<30%** (doesn't cry gap when coverage is good)

**Test file**: `tests/test_gap_detector.py`

```python
def test_gap_detection_recall():
    detected_gaps = 0
    total_gaps = 0
    for case in GAP_TEST_CASES:
        result = gap_detector.analyze(case["query"], case["mock_chunks"])
        for expected in case["expected_gap"]:
            total_gaps += 1
            if any(expected.lower() in gap.lower() for gap in result.gaps):
                detected_gaps += 1
    recall = detected_gaps / total_gaps
    assert recall >= 0.70, f"Gap detection recall {recall:.2%} < 70%"
```

### 2.2 Iterative Retrieval Loop

**Test**: Multiple iterations improve coverage

**Pass criteria**:
- [ ] Multi-hop queries show **improvement after iteration** (more relevant chunks)
- [ ] Simple queries **don't iterate unnecessarily** (≤1 iteration)
- [ ] Max iterations respected (no infinite loops)
- [ ] Total latency for iterative query **<5 seconds** (3 iterations max)

**Test file**: `tests/test_iterative_retrieval.py`

```python
def test_iteration_improves_coverage():
    query = "How does SEAL-RAG address problems identified in basic RAG?"

    # Single shot
    v1_results = search_v1(query, top_k=10)
    v1_entities_covered = count_entities_covered(v1_results, ["SEAL-RAG", "basic RAG", "problems"])

    # Iterative
    v2_results = search_v2_iterative(query, budget=10, max_iter=3)
    v2_entities_covered = count_entities_covered(v2_results, ["SEAL-RAG", "basic RAG", "problems"])

    assert v2_entities_covered >= v1_entities_covered, "Iteration should not reduce coverage"

def test_simple_query_no_unnecessary_iteration():
    query = "What is RAG?"
    result = search_v2_iterative(query, budget=10, max_iter=3)
    assert result.iterations <= 1, "Simple query should not need multiple iterations"

def test_iteration_latency():
    for query in MULTI_HOP_QUERIES:
        start = time.time()
        search_v2_iterative(query, budget=10, max_iter=3)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Iterative search took {elapsed:.1f}s > 5s"
```

### 2.3 Chunk Quality Scoring

**Test**: Chunk scorer ranks high-quality chunks above low-quality

**Setup**: Manually labeled chunks (high/medium/low quality)

**Pass criteria**:
- [ ] High-quality chunks score **higher than low-quality** in ≥85% of comparisons
- [ ] Scoring latency **<10ms per chunk**

---

## Phase 3: Knowledge Graph

### 3.1 Entity Extraction at Ingest

**Test**: Extract entities from paper content

**Pass criteria**:
- [ ] **≥70% recall** on method names (SEAL-RAG, CRAG, DPR, etc.)
- [ ] **≥70% recall** on dataset names (HotpotQA, NQ, MS MARCO, etc.)
- [ ] **≥50% recall** on paper references (cited works)
- [ ] Extraction time **<2 seconds per document**

**Test file**: `tests/test_entity_extraction.py`

```python
KNOWN_ENTITIES_BY_DOC = {
    "2512.10787v1.pdf": {  # SEAL-RAG paper
        "methods": ["SEAL-RAG", "CRAG", "Self-RAG", "Basic RAG"],
        "datasets": ["2WikiMultiHopQA", "HotpotQA", "MuSiQue"],
    },
    # ... more labeled documents
}

def test_entity_extraction_recall():
    for doc_id, expected in KNOWN_ENTITIES_BY_DOC.items():
        extracted = entity_extractor.extract(get_doc_content(doc_id))

        method_recall = len(set(extracted.methods) & set(expected["methods"])) / len(expected["methods"])
        dataset_recall = len(set(extracted.datasets) & set(expected["datasets"])) / len(expected["datasets"])

        assert method_recall >= 0.70, f"Method recall {method_recall:.2%} < 70%"
        assert dataset_recall >= 0.70, f"Dataset recall {dataset_recall:.2%} < 70%"
```

### 3.2 Relationship Extraction

**Test**: Extract relationships between entities

**Pass criteria**:
- [ ] **≥50% recall** on "X outperforms Y" relationships
- [ ] **≥50% recall** on "X cites Y" relationships
- [ ] **<30% false positive rate** on relationships

### 3.3 Graph-Augmented Retrieval

**Test**: Graph traversal surfaces relevant chunks that vector search misses

**Setup**: Queries where related entity is not in query text but connected in graph

**Pass criteria**:
- [ ] Graph retrieval finds **≥1 additional relevant chunk** in 50% of test cases
- [ ] Graph retrieval **does not significantly hurt** simple query performance
- [ ] Graph traversal adds **<200ms** to retrieval time

**Test file**: `tests/test_graph_retrieval.py`

```python
def test_graph_finds_related():
    # Query mentions SEAL-RAG, graph should find CRAG (since SEAL-RAG compares to it)
    query = "Explain the SEAL-RAG approach"

    vector_only = search_vector_only(query)
    graph_augmented = search_with_graph(query)

    # Check if graph found CRAG-related chunks
    vector_has_crag = any("CRAG" in c["content"] for c in vector_only)
    graph_has_crag = any("CRAG" in c["content"] for c in graph_augmented)

    # Graph should find related entities even if not in query
    if not vector_has_crag:
        assert graph_has_crag, "Graph should discover related entity CRAG"
```

---

## Phase 4: Contradiction Detection

### 4.1 Claim Extraction

**Test**: Extract factual claims from chunks

**Pass criteria**:
- [ ] Extracts **≥1 claim per chunk** on average (for paper content)
- [ ] Claims are **factual statements** not opinions/hedges
- [ ] Extraction latency **<100ms per chunk**

### 4.2 Contradiction Detection

**Test**: Detect actual contradictions between chunks

**Setup**: Manually create contradiction test cases

```python
CONTRADICTION_CASES = [
    {
        "chunk_a": "SEAL-RAG achieves 61% accuracy on 2WikiMultiHopQA",
        "chunk_b": "Basic RAG achieves 61% accuracy on 2WikiMultiHopQA",  # Contradiction!
        "expected": True,
    },
    {
        "chunk_a": "Dense retrieval outperforms BM25 on NQ",
        "chunk_b": "Dense retrieval uses learned embeddings",  # Not a contradiction
        "expected": False,
    },
]
```

**Pass criteria**:
- [ ] **≥70% precision** (detected contradictions are real)
- [ ] **≥50% recall** (finds at least half of real contradictions)
- [ ] Detection latency **<500ms** for 10 chunk pairs

---

## Phase 5: Adaptive Retrieval

### 5.1 Confidence Estimation

**Test**: Confidence correlates with actual result quality

**Pass criteria**:
- [ ] High-confidence results have **better relevance scores** than low-confidence
- [ ] Confidence estimation latency **<50ms**

### 5.2 Adaptive Depth

**Test**: System retrieves more for hard queries, less for easy

**Pass criteria**:
- [ ] Simple queries retrieve **≤10 chunks** on average
- [ ] Complex queries adaptively retrieve **more chunks** when needed
- [ ] Never exceeds **max_k budget**

---

## Phase 6: End-to-End Integration

### 6.1 v1 Non-Regression

**Test**: v1 endpoint still works identically

**Pass criteria**:
- [ ] `/search` endpoint returns **same results** as before
- [ ] Latency **within 10%** of baseline

### 6.2 v2 Endpoint Functionality

**Test**: New endpoint works with all features

**Pass criteria**:
- [ ] Returns valid response structure
- [ ] All optional parameters work (mode, iterations, etc.)
- [ ] Metadata includes expected fields (query_type, iterations, etc.)

### 6.3 Overall Performance

**Comparison**: v1 vs v2 on test query set

| Query Type | v1 Baseline | v2 Target | Metric |
|------------|-------------|-----------|--------|
| Simple | <500ms | <600ms | Latency p95 |
| Simple | N/A | No regression | Result relevance |
| Multi-hop | ~40% | **≥60%** | Entity coverage |
| Multi-hop | N/A | <5s | Latency p95 |
| Comparative | Poor | **Improved** | Subjective quality |
| Comparative | N/A | <5s | Latency p95 |

### 6.4 Stress Test

**Test**: System handles concurrent requests

**Pass criteria**:
- [ ] **10 concurrent requests** complete successfully
- [ ] No errors or crashes
- [ ] Latency degradation **<2x** under load

---

## Test Data Requirements

### Labeled Query Set (50 queries)
- 15 simple
- 15 multi-hop
- 10 comparative
- 10 exploratory

### Labeled Entity Set (10 documents)
- Known entities per document
- Known relationships per document

### Labeled Chunk Quality (30 chunks)
- 10 high quality
- 10 medium quality
- 10 low quality

### Contradiction Pairs (20 pairs)
- 10 actual contradictions
- 10 non-contradictions

---

## Acceptance Criteria Summary

| Phase | Must Pass |
|-------|-----------|
| 1 | Query classification ≥90%, decomposition produces valid sub-queries |
| 2 | Iteration improves multi-hop coverage, no unnecessary iteration on simple |
| 3 | Entity extraction ≥70% recall, graph finds related chunks |
| 4 | Contradiction precision ≥70%, recall ≥50% |
| 5 | Confidence correlates with quality, adaptive depth works |
| 6 | v1 non-regression, v2 functional, multi-hop improvement ≥20% |

---

## Open Questions for Discussion

1. **How do we measure "relevance" objectively?**
   - Option A: Manual annotation (gold standard but slow)
   - Option B: LLM-as-judge (scalable but circular)
   - Option C: Proxy metrics (entity coverage, chunk overlap with known-good)

Human Note: The only way humans do this internally is with multiplel different approaches. We measure against what we know, what we suspect, and what we can prove and then make a choice based on that. I think the conclusion I am coming to is that it is only through a series of systems that we will get anywhere close to general intelligence. 

2. **What's acceptable latency for v2?**
   - Current v1: ~500ms
   - v2 with iteration: could be 2-5s
   - Is 5s acceptable for complex queries? 
   Human Note: Lets set a multi tiered approach. This is why there are thinking models after all. Lets say 500 ms for a basic factoid, 10 seconds for one of medium complexity, and up to 5 minutes for a multi hop exploratory approach so long as we don't overwhelm the models and context. 

3. **Should contradiction detection be opt-in or default?**
   - Default ON: More informative but adds latency
   - Default OFF: Faster but user must know to enable
Human Note: Turn it on by default but make sure the user knows about it. Do not let the llm decide. 
4. **How many labeled examples do we need?**
   - Proposed: 50 queries, 10 docs, 30 chunks, 20 contradiction pairs
   - Is this enough? Too much?
This is sufficient. 