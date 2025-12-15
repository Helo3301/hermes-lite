# HERMES v2 vs State-of-the-Art RAG: Analysis & Improvement Roadmap

## 1. Literature Benchmarks (From Papers in Hermes)

### SEAL-RAG Results (2WikiMultiHopQA, k=5)
| Method | Precision | Accuracy Δ |
|--------|-----------|------------|
| Basic RAG | 22% | baseline |
| CRAG | 22% | +2% |
| Self-RAG | ~35% | +4% |
| Adaptive-k | ~45% | +5% |
| **SEAL-RAG** | **96%** | **+8%** |

### FAIR-RAG Results (HotpotQA)
| Method | F1-Score | LLM-Judge Acc |
|--------|----------|---------------|
| Standard RAG | ~55% | ~60% |
| Self-RAG | ~62% | ~68% |
| Iter-Retgen | ~64% | ~70% |
| **FAIR-RAG** | **67%** | **72%** |

### Key Insights from Knowledge Graph
```
Relationships found:
- RQ-RAG outperforms Self-RAG
- MRAG extends RAG (multimodal)
- RAG compares GraphRAG
- RAG outperforms FT (fine-tuning alone)
```

---

## 2. HERMES v2 Current Implementation

### What We Have
| Component | Implementation | Status |
|-----------|----------------|--------|
| Query Classification | Pattern-based + entity extraction | ✅ Working |
| Query Decomposition | Sub-query generation | ✅ Working |
| Iterative Retrieval | SEAL-RAG style fixed-budget | ✅ Working |
| Gap Detection | Entity coverage + aspect analysis | ✅ Working |
| Chunk Scoring | Multi-dimensional (relevance, specificity, factual density) | ✅ Working |
| Knowledge Graph | Entity/relationship extraction | ✅ Working |
| Contradiction Detection | Claim extraction + conflict surfacing | ✅ Working |
| Confidence Estimation | Score distribution + entity coverage | ✅ Working |

### What's Missing (vs SEAL-RAG)
| SEAL-RAG Feature | Our Status | Gap |
|------------------|------------|-----|
| **Micro-Query Policy** | Basic sub-queries | No targeted entity-specific queries |
| **Entity-First Ranking** | General scoring | No entity salience weighting |
| **Scope-Aware Sufficiency** | Simple coverage check | No semantic sufficiency model |
| **Loop-Adaptive Extraction** | Fixed extraction | No dynamic extraction based on iteration |
| **Active Repair** | Replace low-quality | No explicit "bridge page" detection |

---

## 3. Demonstrably Improvable Areas

### 3.1 Micro-Query Generation (HIGH IMPACT)

**Current State**: We generate sub-queries from query decomposition, but they're generic.

**SEAL-RAG Approach**: Generate targeted micro-queries for *specific missing entities*.

**Proposed Enhancement**:
```python
def generate_micro_queries(query, gap_analysis):
    """Generate entity-specific micro-queries for missing information."""
    micro_queries = []

    for missing_entity in gap_analysis.missing_entities:
        # Template-based micro-queries
        micro_queries.append(f"What is {missing_entity}?")
        micro_queries.append(f"{missing_entity} definition and properties")

        # Relationship-based micro-queries
        for known_entity in gap_analysis.found_entities:
            micro_queries.append(f"relationship between {missing_entity} and {known_entity}")

    return micro_queries
```

**Expected Improvement**: +5-10% precision on multi-hop queries (based on SEAL-RAG ablations showing L=3 loops optimal)

---

### 3.2 Entity-First Ranking (HIGH IMPACT)

**Current State**: Chunk scoring uses general relevance + factual density.

**SEAL-RAG Approach**: Rank chunks by *entity salience* - prioritize chunks that mention query entities in prominent positions.

**Proposed Enhancement**:
```python
def entity_first_score(chunk, query_entities):
    """Score chunk based on entity salience."""
    content = chunk['content']
    score = 0.0

    for entity in query_entities:
        if entity.lower() in content.lower():
            # Position bonus: earlier = better
            pos = content.lower().find(entity.lower())
            position_score = 1.0 - (pos / len(content))

            # Frequency bonus
            freq = content.lower().count(entity.lower())
            freq_score = min(freq / 3, 1.0)

            # Title/header bonus (if entity in first 100 chars)
            title_bonus = 0.3 if pos < 100 else 0.0

            score += position_score * 0.4 + freq_score * 0.3 + title_bonus

    return score / max(len(query_entities), 1)
```

**Expected Improvement**: +3-5% precision by reducing "distractor" chunks

---

### 3.3 Bridge Page Detection (MEDIUM IMPACT)

**Current State**: Gap detector identifies missing entities but doesn't distinguish *bridge* entities.

**SEAL-RAG Insight**: Multi-hop questions have a "bridge" entity that connects the question entity to the answer entity. E.g., "What team did the director of Inception play for?" - "Inception director" → Christopher Nolan (bridge) → answer.

**Proposed Enhancement**:
```python
def identify_bridge_entities(query, initial_results):
    """Identify potential bridge entities for multi-hop reasoning."""
    # Extract entities from query
    query_entities = extract_entities(query)

    # Extract entities from results
    result_entities = set()
    for r in initial_results:
        result_entities.update(extract_entities(r['content']))

    # Bridge entities: appear in results but not in query
    # AND are mentioned alongside query entities
    bridge_candidates = result_entities - set(query_entities)

    bridge_entities = []
    for candidate in bridge_candidates:
        # Check if candidate co-occurs with query entities
        for r in initial_results:
            if candidate in r['content']:
                for qe in query_entities:
                    if qe in r['content']:
                        bridge_entities.append({
                            'entity': candidate,
                            'connects': qe,
                            'source': r['id']
                        })
                        break

    return bridge_entities
```

**Expected Improvement**: +2-4% on multi-hop questions by explicitly seeking bridge information

---

### 3.4 Confidence-Based Early Stopping (LOW-MEDIUM IMPACT)

**Current State**: We have confidence estimation but use fixed iteration count.

**Proposed Enhancement**: Stop iterating when confidence plateaus (diminishing returns).

```python
def adaptive_iteration(query, budget, confidence_threshold=0.7):
    """Iterate until confidence stabilizes or threshold met."""
    prev_confidence = 0.0
    improvement_threshold = 0.05  # Stop if improvement < 5%

    for iteration in range(1, max_iterations + 1):
        results = retrieve(query, budget)
        confidence = estimate_confidence(results)

        improvement = confidence - prev_confidence

        if confidence >= confidence_threshold:
            return results, "threshold_met"

        if improvement < improvement_threshold and iteration > 1:
            return results, "diminishing_returns"

        prev_confidence = confidence
        # Generate micro-queries for next iteration
```

**Expected Improvement**: 20-30% latency reduction on simple queries without accuracy loss

---

### 3.5 Knowledge Graph Integration (MEDIUM IMPACT)

**Current State**: We extract entities and relationships but don't use them in retrieval.

**Proposed Enhancement**: Use KG relationships to expand queries.

```python
def kg_augmented_search(query, db):
    """Augment search with knowledge graph relationships."""
    query_entities = extract_entities(query)

    expanded_entities = set(query_entities)
    for entity in query_entities:
        # Get related entities from KG
        entity_record = db.get_entity_by_name(entity)
        if entity_record:
            related = db.get_related_entities(entity_record['id'])
            for rel in related:
                if rel['relationship_type'] in ['uses', 'extends', 'outperforms']:
                    expanded_entities.add(rel['name'])

    # Search for original query + related entities
    results = []
    results.extend(search(query))
    for entity in expanded_entities - set(query_entities):
        results.extend(search(f"{query} {entity}", top_k=3))

    return deduplicate_and_rank(results)
```

**Expected Improvement**: +3-5% recall on comparative queries

---

## 4. Implementation Priority

| Enhancement | Impact | Effort | Priority |
|------------|--------|--------|----------|
| Micro-Query Generation | HIGH | Medium | 1 |
| Entity-First Ranking | HIGH | Low | 2 |
| Confidence Early Stopping | MEDIUM | Low | 3 |
| KG-Augmented Search | MEDIUM | Medium | 4 |
| Bridge Page Detection | MEDIUM | High | 5 |

---

## 5. Measurable Success Criteria

### Benchmark Targets (2WikiMultiHopQA)
| Metric | Current v2 | Target | SEAL-RAG |
|--------|------------|--------|----------|
| Precision@5 | ~50%* | 75% | 96% |
| Entity Coverage | 90% | 95% | ~95% |
| Answer Accuracy | ~45%* | 55% | 53% |

*Estimated based on confidence scores and retrieval patterns

### Latency Targets
| Query Type | Current v2 | Target |
|------------|------------|--------|
| Simple | 6s | 3s |
| Multi-hop | 17s | 10s |
| Comparative | 14s | 8s |

---

## 6. Next Steps

1. **Implement Entity-First Ranking** (1-2 hours)
   - Add entity salience scoring to ChunkScorer
   - Weight by position and frequency

2. **Implement Micro-Query Generation** (2-3 hours)
   - Enhance GapDetector to generate entity-specific queries
   - Add to iterative retrieval loop

3. **Add Confidence Early Stopping** (1 hour)
   - Modify AdaptiveRetriever to check improvement delta
   - Add "diminishing_returns" status

4. **Benchmark After Each Change**
   - Run quick_benchmark.py
   - Track precision/latency improvements

5. **KG Integration** (3-4 hours)
   - Add graph traversal to search pipeline
   - Test on comparative queries
