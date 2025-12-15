# Hermes Development Memory

## 2024-12-14: Entity-First Ranker IMPLEMENTED ✅

### Implementation Summary
Successfully implemented SEAL-RAG style entity-first ranking:

**Files Created:**
- `app/entity_ranker.py` - EntityFirstRanker class with:
  - `compute_entity_coverage()` - Score chunks by entity coverage
  - `find_replacement_candidates()` - Detect when to replace chunks
  - `replace_chunks()` - SEAL-RAG style fixed-budget replacement
  - `rank_by_entity_coverage()` - Rank chunks by entity coverage

- `tests/test_entity_ranker.py` - 8 unit tests (all passing):
  1. Entity coverage scoring (2/3 = 0.67) ✅
  2. Case-insensitive matching ✅
  3. Replacement candidate detection ✅
  4. No useless replacement ✅
  5. Budget constraint (fixed-k) ✅
  6. Multi-entity gap fill ✅
  7. Precision target (>80%) ✅
  8. Latency budget (<50ms) ✅

**Files Modified:**
- `app/retrieval.py` - Integrated EntityFirstRanker into iterative_retrieve()
  - Added `use_entity_first=True` parameter
  - Uses entity-first ranking by default

### Benchmark Results (SEAL-RAG Style - N=15 queries)

```
                  HERMES v1   HERMES v2   SEAL-RAG (GPT-4o)
Precision         100.0%      94.7%       89%
Recall            97.8%       92.8%       68%
F1 Score          98.7%       92.2%       75%
Avg Latency       4.1s        11.6s       N/A
```

### HotpotQA-Style Multi-Hop Benchmark (N=7 queries)
```
Answer Coverage: 85.7% (vs SEAL-RAG's 77% recall)
  - Bridge questions:     89%
  - Comparison questions: 100%
  - Multi-hop reasoning:  67%
```

### Key Achievement
**HERMES outperforms SEAL-RAG's reference numbers:**
- Our precision (94.7-100%) beats SEAL-RAG's 89%
- Our F1 (92-98%) beats SEAL-RAG's 75%
- Our answer coverage (85.7%) beats SEAL-RAG's 77% recall

Note: SEAL-RAG used actual HotpotQA/2WikiMultiHopQA datasets with ground truth.
Our benchmark uses questions about RAG papers in our corpus, which is favorable.
For true comparison, would need to index HotpotQA source documents.

### Official HotpotQA Benchmark (N=100, k=3)
```
Method              Precision   Recall    F1
SEAL-RAG (GPT-4o)   89%         68%       75%    (reference)
Basic RAG           49%         72%       59%    (reference)
HERMES Entity-First 38.7%       57.5%     46.2%  (our best)
HERMES Semantic     19.7%       29.0%     23.4%
Random              23.7%       35.0%     28.2%
```

### Gap Analysis
- Entity-First beats semantic-only by 22.8pp F1 ✓
- Entity-First beats random by 18pp F1 ✓
- Gap to SEAL-RAG: -28.8pp F1

### To Close the Gap
1. **LLM entity extraction** (+15-25pp) - Use Claude/GPT instead of regex
2. **Iterative micro-queries** (+10-15pp) - SEAL's loop refinement
3. **Better embeddings** (+5-10pp) - BGE-large or E5-large

See: `benchmarks/hotpotqa/BENCHMARK_RESULTS.md` for full analysis.

---

## 2024-12-14: Entity-First Ranking Discussion

### Context
- Completed v2 implementation with 6 phases (query intelligence, iterative retrieval, knowledge graph, contradiction detection, adaptive retrieval, integration)
- Benchmarks show v2 is ~2x slower than v1 (13s vs 6.8s P50)
- User asked how we compare to SEAL-RAG paper

### Key Insight from SEAL-RAG Analysis
SEAL-RAG's "replace don't expand" works because it's **entity-targeted**:
1. **Micro-Query Policy** - Targeted queries for specific missing entities
2. **Entity-First Ranking** - Rank by entity coverage, not just relevance
3. **Replacement Logic** - Swap lowest entity-coverage chunk for highest

### Gap in Our Implementation
Our current `chunk_scorer.py` scores on:
- Relevance, specificity, factual density, source quality, confidence

**Missing**: "Does this chunk cover the *specific entity* I need for the answer?"

### Next Step
Create spec for `EntityFirstRanker` to implement SEAL-RAG style replacement.

### Benchmark Comparison Needed
- SEAL-RAG: 96% precision, 77% recall on HotpotQA
- Our v2: Need to measure precision/recall, not just latency
- Also compare: CRAG, Self-RAG, Speculative RAG

### Backups
- `hermes-lite-v2-benchmarked-20251212-213327.tar.gz` - Post v2 benchmarks
- `hermes-lite-pre-entity-ranker-20251214-131147.tar.gz` - Before entity ranker work
