# HERMES Development Kanban

## Backlog

### [HIGH] LLM Entity Extraction
- **Impact:** +20-25pp precision
- **Effort:** 2-3 hours
- **Description:** Replace regex-based entity extraction with LLM (Claude/Ollama)
- **Files:** `app/entity_extractor.py`, `app/query_analyzer.py`
- **Acceptance:** HotpotQA precision improves from 38% to 60%+

### [HIGH] Iterative Micro-Query Loop
- **Impact:** +10-15pp recall
- **Effort:** 4-6 hours
- **Description:** Implement full SEAL cycle (Search→Extract→Assess→Loop)
- **Files:** `app/retrieval.py`, `app/gap_detector.py`
- **Acceptance:** Multi-hop questions get targeted follow-up queries

### [MEDIUM] Better Embedding Model
- **Impact:** +5-10pp baseline
- **Effort:** 2 hours
- **Description:** Switch from nomic-embed-text to BGE-large or E5-large
- **Files:** `app/embed.py`, `config.yaml`
- **Acceptance:** Semantic-only baseline improves on HotpotQA

### [MEDIUM] Bridge Entity Detection
- **Impact:** +2-4pp on multi-hop
- **Effort:** 3-4 hours
- **Description:** Detect "bridge" entities that connect question to answer
- **Files:** `app/gap_detector.py`
- **Acceptance:** Bridge entities identified and used in micro-queries

### [LOW] Confidence Early Stopping
- **Impact:** -30% latency
- **Effort:** 1-2 hours
- **Description:** Stop iterating when confidence plateaus
- **Files:** `app/confidence_estimator.py`, `app/retrieval.py`
- **Acceptance:** Simple queries complete faster without accuracy loss

### [LOW] KG-Augmented Search
- **Impact:** +3-5pp on comparative queries
- **Effort:** 3-4 hours
- **Description:** Use knowledge graph to expand queries with related entities
- **Files:** `app/graph_retrieval.py`, `app/search_v2.py`
- **Acceptance:** "Compare X and Y" queries find related papers

---

## In Progress

### [DONE TODAY] Entity-First Ranker
- **Status:** ✅ Implemented & Tested
- **Files:** `app/entity_ranker.py`, `tests/test_entity_ranker.py`
- **Result:** 46.2% F1 on HotpotQA (beats semantic-only by 22.8pp)

---

## Done

### Phase 1: Query Intelligence ✅
- Query classification (simple, multi-hop, comparative, exploratory)
- Entity extraction from queries
- Query decomposition into sub-queries

### Phase 2: Iterative Retrieval ✅
- Fixed-budget retrieval (SEAL-RAG style)
- Gap detection for missing entities
- Multi-dimensional chunk scoring

### Phase 3: Knowledge Graph ✅
- Entity extraction from documents
- Relationship extraction
- Graph storage and queries

### Phase 4: Contradiction Detection ✅
- Claim extraction from chunks
- Conflict detection between claims
- Contradiction surfacing in results

### Phase 5: Adaptive Retrieval ✅
- Confidence estimation
- Query complexity assessment
- Adaptive depth based on confidence

### Phase 6: Integration ✅
- v2 search pipeline
- All components working together
- Benchmarking infrastructure

### Entity-First Ranker ✅ (Dec 14, 2024)
- `compute_entity_coverage()` - Score by entity coverage
- `find_replacement_candidates()` - Detect beneficial swaps
- `replace_chunks()` - Fixed-budget replacement
- `rank_by_entity_coverage()` - Rank by entities
- 8 unit tests passing
- HotpotQA benchmark suite created

---

## Blocked

*None currently*

---

## Metrics Dashboard

| Metric | Current | Target | SEAL-RAG |
|--------|---------|--------|----------|
| Precision@5 (HotpotQA) | 38.7% | **75%** | 89% |
| F1 Score (HotpotQA) | 46.2% | **70%** | 75% |
| Precision (Our Corpus) | 94.7% | 95%+ | 89% |
| Latency P50 | 13s | **5s** | N/A |
| Papers Indexed | 747 | 1000+ | N/A |

---

## Quick Commands

```bash
# Run HotpotQA benchmark
cd ~/hermes-lite/benchmarks/hotpotqa
python hotpotqa_hybrid_benchmark.py --sample 100 --top-k 3

# Run unit tests
python tests/test_entity_ranker.py

# Start HERMES
docker compose up -d

# Check logs
docker compose logs -f

# Rebuild after changes
docker compose down && docker compose build && docker compose up -d
```

---

## References

- SEAL-RAG Paper: "Replace, Don't Expand: Mitigating Context Dilution"
- HotpotQA: https://hotpotqa.github.io/
- Memory file: `memory.md`
- Benchmark results: `benchmarks/hotpotqa/BENCHMARK_RESULTS.md`
