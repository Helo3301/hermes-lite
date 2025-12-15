# Hermes v2: Advanced RAG Implementation Plan

## Overview

Transform Hermes from a solid baseline RAG system into a state-of-the-art retrieval pipeline incorporating insights from recent literature (SEAL-RAG, FVA-RAG, NeuroPath, Graph RAG).

**Current State**: 747 papers, hybrid search (vector + BM25), RRF fusion, reranking, query expansion, diversity constraints

**Target State**: Iterative retrieval, gap detection, query decomposition, lightweight knowledge graph, contradiction detection, adaptive retrieval

---

## Phase 1: Query Intelligence Layer

### 1.1 Query Classification
**File**: `app/query_analyzer.py` (new)

Classify incoming queries to route them appropriately:
- **Simple factoid**: "What is RAG?" → single-shot retrieval
- **Multi-hop**: "How does SEAL-RAG improve on CRAG?" → iterative retrieval
- **Comparative**: "Compare dense vs sparse retrieval" → query decomposition
- **Exploratory**: "What are the latest advances in RAG?" → broad retrieval + clustering

Human Note: Love this approach.

Implementation:
```python
class QueryAnalyzer:
    def classify(self, query: str) -> QueryType
    def estimate_complexity(self, query: str) -> int  # 1-5 scale
    def extract_entities(self, query: str) -> list[str]
    def detect_intent(self, query: str) -> Intent  # lookup, compare, explain, explore
```

### 1.2 Query Decomposition
**File**: `app/query_analyzer.py`

For complex queries, decompose into sub-queries:
```python
def decompose_query(self, query: str) -> list[SubQuery]:
    """
    "Compare SEAL-RAG and CRAG" ->
    [
        SubQuery("What is the SEAL-RAG approach?", type="lookup"),
        SubQuery("What is the CRAG approach?", type="lookup"),
        SubQuery("key differences RAG approaches", type="compare")
    ]
    """
```

Uses: Small LLM call (llama3.2) or rule-based patterns
Human Note: Love this approach.
---

## Phase 2: Iterative Gap-Fill Retrieval (SEAL-RAG)

### 2.1 Gap Detection
**File**: `app/gap_detector.py` (new)

After initial retrieval, analyze what's missing:
```python
class GapDetector:
    def analyze_coverage(
        self,
        query: str,
        retrieved_chunks: list[dict],
        query_entities: list[str]
    ) -> GapAnalysis:
        """
        Returns:
        - missing_entities: entities mentioned in query but not in chunks
        - unanswered_aspects: parts of query not addressed
        - confidence_score: 0-1 how well query is covered
        - suggested_subqueries: targeted queries to fill gaps
        """
```
Human Note: This makes logical sense
### 2.2 Evidence Replacement Strategy
**File**: `app/retrieval.py` (modify)

Fixed-budget retrieval with replacement:
```python
def iterative_retrieve(
    self,
    query: str,
    budget: int = 10,  # fixed number of chunks
    max_iterations: int = 3
) -> list[dict]:
    """
    1. Initial retrieval (budget chunks)
    2. Gap detection
    3. If gaps found:
       - Generate sub-queries for gaps
       - Retrieve candidates for sub-queries
       - Score all chunks (original + new)
       - Keep top budget chunks (replacement, not expansion)
    4. Repeat until confident or max_iterations
    """
```

### 2.3 Chunk Quality Scoring
**File**: `app/chunk_scorer.py` (new)

Score chunks on multiple dimensions:
```python
class ChunkScorer:
    def score(self, chunk: dict, query: str) -> ChunkScore:
        return ChunkScore(
            relevance=self.semantic_score(chunk, query),
            specificity=self.measure_specificity(chunk),  # concrete vs vague
            factual_density=self.count_facts(chunk),  # entities, numbers
            source_quality=self.score_source(chunk),  # recency, citations
            confidence_signals=self.detect_confidence(chunk)  # hedging language
        )

    def combined_score(self, scores: ChunkScore, weights: dict) -> float
```

---

## Phase 3: Lightweight Knowledge Graph

### 3.1 Entity Extraction at Ingest
**File**: `app/entity_extractor.py` (new)

Extract entities when papers are ingested:
```python
class EntityExtractor:
    def extract(self, text: str, doc_id: int) -> list[Entity]:
        """
        Extract:
        - Paper titles (referenced works)
        - Method names (SEAL-RAG, CRAG, DPR, etc.)
        - Dataset names (HotpotQA, NQ, MS MARCO)
        - Author names
        - Key concepts (multi-hop, dense retrieval, etc.)
        """
```

### 3.2 Relationship Extraction
**File**: `app/entity_extractor.py`

Extract relationships between entities:
```python
class RelationshipExtractor:
    def extract(self, text: str, entities: list[Entity]) -> list[Relationship]:
        """
        Relationship types:
        - CITES: Paper A cites Paper B
        - EXTENDS: Method A builds on Method B
        - COMPARES: Paper compares A vs B
        - USES: Paper uses Dataset X
        - OUTPERFORMS: Method A beats Method B on X
        """
```

### 3.3 Database Schema Updates
**File**: `app/database.py` (modify)

Add tables for knowledge graph:
```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- paper, method, dataset, author, concept
    doc_id INTEGER REFERENCES documents(id),
    chunk_id INTEGER REFERENCES chunks(id),
    embedding BLOB
);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER REFERENCES entities(id),
    target_entity_id INTEGER REFERENCES entities(id),
    relationship_type TEXT NOT NULL,
    confidence REAL,
    evidence_chunk_id INTEGER REFERENCES chunks(id)
);

CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
```
Human Note: We should probably load up some data structure docs too. See if these indexes are really what we want.
### 3.4 Graph-Augmented Retrieval
**File**: `app/graph_retrieval.py` (new)

Combine vector search with graph traversal:
```python
class GraphRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        # 1. Standard vector retrieval
        vector_results = self.vector_search(query, top_k * 2)

        # 2. Extract entities from query
        query_entities = self.extract_entities(query)

        # 3. Find entity matches in graph
        matched_entities = self.match_entities(query_entities)

        # 4. Traverse 1-hop relationships
        related_chunks = self.traverse_graph(matched_entities, hops=1)

        # 5. Merge and deduplicate
        all_chunks = self.merge_results(vector_results, related_chunks)

        # 6. Re-rank combined results
        return self.rerank(query, all_chunks, top_k)
```

---

## Phase 4: Contradiction Detection (FVA-RAG lite)

### 4.1 Claim Extraction
**File**: `app/contradiction_detector.py` (new)

Extract factual claims from chunks:
```python
class ClaimExtractor:
    def extract_claims(self, chunk: dict) -> list[Claim]:
        """
        Extract verifiable claims:
        - "SEAL-RAG achieves 61% accuracy on 2WikiMultiHopQA"
        - "Context dilution degrades performance as k increases"
        """
```

### 4.2 Contradiction Detection
**File**: `app/contradiction_detector.py`

Detect conflicts between retrieved chunks:
```python
class ContradictionDetector:
    def detect(self, chunks: list[dict]) -> list[Contradiction]:
        """
        Compare claims across chunks:
        - Identify conflicting numbers/statistics
        - Detect opposing conclusions
        - Flag hedged vs confident claims on same topic

        Returns list of Contradiction objects with:
        - claim_a, claim_b
        - chunk_a, chunk_b
        - conflict_type: factual, methodological, interpretive
        - severity: high, medium, low
        """
```

### 4.3 Conflict Resolution
**File**: `app/contradiction_detector.py`

Surface conflicts to user or attempt resolution:
```python
def resolve_or_surface(
    self,
    contradictions: list[Contradiction],
    strategy: str = "surface"  # or "resolve", "newest_wins", "most_cited"
) -> ResolutionResult:
    """
    Options:
    - surface: Include note in response about disagreement
    - resolve: Retrieve more evidence, pick winner
    - newest_wins: Prefer more recent paper
    - most_cited: Prefer more authoritative source
    """
```

---

## Phase 5: Adaptive Retrieval

### 5.1 Confidence Estimation
**File**: `app/confidence_estimator.py` (new)

Estimate confidence in retrieved results:
```python
class ConfidenceEstimator:
    def estimate(
        self,
        query: str,
        chunks: list[dict],
        scores: list[float]
    ) -> ConfidenceScore:
        """
        Factors:
        - Score distribution (tight cluster = confident)
        - Query entity coverage
        - Chunk agreement (multiple chunks say same thing)
        - Source diversity (multiple papers agree)
        """
```

### 5.2 Adaptive Depth
**File**: `app/search.py` (modify)

Dynamically adjust retrieval depth:
```python
def adaptive_search(
    self,
    query: str,
    min_k: int = 5,
    max_k: int = 30,
    confidence_threshold: float = 0.8
) -> list[dict]:
    """
    1. Start with min_k
    2. Estimate confidence
    3. If below threshold, increase k
    4. Repeat until confident or max_k reached
    """
```

---

## Phase 6: Integration & API Updates

### 6.1 New Search Endpoint
**File**: `app/main.py` (modify)

Add advanced search endpoint:
```python
@app.post("/search/v2")
async def search_v2(
    query: str,
    mode: str = "auto",  # auto, simple, iterative, comparative
    max_iterations: int = 3,
    budget: int = 10,
    detect_contradictions: bool = True,
    use_graph: bool = True,
    explain: bool = False  # return reasoning trace
) -> SearchV2Response:
    """
    Returns:
    - results: list of chunks
    - metadata:
      - query_type: detected query classification
      - iterations: how many retrieval rounds
      - gaps_filled: what was missing and how it was addressed
      - contradictions: any detected conflicts
      - confidence: overall confidence score
      - reasoning_trace: (if explain=True) step-by-step decisions
    """
```

### 6.2 Batch Re-indexing for Graph
**File**: `app/main.py` (modify)

Endpoint to build knowledge graph from existing documents:
```python
@app.post("/admin/build-graph")
async def build_knowledge_graph(
    collection: str = "ai-papers",
    batch_size: int = 50
) -> dict:
    """
    Process all documents to extract entities and relationships.
    Can be run incrementally.
    """
```

---

## Implementation Order

### Weekend Sprint 1: Foundation (Day 1)
1. [ ] Query Analyzer (classification, decomposition)
2. [ ] Database schema updates (entities, relationships tables)
3. [ ] Basic entity extraction (method names, datasets)
4. [ ] Unit tests for new modules

### Weekend Sprint 2: Core Iteration (Day 2)
5. [ ] Gap Detector
6. [ ] Chunk Scorer
7. [ ] Iterative retrieval loop (SEAL-RAG style)
8. [ ] Integration tests

### Weekend Sprint 3: Graph & Contradictions (Day 3)
9. [ ] Relationship extraction
10. [ ] Graph-augmented retrieval
11. [ ] Contradiction detection
12. [ ] Adaptive retrieval depth

### Final Integration (Day 4)
13. [ ] New /search/v2 endpoint
14. [ ] Build graph for existing documents
15. [ ] End-to-end testing
16. [ ] Docker rebuild and deployment
17. [ ] Performance benchmarking

---

## Testing Strategy

### Unit Tests
- Query classification accuracy
- Entity extraction recall
- Gap detection on known multi-hop queries
- Contradiction detection precision

### Integration Tests
- Full pipeline on sample queries
- Comparison: v1 vs v2 on same queries
- Latency measurements

### Evaluation Queries
```python
TEST_QUERIES = [
    # Simple (should stay fast)
    "What is RAG?",
    "Define context dilution",

    # Multi-hop (should improve)
    "How does SEAL-RAG address the problems identified in basic RAG?",
    "What datasets are used to evaluate multi-hop RAG systems?",

    # Comparative (should improve significantly)
    "Compare SEAL-RAG, CRAG, and Self-RAG approaches",
    "What are the tradeoffs between dense and sparse retrieval?",

    # Exploratory
    "What are the main challenges in RAG systems according to recent papers?",
    "How has RAG research evolved in 2025?",
]
```

---

## Rollback Plan

If issues arise:
1. v1 baseline preserved in `/backups/v1-baseline/`
2. New code in separate modules (not modifying core until integration)
3. Feature flags for new functionality
4. Can revert Docker image to previous version

---

## Success Criteria

| Metric | v1 Baseline | v2 Target |
|--------|-------------|-----------|
| Simple query latency | ~500ms | <600ms (slight increase OK) |
| Multi-hop query accuracy | ~40% | >70% |
| Comparative query quality | Poor | Good (subjective) |
| Contradiction surfacing | None | Detected when present |
| Graph coverage | 0% | >80% of papers have entities |

---

## Files to Create

| File | Purpose |
|------|---------|
| `app/query_analyzer.py` | Query classification and decomposition |
| `app/gap_detector.py` | Missing information detection |
| `app/chunk_scorer.py` | Multi-dimensional chunk scoring |
| `app/entity_extractor.py` | Entity and relationship extraction |
| `app/graph_retrieval.py` | Graph-augmented search |
| `app/contradiction_detector.py` | Claim extraction and conflict detection |
| `app/confidence_estimator.py` | Retrieval confidence scoring |
| `tests/test_v2_*.py` | Test suites for each module |

## Files to Modify

| File | Changes |
|------|---------|
| `app/database.py` | Add entity/relationship tables and methods |
| `app/search.py` | Integrate iterative and adaptive retrieval |
| `app/main.py` | Add /search/v2 and /admin/build-graph endpoints |
| `app/ingest.py` | Hook entity extraction into ingestion pipeline |

---

## Notes

- Use Ollama (llama3.2) for LLM calls to keep it local
- Entity extraction can use spaCy + custom patterns (no external API)
- Start with rule-based approaches, upgrade to ML if needed
- Keep v1 endpoint working alongside v2
- Log everything for debugging iteration behavior
