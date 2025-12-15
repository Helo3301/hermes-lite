# HERMES v2 Implementation Summary

## Overview

HERMES v2 implements advanced RAG (Retrieval Augmented Generation) techniques based on insights from:
- **SEAL-RAG**: Fixed-budget evidence assembly, "replace don't expand"
- **FVA-RAG**: Fact verification and analysis
- **NeuroPath**: Neural pathway reasoning
- **Graph RAG**: Knowledge graph augmentation

## Components Implemented

### Phase 1: Query Intelligence (`app/query_analyzer.py`)
- **Query Classification**: Categorizes queries as simple, multi-hop, comparative, or exploratory
- **Query Decomposition**: Breaks complex queries into sub-queries for targeted retrieval
- **Entity Extraction**: Identifies key entities (methods, datasets, concepts) from queries
- **Test Results**: 84% classification accuracy (threshold: 80%)

### Phase 2: Iterative Gap-Fill (`app/gap_detector.py`, `app/chunk_scorer.py`, `app/retrieval.py`)
- **Gap Detection**: Analyzes retrieved chunks to identify missing information
- **Chunk Scoring**: Multi-dimensional scoring (relevance, specificity, factual density, source quality)
- **Iterative Retrieval**: SEAL-RAG style fixed-budget retrieval that replaces low-quality chunks
- **Test Results**: 8/8 tests passing

### Phase 3: Knowledge Graph (`app/entity_extractor.py`, `app/graph_retrieval.py`)
- **Entity Extraction**: Pattern-based extraction of methods, datasets, concepts
- **Relationship Extraction**: Identifies relationships (outperforms, uses, extends, compares)
- **Graph-Augmented Search**: Combines vector search with knowledge graph traversal
- **Test Results**: 6/6 tests passing

### Phase 4: Contradiction Detection (`app/contradiction_detector.py`)
- **Claim Extraction**: Extracts factual claims from text (statistics, comparisons, methodologies)
- **Conflict Detection**: Identifies contradictions between claims from different sources
- **Surfacing**: Presents conflicts to users with source attribution
- **Test Results**: 6/6 tests passing

### Phase 5: Adaptive Retrieval (`app/confidence_estimator.py`)
- **Confidence Estimation**: Multi-factor confidence scoring
  - Score distribution (consistency of retrieval scores)
  - Entity coverage (query entities found in results)
  - Source agreement (overlap between sources)
  - Source diversity (results from multiple documents)
- **Adaptive Depth**: Dynamically adjusts retrieval depth based on confidence
- **Test Results**: 9/9 tests passing

### Phase 6: Integration (`app/search_v2.py`, `app/main.py`)
- **SearchV2Pipeline**: Unified pipeline combining all components
- **`/search/v2` Endpoint**: New API endpoint with advanced features
- **Test Results**: 10/10 tests passing

## New API Endpoint

### GET `/search/v2`

**Parameters:**
- `query` (required): Natural language search query
- `top_k` (default: 10): Number of results
- `rerank` (default: true): Use cross-encoder reranking
- `detect_contradictions` (default: true): Surface conflicting claims
- `doc_filter` (optional): Filename pattern filter

**Response:**
```json
{
  "query": "How does SEAL-RAG work?",
  "query_type": "simple",
  "sub_queries": [],
  "entities": ["SEAL-RAG", "RAG"],
  "results": [...],
  "confidence": {
    "score": 0.85,
    "explanation": "Based on 5 results: results have high relevance scores; covers all query entities"
  },
  "contradictions": null,
  "metadata": {
    "iterations": 1,
    "timing_ms": 450.5,
    "status": "success"
  }
}
```

## Configuration

Add to `config.yaml`:
```yaml
search_v2:
  use_query_analysis: true
  decompose_multi_hop: true
  use_iterative: true
  max_iterations: 3
  budget: 10
  use_adaptive: true
  min_k: 5
  max_k: 30
  confidence_threshold: 0.6
  detect_contradictions: true
  surface_contradictions: true
  use_llm: false
```

## Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| `app/query_analyzer.py` | Query classification, decomposition, entity extraction |
| `app/gap_detector.py` | Gap detection after retrieval |
| `app/chunk_scorer.py` | Multi-dimensional chunk scoring |
| `app/entity_extractor.py` | Entity and relationship extraction |
| `app/graph_retrieval.py` | Graph-augmented search |
| `app/contradiction_detector.py` | Claim extraction and conflict detection |
| `app/confidence_estimator.py` | Confidence estimation and adaptive retrieval |
| `app/search_v2.py` | Unified v2 search pipeline |
| `scripts/rebuild_graph.py` | Knowledge graph rebuild utility |

### Modified Files
| File | Changes |
|------|---------|
| `app/database.py` | Added entity/relationship tables and methods |
| `app/retrieval.py` | Added `iterative_retrieve` function |
| `app/main.py` | Added `/search/v2` endpoint and v2 initialization |

### Test Files
| File | Tests |
|------|-------|
| `tests/test_query_analyzer.py` | 6 tests |
| `tests/test_iterative_retrieval.py` | 8 tests |
| `tests/test_knowledge_graph.py` | 6 tests |
| `tests/test_contradiction_detector.py` | 6 tests |
| `tests/test_confidence_estimator.py` | 9 tests |
| `tests/test_search_v2.py` | 10 tests |

## Test Summary

| Phase | Component | Tests | Status |
|-------|-----------|-------|--------|
| 1 | Query Intelligence | 6/6 | ✅ Pass |
| 2 | Iterative Gap-Fill | 8/8 | ✅ Pass |
| 3 | Knowledge Graph | 6/6 | ✅ Pass |
| 4 | Contradiction Detection | 6/6 | ✅ Pass |
| 5 | Adaptive Retrieval | 9/9 | ✅ Pass |
| 6 | Integration | 10/10 | ✅ Pass |
| **Total** | | **45/45** | ✅ **Pass** |

## Usage

### Basic Usage
```bash
# Search with v2 pipeline
curl "http://localhost:8000/search/v2?query=How%20does%20SEAL-RAG%20work?"

# Comparative query
curl "http://localhost:8000/search/v2?query=Compare%20SEAL-RAG%20vs%20CRAG"

# Multi-hop query
curl "http://localhost:8000/search/v2?query=What%20are%20the%20key%20differences%20between%20SEAL-RAG%20and%20CRAG%20on%20multi-hop%20benchmarks?"
```

### Rebuild Knowledge Graph
```bash
cd /home/hestiasadmin/hermes-lite
python scripts/rebuild_graph.py --db /path/to/hermes.db -v
```

## Performance Targets

| Query Type | Target Latency | Notes |
|------------|---------------|-------|
| Simple | 500ms | Single-pass retrieval |
| Multi-hop | 10s | Up to 3 iterations |
| Exploratory | 5 min | Comprehensive search |

## V1 Baseline (for comparison)

From `benchmarks/results/v1_baseline/BASELINE_SUMMARY.md`:
- Latency P50: 6,695ms
- Entity Coverage: 89.76%
- Concept Relevance: 36.82%

V2 improvements target:
- Reduced latency for simple queries
- Improved relevance through iterative refinement
- Better handling of complex multi-hop queries
- Contradiction detection for conflicting sources

## Next Steps

1. Run full benchmarks against v2 endpoint
2. Compare v2 metrics to v1 baseline
3. Fine-tune confidence thresholds based on real-world usage
4. Consider enabling LLM-enhanced extraction for improved accuracy
