# Hermes-Lite Upgrade Specifications

Based on analysis of 2,165 RAG research papers, these specs outline the top 3 recommended upgrades for Hermes-Lite.

## Specs Overview

| # | Upgrade | Impact | Effort | File |
|---|---------|--------|--------|------|
| 1 | **HyDE Query Expansion** | +15-20% recall | Low | [01-HyDE-spec.md](01-HyDE-spec.md) |
| 2 | **Context Compression** | 5-10x token reduction | Medium | [02-context-compression-spec.md](02-context-compression-spec.md) |
| 3 | **Embedding Model Upgrade** | +4 MTEB points | High | [03-embedding-upgrade-spec.md](03-embedding-upgrade-spec.md) |

## Priority Order

### Immediate (High ROI, Low Risk)
1. **HyDE** - Add ~50 lines of code, significant recall improvement

### Short-term (1-2 weeks)
2. **Context Compression** - Reduce LLM costs and latency

### Medium-term (2-4 weeks)
3. **Embedding Upgrade** - Requires migration, but long-term quality boost

## Research Foundation

These specs are informed by papers including:
- HyDE: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- Compression: ICPC, LLMLingua, Selective Context papers
- Embeddings: BGE-M3, Matryoshka Representation Learning

## Current Hermes-Lite Architecture

Already implemented:
- Hybrid search (RRF fusion)
- Cross-encoder reranking (bge-reranker-v2-m3)
- Query decomposition for multi-hop
- Iterative retrieval
- Contradiction detection
- Adaptive retrieval depth

## Quick Start

```bash
# View specs
cat specs/01-HyDE-spec.md

# Implement HyDE (simplest upgrade)
# 1. Create app/hyde.py (see spec)
# 2. Add to search pipeline
# 3. Test and tune blend_weight
```

## Stats

- **Papers analyzed**: 2,165
- **Chunks indexed**: 86,513
- **Database size**: 616 MB
- **Date**: December 2024
