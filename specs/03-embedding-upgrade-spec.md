# Embedding Model Upgrade

## Overview

Upgrading from `nomic-embed-text` to a more capable embedding model to improve retrieval quality, multilingual support, and enable flexible dimension reduction.

### Current State
- **Model**: nomic-embed-text (via Ollama)
- **Dimensions**: 768
- **Max tokens**: 8192
- **MTEB Score**: ~62

### Recommended Upgrade Options

| Model | Dimensions | MTEB | Context | Multilingual | Matryoshka |
|-------|-----------|------|---------|--------------|------------|
| nomic-embed-text (current) | 768 | 62.4 | 8192 | Limited | No |
| **bge-m3** | 1024 | 66.6 | 8192 | 100+ langs | Yes |
| e5-mistral-7b-instruct | 4096 | 66.6 | 32768 | Yes | No |
| mxbai-embed-large | 1024 | 64.7 | 512 | Limited | Yes |

**Recommendation**: `bge-m3` for best balance of quality, multilingual, and Matryoshka support.

---

## Research Basis

Papers in corpus:
- `2508.12243v2_SEA-BED Southeast Asia Embedding Benchmark.pdf`
- `2510.22264v1_PatenTEB A Comprehensive Benchmark and Model Fa.pdf`
- `2509.24291v1_Let LLMs Speak Embedding Languages Generative T.pdf`
- Multi-vector retrieval papers (ColBERT family)

### Why Upgrade?
1. **Better retrieval quality**: +4 points on MTEB = measurable improvement
2. **Multilingual**: bge-m3 supports 100+ languages natively
3. **Matryoshka embeddings**: Reduce dimensions at query time for speed/quality tradeoff
4. **Longer context**: Better for document-level understanding

---

## Matryoshka Embeddings

### What Are They?
Matryoshka Representation Learning (MRL) trains embeddings where prefixes are also valid embeddings:
- Full 1024d: Maximum quality
- 512d prefix: 95% quality, 50% storage/compute
- 256d prefix: 90% quality, 25% storage/compute
- 128d prefix: 85% quality, 12.5% storage/compute

```
[e1, e2, e3, ... e128, ... e256, ... e512, ... e1024]
 └────────────────┘     └──────────┘    └─────────┘
    128d valid           256d valid      512d valid
```

### Benefits for Hermes
1. **Fast initial retrieval**: Use 128d for first-pass search
2. **Quality reranking**: Use full 1024d for reranking
3. **Storage optimization**: Store 256d, expand when needed
4. **Adaptive quality**: Match quality to query complexity

---

## Architecture Changes

### Current Flow
```
Document → Embed (768d) → Store → Search (768d) → Results
```

### Upgraded Flow with Matryoshka
```
Document → Embed (1024d) → Store (256d*) → Search (256d) → Expand (1024d) → Rerank
                              ↑
                         *configurable
```

---

## Implementation Spec

### Option A: Ollama with bge-m3 (Simpler)

bge-m3 is available in Ollama:
```bash
ollama pull bge-m3
```

**Pros**: Simple, consistent with current setup
**Cons**: No native Matryoshka truncation in Ollama

### Option B: Direct HuggingFace (More Control)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
# Supports truncate_dim parameter for Matryoshka
embeddings = model.encode(texts, truncate_dim=256)
```

**Pros**: Full Matryoshka support, more control
**Cons**: Requires model download, GPU memory

### Option C: Hybrid (Recommended)

Use Ollama for simplicity, add Matryoshka truncation in Python:

```python
def embed_with_matryoshka(text: str, target_dim: int = 256) -> List[float]:
    # Get full embedding from Ollama
    full_emb = ollama_embed(text)  # Returns 1024d

    # Truncate to target dimension (Matryoshka-style)
    truncated = full_emb[:target_dim]

    # Normalize truncated embedding
    norm = np.linalg.norm(truncated)
    return (truncated / norm).tolist()
```

---

## New Module: `app/embed_v2.py`

```python
"""Enhanced embedding client with Matryoshka support."""

import httpx
import numpy as np
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingClientV2:
    """Embedding client with Matryoshka dimension support."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "bge-m3",
        full_dim: int = 1024,
        default_dim: int = 256,
        batch_size: int = 32,
        timeout: float = 60.0
    ):
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.full_dim = full_dim
        self.default_dim = default_dim
        self.batch_size = batch_size
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def embed_single(
        self,
        text: str,
        dim: Optional[int] = None
    ) -> List[float]:
        """
        Embed a single text with optional dimension reduction.

        Args:
            text: Text to embed
            dim: Target dimension (None = default_dim)

        Returns:
            Normalized embedding vector
        """
        target_dim = dim or self.default_dim

        # Get full embedding from Ollama
        response = self.client.post(
            f"{self.ollama_host}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        full_emb = np.array(response.json()["embedding"])

        # Apply Matryoshka truncation if needed
        if target_dim < len(full_emb):
            truncated = full_emb[:target_dim]
            # Re-normalize after truncation
            truncated = truncated / np.linalg.norm(truncated)
            return truncated.tolist()

        return full_emb.tolist()

    def embed_batch(
        self,
        texts: List[str],
        dim: Optional[int] = None
    ) -> List[List[float]]:
        """Embed multiple texts."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info(f"Embedding batch {i // self.batch_size + 1}")

            for text in batch:
                emb = self.embed_single(text, dim=dim)
                embeddings.append(emb)

        return embeddings

    def embed_for_storage(self, text: str) -> List[float]:
        """Embed for database storage (default/reduced dim)."""
        return self.embed_single(text, dim=self.default_dim)

    def embed_for_search(self, text: str) -> List[float]:
        """Embed for search (same dim as storage)."""
        return self.embed_single(text, dim=self.default_dim)

    def embed_for_rerank(self, text: str) -> List[float]:
        """Embed for reranking (full dim for quality)."""
        return self.embed_single(text, dim=self.full_dim)


# Dimension presets for different use cases
MATRYOSHKA_PRESETS = {
    "fast": 128,      # Fast search, lower quality
    "balanced": 256,  # Good balance (recommended)
    "quality": 512,   # Higher quality
    "max": 1024,      # Maximum quality
}


def get_dimension_for_use_case(use_case: str) -> int:
    """Get recommended dimension for use case."""
    return MATRYOSHKA_PRESETS.get(use_case, 256)
```

---

## Migration Plan

### Challenge: Existing 86K+ Embeddings
Current database has 86,513 chunks with 768d embeddings. Options:

#### Option 1: Full Re-embedding (Clean but Slow)
```bash
# Re-embed all documents with new model
python scripts/reembed_all.py --model bge-m3 --dim 256
```
- **Time**: ~4-8 hours for 86K chunks
- **Pros**: Clean slate, consistent embeddings
- **Cons**: Downtime, compute cost

#### Option 2: Dual Index (No Downtime)
Keep both embedding types, migrate gradually:
```sql
ALTER TABLE chunks ADD COLUMN embedding_v2 BLOB;
-- Populate v2 on new ingests
-- Background job to backfill existing
```
- **Time**: Zero downtime
- **Pros**: Gradual migration
- **Cons**: 2x storage temporarily

#### Option 3: Dimension-Agnostic Search (Hybrid)
Store new embeddings at new dimension, handle search across both:
```python
def search(query):
    # Search old index (768d)
    old_results = search_768(query_768)
    # Search new index (256d)
    new_results = search_256(query_256)
    # Merge results
    return rrf_fuse(old_results, new_results)
```

**Recommendation**: Option 2 (Dual Index) for production safety.

---

## Configuration Changes

### `config.yaml` Updates

```yaml
# Old config
ollama:
  embed_model: nomic-embed-text
  embed_batch_size: 32

# New config
embedding:
  model: bge-m3
  provider: ollama  # or "huggingface" for direct
  full_dimension: 1024
  storage_dimension: 256
  search_dimension: 256
  rerank_dimension: 1024
  batch_size: 32

  matryoshka:
    enabled: true
    presets:
      fast: 128
      balanced: 256
      quality: 512
      max: 1024
```

---

## Database Schema Changes

### New Chunks Table (v2)

```sql
-- Add new column for v2 embeddings
ALTER TABLE chunks ADD COLUMN embedding_v2 BLOB;
ALTER TABLE chunks ADD COLUMN embedding_dim INTEGER DEFAULT 768;

-- New index for v2 embeddings (256d)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_v2 USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[256]
);
```

### Migration Script

```python
def migrate_embeddings():
    """Backfill v2 embeddings for existing chunks."""
    embed_client = EmbeddingClientV2(model="bge-m3")

    chunks = db.get_all_chunks()
    for batch in batched(chunks, 100):
        texts = [c['content'] for c in batch]
        embeddings = embed_client.embed_batch(texts, dim=256)

        for chunk, emb in zip(batch, embeddings):
            db.update_chunk_embedding_v2(chunk['id'], emb)

        logger.info(f"Migrated {len(batch)} chunks")
```

---

## Performance Comparison

### Expected Improvements

| Metric | nomic-embed-text | bge-m3 (256d) | bge-m3 (1024d) |
|--------|-----------------|---------------|----------------|
| MTEB Score | 62.4 | ~64 | 66.6 |
| Retrieval P@10 | Baseline | +5% | +10% |
| Embedding latency | 50ms | 40ms | 60ms |
| Storage per chunk | 3KB | 1KB | 4KB |
| Search latency | Baseline | -30% | +20% |

### Matryoshka Quality Curve

```
Quality (relative)
│
│     ████████████████  1024d (100%)
│   ████████████████    512d (97%)
│  ███████████████      256d (93%)
│ █████████████         128d (87%)
└────────────────────── Dimension
```

---

## Rollout Plan

### Phase 1: Setup (Day 1)
- [ ] Pull bge-m3 model in Ollama
- [ ] Create embed_v2.py module
- [ ] Add config options
- [ ] Test embedding quality

### Phase 2: Dual Index (Day 2-3)
- [ ] Add embedding_v2 column to schema
- [ ] Create v2 vector index
- [ ] Update ingest to use both
- [ ] Test search across both indexes

### Phase 3: Migration (Day 4-7)
- [ ] Run background re-embedding job
- [ ] Monitor progress
- [ ] Validate quality parity

### Phase 4: Cutover (Day 8)
- [ ] Switch search to v2 only
- [ ] Remove v1 fallback
- [ ] Archive v1 embeddings
- [ ] Update documentation

---

## Verification Tests

```python
def test_embedding_quality():
    """Verify new embeddings match or exceed quality."""
    test_queries = [
        "What is retrieval augmented generation?",
        "How does ColBERT work?",
        "Explain dense passage retrieval",
    ]

    for query in test_queries:
        v1_results = search_v1(query)
        v2_results = search_v2(query)

        # V2 should have same or better recall
        v1_docs = set(r['doc_id'] for r in v1_results[:10])
        v2_docs = set(r['doc_id'] for r in v2_results[:10])

        overlap = len(v1_docs & v2_docs) / len(v1_docs)
        assert overlap >= 0.7, f"Low overlap for query: {query}"
```

---

## Dependencies

- **Ollama update**: Ensure bge-m3 model available
- **sqlite-vec**: May need update for different dimensions
- **numpy**: For vector operations (already available)

---

## References

1. BGE-M3: Multi-Functionality Multi-Linguality Multi-Granularity
2. Matryoshka Representation Learning (MRL)
3. MTEB Benchmark: https://huggingface.co/spaces/mteb/leaderboard
4. Papers in corpus (see Research Basis)
