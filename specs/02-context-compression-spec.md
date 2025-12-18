# Context Compression Module

## Overview

**Context Compression** reduces the number of tokens sent to the LLM while preserving semantic meaning. Research shows 5-10x compression is achievable with minimal quality loss.

### Research Basis
Papers in corpus:
- `2501.01625v1_ICPC In-context Prompt Compression with Faster Inf.pdf`
- `2505.00019v1_An Empirical Study on Prompt Compression for La.pdf`
- `2409.15395v3_Parse Trees Guided LLM Prompt Compression.pdf`
- `2501.12959v2_Efficient Prompt Compression with Evaluator Hea.pdf`
- `2502.13374v1_Task-agnostic Prompt Compression with Context-awar.pdf`
- `2503.07956v1_EFPC Towards Efficient and Flexible Prompt Compres.pdf`

### Why Compress?
1. **Cost**: Fewer tokens = lower API costs
2. **Latency**: Less to process = faster responses
3. **Context window**: Fit more relevant content in limited window
4. **Quality**: Remove noise, keep signal

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Retrieved Chunks (10-20)                   │
│  [chunk1: 512 tokens] [chunk2: 512 tokens] ... [total: 8K]   │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                   Context Compression Pipeline                │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Sentence   │─▶│  Relevance  │─▶│   Token     │          │
│  │  Scoring    │  │  Filtering  │  │  Pruning    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                 Compressed Context (~1-2K tokens)             │
│           [Most relevant sentences, key information]          │
└──────────────────────────────────────────────────────────────┘
```

---

## Compression Strategies

### Strategy 1: Sentence-Level Scoring (Fast)
Score each sentence by relevance to query, keep top-k.

```python
def score_sentences(query: str, sentences: List[str]) -> List[float]:
    query_emb = embed(query)
    return [cosine_sim(query_emb, embed(s)) for s in sentences]
```

**Pros**: Fast, simple
**Cons**: May miss cross-sentence context

### Strategy 2: Token-Level Importance (LLMLingua-style)
Use small model perplexity to identify important tokens.

```python
def get_token_importance(text: str, model) -> List[float]:
    # Tokens with HIGH perplexity are more informative
    # Tokens with LOW perplexity are predictable/redundant
    return model.get_perplexity_per_token(text)
```

**Pros**: Fine-grained, preserves key terms
**Cons**: Requires model inference

### Strategy 3: Extractive Summary (Best Quality)
Extract key sentences using TextRank or similar.

**Pros**: Coherent output
**Cons**: Slower, may miss details

---

## Implementation Spec

### New File: `app/compression.py`

```python
"""Context compression for RAG pipelines."""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    strategy: str = "sentence"  # sentence, token, extractive
    target_ratio: float = 0.3   # Keep 30% of original tokens
    min_sentences: int = 3      # Minimum sentences to keep
    max_tokens: int = 2000      # Hard limit
    preserve_structure: bool = True  # Keep document boundaries


@dataclass
class CompressedContext:
    """Result of compression."""
    text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    kept_sentences: int
    total_sentences: int


class ContextCompressor:
    """Compress retrieved context before sending to LLM."""

    def __init__(
        self,
        embed_fn,
        config: Optional[CompressionConfig] = None
    ):
        self.embed_fn = embed_fn
        self.config = config or CompressionConfig()

    def compress(
        self,
        query: str,
        chunks: List[Dict],
        target_tokens: Optional[int] = None
    ) -> CompressedContext:
        """
        Compress retrieved chunks into condensed context.

        Args:
            query: The user query
            chunks: List of retrieved chunks with 'content' field
            target_tokens: Override default max tokens

        Returns:
            CompressedContext with compressed text and stats
        """
        target = target_tokens or self.config.max_tokens

        # Combine chunks
        full_text = "\n\n".join(c.get("content", "") for c in chunks)
        original_tokens = self._count_tokens(full_text)

        if original_tokens <= target:
            # No compression needed
            return CompressedContext(
                text=full_text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                kept_sentences=len(self._split_sentences(full_text)),
                total_sentences=len(self._split_sentences(full_text))
            )

        # Apply compression strategy
        if self.config.strategy == "sentence":
            compressed = self._sentence_compression(query, full_text, target)
        elif self.config.strategy == "token":
            compressed = self._token_compression(query, full_text, target)
        else:
            compressed = self._extractive_compression(query, full_text, target)

        compressed_tokens = self._count_tokens(compressed)

        return CompressedContext(
            text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens,
            kept_sentences=len(self._split_sentences(compressed)),
            total_sentences=len(self._split_sentences(full_text))
        )

    def _sentence_compression(
        self,
        query: str,
        text: str,
        target_tokens: int
    ) -> str:
        """Compress by selecting most relevant sentences."""
        sentences = self._split_sentences(text)
        if len(sentences) <= self.config.min_sentences:
            return text

        # Score sentences by relevance to query
        query_emb = np.array(self.embed_fn(query))
        scored = []

        for i, sent in enumerate(sentences):
            if len(sent.strip()) < 10:
                continue
            sent_emb = np.array(self.embed_fn(sent))
            score = self._cosine_sim(query_emb, sent_emb)
            scored.append((i, sent, score))

        # Sort by score
        scored.sort(key=lambda x: x[2], reverse=True)

        # Select top sentences until target reached
        selected = []
        current_tokens = 0

        for idx, sent, score in scored:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > target_tokens:
                if len(selected) >= self.config.min_sentences:
                    break
            selected.append((idx, sent))
            current_tokens += sent_tokens

        # Sort by original order to maintain coherence
        selected.sort(key=lambda x: x[0])

        return " ".join(sent for _, sent in selected)

    def _token_compression(
        self,
        query: str,
        text: str,
        target_tokens: int
    ) -> str:
        """
        Compress by removing low-importance tokens.
        Simplified version - removes filler words and redundant phrases.
        """
        # Remove common filler patterns
        patterns = [
            r'\b(basically|essentially|actually|really|very|quite)\b',
            r'\b(in order to)\b',
            r'\b(it is worth noting that|it should be noted that)\b',
            r'\b(as mentioned (earlier|above|before))\b',
            r'\b(the fact that)\b',
        ]

        compressed = text
        for pattern in patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)

        # Remove extra whitespace
        compressed = re.sub(r'\s+', ' ', compressed).strip()

        # If still too long, fall back to sentence compression
        if self._count_tokens(compressed) > target_tokens:
            return self._sentence_compression(query, compressed, target_tokens)

        return compressed

    def _extractive_compression(
        self,
        query: str,
        text: str,
        target_tokens: int
    ) -> str:
        """Extract key sentences using TextRank-style scoring."""
        sentences = self._split_sentences(text)
        if len(sentences) <= self.config.min_sentences:
            return text

        # Build similarity matrix
        embeddings = [np.array(self.embed_fn(s)) for s in sentences if len(s) > 10]
        n = len(embeddings)

        if n == 0:
            return text

        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = self._cosine_sim(embeddings[i], embeddings[j])

        # TextRank scoring
        scores = np.ones(n)
        damping = 0.85

        for _ in range(10):  # Iterate until convergence
            new_scores = (1 - damping) + damping * sim_matrix.T @ scores
            new_scores = new_scores / np.linalg.norm(new_scores)
            if np.allclose(scores, new_scores):
                break
            scores = new_scores

        # Also factor in query relevance
        query_emb = np.array(self.embed_fn(query))
        query_scores = [self._cosine_sim(query_emb, e) for e in embeddings]

        # Combined score: 50% TextRank + 50% query relevance
        combined = 0.5 * scores + 0.5 * np.array(query_scores)

        # Select top sentences
        indices = np.argsort(combined)[::-1]
        selected = []
        current_tokens = 0

        valid_sentences = [s for s in sentences if len(s) > 10]
        for idx in indices:
            sent = valid_sentences[idx]
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > target_tokens:
                if len(selected) >= self.config.min_sentences:
                    break
            selected.append((idx, sent))
            current_tokens += sent_tokens

        # Restore original order
        selected.sort(key=lambda x: x[0])

        return " ".join(sent for _, sent in selected)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
```

---

## Integration Points

### 1. Add to Search Pipeline

```python
# In search.py or search_v2.py

class SearchEngine:
    def __init__(self, ...):
        self.compressor = ContextCompressor(
            embed_fn=self.embed_client.embed_single,
            config=CompressionConfig(
                strategy="sentence",
                target_ratio=0.3,
                max_tokens=2000
            )
        )

    def search_with_compression(
        self,
        query: str,
        top_k: int = 10,
        compress: bool = True
    ) -> dict:
        # Get raw results
        results = self.search(query, top_k=top_k)

        if compress:
            compressed = self.compressor.compress(query, results)
            return {
                "results": results,
                "compressed_context": compressed.text,
                "compression_stats": {
                    "original_tokens": compressed.original_tokens,
                    "compressed_tokens": compressed.compressed_tokens,
                    "ratio": compressed.compression_ratio
                }
            }
        return {"results": results}
```

### 2. Add to Config

```yaml
compression:
  enabled: true
  strategy: "sentence"  # sentence, token, extractive
  target_ratio: 0.3
  max_tokens: 2000
  min_sentences: 3
```

### 3. API Enhancement

```python
@app.get("/search")
async def search(
    query: str,
    compress: bool = True,
    max_context_tokens: int = 2000,
    ...
):
    """Search with optional context compression."""
```

---

## Performance Characteristics

### Compression Ratios by Strategy

| Strategy | Compression | Quality | Latency |
|----------|-------------|---------|---------|
| Sentence | 3-5x | Good | +50ms |
| Token | 2-3x | Better | +100ms |
| Extractive | 4-6x | Best | +200ms |

### Benchmarks (Expected)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Avg tokens | 6000 | 1500 | -75% |
| LLM latency | 3s | 1s | -66% |
| Quality (F1) | 0.85 | 0.82 | -3% |

---

## Advanced Features (Future)

### 1. Query-Aware Compression
Different compression levels based on query complexity:
- Simple factual: Heavy compression (5x)
- Complex reasoning: Light compression (2x)

### 2. Hierarchical Compression
Keep full detail for top-3 chunks, compress rest:
```python
def hierarchical_compress(chunks):
    top_chunks = chunks[:3]  # Full detail
    rest = compress(chunks[3:])  # Heavy compression
    return top_chunks + rest
```

### 3. Iterative Refinement
If LLM response quality is low, decompress and retry:
```python
def iterative_search(query):
    compressed = compress(chunks, ratio=0.3)
    response = llm(compressed)
    if quality_score(response) < threshold:
        compressed = compress(chunks, ratio=0.5)  # Less compression
        response = llm(compressed)
    return response
```

---

## Evaluation Plan

### Metrics
1. **Compression ratio**: tokens_after / tokens_before
2. **Information retention**: % of key facts preserved
3. **Answer quality**: F1/EM on QA benchmark
4. **Latency reduction**: Time saved on LLM call

### Test Methodology
1. Select 100 queries from corpus
2. Run with/without compression
3. Compare answer quality and latency
4. Tune parameters based on results

---

## Dependencies

- numpy (vector operations)
- Embedding function (already available)
- No additional models required for sentence strategy

---

## Rollout Plan

### Phase 1: Basic Implementation
- [ ] Create `compression.py` module
- [ ] Implement sentence-level compression
- [ ] Add to search pipeline

### Phase 2: Testing
- [ ] Unit tests for compressor
- [ ] Integration tests
- [ ] Quality benchmarks

### Phase 3: Advanced Strategies
- [ ] Implement token-level compression
- [ ] Implement extractive compression
- [ ] Add adaptive compression

### Phase 4: Production
- [ ] Feature flag for gradual rollout
- [ ] Monitoring for quality degradation
- [ ] A/B testing

---

## References

1. LLMLingua: Compressing Prompts for Accelerated Inference
2. Selective Context for Efficient Inference
3. ICPC: In-context Prompt Compression
4. Papers in corpus (see Research Basis section)
