# HyDE: Hypothetical Document Embeddings

## Overview

**HyDE (Hypothetical Document Embeddings)** is a query expansion technique that generates a hypothetical answer to a query, embeds that answer, and uses it for retrieval instead of (or in addition to) the original query embedding.

### Research Basis
- Original Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022) - arXiv:2212.10496
- Found in corpus: `2212.10496v1_Precise Zero-Shot Dense Retrieval without Relevanc.pdf`

### Why It Works
The query "What causes climate change?" is semantically distant from documents about greenhouse gases. But a hypothetical answer like "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels..." is much closer in embedding space to relevant documents.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  LLM (small)│────▶│ Hypothetical│
│             │     │  Generate   │     │   Answer    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Retrieved  │◀────│   Vector    │◀────│   Embed     │
│  Documents  │     │   Search    │     │   Answer    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Variants

1. **Pure HyDE**: Only use hypothetical embedding
2. **Hybrid HyDE**: Combine original query + hypothetical embeddings (recommended)
3. **Multi-HyDE**: Generate multiple hypotheticals, average embeddings

---

## Implementation Spec

### New File: `app/hyde.py`

```python
"""HyDE: Hypothetical Document Embeddings for query expansion."""

import logging
from typing import Optional, List, Callable
import numpy as np

logger = logging.getLogger(__name__)


class HyDEExpander:
    """Generate hypothetical documents for improved retrieval."""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[str], List[float]],
        num_hypotheticals: int = 1,
        blend_weight: float = 0.7,  # Weight for hypothetical vs original
    ):
        """
        Initialize HyDE expander.

        Args:
            llm_fn: Function that takes prompt, returns generated text
            embed_fn: Function that takes text, returns embedding vector
            num_hypotheticals: Number of hypothetical docs to generate
            blend_weight: Weight for hypothetical embedding (0-1)
        """
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.num_hypotheticals = num_hypotheticals
        self.blend_weight = blend_weight

    def generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Write a short, factual passage (2-3 sentences) that directly answers this question:

Question: {query}

Passage:"""

        return self.llm_fn(prompt)

    def expand_query(
        self,
        query: str,
        return_hypothetical: bool = False
    ) -> List[float]:
        """
        Expand query using HyDE.

        Args:
            query: Original user query
            return_hypothetical: If True, also return generated text

        Returns:
            Blended embedding vector (and optionally the hypothetical text)
        """
        # Generate hypothetical document(s)
        hypotheticals = []
        for _ in range(self.num_hypotheticals):
            hypo = self.generate_hypothetical(query)
            hypotheticals.append(hypo)
            logger.debug(f"Generated hypothetical: {hypo[:100]}...")

        # Embed original query
        query_embedding = np.array(self.embed_fn(query))

        # Embed hypotheticals and average
        hypo_embeddings = [
            np.array(self.embed_fn(h)) for h in hypotheticals
        ]
        avg_hypo_embedding = np.mean(hypo_embeddings, axis=0)

        # Blend embeddings
        blended = (
            self.blend_weight * avg_hypo_embedding +
            (1 - self.blend_weight) * query_embedding
        )

        # Normalize
        blended = blended / np.linalg.norm(blended)

        if return_hypothetical:
            return blended.tolist(), hypotheticals[0]
        return blended.tolist()


# Prompt templates for different query types
HYDE_PROMPTS = {
    "factual": """Write a short, factual passage (2-3 sentences) that directly answers:
Question: {query}
Passage:""",

    "technical": """Write a technical explanation (2-3 sentences) that answers:
Question: {query}
Technical answer:""",

    "comparison": """Write a balanced comparison (2-3 sentences) addressing:
Question: {query}
Comparison:""",

    "definition": """Write a clear definition (1-2 sentences) for:
Question: {query}
Definition:""",
}


def get_hyde_prompt(query: str, query_type: str = "factual") -> str:
    """Get appropriate HyDE prompt based on query type."""
    template = HYDE_PROMPTS.get(query_type, HYDE_PROMPTS["factual"])
    return template.format(query=query)
```

---

## Integration Points

### 1. Add to `search.py`

```python
from .hyde import HyDEExpander

class SearchEngine:
    def __init__(self, ..., use_hyde: bool = True):
        self.use_hyde = use_hyde
        if use_hyde:
            self.hyde = HyDEExpander(
                llm_fn=self._call_llm,
                embed_fn=self.embed_client.embed_single,
                blend_weight=0.7
            )

    def search(self, query: str, ...):
        if self.use_hyde:
            query_embedding = self.hyde.expand_query(query)
        else:
            query_embedding = self.embed_client.embed_single(query)
        # ... rest of search
```

### 2. Add to `config.yaml`

```yaml
hyde:
  enabled: true
  num_hypotheticals: 1
  blend_weight: 0.7
  llm_model: "llama3.2:3b"  # Small, fast model
  prompt_type: "auto"  # auto-detect from query analysis
```

### 3. Add to API (optional)

```python
@app.get("/search")
async def search(
    query: str,
    use_hyde: bool = True,  # New parameter
    ...
):
```

---

## Performance Considerations

### Latency Impact
- **Without HyDE**: ~100ms (embedding only)
- **With HyDE**: ~500-1000ms (LLM generation + embedding)

### Mitigation Strategies
1. **Use small LLM**: llama3.2:1b or 3b, not 70b
2. **Cache hypotheticals**: Hash query → cached hypothetical
3. **Async generation**: Generate while user types (speculative)
4. **Skip for simple queries**: Only use HyDE for complex queries

### Quality vs Speed Matrix

| Mode | Latency | Quality | Use Case |
|------|---------|---------|----------|
| No HyDE | ~100ms | Baseline | Simple lookups |
| HyDE (1 hypo) | ~500ms | +15% recall | Default |
| HyDE (3 hypo) | ~1.5s | +20% recall | Complex queries |

---

## Evaluation Plan

### Metrics
1. **Recall@10**: % of relevant docs in top 10
2. **MRR**: Mean Reciprocal Rank
3. **Latency P50/P95**: Response time distribution

### Test Cases
1. Factual questions: "What is ColBERT?"
2. Complex questions: "How does HyDE compare to query expansion?"
3. Multi-hop: "What papers cite the original RAG paper?"

### A/B Test Design
```python
# 50/50 split
if hash(query) % 2 == 0:
    use_hyde = True
else:
    use_hyde = False
```

---

## Dependencies

- **LLM Backend**: Ollama (already configured)
- **Model**: llama3.2:3b or similar small model
- **Numpy**: For vector operations

---

## Rollout Plan

### Phase 1: Implementation (1-2 days)
- [ ] Create `hyde.py` module
- [ ] Add integration to search pipeline
- [ ] Add config options

### Phase 2: Testing (1 day)
- [ ] Unit tests for HyDEExpander
- [ ] Integration tests with search
- [ ] Benchmark latency impact

### Phase 3: Evaluation (2-3 days)
- [ ] Run on test query set
- [ ] Compare recall/MRR with baseline
- [ ] Tune blend_weight parameter

### Phase 4: Production (1 day)
- [ ] Add feature flag
- [ ] Deploy with monitoring
- [ ] Document usage

---

## References

1. Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
2. Papers in corpus: 2212.10496v1
3. HyDE implementations: LangChain, LlamaIndex
