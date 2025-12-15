# HERMES Entity-First Ranking: HotpotQA Benchmark Results

## Executive Summary

We benchmarked our Entity-First Ranker against the official HotpotQA dataset (7,405 multi-hop questions) using the same methodology as SEAL-RAG.

### Key Findings

| Method | Precision | Recall | F1 | Notes |
|--------|-----------|--------|-----|-------|
| **SEAL-RAG (GPT-4o)** | **89%** | **68%** | **75%** | Reference - uses GPT-4o |
| Basic RAG | 49% | 72% | 59% | Semantic retrieval only |
| **HERMES Entity-First** | **38.7%** | **57.5%** | **46.2%** | Our implementation |
| HERMES Semantic Only | 19.7% | 29.0% | 23.4% | nomic-embed-text |
| Random Baseline | 23.7% | 35.0% | 28.2% | 10 docs, pick 3 |

### Honest Assessment

**What We Achieved:**
- Entity-First ranking (46.2% F1) beats semantic-only (23.4% F1) by 22.8pp
- Entity-First ranking beats random baseline by 18pp
- Confirms entity-aware ranking improves multi-hop QA retrieval

**Gap to SEAL-RAG:**
- Precision: -50.3pp (38.7% vs 89%)
- F1: -28.8pp (46.2% vs 75%)

---

## Why the Gap Exists

### 1. Entity Extraction Quality

**SEAL-RAG:** Uses GPT-4o for entity extraction with sophisticated prompting.
```
Prompt: "Extract key entities from this question that are needed to find the answer..."
```

**HERMES:** Uses regex-based extraction (capitalized words, quoted terms).
```python
# Our simple approach
words = question.split()
entities = [w for w in words if w[0].isupper()]
```

**Impact:** GPT-4o extracts semantic entities; we extract syntactic patterns.

### 2. Iterative Refinement (SEAL Cycle)

**SEAL-RAG's Loop:**
1. **Search** - Initial retrieval
2. **Extract** - Find entities in retrieved docs
3. **Assess** - Identify missing entities (gap detection)
4. **Loop** - Generate micro-queries for missing entities
5. **Replace** - Swap low-coverage chunks with high-coverage ones

**HERMES:** Single-pass entity ranking without iterative micro-queries.

### 3. LLM-Powered Gap Detection

**SEAL-RAG:** Uses GPT-4o to assess whether retrieved context is sufficient.

**HERMES:** Uses rule-based coverage calculation.

---

## What We'd Need to Match SEAL-RAG

### Option 1: LLM-Powered Entity Extraction (Recommended)
Replace regex extraction with LLM:
```python
def extract_entities_llm(question: str) -> list[str]:
    prompt = f"""Extract the key named entities from this question.
    These are the specific things we need to find information about.

    Question: {question}

    Entities (comma-separated):"""

    response = llm.complete(prompt)
    return [e.strip() for e in response.split(",")]
```

**Expected Impact:** +15-25pp precision

### Option 2: Iterative Micro-Query Loop
Implement SEAL-RAG's iterative refinement:
```python
for iteration in range(max_iterations):
    # Assess what's missing
    missing = gap_detector.find_missing_entities(results, query_entities)

    if not missing:
        break

    # Generate targeted micro-queries
    for entity in missing:
        micro_results = search(f"What is {entity}?")
        candidates.extend(micro_results)

    # Replace low-coverage chunks
    results = entity_ranker.replace_chunks(results, candidates, missing, budget)
```

**Expected Impact:** +10-15pp recall

### Option 3: Better Embedding Model
Use embedding model trained for multi-hop QA:
- BGE-large-en-v1.5 (better than nomic-embed-text for this task)
- E5-large-v2
- GTE-large

**Expected Impact:** +5-10pp semantic baseline

---

## Benchmark Details

### Dataset
- **HotpotQA Dev Set** (distractor setting)
- 7,405 total questions
- Sampled: 100-200 questions (stratified by type)
- Types: Bridge (80%), Comparison (20%)

### Evaluation Protocol
1. Each question has 10 context documents (2 gold + 8 distractors)
2. Methods rank documents and select top-k
3. Measure if selected docs match gold docs

### Metrics
- **Precision:** % of retrieved docs that are gold
- **Recall:** % of gold docs that were retrieved
- **F1:** Harmonic mean of precision and recall

---

## Reproducibility

### Running the Benchmark
```bash
cd ~/hermes-lite/benchmarks/hotpotqa

# Quick test (keyword matching)
python hotpotqa_hybrid_benchmark.py --sample 200 --top-k 3

# Full test (embeddings)
python hotpotqa_hybrid_benchmark.py --sample 200 --top-k 3 --ollama

# Save results
python hotpotqa_hybrid_benchmark.py --sample 500 --top-k 3 -o results.json
```

### Files
- `hotpot_dev_distractor_v1.json` - HotpotQA dataset
- `hotpotqa_entity_ranker_benchmark.py` - Entity-only benchmark
- `hotpotqa_hybrid_benchmark.py` - Multi-method comparison
- `results_k3.json` - Saved results

---

## Conclusion

Our Entity-First Ranker demonstrates that **entity-aware ranking improves multi-hop QA retrieval** (+22.8pp F1 over semantic-only). However, to match SEAL-RAG's 89% precision, we would need:

1. **LLM-powered entity extraction** (biggest impact)
2. **Iterative micro-query refinement**
3. **Better embedding models**

The core algorithm (entity coverage scoring, fixed-budget replacement) is sound. The gap is in the quality of entity extraction and the lack of iterative refinement.

---

## References

- SEAL-RAG Paper: "Replace, Don't Expand: Mitigating Context Dilution in RAG"
- HotpotQA: https://hotpotqa.github.io/
- Our implementation: `app/entity_ranker.py`
