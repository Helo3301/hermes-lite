# Hermes v2 Benchmark Specification

## Overview

Dual evaluation strategy:
1. **Standard Benchmarks** — Validate methods against published baselines
2. **Hermes-Specific Tests** — Validate on our actual paper corpus

---

## Part 1: Standard Benchmark Integration

### 1.1 Dataset Selection

| Dataset | Why | Size | Source |
|---------|-----|------|--------|
| **HotpotQA** | Multi-hop, most cited | 500 samples | [HuggingFace](https://huggingface.co/datasets/hotpot_qa) |
| **2WikiMultiHopQA** | Multi-hop, comparison focus | 500 samples | [GitHub](https://github.com/Alab-NII/2wikimultihop) |
| **NQ (Natural Questions)** | Simple factoid baseline | 500 samples | [HuggingFace](https://huggingface.co/datasets/natural_questions) |

**Total**: 1500 test queries with gold answers

### 1.2 Corpus Setup

For standard benchmarks, we need their associated corpus:

```python
# Download Wikipedia corpus used by these benchmarks
# KILT Wikipedia snapshot (used by HotpotQA, 2Wiki, NQ)
KILT_CORPUS_URL = "https://huggingface.co/datasets/facebook/kilt_wikipedia"

# Ingest into separate Hermes collection
hermes.create_collection("benchmark-corpus", "KILT Wikipedia for benchmark evaluation")
```

### 1.3 Evaluation Protocol

```python
class StandardBenchmarkEvaluator:
    """Run standard RAG benchmarks against Hermes."""

    def __init__(self, hermes_url: str, corpus_collection: str):
        self.hermes = HermesClient(hermes_url)
        self.corpus = corpus_collection

    def evaluate_dataset(
        self,
        dataset_name: str,
        samples: list[dict],  # [{question, answer, supporting_facts}, ...]
        search_version: str = "v1"  # or "v2"
    ) -> BenchmarkResult:

        results = []
        for sample in samples:
            # Retrieve
            if search_version == "v1":
                chunks = self.hermes.search(sample["question"], top_k=10)
            else:
                chunks = self.hermes.search_v2(sample["question"], mode="auto")

            # Generate answer (using Ollama)
            answer = self.generate_answer(sample["question"], chunks)

            # Score
            results.append({
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "predicted_answer": answer,
                "retrieved_chunks": chunks,
                "supporting_facts": sample.get("supporting_facts", []),
            })

        return self.compute_metrics(results)

    def compute_metrics(self, results: list[dict]) -> BenchmarkResult:
        return BenchmarkResult(
            exact_match=self._compute_em(results),
            f1_score=self._compute_f1(results),
            retrieval_recall=self._compute_retrieval_recall(results),
            retrieval_precision=self._compute_retrieval_precision(results),
        )
```

### 1.4 Expected Baselines (from literature)

| Method | HotpotQA EM | 2Wiki EM | NQ EM |
|--------|-------------|----------|-------|
| Basic RAG | 0.28 | 0.25 | 0.33 |
| Iterative RAG | 0.35 | 0.35 | 0.35 |
| SEAL-RAG | 0.44 | 0.52 | — |
| **Hermes v1 (target)** | ≥0.28 | ≥0.25 | ≥0.33 |
| **Hermes v2 (target)** | ≥0.40 | ≥0.40 | ≥0.35 |

### 1.5 Pass Criteria for Standard Benchmarks

```python
STANDARD_BENCHMARK_CRITERIA = {
    "hotpotqa": {
        "v1": {"em": 0.25, "f1": 0.35},  # At least basic RAG level
        "v2": {"em": 0.40, "f1": 0.50},  # Significant improvement
    },
    "2wikimultihop": {
        "v1": {"em": 0.22, "f1": 0.30},
        "v2": {"em": 0.38, "f1": 0.45},
    },
    "nq": {
        "v1": {"em": 0.30, "f1": 0.40},
        "v2": {"em": 0.33, "f1": 0.43},  # Smaller gain expected (simple queries)
    },
}
```

---

## Part 2: Hermes-Specific Benchmark

### 2.1 Query Categories

Design queries that test what Hermes is actually for:

#### Category A: Paper Lookup (Simple)
```python
PAPER_LOOKUP_QUERIES = [
    {
        "question": "What is the main contribution of the SEAL-RAG paper?",
        "answer_must_contain": ["replace", "context dilution", "fixed budget"],
        "gold_doc": "2512.10787v1.pdf",
        "difficulty": "simple",
    },
    {
        "question": "What datasets does SEAL-RAG evaluate on?",
        "answer_must_contain": ["2WikiMultiHopQA", "HotpotQA", "MuSiQue"],
        "gold_doc": "2512.10787v1.pdf",
        "difficulty": "simple",
    },
    # ... 15 more
]
```

#### Category B: Cross-Paper Reasoning (Multi-hop)
```python
CROSS_PAPER_QUERIES = [
    {
        "question": "How does SEAL-RAG's approach to context dilution differ from CRAG's corrective retrieval?",
        "answer_must_contain": ["replace vs expand", "fixed budget", "corrective"],
        "gold_docs": ["2512.10787v1.pdf", "crag_paper.pdf"],
        "difficulty": "multi_hop",
    },
    {
        "question": "Which multi-hop QA benchmarks are used by both SEAL-RAG and NeuroPath?",
        "answer_must_contain": ["HotpotQA", "2WikiMultiHopQA"],
        "gold_docs": ["2512.10787v1.pdf", "2511.14096v1.pdf"],
        "difficulty": "multi_hop",
    },
    # ... 15 more
]
```

#### Category C: Comparative Analysis
```python
COMPARATIVE_QUERIES = [
    {
        "question": "Compare the performance of graph-based RAG methods vs vector-only RAG on multi-hop questions",
        "answer_must_mention": ["graph RAG", "vector", "multi-hop", "improvement"],
        "requires_synthesis": True,
        "difficulty": "comparative",
    },
    {
        "question": "What are the tradeoffs between iterative retrieval and single-shot retrieval?",
        "answer_must_mention": ["latency", "accuracy", "iterations"],
        "requires_synthesis": True,
        "difficulty": "comparative",
    },
    # ... 10 more
]
```

#### Category D: Exploratory/Survey
```python
EXPLORATORY_QUERIES = [
    {
        "question": "What are the main approaches to reducing hallucination in RAG systems?",
        "answer_must_mention": ["grounding", "verification", "retrieval quality"],
        "min_sources": 3,
        "difficulty": "exploratory",
    },
    {
        "question": "How has multi-hop RAG research evolved in 2025?",
        "answer_must_mention": ["iterative", "graph", "reasoning"],
        "min_sources": 3,
        "difficulty": "exploratory",
    },
    # ... 10 more
]
```

### 2.2 Ground Truth Construction

For each Hermes-specific query, we need:

```python
@dataclass
class HermesTestCase:
    question: str
    difficulty: str  # simple, multi_hop, comparative, exploratory

    # Answer validation
    answer_must_contain: list[str]  # Required keywords/phrases
    answer_must_not_contain: list[str]  # Hallucination indicators

    # Retrieval validation
    gold_chunks: list[int]  # Chunk IDs that MUST be retrieved
    gold_docs: list[str]  # Document filenames that should appear
    min_relevant_chunks: int  # Minimum relevant chunks needed

    # Timing expectations
    max_latency_ms: int  # Based on difficulty tier
    max_iterations: int  # For v2 iterative retrieval

    # Metadata
    created_by: str  # "human" or "synthetic"
    validated: bool  # Has a human verified this test case?
```

### 2.3 Hermes-Specific Metrics

Beyond standard EM/F1, we measure:

```python
class HermesMetrics:
    """Metrics specific to Hermes use cases."""

    def entity_coverage(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[dict]
    ) -> float:
        """What % of question entities appear in retrieval + answer?"""
        question_entities = extract_entities(question)
        found_entities = extract_entities(answer + " ".join(c["content"] for c in retrieved_chunks))
        return len(question_entities & found_entities) / len(question_entities)

    def source_diversity(self, retrieved_chunks: list[dict]) -> float:
        """How many unique papers are represented?"""
        unique_docs = set(c["doc_id"] for c in retrieved_chunks)
        return len(unique_docs) / len(retrieved_chunks)

    def citation_accuracy(
        self,
        answer: str,
        retrieved_chunks: list[dict]
    ) -> float:
        """Are claims in answer actually supported by retrieved chunks?"""
        claims = extract_claims(answer)
        supported = sum(1 for c in claims if self.claim_supported(c, retrieved_chunks))
        return supported / len(claims) if claims else 1.0

    def contradiction_detection_rate(
        self,
        retrieved_chunks: list[dict],
        known_contradictions: list[tuple]
    ) -> float:
        """Did we detect known contradictions in our corpus?"""
        detected = self.detect_contradictions(retrieved_chunks)
        return len(detected & known_contradictions) / len(known_contradictions)
```

### 2.4 Hermes Benchmark Test Sets

| Category | Count | Latency Target | v1 Baseline | v2 Target |
|----------|-------|----------------|-------------|-----------|
| Paper Lookup | 20 | <500ms | 80% entity coverage | 85% |
| Cross-Paper | 20 | <10s | 50% entity coverage | 70% |
| Comparative | 15 | <10s | 40% (subjective) | 70% |
| Exploratory | 10 | <5min | 60% source diversity | 75% |

**Total**: 65 Hermes-specific test queries

### 2.5 Automated Test Generation

To scale beyond manual queries, generate synthetic tests:

```python
class HermesTestGenerator:
    """Generate test queries from known paper content."""

    def generate_lookup_query(self, doc: dict) -> HermesTestCase:
        """Generate simple lookup query from paper abstract."""
        # Extract key claim from abstract
        claim = self.extract_main_claim(doc["clean_md"])

        # Generate question that would retrieve this
        question = self.llm.generate(f"""
            Given this claim from a paper: "{claim}"
            Generate a natural question that would lead to this answer.
            Question:
        """)

        return HermesTestCase(
            question=question,
            difficulty="simple",
            answer_must_contain=self.extract_keywords(claim),
            gold_docs=[doc["filename"]],
            max_latency_ms=500,
        )

    def generate_cross_paper_query(
        self,
        doc_a: dict,
        doc_b: dict,
        relationship: str  # "cites", "contradicts", "extends"
    ) -> HermesTestCase:
        """Generate multi-hop query requiring both papers."""
        # ... implementation
```

---

## Part 3: Evaluation Pipeline

### 3.1 Directory Structure

```
hermes-lite/
├── benchmarks/
│   ├── standard/
│   │   ├── hotpotqa_500.json
│   │   ├── 2wikimultihop_500.json
│   │   └── nq_500.json
│   ├── hermes/
│   │   ├── paper_lookup.json
│   │   ├── cross_paper.json
│   │   ├── comparative.json
│   │   └── exploratory.json
│   └── corpus/
│       └── kilt_wikipedia/  # For standard benchmarks
├── evaluation/
│   ├── run_benchmarks.py
│   ├── metrics.py
│   ├── report.py
│   └── compare_versions.py
└── results/
    ├── v1_baseline/
    └── v2_results/
```

### 3.2 Benchmark Runner

```python
# evaluation/run_benchmarks.py

class BenchmarkRunner:
    def __init__(self, hermes_url: str):
        self.hermes = HermesClient(hermes_url)
        self.results_dir = Path("results")

    def run_all(self, version: str = "v1") -> dict:
        """Run complete benchmark suite."""
        results = {}

        # Standard benchmarks
        for dataset in ["hotpotqa", "2wikimultihop", "nq"]:
            print(f"Running {dataset}...")
            results[f"standard_{dataset}"] = self.run_standard_benchmark(
                dataset, version
            )

        # Hermes-specific benchmarks
        for category in ["paper_lookup", "cross_paper", "comparative", "exploratory"]:
            print(f"Running hermes_{category}...")
            results[f"hermes_{category}"] = self.run_hermes_benchmark(
                category, version
            )

        # Save results
        self.save_results(results, version)

        return results

    def compare_versions(self, v1_results: dict, v2_results: dict) -> ComparisonReport:
        """Generate comparison report between versions."""
        return ComparisonReport(
            improvements=self._find_improvements(v1_results, v2_results),
            regressions=self._find_regressions(v1_results, v2_results),
            summary=self._generate_summary(v1_results, v2_results),
        )
```

### 3.3 Report Format

```python
# Example output from benchmark run

BENCHMARK_REPORT = """
# Hermes v2 Benchmark Results
Generated: 2025-12-13 14:30:00

## Standard Benchmarks

| Dataset | Version | EM | F1 | Retrieval Recall | Latency p95 |
|---------|---------|-----|-----|------------------|-------------|
| HotpotQA | v1 | 0.28 | 0.38 | 0.65 | 480ms |
| HotpotQA | v2 | 0.42 | 0.53 | 0.78 | 3200ms |
| 2WikiMultiHop | v1 | 0.24 | 0.33 | 0.58 | 510ms |
| 2WikiMultiHop | v2 | 0.41 | 0.49 | 0.75 | 4100ms |
| NQ | v1 | 0.32 | 0.42 | 0.72 | 450ms |
| NQ | v2 | 0.35 | 0.45 | 0.74 | 520ms |

## Hermes-Specific Benchmarks

| Category | Version | Entity Coverage | Source Diversity | Citation Accuracy | Latency p95 |
|----------|---------|-----------------|------------------|-------------------|-------------|
| Paper Lookup | v1 | 78% | 0.42 | 0.85 | 420ms |
| Paper Lookup | v2 | 86% | 0.48 | 0.91 | 480ms |
| Cross-Paper | v1 | 52% | 0.35 | 0.72 | 890ms |
| Cross-Paper | v2 | 71% | 0.58 | 0.84 | 4200ms |
| Comparative | v1 | 45% | 0.28 | 0.68 | 920ms |
| Comparative | v2 | 69% | 0.62 | 0.81 | 6100ms |
| Exploratory | v1 | 58% | 0.52 | 0.75 | 2100ms |
| Exploratory | v2 | 74% | 0.71 | 0.83 | 45000ms |

## Summary

### Improvements (v1 → v2)
- Multi-hop EM: +50% (0.28 → 0.42 on HotpotQA)
- Cross-paper entity coverage: +37% (52% → 71%)
- Source diversity on comparative: +121% (0.28 → 0.62)

### Tradeoffs
- Simple query latency: +15% (acceptable)
- Multi-hop latency: +600% (within 10s budget)
- Exploratory latency: +2000% (within 5min budget)

### Pass/Fail Status
✅ Standard benchmarks: PASS (all targets met)
✅ Hermes paper lookup: PASS
✅ Hermes cross-paper: PASS
✅ Hermes comparative: PASS
✅ Hermes exploratory: PASS

OVERALL: PASS
"""
```

---

## Part 4: Implementation Checklist

### Phase 0: Benchmark Setup (Before v2 development)

- [ ] Download HotpotQA sample (500 questions)
- [ ] Download 2WikiMultiHopQA sample (500 questions)
- [ ] Download NQ sample (500 questions)
- [ ] Download KILT Wikipedia corpus
- [ ] Ingest corpus into Hermes `benchmark-corpus` collection
- [ ] Create 65 Hermes-specific test cases:
  - [ ] 20 paper lookup queries
  - [ ] 20 cross-paper queries
  - [ ] 15 comparative queries
  - [ ] 10 exploratory queries
- [ ] Implement benchmark runner
- [ ] Run v1 baseline on all benchmarks
- [ ] Save v1 results

### Phase 6: Final Evaluation (After v2 development)

- [ ] Run v2 on all benchmarks
- [ ] Generate comparison report
- [ ] Verify all pass criteria met
- [ ] Document any regressions
- [ ] Performance profiling

---

## Part 5: RAGAS Integration

Use RAGAS framework for automated quality scoring:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str]
) -> dict:
    """Run RAGAS evaluation suite."""

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return result.to_pandas().mean().to_dict()
```

### RAGAS Pass Criteria

| Metric | v1 Target | v2 Target |
|--------|-----------|-----------|
| Faithfulness | ≥0.70 | ≥0.80 |
| Answer Relevancy | ≥0.70 | ≥0.80 |
| Context Precision | ≥0.60 | ≥0.75 |
| Context Recall | ≥0.50 | ≥0.70 |

---

## Summary

| Component | Count | Purpose |
|-----------|-------|---------|
| Standard benchmark queries | 1500 | Validate against literature |
| Hermes-specific queries | 65 | Validate on our corpus |
| Metrics tracked | 10+ | Comprehensive evaluation |
| RAGAS metrics | 4 | Industry-standard quality |

**Total evaluation**: ~1565 test queries across two benchmark suites

This gives us:
1. **Credibility** — We can cite real benchmark numbers
2. **Coverage** — We test our actual use case
3. **Automation** — Reproducible, no manual scoring
