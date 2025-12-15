"""
SEAL-RAG Style Benchmark for HERMES Entity-First Ranking

Benchmarks our implementation against the same metrics used in SEAL-RAG paper:
- HotpotQA style multi-hop questions
- 2WikiMultiHopQA style questions

Metrics:
- Precision: % of retrieved chunks that are relevant
- Recall: % of required entities covered
- F1: Harmonic mean of precision and recall
- Entity Coverage: % of query entities found in results

SEAL-RAG Reference Results (GPT-4o, k=3):
- HotpotQA: 89% Precision, 68% Recall, 75% F1
- 2WikiMultiHopQA (k=5): 96% Precision
"""

import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Optional
import sys


@dataclass
class BenchmarkResult:
    query: str
    query_type: str
    precision: float
    recall: float
    f1: float
    entity_coverage: float
    latency_ms: float
    num_results: int
    relevant_count: int
    target_entities: list
    found_entities: list


# Multi-hop questions inspired by HotpotQA and 2WikiMultiHopQA
# Each question has target entities that should appear in relevant results
BENCHMARK_QUESTIONS = [
    # HotpotQA-style: Bridge entity questions
    {
        "query": "How does SEAL-RAG's entity extraction improve precision compared to basic RAG?",
        "target_entities": ["SEAL-RAG", "entity", "precision", "RAG"],
        "type": "multi_hop",
        "description": "SEAL-RAG vs RAG comparison"
    },
    {
        "query": "What benchmark datasets are used to evaluate multi-hop question answering systems?",
        "target_entities": ["HotpotQA", "2WikiMultiHopQA", "benchmark", "multi-hop"],
        "type": "multi_hop",
        "description": "QA benchmark datasets"
    },
    {
        "query": "How do Self-RAG and CRAG differ in their approach to retrieval augmentation?",
        "target_entities": ["Self-RAG", "CRAG", "retrieval"],
        "type": "comparative",
        "description": "RAG method comparison"
    },
    {
        "query": "What is context dilution and how does fixed-budget retrieval address it?",
        "target_entities": ["context", "dilution", "budget", "retrieval"],
        "type": "multi_hop",
        "description": "Context dilution explanation"
    },
    {
        "query": "How does the replace-don't-expand strategy improve retrieval precision?",
        "target_entities": ["replace", "expand", "precision", "retrieval"],
        "type": "multi_hop",
        "description": "Replace strategy"
    },

    # 2WikiMultiHopQA-style: Multi-hop reasoning
    {
        "query": "What retrieval methods achieve highest precision on HotpotQA benchmark?",
        "target_entities": ["precision", "HotpotQA", "retrieval"],
        "type": "multi_hop",
        "description": "HotpotQA precision leaders"
    },
    {
        "query": "How do dense retrieval methods like DPR compare to sparse methods like BM25?",
        "target_entities": ["DPR", "BM25", "dense", "sparse", "retrieval"],
        "type": "comparative",
        "description": "Dense vs sparse retrieval"
    },
    {
        "query": "What role does the knowledge graph play in RAG systems?",
        "target_entities": ["knowledge", "graph", "RAG"],
        "type": "exploratory",
        "description": "KG in RAG"
    },
    {
        "query": "How does iterative retrieval improve answer quality for complex questions?",
        "target_entities": ["iterative", "retrieval", "answer", "quality"],
        "type": "multi_hop",
        "description": "Iterative retrieval benefits"
    },
    {
        "query": "What are the main limitations of single-shot retrieval for multi-hop QA?",
        "target_entities": ["single", "retrieval", "multi-hop", "limitation"],
        "type": "multi_hop",
        "description": "Single-shot limitations"
    },

    # Additional precision-focused questions
    {
        "query": "How does SEAL-RAG achieve 96% precision on 2WikiMultiHopQA?",
        "target_entities": ["SEAL-RAG", "precision", "2WikiMultiHopQA"],
        "type": "multi_hop",
        "description": "SEAL-RAG precision method"
    },
    {
        "query": "What is the SEAL cycle and how does it work?",
        "target_entities": ["SEAL", "cycle", "search", "extract", "assess"],
        "type": "multi_hop",
        "description": "SEAL cycle explanation"
    },
    {
        "query": "How do micro-queries help fill knowledge gaps in retrieval?",
        "target_entities": ["micro", "query", "gap", "retrieval"],
        "type": "multi_hop",
        "description": "Micro-query strategy"
    },
    {
        "query": "What metrics are used to evaluate RAG system performance?",
        "target_entities": ["precision", "recall", "F1", "accuracy", "metric"],
        "type": "exploratory",
        "description": "RAG evaluation metrics"
    },
    {
        "query": "How does entity-first ranking differ from relevance-based ranking?",
        "target_entities": ["entity", "ranking", "relevance"],
        "type": "comparative",
        "description": "Entity vs relevance ranking"
    },
]


def query_hermes(query: str, top_k: int = 5, endpoint: str = "search/v2") -> dict:
    """Query HERMES and return results."""
    url = f"http://localhost:8780/{endpoint}?query={urllib.parse.quote(query)}&top_k={top_k}"

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read())
        latency = (time.time() - start) * 1000
        data['latency_ms'] = latency
        return data
    except Exception as e:
        return {"error": str(e), "results": [], "latency_ms": 0}


def calculate_metrics(results: list, target_entities: list) -> dict:
    """Calculate precision, recall, and entity coverage."""
    if not results:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "entity_coverage": 0.0,
            "relevant_count": 0,
            "found_entities": []
        }

    # Precision: % of results containing at least one target entity
    relevant_count = 0
    all_found_entities = set()

    for chunk in results:
        content = chunk.get("content", "").lower()
        chunk_relevant = False

        for entity in target_entities:
            entity_lower = entity.lower()
            if entity_lower in content:
                all_found_entities.add(entity)
                chunk_relevant = True

        if chunk_relevant:
            relevant_count += 1

    precision = relevant_count / len(results)

    # Recall: % of target entities found in ANY result
    recall = len(all_found_entities) / len(target_entities) if target_entities else 0

    # F1: Harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Entity coverage (same as recall for our purposes)
    entity_coverage = recall

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "entity_coverage": entity_coverage,
        "relevant_count": relevant_count,
        "found_entities": list(all_found_entities)
    }


def run_benchmark(top_k: int = 5, endpoint: str = "search/v2") -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    results = []

    for q in BENCHMARK_QUESTIONS:
        print(f"  Testing: {q['description'][:40]}...", end=" ", flush=True)

        response = query_hermes(q["query"], top_k=top_k, endpoint=endpoint)

        if "error" in response:
            print(f"ERROR: {response['error']}")
            continue

        search_results = response.get("results", [])
        metrics = calculate_metrics(search_results, q["target_entities"])

        result = BenchmarkResult(
            query=q["query"],
            query_type=q["type"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            entity_coverage=metrics["entity_coverage"],
            latency_ms=response.get("latency_ms", 0),
            num_results=len(search_results),
            relevant_count=metrics["relevant_count"],
            target_entities=q["target_entities"],
            found_entities=metrics["found_entities"]
        )
        results.append(result)

        print(f"P={result.precision:.0%} R={result.recall:.0%} F1={result.f1:.0%} ({result.latency_ms:.0f}ms)")

    return results


def print_summary(results: list[BenchmarkResult], title: str):
    """Print benchmark summary."""
    if not results:
        print("No results to summarize")
        return

    # Calculate averages
    avg_precision = sum(r.precision for r in results) / len(results)
    avg_recall = sum(r.recall for r in results) / len(results)
    avg_f1 = sum(r.f1 for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    # By query type
    by_type = {}
    for r in results:
        if r.query_type not in by_type:
            by_type[r.query_type] = []
        by_type[r.query_type].append(r)

    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()
    print(f"Overall Results (N={len(results)} queries):")
    print(f"  Precision:  {avg_precision:.1%}")
    print(f"  Recall:     {avg_recall:.1%}")
    print(f"  F1 Score:   {avg_f1:.1%}")
    print(f"  Avg Latency: {avg_latency:.0f}ms")
    print()

    print("By Query Type:")
    for qtype, type_results in sorted(by_type.items()):
        type_p = sum(r.precision for r in type_results) / len(type_results)
        type_r = sum(r.recall for r in type_results) / len(type_results)
        type_f1 = sum(r.f1 for r in type_results) / len(type_results)
        print(f"  {qtype:12} P={type_p:.0%} R={type_r:.0%} F1={type_f1:.0%} (n={len(type_results)})")

    print()
    print("-" * 70)
    print("SEAL-RAG Reference (GPT-4o, k=3):")
    print("  HotpotQA:        P=89% R=68% F1=75%")
    print("  2WikiMultiHopQA: P=96% R=77% F1=85%")
    print("-" * 70)
    print()

    # Highlight best and worst
    best = max(results, key=lambda r: r.precision)
    worst = min(results, key=lambda r: r.precision)

    print(f"Best:  {best.precision:.0%} precision - {best.query[:50]}...")
    print(f"Worst: {worst.precision:.0%} precision - {worst.query[:50]}...")


def main():
    print()
    print("=" * 70)
    print(" SEAL-RAG Style Benchmark for HERMES Entity-First Ranking")
    print("=" * 70)
    print()

    # Check if HERMES is available
    try:
        health = urllib.request.urlopen("http://localhost:8780/health", timeout=5)
        print("HERMES is healthy. Starting benchmark...")
        print()
    except:
        print("ERROR: HERMES is not available at localhost:8780")
        sys.exit(1)

    # Run v2 benchmark (with entity-first ranking)
    print("Running v2 benchmark (entity-first ranking enabled)...")
    print("-" * 70)
    v2_results = run_benchmark(top_k=5, endpoint="search/v2")
    print_summary(v2_results, "HERMES v2 with Entity-First Ranking")

    # Run v1 benchmark for comparison
    print()
    print("Running v1 benchmark (baseline - no entity-first ranking)...")
    print("-" * 70)
    v1_results = run_benchmark(top_k=5, endpoint="search")
    print_summary(v1_results, "HERMES v1 Baseline (No Entity-First)")

    # Comparison
    if v1_results and v2_results:
        v1_p = sum(r.precision for r in v1_results) / len(v1_results)
        v2_p = sum(r.precision for r in v2_results) / len(v2_results)
        v1_f1 = sum(r.f1 for r in v1_results) / len(v1_results)
        v2_f1 = sum(r.f1 for r in v2_results) / len(v2_results)

        print()
        print("=" * 70)
        print(" IMPROVEMENT SUMMARY")
        print("=" * 70)
        print(f"  Precision: {v1_p:.1%} -> {v2_p:.1%} ({(v2_p-v1_p)*100:+.1f}pp)")
        print(f"  F1 Score:  {v1_f1:.1%} -> {v2_f1:.1%} ({(v2_f1-v1_f1)*100:+.1f}pp)")
        print()

        # Compare to SEAL-RAG
        seal_p = 0.89  # SEAL-RAG HotpotQA precision
        seal_f1 = 0.75
        print(f"  vs SEAL-RAG (HotpotQA):")
        print(f"    Precision: {seal_p:.0%} (ours: {v2_p:.1%}, gap: {(seal_p-v2_p)*100:+.1f}pp)")
        print(f"    F1 Score:  {seal_f1:.0%} (ours: {v2_f1:.1%}, gap: {(seal_f1-v2_f1)*100:+.1f}pp)")
        print("=" * 70)


if __name__ == "__main__":
    main()
