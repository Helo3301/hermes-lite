#!/usr/bin/env python3
"""Run benchmarks against Hermes RAG system."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse

from metrics import (
    compute_retrieval_metrics,
    aggregate_metrics,
)

HERMES_URL = "http://localhost:8780"
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def search_hermes(query: str, top_k: int = 10) -> tuple[list[dict], float]:
    """Search Hermes and return results with latency."""
    url = f"{HERMES_URL}/search?query={urllib.parse.quote(query)}&top_k={top_k}"

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            latency_ms = (time.time() - start) * 1000
            return data.get("results", []), latency_ms
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        print(f"  Error searching for '{query[:50]}...': {e}")
        return [], latency_ms


def run_hermes_benchmark(version: str = "v1") -> dict:
    """Run the Hermes-specific benchmark."""
    print(f"\n{'='*60}")
    print(f"Running Hermes-Specific Benchmark ({version})")
    print(f"{'='*60}")

    queries_path = BENCHMARK_DIR / "hermes" / "test_queries.json"
    with open(queries_path) as f:
        test_data = json.load(f)

    all_results = {
        "metadata": {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "hermes_url": HERMES_URL,
        },
        "by_category": {},
        "detailed_results": [],
    }

    for category in ["paper_lookup", "cross_paper", "comparative", "exploratory"]:
        print(f"\n--- {category.upper()} ---")
        queries = test_data.get(category, [])
        category_metrics = []

        for q in queries:
            query_id = q["id"]
            query_text = q["query"]

            print(f"  [{query_id}] {query_text[:50]}...", end=" ")

            results, latency = search_hermes(query_text, top_k=10)

            # Determine expected values based on category
            expected_entities = q.get("expected_entities", q.get("expected_connections", []))
            expected_concepts = q.get("expected_concepts", q.get("expected_topics", []))

            metrics = compute_retrieval_metrics(
                query=query_text,
                retrieved_chunks=results,
                expected_entities=expected_entities,
                expected_concepts=expected_concepts,
                latency_ms=latency,
            )

            category_metrics.append(metrics)
            all_results["detailed_results"].append({
                "id": query_id,
                "query": query_text,
                "category": category,
                "metrics": metrics,
                "num_results": len(results),
            })

            print(f"({latency:.0f}ms, {len(results)} chunks)")

        # Aggregate for this category
        all_results["by_category"][category] = aggregate_metrics(category_metrics)

    # Overall aggregation
    all_metrics = [r["metrics"] for r in all_results["detailed_results"]]
    all_results["overall"] = aggregate_metrics(all_metrics)

    return all_results


def run_standard_benchmark(dataset: str, sample_limit: int = 100, version: str = "v1") -> dict:
    """Run benchmark on a standard dataset (HotpotQA, 2WikiMultiHopQA, NQ)."""
    print(f"\n{'='*60}")
    print(f"Running {dataset} Benchmark ({version}, n={sample_limit})")
    print(f"{'='*60}")

    data_path = BENCHMARK_DIR / "standard" / f"{dataset}_500.json"
    if not data_path.exists():
        print(f"  Dataset not found: {data_path}")
        return {}

    with open(data_path) as f:
        samples = json.load(f)

    # Limit samples for quick benchmarks
    samples = samples[:sample_limit]

    all_results = {
        "metadata": {
            "dataset": dataset,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(samples),
        },
        "detailed_results": [],
    }

    all_metrics = []

    for i, sample in enumerate(samples):
        query = sample["question"]
        answer = sample.get("answer", "")

        print(f"  [{i+1}/{len(samples)}] {query[:50]}...", end=" ")

        results, latency = search_hermes(query, top_k=10)

        # For standard benchmarks, check if answer appears in retrieved text
        all_text = " ".join(r.get("content", "") for r in results).lower()
        answer_in_context = 1.0 if answer.lower() in all_text else 0.0

        # Get supporting facts if available (HotpotQA, 2WikiMultiHopQA)
        supporting_facts = []
        if "supporting_facts" in sample:
            sf = sample["supporting_facts"]
            if isinstance(sf, dict) and "titles" in sf:
                supporting_facts = [{"title": t} for t in sf["titles"]]
            elif isinstance(sf, list):
                supporting_facts = sf

        metrics = compute_retrieval_metrics(
            query=query,
            retrieved_chunks=results,
            supporting_facts=supporting_facts if supporting_facts else None,
            latency_ms=latency,
        )
        metrics["answer_in_context"] = answer_in_context

        all_metrics.append(metrics)
        all_results["detailed_results"].append({
            "id": sample.get("id", i),
            "query": query,
            "answer": answer,
            "metrics": metrics,
        })

        status = "✓" if answer_in_context else "✗"
        print(f"{status} ({latency:.0f}ms)")

    all_results["overall"] = aggregate_metrics(all_metrics)

    # Compute answer-in-context rate
    aic_values = [m["answer_in_context"] for m in all_metrics]
    all_results["overall"]["answer_in_context_rate"] = sum(aic_values) / len(aic_values) if aic_values else 0

    return all_results


def save_results(results: dict, name: str, version: str = "v1"):
    """Save benchmark results to file."""
    output_dir = RESULTS_DIR / f"{version}_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}_{version}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")


def print_summary(results: dict, name: str):
    """Print a summary of benchmark results."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {name}")
    print(f"{'='*60}")

    overall = results.get("overall", {})

    print(f"\nTotal queries: {overall.get('total_queries', 0)}")

    if "latency_p50" in overall:
        print(f"\nLatency:")
        print(f"  P50: {overall['latency_p50']:.0f}ms")
        print(f"  P95: {overall['latency_p95']:.0f}ms")
        print(f"  P99: {overall['latency_p99']:.0f}ms")

    if "entity_coverage_mean" in overall:
        print(f"\nEntity Coverage: {overall['entity_coverage_mean']:.2%}")

    if "concept_relevance_mean" in overall:
        print(f"Concept Relevance: {overall['concept_relevance_mean']:.2%}")

    if "answer_in_context_rate" in overall:
        print(f"Answer in Context: {overall['answer_in_context_rate']:.2%}")

    if "supporting_fact_recall_mean" in overall:
        print(f"Supporting Fact Recall: {overall['supporting_fact_recall_mean']:.2%}")

    # Category breakdown for Hermes
    if "by_category" in results:
        print("\nBy Category:")
        for cat, metrics in results["by_category"].items():
            entity_cov = metrics.get("entity_coverage_mean", 0)
            latency = metrics.get("latency_ms_mean", 0)
            print(f"  {cat}: entity_cov={entity_cov:.2%}, latency={latency:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Run Hermes RAG benchmarks")
    parser.add_argument(
        "--version", "-v",
        default="v1",
        help="Version identifier (v1, v2, etc.)"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["hermes", "hotpotqa", "2wikimultihop", "nq"],
        help="Datasets to benchmark"
    )
    parser.add_argument(
        "--sample-limit", "-n",
        type=int,
        default=100,
        help="Max samples per standard dataset"
    )
    args = parser.parse_args()

    print(f"\nHermes RAG Benchmark Suite")
    print(f"Version: {args.version}")
    print(f"Datasets: {args.datasets}")
    print(f"Sample limit: {args.sample_limit}")

    for dataset in args.datasets:
        if dataset == "hermes":
            results = run_hermes_benchmark(version=args.version)
            if results:
                save_results(results, "hermes", args.version)
                print_summary(results, "Hermes-Specific")
        else:
            results = run_standard_benchmark(
                dataset=dataset,
                sample_limit=args.sample_limit,
                version=args.version
            )
            if results:
                save_results(results, dataset, args.version)
                print_summary(results, dataset.upper())

    print(f"\n{'='*60}")
    print("Benchmarking complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
