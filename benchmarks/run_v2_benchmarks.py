#!/usr/bin/env python3
"""Run benchmarks comparing v1 and v2 search endpoints."""

import json
import time
import requests
import statistics
from pathlib import Path
from datetime import datetime

HERMES_URL = "http://localhost:8780"
RESULTS_DIR = Path(__file__).parent / "results" / "v2"


def load_test_queries():
    """Load Hermes-specific test queries."""
    queries_path = Path(__file__).parent / "hermes" / "test_queries.json"
    with open(queries_path) as f:
        data = json.load(f)

    # Flatten the queries from different types
    all_queries = []
    for query_type in ["paper_lookup", "cross_paper", "comparative", "exploratory"]:
        if query_type in data:
            for q in data[query_type]:
                q["type"] = query_type
                all_queries.append(q)

    return all_queries


def search_v1(query: str, top_k: int = 10) -> dict:
    """Search using v1 endpoint."""
    start = time.time()
    try:
        resp = requests.get(
            f"{HERMES_URL}/search",
            params={"query": query, "top_k": top_k, "rerank": True},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "latency_ms": elapsed,
                "results": data.get("results", []),
                "count": len(data.get("results", [])),
            }
        return {"success": False, "latency_ms": elapsed, "error": resp.text}
    except Exception as e:
        return {"success": False, "latency_ms": (time.time() - start) * 1000, "error": str(e)}


def search_v2(query: str, top_k: int = 10) -> dict:
    """Search using v2 endpoint."""
    start = time.time()
    try:
        resp = requests.get(
            f"{HERMES_URL}/search/v2",
            params={"query": query, "top_k": top_k, "rerank": True, "detect_contradictions": True},
            timeout=60
        )
        elapsed = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "latency_ms": elapsed,
                "results": data.get("results", []),
                "count": len(data.get("results", [])),
                "query_type": data.get("query_type"),
                "entities": data.get("entities", []),
                "confidence": data.get("confidence", {}).get("score", 0),
                "iterations": data.get("metadata", {}).get("iterations", 1),
                "contradictions": data.get("contradictions"),
            }
        return {"success": False, "latency_ms": elapsed, "error": resp.text}
    except Exception as e:
        return {"success": False, "latency_ms": (time.time() - start) * 1000, "error": str(e)}


def calculate_entity_coverage(results: list, expected_entities: list) -> float:
    """Calculate what percentage of expected entities appear in results."""
    if not expected_entities:
        return 1.0

    all_content = " ".join(r.get("content", "") for r in results).lower()
    found = sum(1 for e in expected_entities if e.lower() in all_content)
    return found / len(expected_entities)


def run_benchmarks():
    """Run comprehensive benchmarks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_queries = load_test_queries()

    print("=" * 70)
    print("HERMES v2 Benchmark Suite")
    print("=" * 70)
    print(f"Testing {len(test_queries)} queries")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    v1_results = []
    v2_results = []

    by_type = {"paper_lookup": [], "cross_paper": [], "comparative": [], "exploratory": []}

    for i, query_data in enumerate(test_queries):
        query = query_data["query"]
        query_type = query_data["type"]
        expected_entities = query_data.get("expected_entities", [])

        print(f"[{i+1}/{len(test_queries)}] {query[:60]}...")

        # Run v1
        v1 = search_v1(query)
        v1["expected_entities"] = expected_entities
        if v1["success"]:
            v1["entity_coverage"] = calculate_entity_coverage(v1["results"], expected_entities)
        v1_results.append(v1)

        # Run v2
        v2 = search_v2(query)
        v2["expected_entities"] = expected_entities
        if v2["success"]:
            v2["entity_coverage"] = calculate_entity_coverage(v2["results"], expected_entities)
        v2_results.append(v2)

        # Track by type
        by_type[query_type].append({
            "query": query,
            "v1_latency": v1.get("latency_ms"),
            "v2_latency": v2.get("latency_ms"),
            "v1_coverage": v1.get("entity_coverage", 0),
            "v2_coverage": v2.get("entity_coverage", 0),
            "v2_confidence": v2.get("confidence", 0),
            "v2_type": v2.get("query_type", "unknown"),
        })

        # Brief status
        print(f"   v1: {v1['latency_ms']:.0f}ms, v2: {v2['latency_ms']:.0f}ms, " +
              f"type: {v2.get('query_type', 'N/A')}, conf: {v2.get('confidence', 0):.2f}")

    # Calculate statistics
    def calc_stats(results, key):
        values = [r.get(key, 0) for r in results if r.get("success")]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0],
            "min": min(values),
            "max": max(values),
        }

    v1_latency = calc_stats(v1_results, "latency_ms")
    v2_latency = calc_stats(v2_results, "latency_ms")
    v1_coverage = calc_stats(v1_results, "entity_coverage")
    v2_coverage = calc_stats(v2_results, "entity_coverage")
    v2_confidence = calc_stats(v2_results, "confidence")

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n## Latency (ms)")
    print(f"{'Metric':<15} {'v1':>12} {'v2':>12} {'Δ':>12}")
    print("-" * 51)
    print(f"{'P50 (median)':<15} {v1_latency['median']:>12.0f} {v2_latency['median']:>12.0f} {v2_latency['median'] - v1_latency['median']:>+12.0f}")
    print(f"{'P95':<15} {v1_latency['p95']:>12.0f} {v2_latency['p95']:>12.0f} {v2_latency['p95'] - v1_latency['p95']:>+12.0f}")
    print(f"{'Mean':<15} {v1_latency['mean']:>12.0f} {v2_latency['mean']:>12.0f} {v2_latency['mean'] - v1_latency['mean']:>+12.0f}")

    print("\n## Entity Coverage")
    print(f"{'Metric':<15} {'v1':>12} {'v2':>12} {'Δ':>12}")
    print("-" * 51)
    print(f"{'Mean':<15} {v1_coverage['mean']:>12.2%} {v2_coverage['mean']:>12.2%} {v2_coverage['mean'] - v1_coverage['mean']:>+12.2%}")

    print("\n## v2-Specific Metrics")
    print(f"{'Metric':<20} {'Value':>12}")
    print("-" * 32)
    print(f"{'Mean Confidence':<20} {v2_confidence['mean']:>12.2f}")

    # Query type classification accuracy
    correct = sum(1 for r in v2_results if r.get("success") and r.get("query_type") != "unknown")
    print(f"{'Queries Classified':<20} {correct:>12}")

    # Contradictions found
    with_contradictions = sum(1 for r in v2_results if r.get("contradictions") and r["contradictions"].get("has_contradictions"))
    print(f"{'With Contradictions':<20} {with_contradictions:>12}")

    # By query type breakdown
    print("\n## Results by Query Type")
    print(f"{'Type':<15} {'Count':>8} {'v1 P50':>10} {'v2 P50':>10} {'v1 Cov':>10} {'v2 Cov':>10}")
    print("-" * 63)

    for qtype, items in by_type.items():
        if not items:
            continue
        v1_lats = sorted([x["v1_latency"] for x in items if x["v1_latency"]])
        v2_lats = sorted([x["v2_latency"] for x in items if x["v2_latency"]])
        v1_covs = [x["v1_coverage"] for x in items if x["v1_coverage"] is not None]
        v2_covs = [x["v2_coverage"] for x in items if x["v2_coverage"] is not None]

        v1_p50 = v1_lats[len(v1_lats)//2] if v1_lats else 0
        v2_p50 = v2_lats[len(v2_lats)//2] if v2_lats else 0
        v1_cov = statistics.mean(v1_covs) if v1_covs else 0
        v2_cov = statistics.mean(v2_covs) if v2_covs else 0

        print(f"{qtype:<15} {len(items):>8} {v1_p50:>10.0f} {v2_p50:>10.0f} {v1_cov:>10.2%} {v2_cov:>10.2%}")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "v1_latency": v1_latency,
            "v2_latency": v2_latency,
            "v1_entity_coverage": v1_coverage,
            "v2_entity_coverage": v2_coverage,
            "v2_confidence": v2_confidence,
        },
        "by_type": by_type,
        "v1_results": v1_results,
        "v2_results": v2_results,
    }

    results_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    # Create summary markdown
    summary_md = f"""# HERMES v2 Benchmark Results

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Queries Tested**: {len(test_queries)}

## Latency Comparison

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| P50 (median) | {v1_latency['median']:.0f}ms | {v2_latency['median']:.0f}ms | {v2_latency['median'] - v1_latency['median']:+.0f}ms |
| P95 | {v1_latency['p95']:.0f}ms | {v2_latency['p95']:.0f}ms | {v2_latency['p95'] - v1_latency['p95']:+.0f}ms |
| Mean | {v1_latency['mean']:.0f}ms | {v2_latency['mean']:.0f}ms | {v2_latency['mean'] - v1_latency['mean']:+.0f}ms |

## Entity Coverage

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| Mean | {v1_coverage['mean']:.2%} | {v2_coverage['mean']:.2%} | {v2_coverage['mean'] - v1_coverage['mean']:+.2%} |

## v2-Specific Metrics

| Metric | Value |
|--------|-------|
| Mean Confidence | {v2_confidence['mean']:.2f} |
| Queries with Contradictions | {with_contradictions} |

## Results by Query Type

| Type | Count | v1 P50 | v2 P50 | v1 Coverage | v2 Coverage |
|------|-------|--------|--------|-------------|-------------|
"""

    for qtype, items in by_type.items():
        if not items:
            continue
        v1_lats = sorted([x["v1_latency"] for x in items if x["v1_latency"]])
        v2_lats = sorted([x["v2_latency"] for x in items if x["v2_latency"]])
        v1_covs = [x["v1_coverage"] for x in items if x["v1_coverage"] is not None]
        v2_covs = [x["v2_coverage"] for x in items if x["v2_coverage"] is not None]

        v1_p50 = v1_lats[len(v1_lats)//2] if v1_lats else 0
        v2_p50 = v2_lats[len(v2_lats)//2] if v2_lats else 0
        v1_cov = statistics.mean(v1_covs) if v1_covs else 0
        v2_cov = statistics.mean(v2_covs) if v2_covs else 0

        summary_md += f"| {qtype} | {len(items)} | {v1_p50:.0f}ms | {v2_p50:.0f}ms | {v1_cov:.2%} | {v2_cov:.2%} |\n"

    summary_file = RESULTS_DIR / f"BENCHMARK_SUMMARY.md"
    with open(summary_file, "w") as f:
        f.write(summary_md)

    print(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    run_benchmarks()
