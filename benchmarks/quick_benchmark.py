#!/usr/bin/env python3
"""Quick benchmark comparing v1 and v2 search endpoints."""

import json
import time
import requests
import statistics
from datetime import datetime
from pathlib import Path

HERMES_URL = "http://localhost:8780"
RESULTS_DIR = Path(__file__).parent / "results" / "v2"

def load_test_queries():
    """Load test queries."""
    queries_path = Path(__file__).parent / "hermes" / "test_queries.json"
    with open(queries_path) as f:
        data = json.load(f)

    all_queries = []
    for query_type in ["paper_lookup", "cross_paper", "comparative", "exploratory"]:
        if query_type in data:
            for q in data[query_type]:
                q["type"] = query_type
                all_queries.append(q)
    return all_queries

def benchmark_query(query, top_k=10):
    """Benchmark a single query on v1 and v2."""
    # v1
    v1_start = time.time()
    try:
        r = requests.get(f"{HERMES_URL}/search", params={"query": query, "top_k": top_k, "rerank": "true"}, timeout=30)
        v1_lat = (time.time() - v1_start) * 1000
        v1_ok = r.status_code == 200
        v1_results = len(r.json().get("results", [])) if v1_ok else 0
    except Exception as e:
        v1_lat = (time.time() - v1_start) * 1000
        v1_ok = False
        v1_results = 0

    # v2
    v2_start = time.time()
    try:
        r = requests.get(f"{HERMES_URL}/search/v2", params={"query": query, "top_k": top_k, "rerank": "true"}, timeout=120)
        v2_lat = (time.time() - v2_start) * 1000
        v2_ok = r.status_code == 200
        if v2_ok:
            data = r.json()
            v2_results = len(data.get("results", []))
            v2_type = data.get("query_type", "unknown")
            v2_conf = data.get("confidence", {}).get("score", 0)
        else:
            v2_results = 0
            v2_type = "error"
            v2_conf = 0
    except Exception as e:
        v2_lat = (time.time() - v2_start) * 1000
        v2_ok = False
        v2_results = 0
        v2_type = "error"
        v2_conf = 0

    return {
        "v1_latency": v1_lat,
        "v1_ok": v1_ok,
        "v1_results": v1_results,
        "v2_latency": v2_lat,
        "v2_ok": v2_ok,
        "v2_results": v2_results,
        "v2_type": v2_type,
        "v2_confidence": v2_conf,
    }

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_test_queries()
    print(f"Benchmarking {len(queries)} queries...")
    print("=" * 70)

    results = []
    v1_lats = []
    v2_lats = []

    for i, q in enumerate(queries):
        query = q["query"]
        qtype = q["type"]

        r = benchmark_query(query)
        r["query"] = query
        r["expected_type"] = qtype
        results.append(r)

        if r["v1_ok"]:
            v1_lats.append(r["v1_latency"])
        if r["v2_ok"]:
            v2_lats.append(r["v2_latency"])

        print(f"[{i+1:2d}/{len(queries)}] {query[:45]:45s} v1:{r['v1_latency']:6.0f}ms v2:{r['v2_latency']:6.0f}ms type:{r['v2_type']:12s} conf:{r['v2_confidence']:.2f}")

        # Save progress every 10 queries
        if (i + 1) % 10 == 0:
            with open(RESULTS_DIR / "progress.json", "w") as f:
                json.dump(results, f, indent=2)

    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if v1_lats and v2_lats:
        print(f"\nLatency (ms):")
        print(f"  v1 P50: {statistics.median(v1_lats):,.0f}")
        print(f"  v2 P50: {statistics.median(v2_lats):,.0f}")
        print(f"  v1 Mean: {statistics.mean(v1_lats):,.0f}")
        print(f"  v2 Mean: {statistics.mean(v2_lats):,.0f}")

        # By query type
        print(f"\nBy Query Type:")
        for qtype in ["paper_lookup", "cross_paper", "comparative", "exploratory"]:
            type_results = [r for r in results if r["expected_type"] == qtype]
            if type_results:
                v1_type_lats = [r["v1_latency"] for r in type_results if r["v1_ok"]]
                v2_type_lats = [r["v2_latency"] for r in type_results if r["v2_ok"]]
                if v1_type_lats and v2_type_lats:
                    print(f"  {qtype:15s} v1:{statistics.median(v1_type_lats):6.0f}ms v2:{statistics.median(v2_type_lats):6.0f}ms (n={len(type_results)})")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"benchmark_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "results": results,
            "summary": {
                "v1_median_ms": statistics.median(v1_lats) if v1_lats else 0,
                "v2_median_ms": statistics.median(v2_lats) if v2_lats else 0,
                "v1_mean_ms": statistics.mean(v1_lats) if v1_lats else 0,
                "v2_mean_ms": statistics.mean(v2_lats) if v2_lats else 0,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / f'benchmark_{timestamp}.json'}")

if __name__ == "__main__":
    main()
