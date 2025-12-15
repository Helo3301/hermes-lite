#!/usr/bin/env python3
"""
HotpotQA Benchmark for Entity-First Ranking

This benchmark evaluates our EntityFirstRanker directly on the HotpotQA dataset,
using the same evaluation methodology as SEAL-RAG.

For each question:
1. We have 10 context documents (2 gold + 8 distractors)
2. We extract entities from the question
3. We rank the 10 documents using entity-first ranking
4. We measure if top-k contains the gold documents

This is a fair comparison because:
- Same dataset as SEAL-RAG (HotpotQA dev set)
- Same task: select gold docs from pool of 10
- Same metrics: Precision, Recall, F1

SEAL-RAG Reference (GPT-4o, k=3, HotpotQA N=1000):
- Precision: 89%
- Recall: 68%
- F1: 75%
"""

import json
import time
import random
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.entity_ranker import EntityFirstRanker


@dataclass
class BenchmarkResult:
    question_id: str
    question: str
    question_type: str

    # Gold data
    gold_doc_titles: list[str] = field(default_factory=list)

    # Retrieved
    retrieved_titles: list[str] = field(default_factory=list)

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    latency_ms: float = 0.0


def extract_entities_from_question(question: str) -> list[str]:
    """
    Extract entities from a question for entity-first ranking.

    Uses simple heuristics (proper nouns, quoted terms, capitalized words).
    In production, this could use NER or the query analyzer.
    """
    entities = []

    # Quoted terms
    quoted = re.findall(r'"([^"]+)"', question)
    entities.extend(quoted)

    # Capitalized words (likely proper nouns)
    # Skip common question words
    skip_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which',
                  'is', 'are', 'was', 'were', 'did', 'does', 'do',
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'from', 'both', 'same'}

    words = question.replace('?', ' ').replace(',', ' ').split()
    for word in words:
        # Check if capitalized (and not start of sentence)
        if word[0].isupper() and word.lower() not in skip_words:
            clean = word.strip('.,?!"\'-')
            if len(clean) > 1:
                entities.append(clean)

    # Multi-word proper nouns (consecutive capitalized words)
    pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    multi_word = re.findall(pattern, question)
    entities.extend(multi_word)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    return unique


def load_hotpotqa(path: str, sample_size: int = None) -> list[dict]:
    """Load HotpotQA dataset with optional sampling."""
    with open(path) as f:
        data = json.load(f)

    if sample_size and sample_size < len(data):
        # Stratified sample
        bridge = [q for q in data if q['type'] == 'bridge']
        comparison = [q for q in data if q['type'] == 'comparison']

        random.seed(42)
        bridge_n = int(sample_size * 0.8)
        comp_n = sample_size - bridge_n

        sampled = random.sample(bridge, min(bridge_n, len(bridge)))
        sampled += random.sample(comparison, min(comp_n, len(comparison)))
        random.shuffle(sampled)
        return sampled

    return data


def evaluate_question(
    ranker: EntityFirstRanker,
    question: dict,
    top_k: int = 3
) -> BenchmarkResult:
    """Evaluate entity-first ranking on a single question."""

    q_text = question['question']
    q_type = question['type']
    q_id = question['_id']

    # Get gold document titles
    gold_titles = set()
    for sf in question.get('supporting_facts', []):
        gold_titles.add(sf[0])

    # Create chunks from context documents
    chunks = []
    for doc in question.get('context', []):
        title = doc[0]
        sentences = doc[1]
        content = " ".join(sentences)
        chunks.append({
            'title': title,
            'content': content,
            'id': title
        })

    # Extract entities from question
    entities = extract_entities_from_question(q_text)

    # If no entities extracted, fall back to key nouns
    if not entities:
        # Use all capitalized words as fallback
        words = q_text.split()
        entities = [w.strip('.,?!') for w in words if w[0].isupper() and len(w) > 2]

    # Rank chunks by entity coverage
    start = time.time()

    if entities:
        ranked = ranker.rank_by_entity_coverage(chunks, entities, top_k=top_k)
    else:
        # No entities - just take first k chunks
        ranked = chunks[:top_k]

    latency = (time.time() - start) * 1000

    # Get retrieved titles
    retrieved_titles = [c['title'] for c in ranked]

    # Calculate metrics
    retrieved_set = set(retrieved_titles)
    gold_set = gold_titles

    true_positives = len(retrieved_set & gold_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return BenchmarkResult(
        question_id=q_id,
        question=q_text,
        question_type=q_type,
        gold_doc_titles=list(gold_titles),
        retrieved_titles=retrieved_titles,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_ms=latency
    )


def run_benchmark(
    data_path: str,
    sample_size: int = 200,
    top_k: int = 3,
    verbose: bool = False
) -> list[BenchmarkResult]:
    """Run full HotpotQA benchmark."""

    print(f"Loading HotpotQA from {data_path}...")
    questions = load_hotpotqa(data_path, sample_size)
    print(f"Loaded {len(questions)} questions")

    ranker = EntityFirstRanker()
    results = []

    for i, q in enumerate(questions):
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:50]}...")
        else:
            print(f"\r  Processing {i+1}/{len(questions)}...", end="", flush=True)

        result = evaluate_question(ranker, q, top_k)
        results.append(result)

        if verbose:
            print(f"  Entities: {extract_entities_from_question(q['question'])}")
            print(f"  Gold: {result.gold_doc_titles}")
            print(f"  Retrieved: {result.retrieved_titles}")
            print(f"  P={result.precision:.0%} R={result.recall:.0%} F1={result.f1:.0%}")

    print()
    return results


def print_summary(results: list[BenchmarkResult], title: str, top_k: int):
    """Print benchmark summary with SEAL-RAG comparison."""

    if not results:
        print("No results")
        return

    # Calculate averages
    avg_p = sum(r.precision for r in results) / len(results)
    avg_r = sum(r.recall for r in results) / len(results)
    avg_f1 = sum(r.f1 for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    # By type
    bridge = [r for r in results if r.question_type == 'bridge']
    comparison = [r for r in results if r.question_type == 'comparison']

    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()
    print(f"Overall Results (N={len(results)}, k={top_k}):")
    print(f"  Precision:  {avg_p:.1%}")
    print(f"  Recall:     {avg_r:.1%}")
    print(f"  F1 Score:   {avg_f1:.1%}")
    print(f"  Latency:    {avg_latency:.2f}ms")
    print()

    if bridge:
        b_p = sum(r.precision for r in bridge) / len(bridge)
        b_r = sum(r.recall for r in bridge) / len(bridge)
        b_f1 = sum(r.f1 for r in bridge) / len(bridge)
        print(f"Bridge Questions (n={len(bridge)}):")
        print(f"  P={b_p:.1%}  R={b_r:.1%}  F1={b_f1:.1%}")

    if comparison:
        c_p = sum(r.precision for r in comparison) / len(comparison)
        c_r = sum(r.recall for r in comparison) / len(comparison)
        c_f1 = sum(r.f1 for r in comparison) / len(comparison)
        print(f"Comparison Questions (n={len(comparison)}):")
        print(f"  P={c_p:.1%}  R={c_r:.1%}  F1={c_f1:.1%}")

    print()
    print("-" * 70)
    print("SEAL-RAG Reference (GPT-4o, k=3, HotpotQA N=1000):")
    print("  Precision: 89%  |  Recall: 68%  |  F1: 75%")
    print()
    print("Basic RAG Reference (k=3):")
    print("  Precision: 49%  |  Recall: 72%  |  F1: 59%")
    print("-" * 70)

    # Delta calculations
    seal_p, seal_r, seal_f1 = 0.89, 0.68, 0.75
    basic_p, basic_r, basic_f1 = 0.49, 0.72, 0.59

    print()
    print("Comparison:")
    print(f"  vs SEAL-RAG:  P {(avg_p-seal_p)*100:+.1f}pp | R {(avg_r-seal_r)*100:+.1f}pp | F1 {(avg_f1-seal_f1)*100:+.1f}pp")
    print(f"  vs Basic RAG: P {(avg_p-basic_p)*100:+.1f}pp | R {(avg_r-basic_r)*100:+.1f}pp | F1 {(avg_f1-basic_f1)*100:+.1f}pp")
    print()

    # Verdict
    if avg_p >= seal_p and avg_f1 >= seal_f1:
        print("  ✓ MATCHES OR EXCEEDS SEAL-RAG PERFORMANCE")
    elif avg_p >= basic_p * 1.5:
        print("  ✓ SIGNIFICANT IMPROVEMENT OVER BASIC RAG")
    else:
        print("  ○ Room for improvement")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA benchmark for Entity-First Ranking"
    )
    parser.add_argument(
        "--data",
        default="hotpot_dev_distractor_v1.json",
        help="Path to HotpotQA dataset"
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=200,
        help="Sample size (default: 200)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Top-k documents to retrieve (default: 3, matching SEAL-RAG)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON"
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" HotpotQA Benchmark for Entity-First Ranking")
    print(" Direct comparison with SEAL-RAG metrics")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Sample: {args.sample} questions")
    print(f"  Top-k: {args.top_k}")
    print()

    results = run_benchmark(
        args.data,
        args.sample,
        args.top_k,
        args.verbose
    )

    print_summary(results, "Entity-First Ranker on HotpotQA", args.top_k)

    if args.output:
        output = {
            "config": {
                "sample_size": args.sample,
                "top_k": args.top_k
            },
            "summary": {
                "precision": sum(r.precision for r in results) / len(results),
                "recall": sum(r.recall for r in results) / len(results),
                "f1": sum(r.f1 for r in results) / len(results),
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                }
                for r in results
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
