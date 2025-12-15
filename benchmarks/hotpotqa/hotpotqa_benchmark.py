#!/usr/bin/env python3
"""
HotpotQA Benchmark for HERMES Entity-First Ranking

This benchmark uses the official HotpotQA dev set to evaluate retrieval performance.
We measure the same metrics as SEAL-RAG to enable direct comparison.

HotpotQA Structure:
- 7,405 questions (dev set)
- Each question has 10 context documents (2 gold + 8 distractors)
- supporting_facts: List of [doc_title, sentence_idx] pairs that contain answer
- Types: 'bridge' (5,918) and 'comparison' (1,487)

Metrics (matching SEAL-RAG paper):
- Precision: % of retrieved docs that are gold (supporting) documents
- Recall: % of gold documents that were retrieved
- F1: Harmonic mean of precision and recall
- Supporting Fact Recall: % of supporting facts sentences retrieved

SEAL-RAG Reference (GPT-4o, k=3, HotpotQA):
- Precision: 89%
- Recall: 68%
- F1: 75%
"""

import json
import time
import random
import argparse
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question: str
    question_type: str
    answer: str

    # Gold data
    gold_docs: list[str] = field(default_factory=list)
    supporting_facts: list[tuple] = field(default_factory=list)

    # Retrieval results
    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_content: list[str] = field(default_factory=list)

    # Metrics
    doc_precision: float = 0.0
    doc_recall: float = 0.0
    doc_f1: float = 0.0
    sf_recall: float = 0.0  # Supporting fact recall

    latency_ms: float = 0.0
    error: Optional[str] = None


def load_hotpotqa(path: str, sample_size: Optional[int] = None) -> list[dict]:
    """Load HotpotQA dataset."""
    with open(path, "r") as f:
        data = json.load(f)

    if sample_size and sample_size < len(data):
        # Stratified sample by question type
        bridge = [q for q in data if q['type'] == 'bridge']
        comparison = [q for q in data if q['type'] == 'comparison']

        bridge_n = int(sample_size * 0.8)  # 80% bridge (matches distribution)
        comp_n = sample_size - bridge_n

        random.seed(42)  # Reproducible
        sampled = random.sample(bridge, min(bridge_n, len(bridge)))
        sampled += random.sample(comparison, min(comp_n, len(comparison)))
        random.shuffle(sampled)
        return sampled

    return data


def create_document_index(question: dict) -> dict[str, str]:
    """Create title -> content index for a question's context."""
    doc_index = {}
    for doc in question.get('context', []):
        title = doc[0]
        sentences = doc[1]
        content = " ".join(sentences)
        doc_index[title] = content
    return doc_index


def get_gold_documents(question: dict) -> set[str]:
    """Get titles of gold (supporting) documents."""
    gold_docs = set()
    for sf in question.get('supporting_facts', []):
        doc_title = sf[0]
        gold_docs.add(doc_title)
    return gold_docs


def get_supporting_fact_sentences(question: dict) -> list[str]:
    """Get actual text of supporting fact sentences."""
    sf_sentences = []
    doc_index = {doc[0]: doc[1] for doc in question.get('context', [])}

    for sf in question.get('supporting_facts', []):
        doc_title, sent_idx = sf[0], sf[1]
        if doc_title in doc_index:
            sentences = doc_index[doc_title]
            if sent_idx < len(sentences):
                sf_sentences.append(sentences[sent_idx])

    return sf_sentences


def index_question_context(question: dict, hermes_url: str) -> bool:
    """Index a question's context documents into HERMES."""
    doc_index = create_document_index(question)

    for title, content in doc_index.items():
        # Create a document with the context
        payload = {
            "content": content,
            "filename": f"hotpotqa_{title}.txt",
            "metadata": {"title": title, "source": "hotpotqa"}
        }

        try:
            req = urllib.request.Request(
                f"{hermes_url}/ingest/text",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            print(f"  Warning: Failed to index {title}: {e}")
            return False

    return True


def search_hermes(query: str, hermes_url: str, top_k: int = 5,
                  endpoint: str = "search/v2") -> tuple[list[dict], float]:
    """Search HERMES and return results with latency."""
    url = f"{hermes_url}/{endpoint}?query={urllib.parse.quote(query)}&top_k={top_k}"

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read())
        latency = (time.time() - start) * 1000
        return data.get("results", []), latency
    except Exception as e:
        return [], 0


def evaluate_retrieval(
    retrieved: list[dict],
    gold_docs: set[str],
    sf_sentences: list[str]
) -> dict:
    """Evaluate retrieval results against gold standard."""
    if not retrieved:
        return {
            "doc_precision": 0.0,
            "doc_recall": 0.0,
            "doc_f1": 0.0,
            "sf_recall": 0.0,
            "retrieved_docs": []
        }

    # Extract document titles from retrieved chunks
    retrieved_docs = set()
    retrieved_content = []

    for chunk in retrieved:
        # Try to match filename to document title
        filename = chunk.get("filename", "")
        content = chunk.get("content", "")
        retrieved_content.append(content)

        # Check if any gold doc content is in the retrieved chunk
        for gold_doc in gold_docs:
            if gold_doc.lower() in filename.lower() or gold_doc.lower() in content.lower():
                retrieved_docs.add(gold_doc)

    # Document-level metrics
    true_positives = len(retrieved_docs & gold_docs)

    doc_precision = true_positives / len(retrieved) if retrieved else 0
    doc_recall = true_positives / len(gold_docs) if gold_docs else 0
    doc_f1 = (2 * doc_precision * doc_recall / (doc_precision + doc_recall)
              if (doc_precision + doc_recall) > 0 else 0)

    # Supporting fact recall
    sf_found = 0
    all_retrieved_text = " ".join(retrieved_content).lower()
    for sf in sf_sentences:
        # Check if supporting fact sentence appears in any retrieved content
        if sf.lower() in all_retrieved_text:
            sf_found += 1

    sf_recall = sf_found / len(sf_sentences) if sf_sentences else 0

    return {
        "doc_precision": doc_precision,
        "doc_recall": doc_recall,
        "doc_f1": doc_f1,
        "sf_recall": sf_recall,
        "retrieved_docs": list(retrieved_docs)
    }


def run_benchmark(
    data_path: str,
    hermes_url: str = "http://localhost:8780",
    sample_size: int = 200,
    top_k: int = 5,
    endpoint: str = "search/v2",
    verbose: bool = False
) -> list[QuestionResult]:
    """Run the full HotpotQA benchmark."""

    print(f"Loading HotpotQA dataset from {data_path}...")
    questions = load_hotpotqa(data_path, sample_size)
    print(f"Loaded {len(questions)} questions")

    results = []

    for i, q in enumerate(questions):
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:60]}...")
        else:
            print(f"\r  Processing {i+1}/{len(questions)}...", end="", flush=True)

        # Get gold data
        gold_docs = get_gold_documents(q)
        sf_sentences = get_supporting_fact_sentences(q)

        # Search using the question
        retrieved, latency = search_hermes(
            q['question'], hermes_url, top_k, endpoint
        )

        # Evaluate
        metrics = evaluate_retrieval(retrieved, gold_docs, sf_sentences)

        result = QuestionResult(
            question_id=q['_id'],
            question=q['question'],
            question_type=q['type'],
            answer=q['answer'],
            gold_docs=list(gold_docs),
            supporting_facts=q.get('supporting_facts', []),
            retrieved_docs=metrics['retrieved_docs'],
            retrieved_content=[r.get('content', '')[:200] for r in retrieved],
            doc_precision=metrics['doc_precision'],
            doc_recall=metrics['doc_recall'],
            doc_f1=metrics['doc_f1'],
            sf_recall=metrics['sf_recall'],
            latency_ms=latency
        )
        results.append(result)

        if verbose:
            print(f"  P={result.doc_precision:.0%} R={result.doc_recall:.0%} "
                  f"F1={result.doc_f1:.0%} SF={result.sf_recall:.0%} "
                  f"({result.latency_ms:.0f}ms)")

    print()
    return results


def print_results(results: list[QuestionResult], title: str):
    """Print benchmark results summary."""
    if not results:
        print("No results to display")
        return

    # Overall metrics
    avg_precision = sum(r.doc_precision for r in results) / len(results)
    avg_recall = sum(r.doc_recall for r in results) / len(results)
    avg_f1 = sum(r.doc_f1 for r in results) / len(results)
    avg_sf_recall = sum(r.sf_recall for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    # By type
    bridge = [r for r in results if r.question_type == 'bridge']
    comparison = [r for r in results if r.question_type == 'comparison']

    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()
    print(f"Overall Results (N={len(results)}):")
    print(f"  Document Precision:     {avg_precision:.1%}")
    print(f"  Document Recall:        {avg_recall:.1%}")
    print(f"  Document F1:            {avg_f1:.1%}")
    print(f"  Supporting Fact Recall: {avg_sf_recall:.1%}")
    print(f"  Average Latency:        {avg_latency:.0f}ms")
    print()

    if bridge:
        bridge_p = sum(r.doc_precision for r in bridge) / len(bridge)
        bridge_r = sum(r.doc_recall for r in bridge) / len(bridge)
        bridge_f1 = sum(r.doc_f1 for r in bridge) / len(bridge)
        print(f"Bridge Questions (n={len(bridge)}):")
        print(f"  P={bridge_p:.1%} R={bridge_r:.1%} F1={bridge_f1:.1%}")

    if comparison:
        comp_p = sum(r.doc_precision for r in comparison) / len(comparison)
        comp_r = sum(r.doc_recall for r in comparison) / len(comparison)
        comp_f1 = sum(r.doc_f1 for r in comparison) / len(comparison)
        print(f"Comparison Questions (n={len(comparison)}):")
        print(f"  P={comp_p:.1%} R={comp_r:.1%} F1={comp_f1:.1%}")

    print()
    print("-" * 70)
    print("SEAL-RAG Reference (GPT-4o, k=3, HotpotQA N=1000):")
    print("  Precision: 89%  |  Recall: 68%  |  F1: 75%")
    print("-" * 70)

    # Delta vs SEAL-RAG
    seal_p, seal_r, seal_f1 = 0.89, 0.68, 0.75
    print()
    print(f"Delta vs SEAL-RAG:")
    print(f"  Precision: {(avg_precision - seal_p)*100:+.1f}pp")
    print(f"  Recall:    {(avg_recall - seal_r)*100:+.1f}pp")
    print(f"  F1:        {(avg_f1 - seal_f1)*100:+.1f}pp")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="HotpotQA Benchmark for HERMES")
    parser.add_argument("--data", default="hotpot_dev_distractor_v1.json",
                        help="Path to HotpotQA dataset")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of questions to sample (default: 200)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve (default: 5)")
    parser.add_argument("--endpoint", default="search/v2",
                        help="Search endpoint (search or search/v2)")
    parser.add_argument("--url", default="http://localhost:8780",
                        help="HERMES URL")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" HotpotQA Benchmark for HERMES")
    print(" Comparing against SEAL-RAG reference metrics")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Sample size: {args.sample}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Endpoint: {args.endpoint}")
    print()

    # Check HERMES health
    try:
        urllib.request.urlopen(f"{args.url}/health", timeout=5)
        print("HERMES is healthy. Starting benchmark...")
    except:
        print("ERROR: HERMES is not available")
        return

    # Run benchmark
    results = run_benchmark(
        args.data,
        args.url,
        args.sample,
        args.top_k,
        args.endpoint,
        args.verbose
    )

    # Print results
    print_results(results, f"HERMES {args.endpoint} (k={args.top_k})")

    # Save results if requested
    if args.output:
        output_data = {
            "config": {
                "sample_size": args.sample,
                "top_k": args.top_k,
                "endpoint": args.endpoint
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "question_type": r.question_type,
                    "doc_precision": r.doc_precision,
                    "doc_recall": r.doc_recall,
                    "doc_f1": r.doc_f1,
                    "sf_recall": r.sf_recall,
                    "latency_ms": r.latency_ms
                }
                for r in results
            ],
            "summary": {
                "avg_precision": sum(r.doc_precision for r in results) / len(results),
                "avg_recall": sum(r.doc_recall for r in results) / len(results),
                "avg_f1": sum(r.doc_f1 for r in results) / len(results),
                "avg_sf_recall": sum(r.sf_recall for r in results) / len(results),
            }
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
