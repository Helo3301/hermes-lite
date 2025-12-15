"""Evaluation metrics for Hermes RAG benchmarking."""

import re
import string
from collections import Counter
from typing import Optional


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def entity_coverage(
    retrieved_chunks: list[dict],
    expected_entities: list[str]
) -> float:
    """Measure what fraction of expected entities appear in retrieved chunks."""
    if not expected_entities:
        return 1.0

    all_text = " ".join(chunk.get("content", "") for chunk in retrieved_chunks).lower()

    found = 0
    for entity in expected_entities:
        if entity.lower() in all_text:
            found += 1

    return found / len(expected_entities)


def chunk_relevance_score(
    retrieved_chunks: list[dict],
    query: str,
    expected_concepts: list[str]
) -> float:
    """Score relevance of retrieved chunks based on expected concepts."""
    if not retrieved_chunks or not expected_concepts:
        return 0.0

    query_lower = query.lower()
    scores = []

    for chunk in retrieved_chunks:
        content = chunk.get("content", "").lower()

        # Count concept matches
        concept_matches = sum(1 for c in expected_concepts if c.lower() in content)
        concept_score = concept_matches / len(expected_concepts)

        # Combine with retrieval score if available
        retrieval_score = chunk.get("score", 0.5)

        # Weighted combination
        combined = 0.6 * concept_score + 0.4 * min(retrieval_score, 1.0)
        scores.append(combined)

    return sum(scores) / len(scores) if scores else 0.0


def supporting_fact_recall(
    retrieved_chunks: list[dict],
    supporting_facts: list[dict],
) -> float:
    """Measure how many supporting facts are covered by retrieved chunks."""
    if not supporting_facts:
        return 1.0

    all_text = " ".join(chunk.get("content", "") for chunk in retrieved_chunks).lower()

    found = 0
    for fact in supporting_facts:
        # Check if the title/topic of the supporting fact is mentioned
        if isinstance(fact, dict):
            title = fact.get("title", "").lower()
            if title and title in all_text:
                found += 1
        elif isinstance(fact, str):
            if fact.lower() in all_text:
                found += 1

    return found / len(supporting_facts) if supporting_facts else 0.0


def latency_percentile(latencies: list[float], percentile: int) -> float:
    """Compute the given percentile of latencies."""
    if not latencies:
        return 0.0

    sorted_latencies = sorted(latencies)
    index = int((percentile / 100) * len(sorted_latencies))
    index = min(index, len(sorted_latencies) - 1)

    return sorted_latencies[index]


def compute_retrieval_metrics(
    query: str,
    retrieved_chunks: list[dict],
    expected_entities: Optional[list[str]] = None,
    expected_concepts: Optional[list[str]] = None,
    supporting_facts: Optional[list[dict]] = None,
    latency_ms: Optional[float] = None,
) -> dict:
    """Compute comprehensive retrieval metrics for a single query."""
    metrics = {
        "num_chunks": len(retrieved_chunks),
    }

    if expected_entities:
        metrics["entity_coverage"] = entity_coverage(retrieved_chunks, expected_entities)

    if expected_concepts:
        metrics["concept_relevance"] = chunk_relevance_score(
            retrieved_chunks, query, expected_concepts
        )

    if supporting_facts:
        metrics["supporting_fact_recall"] = supporting_fact_recall(
            retrieved_chunks, supporting_facts
        )

    if latency_ms is not None:
        metrics["latency_ms"] = latency_ms

    # Compute average retrieval score
    if retrieved_chunks:
        scores = [c.get("score", 0) for c in retrieved_chunks]
        metrics["avg_retrieval_score"] = sum(scores) / len(scores)
        metrics["max_retrieval_score"] = max(scores) if scores else 0

    return metrics


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate metrics across multiple queries."""
    if not all_metrics:
        return {}

    # Collect all metric names
    metric_names = set()
    for m in all_metrics:
        metric_names.update(m.keys())

    aggregated = {}

    for name in metric_names:
        values = [m[name] for m in all_metrics if name in m and m[name] is not None]
        if values:
            aggregated[f"{name}_mean"] = sum(values) / len(values)
            aggregated[f"{name}_min"] = min(values)
            aggregated[f"{name}_max"] = max(values)

            if name == "latency_ms":
                aggregated["latency_p50"] = latency_percentile(values, 50)
                aggregated["latency_p95"] = latency_percentile(values, 95)
                aggregated["latency_p99"] = latency_percentile(values, 99)

    aggregated["total_queries"] = len(all_metrics)

    return aggregated
