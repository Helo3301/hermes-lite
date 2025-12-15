#!/usr/bin/env python3
"""Download standard RAG benchmark datasets."""

import json
import random
from pathlib import Path
from datasets import load_dataset

BENCHMARK_DIR = Path(__file__).parent / "standard"
SAMPLE_SIZE = 500

def download_hotpotqa():
    """Download HotpotQA distractor setting."""
    print("Downloading HotpotQA...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    # Sample and format
    samples = []
    indices = random.sample(range(len(dataset)), min(SAMPLE_SIZE, len(dataset)))

    for idx in indices:
        item = dataset[idx]
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": {
                "titles": item["supporting_facts"]["title"],
                "sent_ids": item["supporting_facts"]["sent_id"],
            },
            "context": [
                {"title": t, "sentences": s}
                for t, s in zip(item["context"]["title"], item["context"]["sentences"])
            ],
            "type": item["type"],
            "level": item["level"],
        })

    output_path = BENCHMARK_DIR / "hotpotqa_500.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} HotpotQA samples to {output_path}")
    return samples


def download_2wikimultihop():
    """Download 2WikiMultiHopQA."""
    print("Downloading 2WikiMultiHopQA...")
    # This dataset might need different loading
    try:
        dataset = load_dataset("scholarly-shadows/2WikiMultiHopQA", split="validation")
    except:
        # Fallback: try alternative source
        try:
            dataset = load_dataset("xanhho/2WikiMultihopQA", split="dev")
        except Exception as e:
            print(f"Could not load 2WikiMultiHopQA: {e}")
            print("Creating placeholder...")
            # Create placeholder with structure
            samples = [{"id": f"2wiki_{i}", "question": "", "answer": "", "note": "placeholder"}
                      for i in range(SAMPLE_SIZE)]
            output_path = BENCHMARK_DIR / "2wikimultihop_500.json"
            with open(output_path, "w") as f:
                json.dump(samples, f, indent=2)
            return samples

    samples = []
    indices = random.sample(range(len(dataset)), min(SAMPLE_SIZE, len(dataset)))

    for idx in indices:
        item = dataset[idx]
        samples.append({
            "id": item.get("_id", f"2wiki_{idx}"),
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": item.get("supporting_facts", []),
            "context": item.get("context", []),
            "type": item.get("type", "unknown"),
        })

    output_path = BENCHMARK_DIR / "2wikimultihop_500.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} 2WikiMultiHopQA samples to {output_path}")
    return samples


def download_natural_questions():
    """Download Natural Questions."""
    print("Downloading Natural Questions...")
    dataset = load_dataset("natural_questions", "default", split="validation")

    samples = []
    count = 0

    for item in dataset:
        if count >= SAMPLE_SIZE:
            break

        # Extract short answer if available
        annotations = item["annotations"]
        short_answers = annotations["short_answers"][0] if annotations["short_answers"] else []

        if short_answers and short_answers["start_token"]:
            # Get answer text from document tokens
            doc_tokens = item["document"]["tokens"]["token"]
            start = short_answers["start_token"][0]
            end = short_answers["end_token"][0]
            answer = " ".join(doc_tokens[start:end])

            samples.append({
                "id": item["id"],
                "question": item["question"]["text"],
                "answer": answer,
                "document_title": item["document"]["title"],
            })
            count += 1

    output_path = BENCHMARK_DIR / "nq_500.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} NQ samples to {output_path}")
    return samples


if __name__ == "__main__":
    random.seed(42)  # Reproducibility

    print("=" * 60)
    print("Downloading Standard RAG Benchmarks")
    print("=" * 60)

    download_hotpotqa()
    download_2wikimultihop()
    download_natural_questions()

    print("\nDone!")
