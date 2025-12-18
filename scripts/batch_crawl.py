#!/usr/bin/env python3
"""Batch crawl papers using the API."""
import requests
import time
import sys

API = "http://localhost:8780"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
BATCH = 100

# Diverse keywords to find unique papers
KEYWORD_SETS = [
    "language model,GPT,BERT",
    "transformer,attention mechanism",
    "reinforcement learning,policy gradient",
    "computer vision,image classification",
    "natural language processing,NLP,text",
    "speech recognition,ASR,audio",
    "neural network,deep learning",
    "generative model,GAN,VAE",
    "graph neural network,GNN",
    "recommendation system,collaborative filtering",
    "time series,forecasting,prediction",
    "object detection,YOLO,detection",
    "semantic segmentation,image segmentation",
    "machine translation,NMT,translation",
    "question answering,QA,reading comprehension",
    "sentiment analysis,opinion mining",
    "named entity recognition,NER",
    "text classification,document classification",
    "knowledge distillation,model compression",
    "self-supervised learning,contrastive learning",
    "meta-learning,few-shot learning",
    "federated learning,distributed learning",
    "adversarial attack,robustness",
    "explainable AI,interpretability,XAI",
    "AutoML,neural architecture search",
]

CATEGORIES = "cs.LG,cs.CL,cs.CV,cs.AI,cs.NE,stat.ML"

def get_count():
    try:
        r = requests.get(f"{API}/stats", timeout=10)
        return r.json()["documents"]
    except:
        return 0

def crawl(keywords, days_back):
    try:
        r = requests.post(
            f"{API}/crawler/run",
            params={
                "keywords": keywords,
                "categories": CATEGORIES,
                "max_papers": BATCH,
                "days_back": days_back,
                "collection": "ai-papers"
            },
            timeout=300
        )
        return r.json().get("papers_ingested", 0)
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    current = get_count()
    print(f"Starting at {current} documents, target {TARGET}")

    batch_num = 0
    days_back = 180  # Start with 6 months

    while current < TARGET:
        batch_num += 1
        kw_idx = (batch_num - 1) % len(KEYWORD_SETS)
        keywords = KEYWORD_SETS[kw_idx]

        print(f"\nBatch {batch_num}: {current}/{TARGET} | Keywords: {keywords[:30]}... | Days: {days_back}")

        ingested = crawl(keywords, days_back)
        print(f"  Ingested: {ingested}")

        current = get_count()

        if ingested == 0:
            # Try extending days back
            days_back = min(days_back + 30, 730)  # Up to 2 years
            print(f"  No new papers, extending to {days_back} days")

        # Brief pause
        time.sleep(2)

    print(f"\n=== COMPLETE: {current} documents ===")

if __name__ == "__main__":
    main()
