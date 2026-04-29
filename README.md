# HERMES-Lite

Hybrid RAG library for SQLite-backed corpora. Combines dense (bge-m3) + sparse (SQLite FTS5) retrieval with reciprocal rank fusion, optional cross-encoder reranking, and context compression.

Built for cases where you want production-grade hybrid retrieval without spinning up a vector database service.

## What's measured

On a 176,000-chunk internal documentation corpus, single-node deployment:

| Path | P50 | P95 |
|---|---|---|
| Hybrid search (semantic + FTS, no rerank) | **232 ms** | 259 ms |
| Vector-only, cached embeddings | **27 ms** | — |
| Full query incl. embedding (Ollama) | ~123 ms | — |
| + cross-encoder reranking | +~1000 ms | — |
| + HyDE query expansion | +~200–500 ms | — |

INT8 vector quantization:

- **3.97× memory reduction** (19.5 MB → 4.9 MB across 176k chunks at 1024-dim)
- **99.98% mean cosine similarity** preserved
- Rank correlation 0.9996 (search ordering essentially unchanged)

Honest caveat: these numbers come from internal corpora. A public evaluation against standard benchmarks (HotpotQA, BEIR) is roadmap, not done yet.

## What it actually is

- **SQLite + sqlite-vec** for vector storage — no external service
- **FTS5** for keyword search
- **bge-m3** embeddings via Ollama, Matryoshka-capable (scale 1024 → 256 dim if you need to)
- **RRF fusion** combining semantic (0.65) + keyword (0.35) results
- **Optional features**, all with off switches:
  - Cross-encoder reranking (`bge-reranker-v2-m3`) — off by default, ~1 s cost
  - HyDE query expansion — off by default, ~200–500 ms cost, +15–20% recall on complex queries
  - Context compression — on by default, 5–10× token reduction
- **REST API** (FastAPI) and **MCP server** for Claude Code integration

## Quickstart

```bash
git clone https://github.com/Helo3301/hermes-lite
cd hermes-lite
pip install -r requirements.txt

# Run Ollama (local or remote) and pull the embedding model
docker run -d -p 11434:11434 ollama/ollama
docker exec -it ollama ollama pull bge-m3

# Edit config.yaml — point ollama.host at your instance

# Ingest documents (PDF / Markdown / HTML)
python -m app.ingest --path /path/to/your/docs/

# Run the API
uvicorn app.main:app --port 8780
```

```python
import requests

r = requests.post("http://localhost:8780/search", json={
    "query": "what is reciprocal rank fusion",
    "top_k": 10,
})
print(r.json())
```

## Usage patterns

**Default — fastest, fits most cases**

```python
results = search.search(query, top_k=10, rerank=False)
```

Hybrid (semantic + keyword), context-compressed, ~230 ms.

**High-precision**

```python
results = search.search(query, top_k=10, rerank=True)
```

Adds cross-encoder reranking. ~1.2 s end-to-end. Use when offline precision matters more than latency.

**LLM-optimized (token-conscious)**

```python
results = search.search_for_llm(query, max_tokens=2000)
```

Returns context compressed to fit a token budget. 5–10× compression typical, top-2 chunks preserved uncompressed.

**Memory-constrained**

```python
from app.search_lite import LiteSearch
search = LiteSearch(
    db_path="data/hermes.db",
    embed_fn=...,
    preload=True,
    fast_search=False,  # INT8-only, no pre-dequantize
)
```

INT8-only: ~174 MB RAM for 176k chunks. With `fast_search=True`: ~950 MB RAM, faster searches.

## Architecture

```
query → embed (bge-m3) ─┐
                         ├─→ RRF fusion → [optional rerank] → [optional compression] → result
query → FTS5 lookup ─────┘
```

| Component | File | Notes |
|---|---|---|
| Database | `app/database.py` | SQLite + sqlite-vec extension |
| Embedding | `app/embed_v2.py` | Ollama + bge-m3, Matryoshka-capable |
| Search (full) | `app/search.py` | Hybrid + reranker + HyDE + compression |
| Search (lite) | `app/search_lite.py` | INT8-quantized, in-memory, optimized for speed |
| Ingestion | `app/ingest.py` | PDF / Markdown / HTML, dedup via content hash |
| API | `app/main.py` | FastAPI: `/search`, `/search/lite`, `/search/llm` |
| MCP server | `app/mcp_server.py` | Claude Code integration |

Configuration lives in `config.yaml`. Every optional feature has an `enabled:` toggle and a comment explaining the default.

## Trade-offs

**What works**

- Self-contained: SQLite + Ollama, no Pinecone / Weaviate / Qdrant to operate
- Sub-second hybrid search on six-figure chunk counts
- INT8 quantization with measured fidelity (99.98% similarity preserved)
- Every expensive feature has an off switch

**What doesn't**

- **No HNSW indexing.** Brute-force cosine is fine to ~500k chunks; beyond that you want a real vector index.
- **Embedding latency dominates.** Ollama on bge-m3 is ~80 ms per query; cached corpora speed up search, but every fresh query pays the embedding cost.
- **No GPU-optional path baked in.** Reranker and embedding prefer GPU; CPU works but is slower.
- **Reranker and HyDE are off by default.** Their latency cost (~1 s and ~300 ms) wasn't worth it on the corpora I measured. Re-enable them if your precision needs justify it.

## What's not in the repo yet

- Public evaluation against open benchmarks (HotpotQA, BEIR)
- Sample corpus / "hello world" dataset
- HNSW index option for >500k chunks
- ONNX embedding fallback (Ollama-free path)

These are roadmap items. The current code is honest about what it does and doesn't.

## License

See [LICENSE](LICENSE).
