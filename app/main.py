"""HERMES-Lite: Main FastAPI application."""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .database import Database, compute_hash
from .embed_v2 import EmbeddingClientV2 as EmbeddingClient
from .ingest import ingest_document
from .search import SearchEngine, Reranker
from .search_v2 import SearchV2Pipeline, SearchV2Config, create_search_v2_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.environ.get('HERMES_CONFIG', '/app/config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Global instances
db: Database = None
embed_client: EmbeddingClient = None
search_engine: SearchEngine = None
reranker: Reranker = None
search_v2: SearchV2Pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global db, embed_client, search_engine, reranker

    logger.info("Starting HERMES-Lite...")

    # Initialize database
    db_path = config['database']['path']
    db = Database(db_path)
    logger.info(f"Database initialized at {db_path}")

    # Initialize embedding client (bge-m3 with Matryoshka support)
    embed_client = EmbeddingClient(
        ollama_host=config['ollama']['host'],
        model=config['ollama']['embed_model'],
        default_dim=config['ollama'].get('embed_dim', 1024),
        batch_size=config['ollama']['embed_batch_size']
    )
    logger.info(f"Embedding client initialized: {config['ollama']['embed_model']} ({config['ollama'].get('embed_dim', 1024)}d)")

    # Initialize reranker if enabled
    if config['reranker']['enabled']:
        reranker = Reranker(
            model_name=config['reranker']['model'],
            device=config['reranker']['device'],
            use_fp16=config['reranker']['use_fp16']
        )
        logger.info(f"Reranker configured: {config['reranker']['model']}")
    else:
        reranker = None

    # Initialize search engine with HyDE and compression support
    hyde_config = config.get('hyde', {})
    compression_config = config.get('compression', {})
    search_engine = SearchEngine(
        database=db,
        embed_client=embed_client,
        reranker=reranker,
        semantic_weight=config['search']['semantic_weight'],
        keyword_weight=config['search']['keyword_weight'],
        rrf_k=config['search']['rrf_k'],
        hyde_enabled=hyde_config.get('enabled', True),
        hyde_model=hyde_config.get('model', 'llama3.2:3b'),
        hyde_blend_weight=hyde_config.get('blend_weight', 0.7),
        ollama_host=config['ollama']['host'],
        compression_enabled=compression_config.get('enabled', True),
        compression_strategy=compression_config.get('strategy', 'sentence'),
        compression_max_tokens=compression_config.get('max_tokens', 2000)
    )
    logger.info(f"Search engine initialized (HyDE: {hyde_config.get('enabled', True)}, Compression: {compression_config.get('enabled', True)})")

    # Initialize v2 search pipeline
    v2_config = config.get('search_v2', {})
    search_v2 = create_search_v2_fn(
        search_engine,
        config=SearchV2Config(
            use_query_analysis=v2_config.get('use_query_analysis', True),
            decompose_multi_hop=v2_config.get('decompose_multi_hop', True),
            use_iterative=v2_config.get('use_iterative', True),
            max_iterations=v2_config.get('max_iterations', 3),
            budget=v2_config.get('budget', 10),
            use_adaptive=v2_config.get('use_adaptive', True),
            min_k=v2_config.get('min_k', 5),
            max_k=v2_config.get('max_k', 30),
            confidence_threshold=v2_config.get('confidence_threshold', 0.6),
            detect_contradictions=v2_config.get('detect_contradictions', True),
            surface_contradictions=v2_config.get('surface_contradictions', True),
        ),
        use_llm=v2_config.get('use_llm', False),
    )
    logger.info("Search v2 pipeline initialized")

    # Make search_v2 available globally
    globals()['search_v2'] = search_v2

    # Ensure embedding model is loaded
    embed_client.ensure_model_loaded()

    yield

    # Cleanup
    logger.info("Shutting down HERMES-Lite...")
    if db:
        db.close()
    if embed_client:
        embed_client.close()


app = FastAPI(
    title="HERMES-Lite",
    description="Minimal RAG ingestion and query pipeline",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/")
async def index():
    """Serve the web UI."""
    return FileResponse("/app/static/index.html")


@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=512),
    overlap: int = Form(default=64)
):
    """
    Ingest a document into the knowledge base.

    Args:
        file: The document file to ingest
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        Document ID and chunk count
    """
    try:
        # Read file content
        content = await file.read()
        filename = file.filename
        file_path = Path(filename)

        logger.info(f"Ingesting: {filename} ({len(content)} bytes)")

        # Check for duplicates
        content_hash = compute_hash(content.decode('utf-8', errors='replace'))
        if db.document_exists(content_hash):
            return JSONResponse(
                status_code=409,
                content={"error": "Document already exists", "filename": filename}
            )

        # Ingest: convert, clean, chunk
        clean_md, chunks = ingest_document(
            file_path,
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_size=config['chunking']['min_size']
        )

        # Store document
        doc_id = db.insert_document(
            filename=filename,
            source_path=None,
            doc_type=file_path.suffix.lower().lstrip('.'),
            clean_md=clean_md,
            content_hash=content_hash
        )

        # Embed chunks
        logger.info(f"Embedding {len(chunks)} chunks...")
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_client.embed_batch(chunk_texts)

        # Store chunks with embeddings
        chunks_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunks_data.append({
                'doc_id': doc_id,
                'chunk_index': i,
                'content': chunk.text,
                'embedding': embedding,
                'token_count': chunk.tokens,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char
            })

        db.insert_chunks_batch(chunks_data)

        logger.info(f"Ingested {filename}: {len(chunks)} chunks")

        return {
            "success": True,
            "filename": filename,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "total_tokens": sum(c.tokens for c in chunks)
        }

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    doc_filter: str = None,
    expand_query: bool = True,
    diversity: bool = True,
    max_per_doc: int = 3,
    recency_boost: bool = False,
    use_hyde: bool = None
):
    """
    Search the knowledge base with advanced retrieval features.

    Args:
        query: Natural language search query
        top_k: Number of results to return
        rerank: Whether to use cross-encoder reranking
        doc_filter: Optional filename pattern to filter results
        expand_query: Expand query with synonyms/acronyms (RAG->retrieval augmented generation)
        diversity: Apply document diversity (prevents single doc from dominating)
        max_per_doc: Max chunks per document in first pass (soft limit)
        recency_boost: Boost newer documents slightly
        use_hyde: Use HyDE (Hypothetical Document Embeddings) for improved recall.
                  None = use server default, True/False = override

    Returns:
        List of search results
    """
    try:
        results = search_engine.search(
            query=query,
            top_k=top_k,
            rerank=rerank,
            doc_filter=doc_filter,
            use_query_expansion=expand_query,
            use_diversity=diversity,
            max_per_doc=max_per_doc,
            use_recency_weight=recency_boost,
            use_hyde=use_hyde
        )

        return {
            "query": query,
            "results": [
                {
                    "id": r['id'],
                    "content": r['content'],
                    "filename": r['filename'],
                    "doc_id": r['doc_id'],
                    "score": r.get('score'),
                    "match_count": r.get('_match_count', 1)
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/llm")
async def search_for_llm(
    query: str,
    top_k: int = 10,
    max_tokens: int = 2000,
    compress: bool = True,
    rerank: bool = True,
    doc_filter: str = None,
    use_hyde: bool = None
):
    """
    Search optimized for LLM consumption with context compression.

    This endpoint returns compressed context ready to be passed to an LLM,
    reducing token usage while preserving the most relevant information.

    Args:
        query: Natural language search query
        top_k: Number of results to retrieve before compression
        max_tokens: Target token limit for compressed output
        compress: Whether to apply compression (default True)
        rerank: Whether to use cross-encoder reranking
        doc_filter: Optional filename pattern to filter results
        use_hyde: Use HyDE query expansion (None = server default)

    Returns:
        Compressed context text with statistics
    """
    try:
        result = search_engine.search_for_llm(
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
            compress=compress,
            rerank=rerank,
            doc_filter=doc_filter,
            use_hyde=use_hyde
        )

        return {
            "query": result["query"],
            "context": result["text"],
            "compression_stats": result["stats"],
            "source_count": len(result["results"]),
            "sources": [
                {
                    "id": r.get('id'),
                    "filename": r.get('filename'),
                    "doc_id": r.get('doc_id')
                }
                for r in result["results"]
            ]
        }

    except Exception as e:
        logger.exception(f"LLM search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/v2")
async def search_v2_endpoint(
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    detect_contradictions: bool = True,
    doc_filter: str = None,
):
    """
    Advanced v2 search with query intelligence and iterative retrieval.

    Features:
    - Query classification (simple, multi-hop, comparative, exploratory)
    - Query decomposition for complex queries
    - Entity extraction
    - Iterative gap-fill retrieval (SEAL-RAG style)
    - Adaptive retrieval depth based on confidence
    - Contradiction detection between sources

    Args:
        query: Natural language search query
        top_k: Number of results to return
        rerank: Whether to use cross-encoder reranking
        detect_contradictions: Whether to detect conflicting claims
        doc_filter: Optional filename pattern to filter results

    Returns:
        Enhanced search results with analysis metadata
    """
    try:
        global search_v2

        if search_v2 is None:
            raise HTTPException(
                status_code=503,
                detail="Search v2 pipeline not initialized"
            )

        # Build kwargs for base search
        kwargs = {}
        if doc_filter:
            kwargs['doc_filter'] = doc_filter

        # Execute v2 search
        result = search_v2.search(
            query=query,
            top_k=top_k,
            rerank=rerank,
            detect_contradictions=detect_contradictions,
            **kwargs
        )

        return {
            "query": result.query,
            "query_type": result.query_type,
            "sub_queries": result.sub_queries,
            "entities": result.entities,
            "results": [
                {
                    "id": r.get('id'),
                    "content": r.get('content'),
                    "filename": r.get('filename'),
                    "doc_id": r.get('doc_id'),
                    "score": r.get('score'),
                    "v2_score": r.get('v2_score'),
                }
                for r in result.results
            ],
            "confidence": {
                "score": result.confidence,
                "explanation": result.confidence_explanation,
            },
            "contradictions": result.contradictions,
            "metadata": {
                "iterations": result.iterations,
                "timing_ms": result.timing_ms,
                "status": result.status,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Search v2 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    docs = db.get_documents()
    return {"documents": docs}


@app.get("/documents/{doc_id}/content")
async def get_document_content(doc_id: int):
    """Get full markdown content of a document."""
    doc = db.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc_id,
        "filename": doc['filename'],
        "source_url": doc.get('source_url'),
        "ingested_at": doc['ingested_at'],
        "chunk_count": doc['chunk_count'],
        "content": doc['clean_md']
    }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document and its chunks."""
    success = db.delete_document(doc_id)
    if success:
        return {"success": True, "deleted": doc_id}
    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    stats = db.get_stats()
    return stats


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.2.0"}


# ==================== Collection Endpoints ====================

@app.get("/collections")
async def list_collections():
    """List all collections."""
    collections = db.get_collections()
    return {"collections": collections}


@app.post("/collections")
async def create_collection(name: str, description: str = ""):
    """Create a new collection."""
    try:
        collection_id = db.create_collection(name, description)
        return {"success": True, "id": collection_id, "name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{collection_name}/documents")
async def list_collection_documents(collection_name: str):
    """List documents in a collection."""
    collection_id = db.get_collection_id(collection_name)
    if not collection_id:
        raise HTTPException(status_code=404, detail="Collection not found")
    docs = db.get_documents_by_collection(collection_id)
    return {"collection": collection_name, "documents": docs}


# ==================== Whitelist Endpoints ====================

@app.get("/whitelist")
async def list_whitelist():
    """List all whitelisted domains."""
    domains = db.get_whitelist()
    return {"domains": domains}


@app.post("/whitelist")
async def add_to_whitelist(domain: str):
    """Add a domain to the whitelist."""
    try:
        db.add_domain_to_whitelist(domain)
        return {"success": True, "domain": domain}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/whitelist/{domain}")
async def remove_from_whitelist(domain: str):
    """Remove a domain from the whitelist."""
    success = db.remove_domain_from_whitelist(domain)
    if success:
        return {"success": True, "removed": domain}
    raise HTTPException(status_code=404, detail="Domain not found")


# ==================== Crawler Endpoints ====================

@app.post("/crawler/run")
async def run_crawler(
    keywords: str = "RAG,LLM,retrieval augmented",
    categories: str = "cs.AI,cs.LG,cs.CL",
    collection: str = "ai-papers",
    max_papers: int = 5,
    days_back: int = 7
):
    """
    Run the paper crawler manually.

    Args:
        keywords: Comma-separated search keywords
        categories: Comma-separated arXiv categories
        collection: Target collection name
        max_papers: Maximum papers to fetch
        days_back: How many days back to search
    """
    from .crawler import ArxivCrawler

    try:
        crawler = ArxivCrawler(db, embed_client)
        results = crawler.crawl_recent_papers(
            keywords=[k.strip() for k in keywords.split(',')],
            categories=[c.strip() for c in categories.split(',')],
            collection_name=collection,
            max_papers=max_papers,
            days_back=days_back
        )
        crawler.close()

        return {
            "success": True,
            "papers_ingested": len(results),
            "papers": results
        }
    except Exception as e:
        logger.exception(f"Crawler failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Digest Endpoints ====================

# ==================== n8n Webhook Endpoints ====================

@app.post("/webhook/n8n")
async def n8n_webhook(action: str = "crawl"):
    """
    Webhook endpoint for n8n to trigger actions.

    Actions:
        - crawl: Run the daily paper crawler with default settings
        - digest: Generate and return the daily digest

    Returns:
        Action results
    """
    if action == "crawl":
        from .crawler import ArxivCrawler

        crawler_config = config.get('crawler', {})
        crawler = ArxivCrawler(db, embed_client)
        try:
            results = crawler.crawl_recent_papers(
                keywords=crawler_config.get('default_keywords', ['RAG', 'LLM agents']),
                categories=crawler_config.get('default_categories', ['cs.AI', 'cs.LG', 'cs.CL']),
                collection_name='ai-papers',
                max_papers=crawler_config.get('max_papers_per_day', 10),
                days_back=crawler_config.get('days_back', 7)
            )
            crawler.close()
            return {
                "status": "ok",
                "action": "crawl",
                "papers_ingested": len(results),
                "papers": [{"title": p['title'], "url": p['url']} for p in results]
            }
        except Exception as e:
            crawler.close()
            raise HTTPException(status_code=500, detail=str(e))

    elif action == "digest":
        from .digest import DigestGenerator

        generator = DigestGenerator(
            db,
            ollama_host=config['ollama']['host'],
            ollama_model=config['ollama'].get('chat_model', 'llama3.2'),
            use_local=True
        )
        digest = generator.generate_digest(
            collection_name='ai-papers',
            hours_back=24,
            summarize_papers=False
        )
        return {
            "status": "ok",
            "action": "digest",
            "markdown": generator.format_digest_markdown(digest)
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@app.get("/digest")
async def get_digest(
    collection: str = "ai-papers",
    hours_back: int = 24,
    summarize: bool = False,
    format: str = "json"
):
    """
    Generate a digest of recent papers.

    Args:
        collection: Collection to generate digest for
        hours_back: How many hours back to include
        summarize: Whether to generate AI summaries (slower)
        format: Output format (json, markdown, html)
    """
    from .digest import DigestGenerator

    try:
        generator = DigestGenerator(
            db,
            ollama_host=config['ollama']['host'],
            ollama_model=config['ollama'].get('chat_model', 'llama3.2'),
            use_local=True
        )

        digest = generator.generate_digest(
            collection_name=collection,
            hours_back=hours_back,
            summarize_papers=summarize
        )

        if format == "markdown":
            return JSONResponse(
                content={"markdown": generator.format_digest_markdown(digest)},
                media_type="application/json"
            )
        elif format == "html":
            return JSONResponse(
                content={"html": generator.format_digest_html(digest)},
                media_type="application/json"
            )
        else:
            return digest

    except Exception as e:
        logger.exception(f"Digest generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Web Crawler Endpoints ====================

@app.get("/webcrawler/scopes")
async def list_web_crawler_scopes():
    """List available pre-defined subject scopes for web crawling."""
    from .web_crawler import MINECRAFT_REDSTONE_SCOPE

    scopes = {
        "minecraft-redstone": {
            "name": MINECRAFT_REDSTONE_SCOPE.name,
            "collection": MINECRAFT_REDSTONE_SCOPE.collection_name,
            "seed_urls": MINECRAFT_REDSTONE_SCOPE.seed_urls,
            "url_patterns": MINECRAFT_REDSTONE_SCOPE.url_patterns[:5],  # First 5
            "content_keywords": MINECRAFT_REDSTONE_SCOPE.content_keywords[:10]
        }
    }
    return {"scopes": scopes}


@app.post("/webcrawler/crawl/{scope_name}")
async def crawl_subject(
    scope_name: str,
    max_pages: int = 50,
    max_depth: int = 2
):
    """
    Crawl and ingest pages for a pre-defined subject scope.

    Args:
        scope_name: Name of the scope (e.g., 'minecraft-redstone')
        max_pages: Maximum pages to crawl
        max_depth: Maximum link depth from seed URLs
    """
    from .web_crawler import WebCrawler, MINECRAFT_REDSTONE_SCOPE

    # Map scope names to scope objects
    scopes = {
        "minecraft-redstone": MINECRAFT_REDSTONE_SCOPE
    }

    if scope_name not in scopes:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scope: {scope_name}. Available: {list(scopes.keys())}"
        )

    scope = scopes[scope_name]

    try:
        crawler = WebCrawler(
            db, embed_client,
            max_pages=max_pages,
            max_depth=max_depth
        )
        result = crawler.crawl_and_ingest(scope)
        crawler.close()

        return {
            "success": True,
            "subject": result['subject'],
            "collection": result['collection'],
            "pages_crawled": result['pages_crawled'],
            "documents_ingested": result['documents_ingested'],
            "elapsed_seconds": result['elapsed_seconds'],
            "urls": result['crawled_urls']
        }

    except Exception as e:
        logger.exception(f"Web crawl failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webcrawler/custom")
async def crawl_custom_subject(
    name: str,
    seed_urls: str,
    url_patterns: str = "",
    content_keywords: str = "",
    collection_name: str = "",
    max_pages: int = 30,
    max_depth: int = 2
):
    """
    Crawl a custom subject with user-defined scope.

    Args:
        name: Subject name
        seed_urls: Comma-separated seed URLs to start crawling
        url_patterns: Comma-separated regex patterns for URLs to include
        content_keywords: Comma-separated keywords for content filtering
        collection_name: Collection to store results (defaults to name)
        max_pages: Maximum pages to crawl
        max_depth: Maximum link depth
    """
    from .web_crawler import WebCrawler, SubjectScope

    # Parse comma-separated inputs
    seeds = [u.strip() for u in seed_urls.split(',') if u.strip()]
    patterns = [p.strip() for p in url_patterns.split(',') if p.strip()] if url_patterns else []
    keywords = [k.strip() for k in content_keywords.split(',') if k.strip()] if content_keywords else []

    if not seeds:
        raise HTTPException(status_code=400, detail="At least one seed URL required")

    # Validate all seed URLs are from whitelisted domains
    for url in seeds:
        domain = url.split('/')[2] if '://' in url else url.split('/')[0]
        if not db.is_domain_whitelisted(domain):
            raise HTTPException(
                status_code=400,
                detail=f"Domain not whitelisted: {domain}. Add it via POST /whitelist first."
            )

    scope = SubjectScope(
        name=name,
        url_patterns=patterns,
        url_exclude_patterns=[r'/File:', r'/Category:', r'/Template:', r'/User:', r'/Talk:', r'/Special:'],
        content_keywords=keywords,
        seed_urls=seeds,
        collection_name=collection_name or name
    )

    try:
        crawler = WebCrawler(
            db, embed_client,
            max_pages=max_pages,
            max_depth=max_depth
        )
        result = crawler.crawl_and_ingest(scope)
        crawler.close()

        return {
            "success": True,
            "subject": result['subject'],
            "collection": result['collection'],
            "pages_crawled": result['pages_crawled'],
            "documents_ingested": result['documents_ingested'],
            "elapsed_seconds": result['elapsed_seconds'],
            "urls": result['crawled_urls']
        }

    except Exception as e:
        logger.exception(f"Custom web crawl failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config['server']['host'],
        port=config['server']['port'],
        reload=False
    )
