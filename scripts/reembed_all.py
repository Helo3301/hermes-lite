#!/usr/bin/env python3
"""Re-embed all chunks with a new embedding model.

This script re-embeds all chunks in the database using the specified embedding model.
Progress is saved periodically so it can be resumed if interrupted.

Usage:
    python scripts/reembed_all.py --model bge-m3 --batch-size 32
"""

import argparse
import sqlite3
import struct
import sys
import time
import logging
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embed_v2 import EmbeddingClientV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def serialize_embedding(embedding: list) -> bytes:
    """Convert embedding list to bytes for storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def get_chunks_to_process(db_path: str, target_dim: int) -> list:
    """Get all chunks that need re-embedding."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all chunks - we'll re-embed everything
    cursor.execute("""
        SELECT id, content, LENGTH(embedding)/4 as current_dim
        FROM chunks
        ORDER BY id
    """)

    chunks = []
    for row in cursor:
        chunks.append({
            'id': row['id'],
            'content': row['content'],
            'current_dim': row['current_dim']
        })

    conn.close()
    return chunks


def update_embedding(db_path: str, chunk_id: int, embedding: list):
    """Update a single chunk's embedding."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE chunks SET embedding = ? WHERE id = ?",
        (serialize_embedding(embedding), chunk_id)
    )
    conn.commit()
    conn.close()


def update_embeddings_batch(db_path: str, updates: list):
    """Update multiple embeddings in a single transaction."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany(
        "UPDATE chunks SET embedding = ? WHERE id = ?",
        [(serialize_embedding(emb), chunk_id) for chunk_id, emb in updates]
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Re-embed all chunks with new model')
    parser.add_argument('--db-path', default='/home/hestiasadmin/hermes-lite/data/hermes.db',
                        help='Path to database')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                        help='Ollama host URL')
    parser.add_argument('--model', default='bge-m3',
                        help='Embedding model to use')
    parser.add_argument('--dim', type=int, default=1024,
                        help='Target embedding dimension')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from chunk index (for resuming)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')

    args = parser.parse_args()

    logger.info(f"Re-embedding with model: {args.model}")
    logger.info(f"Target dimension: {args.dim}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Ollama host: {args.ollama_host}")

    # Initialize embedding client
    embed_client = EmbeddingClientV2(
        ollama_host=args.ollama_host,
        model=args.model,
        default_dim=args.dim,
        batch_size=args.batch_size,
        timeout=120.0
    )

    # Test embedding
    logger.info("Testing embedding model...")
    if not embed_client.ensure_model_loaded():
        logger.error("Failed to load embedding model")
        sys.exit(1)

    test_emb = embed_client.embed_single("test", dim=args.dim)
    logger.info(f"Test embedding dimension: {len(test_emb)}")

    # Get chunks to process
    logger.info("Loading chunks from database...")
    chunks = get_chunks_to_process(args.db_path, args.dim)
    total_chunks = len(chunks)
    logger.info(f"Total chunks to process: {total_chunks}")

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")
        # Show sample of current dimensions
        dims = {}
        for c in chunks[:100]:
            d = c['current_dim']
            dims[d] = dims.get(d, 0) + 1
        logger.info(f"Current dimension distribution (first 100): {dims}")
        return

    # Process in batches
    start_time = time.time()
    processed = 0
    errors = 0

    for i in range(args.start_from, total_chunks, args.batch_size):
        batch = chunks[i:i + args.batch_size]
        batch_texts = [c['content'] for c in batch]

        try:
            # Embed batch
            embeddings = embed_client.embed_batch(batch_texts, dim=args.dim, show_progress=False)

            # Prepare updates
            updates = [(c['id'], emb) for c, emb in zip(batch, embeddings)]

            # Update database
            update_embeddings_batch(args.db_path, updates)

            processed += len(batch)

            # Progress update
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_chunks - processed) / rate if rate > 0 else 0

            logger.info(
                f"Progress: {processed}/{total_chunks} ({100*processed/total_chunks:.1f}%) "
                f"| Rate: {rate:.1f} chunks/sec | ETA: {eta/60:.1f} min"
            )

        except Exception as e:
            logger.error(f"Error processing batch at {i}: {e}")
            errors += 1
            if errors > 10:
                logger.error("Too many errors, stopping")
                break
            continue

    elapsed = time.time() - start_time
    logger.info(f"Completed: {processed}/{total_chunks} chunks in {elapsed/60:.1f} minutes")
    logger.info(f"Average rate: {processed/elapsed:.1f} chunks/sec")
    if errors > 0:
        logger.warning(f"Errors encountered: {errors}")

    embed_client.close()


if __name__ == "__main__":
    main()
