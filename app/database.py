"""SQLite database setup with sqlite-vec for vector storage."""
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import struct

import sqlite_vec


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with sqlite-vec loaded."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize the database schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)

    conn.executescript("""
        -- Collections table (for organizing documents)
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Domain whitelist for crawler security
        CREATE TABLE IF NOT EXISTS domain_whitelist (
            id INTEGER PRIMARY KEY,
            domain TEXT UNIQUE NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            source_path TEXT,
            source_url TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            doc_type TEXT,
            clean_md TEXT,
            hash TEXT UNIQUE,
            collection_id INTEGER REFERENCES collections(id) ON DELETE SET NULL
        );

        -- Chunks table
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER,
            content TEXT NOT NULL,
            embedding BLOB,
            token_count INTEGER,
            start_char INTEGER,
            end_char INTEGER
        );

        -- FTS5 virtual table for keyword search
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='chunks',
            content_rowid='id'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
            INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);

        -- Insert default whitelisted domains
        INSERT OR IGNORE INTO domain_whitelist (domain) VALUES ('arxiv.org');
        INSERT OR IGNORE INTO domain_whitelist (domain) VALUES ('export.arxiv.org');
        INSERT OR IGNORE INTO domain_whitelist (domain) VALUES ('paperswithcode.com');
        INSERT OR IGNORE INTO domain_whitelist (domain) VALUES ('huggingface.co');
        INSERT OR IGNORE INTO domain_whitelist (domain) VALUES ('openreview.net');
        INSERT OR IGNORE INTO domain_whitelist (domain) VALUES ('semanticscholar.org');

        -- Insert default collection
        INSERT OR IGNORE INTO collections (name, description) VALUES ('default', 'Default collection for manually uploaded documents');
        INSERT OR IGNORE INTO collections (name, description) VALUES ('ai-papers', 'AI/ML research papers from arXiv and other sources');

        -- ==================== Knowledge Graph Tables ====================

        -- Entities table for named entities extracted from documents
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,  -- method, dataset, author, concept, paper
            doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
            confidence REAL DEFAULT 1.0,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Relationships between entities
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY,
            source_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            target_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            relationship_type TEXT NOT NULL,  -- cites, extends, compares, uses, outperforms
            confidence REAL DEFAULT 1.0,
            evidence_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_entity_id, target_entity_id, relationship_type)
        );

        -- Indexes for efficient graph queries
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(normalized_name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_entities_doc ON entities(doc_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
    """)

    conn.commit()
    return conn


def serialize_embedding(embedding: list[float]) -> bytes:
    """Convert embedding list to bytes for storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(blob: bytes) -> list[float]:
    """Convert bytes back to embedding list."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f'{n}f', blob))


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


class Database:
    """Database wrapper for HERMES-Lite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = init_database(db_path)

    def document_exists(self, content_hash: str) -> bool:
        """Check if a document with this hash already exists."""
        cursor = self.conn.execute(
            "SELECT 1 FROM documents WHERE hash = ?", (content_hash,)
        )
        return cursor.fetchone() is not None

    def insert_document(
        self,
        filename: str,
        source_path: Optional[str],
        doc_type: str,
        clean_md: str,
        content_hash: str,
        collection_id: Optional[int] = None,
        source_url: Optional[str] = None
    ) -> int:
        """Insert a new document, returns doc_id."""
        # Default to 'default' collection if none specified
        if collection_id is None:
            collection_id = self.get_collection_id('default')

        cursor = self.conn.execute(
            """INSERT INTO documents (filename, source_path, source_url, doc_type, clean_md, hash, collection_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (filename, source_path, source_url, doc_type, clean_md, content_hash, collection_id)
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_chunk(
        self,
        doc_id: int,
        chunk_index: int,
        content: str,
        embedding: list[float],
        token_count: int,
        start_char: int,
        end_char: int
    ) -> int:
        """Insert a chunk with its embedding."""
        cursor = self.conn.execute(
            """INSERT INTO chunks (doc_id, chunk_index, content, embedding, token_count, start_char, end_char)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, chunk_index, content, serialize_embedding(embedding), token_count, start_char, end_char)
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_chunks_batch(self, chunks_data: list[dict]) -> None:
        """Insert multiple chunks in a single transaction."""
        self.conn.executemany(
            """INSERT INTO chunks (doc_id, chunk_index, content, embedding, token_count, start_char, end_char)
               VALUES (:doc_id, :chunk_index, :content, :embedding, :token_count, :start_char, :end_char)""",
            [
                {
                    **c,
                    'embedding': serialize_embedding(c['embedding'])
                }
                for c in chunks_data
            ]
        )
        self.conn.commit()

    def get_all_chunks_for_search(self) -> list[dict]:
        """Get all chunks with embeddings for vector search."""
        cursor = self.conn.execute(
            """SELECT c.id, c.content, c.doc_id, c.embedding, d.filename
               FROM chunks c
               JOIN documents d ON c.doc_id = d.id"""
        )
        results = []
        for row in cursor:
            results.append({
                'id': row['id'],
                'content': row['content'],
                'doc_id': row['doc_id'],
                'filename': row['filename'],
                'embedding': deserialize_embedding(row['embedding'])
            })
        return results

    def keyword_search(self, query: str, limit: int = 30, doc_filter: Optional[str] = None) -> list[dict]:
        """Perform FTS5 keyword search."""
        # Convert query to FTS5 syntax (OR between terms)
        # Quote each term to prevent FTS5 from interpreting them as column names
        terms = [f'"{term}"' for term in query.split() if term.strip()]
        if not terms:
            return []
        fts_query = " OR ".join(terms)

        filter_clause = ""
        params = [fts_query, limit]
        if doc_filter:
            filter_clause = "AND d.filename LIKE ?"
            params = [fts_query, f"%{doc_filter}%", limit]

        cursor = self.conn.execute(
            f"""SELECT c.id, c.content, c.doc_id, d.filename,
                       bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                JOIN documents d ON c.doc_id = d.id
                WHERE chunks_fts MATCH ?
                {filter_clause}
                ORDER BY score ASC
                LIMIT ?""",
            params
        )
        return [dict(row) for row in cursor]

    def get_documents(self) -> list[dict]:
        """List all documents."""
        cursor = self.conn.execute(
            """SELECT id, filename, source_path, ingested_at, doc_type,
                      (SELECT COUNT(*) FROM chunks WHERE doc_id = documents.id) as chunk_count
               FROM documents
               ORDER BY ingested_at DESC"""
        )
        return [dict(row) for row in cursor]

    def get_document_by_id(self, doc_id: int) -> Optional[dict]:
        """Get a document by ID including full content."""
        cursor = self.conn.execute(
            """SELECT id, filename, source_path, source_url, ingested_at, doc_type, clean_md,
                      (SELECT COUNT(*) FROM chunks WHERE doc_id = documents.id) as chunk_count
               FROM documents
               WHERE id = ?""",
            (doc_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and its chunks."""
        cursor = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_stats(self) -> dict:
        """Get database statistics."""
        doc_count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        return {
            'documents': doc_count,
            'chunks': chunk_count,
            'db_size_mb': Path(self.db_path).stat().st_size / (1024 * 1024) if Path(self.db_path).exists() else 0
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()

    # ==================== Collection Methods ====================

    def get_collection_id(self, name: str) -> Optional[int]:
        """Get collection ID by name."""
        cursor = self.conn.execute(
            "SELECT id FROM collections WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        return row['id'] if row else None

    def create_collection(self, name: str, description: str = "") -> int:
        """Create a new collection, returns collection_id."""
        cursor = self.conn.execute(
            "INSERT INTO collections (name, description) VALUES (?, ?)",
            (name, description)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_collections(self) -> list[dict]:
        """List all collections with document counts."""
        cursor = self.conn.execute(
            """SELECT c.id, c.name, c.description, c.created_at,
                      (SELECT COUNT(*) FROM documents WHERE collection_id = c.id) as doc_count
               FROM collections c
               ORDER BY c.name"""
        )
        return [dict(row) for row in cursor]

    def delete_collection(self, collection_id: int) -> bool:
        """Delete a collection (documents will have collection_id set to NULL)."""
        cursor = self.conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_documents_by_collection(self, collection_id: int) -> list[dict]:
        """List documents in a collection."""
        cursor = self.conn.execute(
            """SELECT id, filename, source_path, source_url, ingested_at, doc_type,
                      (SELECT COUNT(*) FROM chunks WHERE doc_id = documents.id) as chunk_count
               FROM documents
               WHERE collection_id = ?
               ORDER BY ingested_at DESC""",
            (collection_id,)
        )
        return [dict(row) for row in cursor]

    def get_all_chunks_for_search_by_collection(self, collection_id: Optional[int] = None) -> list[dict]:
        """Get all chunks with embeddings for vector search, optionally filtered by collection."""
        if collection_id:
            cursor = self.conn.execute(
                """SELECT c.id, c.content, c.doc_id, c.embedding, d.filename
                   FROM chunks c
                   JOIN documents d ON c.doc_id = d.id
                   WHERE d.collection_id = ?""",
                (collection_id,)
            )
        else:
            cursor = self.conn.execute(
                """SELECT c.id, c.content, c.doc_id, c.embedding, d.filename
                   FROM chunks c
                   JOIN documents d ON c.doc_id = d.id"""
            )
        results = []
        for row in cursor:
            results.append({
                'id': row['id'],
                'content': row['content'],
                'doc_id': row['doc_id'],
                'filename': row['filename'],
                'embedding': deserialize_embedding(row['embedding'])
            })
        return results

    # ==================== Whitelist Methods ====================

    def is_domain_whitelisted(self, domain: str) -> bool:
        """Check if a domain is in the whitelist and enabled."""
        # Check exact match and parent domains (e.g., export.arxiv.org matches arxiv.org)
        cursor = self.conn.execute(
            """SELECT 1 FROM domain_whitelist
               WHERE enabled = TRUE AND (? = domain OR ? LIKE '%.' || domain)""",
            (domain, domain)
        )
        return cursor.fetchone() is not None

    def add_domain_to_whitelist(self, domain: str) -> int:
        """Add a domain to the whitelist."""
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO domain_whitelist (domain) VALUES (?)",
            (domain,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def remove_domain_from_whitelist(self, domain: str) -> bool:
        """Remove a domain from the whitelist."""
        cursor = self.conn.execute(
            "DELETE FROM domain_whitelist WHERE domain = ?", (domain,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_whitelist(self) -> list[dict]:
        """Get all whitelisted domains."""
        cursor = self.conn.execute(
            "SELECT id, domain, enabled, added_at FROM domain_whitelist ORDER BY domain"
        )
        return [dict(row) for row in cursor]

    def set_domain_enabled(self, domain: str, enabled: bool) -> bool:
        """Enable or disable a whitelisted domain."""
        cursor = self.conn.execute(
            "UPDATE domain_whitelist SET enabled = ? WHERE domain = ?",
            (enabled, domain)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ==================== Knowledge Graph Methods ====================

    def insert_entity(
        self,
        name: str,
        entity_type: str,
        doc_id: Optional[int] = None,
        chunk_id: Optional[int] = None,
        confidence: float = 1.0,
        embedding: Optional[list[float]] = None
    ) -> int:
        """Insert an entity, returns entity_id."""
        normalized = name.lower().strip()
        emb_blob = serialize_embedding(embedding) if embedding else None

        cursor = self.conn.execute(
            """INSERT INTO entities (name, normalized_name, entity_type, doc_id, chunk_id, confidence, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, normalized, entity_type, doc_id, chunk_id, confidence, emb_blob)
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_entities_batch(self, entities: list[dict]) -> list[int]:
        """Insert multiple entities in a batch."""
        entity_ids = []
        for entity in entities:
            emb_blob = serialize_embedding(entity.get('embedding', [])) if entity.get('embedding') else None
            normalized = entity['name'].lower().strip()

            cursor = self.conn.execute(
                """INSERT INTO entities (name, normalized_name, entity_type, doc_id, chunk_id, confidence, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entity['name'], normalized, entity['entity_type'],
                 entity.get('doc_id'), entity.get('chunk_id'),
                 entity.get('confidence', 1.0), emb_blob)
            )
            entity_ids.append(cursor.lastrowid)
        self.conn.commit()
        return entity_ids

    def get_entity_by_name(self, name: str) -> Optional[dict]:
        """Find an entity by name (case-insensitive)."""
        normalized = name.lower().strip()
        cursor = self.conn.execute(
            """SELECT id, name, normalized_name, entity_type, doc_id, chunk_id, confidence
               FROM entities WHERE normalized_name = ?""",
            (normalized,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        """Get all entities of a given type."""
        cursor = self.conn.execute(
            """SELECT id, name, normalized_name, entity_type, doc_id, chunk_id, confidence
               FROM entities WHERE entity_type = ?
               ORDER BY name""",
            (entity_type,)
        )
        return [dict(row) for row in cursor]

    def get_entities_for_document(self, doc_id: int) -> list[dict]:
        """Get all entities from a document."""
        cursor = self.conn.execute(
            """SELECT id, name, normalized_name, entity_type, chunk_id, confidence
               FROM entities WHERE doc_id = ?
               ORDER BY entity_type, name""",
            (doc_id,)
        )
        return [dict(row) for row in cursor]

    def insert_relationship(
        self,
        source_entity_id: int,
        target_entity_id: int,
        relationship_type: str,
        evidence_chunk_id: Optional[int] = None,
        confidence: float = 1.0
    ) -> Optional[int]:
        """Insert a relationship between entities. Returns None if already exists."""
        try:
            cursor = self.conn.execute(
                """INSERT INTO relationships (source_entity_id, target_entity_id, relationship_type, evidence_chunk_id, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_entity_id, target_entity_id, relationship_type, evidence_chunk_id, confidence)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Relationship already exists
            return None

    def get_related_entities(
        self,
        entity_id: int,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> list[dict]:
        """Get entities related to a given entity.

        Args:
            entity_id: The source entity ID
            relationship_type: Filter by relationship type (optional)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of related entities with relationship info
        """
        results = []

        if direction in ("outgoing", "both"):
            type_filter = "AND r.relationship_type = ?" if relationship_type else ""
            params = [entity_id, relationship_type] if relationship_type else [entity_id]

            cursor = self.conn.execute(
                f"""SELECT e.id, e.name, e.entity_type, r.relationship_type, r.confidence, 'outgoing' as direction
                   FROM relationships r
                   JOIN entities e ON r.target_entity_id = e.id
                   WHERE r.source_entity_id = ? {type_filter}""",
                params
            )
            results.extend([dict(row) for row in cursor])

        if direction in ("incoming", "both"):
            type_filter = "AND r.relationship_type = ?" if relationship_type else ""
            params = [entity_id, relationship_type] if relationship_type else [entity_id]

            cursor = self.conn.execute(
                f"""SELECT e.id, e.name, e.entity_type, r.relationship_type, r.confidence, 'incoming' as direction
                   FROM relationships r
                   JOIN entities e ON r.source_entity_id = e.id
                   WHERE r.target_entity_id = ? {type_filter}""",
                params
            )
            results.extend([dict(row) for row in cursor])

        return results

    def get_chunks_for_entity(self, entity_name: str) -> list[dict]:
        """Get chunks that mention a specific entity."""
        normalized = entity_name.lower().strip()

        # First find chunks directly linked to the entity
        cursor = self.conn.execute(
            """SELECT DISTINCT c.id, c.content, c.doc_id, d.filename
               FROM entities e
               JOIN chunks c ON e.chunk_id = c.id
               JOIN documents d ON c.doc_id = d.id
               WHERE e.normalized_name = ?""",
            (normalized,)
        )
        results = [dict(row) for row in cursor]

        # Also search for entity name in chunk content (backup)
        if not results:
            cursor = self.conn.execute(
                """SELECT c.id, c.content, c.doc_id, d.filename
                   FROM chunks c
                   JOIN documents d ON c.doc_id = d.id
                   WHERE c.content LIKE ?
                   LIMIT 20""",
                (f"%{entity_name}%",)
            )
            results = [dict(row) for row in cursor]

        return results

    def get_graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        entity_count = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relationship_count = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

        entity_types = {}
        cursor = self.conn.execute(
            "SELECT entity_type, COUNT(*) as count FROM entities GROUP BY entity_type"
        )
        for row in cursor:
            entity_types[row['entity_type']] = row['count']

        rel_types = {}
        cursor = self.conn.execute(
            "SELECT relationship_type, COUNT(*) as count FROM relationships GROUP BY relationship_type"
        )
        for row in cursor:
            rel_types[row['relationship_type']] = row['count']

        return {
            'entities': entity_count,
            'relationships': relationship_count,
            'entity_types': entity_types,
            'relationship_types': rel_types,
        }

    def clear_entities_for_document(self, doc_id: int) -> int:
        """Remove all entities for a document (for re-extraction)."""
        cursor = self.conn.execute("DELETE FROM entities WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount
