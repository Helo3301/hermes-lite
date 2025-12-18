#!/usr/bin/env python3
"""Rebuild the knowledge graph from existing documents in the database.

This script:
1. Reads all documents from the database
2. Extracts entities and relationships using DocumentEntityExtractor
3. Stores them in the entity/relationship tables
4. Prints statistics about the rebuilt graph
"""

import os
import sys
import time
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from app.entity_extractor import DocumentEntityExtractor


def load_config(config_path=None):
    """Load configuration from yaml file."""
    if config_path is None:
        config_path = os.environ.get('HERMES_CONFIG', '/app/config.yaml')
        # Fall back to local config for development
        if not os.path.exists(config_path):
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config.yaml'
            )
    with open(config_path) as f:
        return yaml.safe_load(f)


def rebuild_knowledge_graph(db_path: str, use_llm: bool = False, verbose: bool = False):
    """
    Rebuild the knowledge graph from all documents.

    Args:
        db_path: Path to the SQLite database
        use_llm: Whether to use LLM for enhanced extraction
        verbose: Print detailed progress

    Returns:
        Dictionary with statistics
    """
    # Import database here to avoid initialization issues
    from app.database import Database

    print(f"Opening database: {db_path}")
    db = Database(db_path)

    # Initialize extractor
    extractor = DocumentEntityExtractor(use_llm=use_llm)

    # Get all documents
    documents = db.get_documents()
    print(f"Found {len(documents)} documents")

    stats = {
        "documents_processed": 0,
        "entities_extracted": 0,
        "relationships_extracted": 0,
        "entities_stored": 0,
        "relationships_stored": 0,
        "errors": 0,
    }

    for doc in documents:
        doc_id = doc['id']
        filename = doc['filename']

        try:
            # Get full document content
            full_doc = db.get_document_by_id(doc_id)
            if not full_doc or not full_doc.get('clean_md'):
                if verbose:
                    print(f"  Skipping {filename}: no content")
                continue

            content = full_doc['clean_md']

            if verbose:
                print(f"\nProcessing: {filename} ({len(content)} chars)")

            start_time = time.time()

            # Extract entities and relationships
            entities, relationships = extractor.process_document(
                text=content,
                doc_id=doc_id,
            )

            stats["entities_extracted"] += len(entities)
            stats["relationships_extracted"] += len(relationships)

            # Store entities
            entity_id_map = {}  # name -> entity_id
            for entity in entities:
                # Check if entity already exists
                existing = db.get_entity_by_name(entity.name)
                if existing:
                    entity_id_map[entity.name] = existing['id']
                else:
                    entity_id = db.insert_entity(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        doc_id=doc_id,
                        confidence=entity.confidence,
                    )
                    if entity_id:
                        entity_id_map[entity.name] = entity_id
                        stats["entities_stored"] += 1

            # Store relationships
            for rel in relationships:
                source_id = entity_id_map.get(rel.source)
                target_id = entity_id_map.get(rel.target)

                if source_id and target_id:
                    rel_id = db.insert_relationship(
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relationship_type=rel.relationship_type,
                        evidence_chunk_id=None,  # No chunk ID in rebuild context
                        confidence=rel.confidence,
                    )
                    if rel_id:
                        stats["relationships_stored"] += 1

            elapsed = time.time() - start_time
            stats["documents_processed"] += 1

            if verbose:
                print(f"  Entities: {len(entities)}, Relationships: {len(relationships)}")
                print(f"  Time: {elapsed*1000:.0f}ms")

        except Exception as e:
            stats["errors"] += 1
            print(f"Error processing {filename}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    # Get final graph stats
    try:
        graph_stats = db.get_graph_stats()
        stats["total_entities"] = graph_stats.get("entities", 0)
        stats["total_relationships"] = graph_stats.get("relationships", 0)
    except:
        pass

    db.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description='Rebuild knowledge graph from documents')
    parser.add_argument('--db', type=str, help='Database path')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for extraction')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Determine database path
    db_path = args.db
    if not db_path:
        config = load_config(args.config)
        db_path = config['database']['path']

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    print("=" * 60)
    print("Knowledge Graph Rebuild")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Use LLM: {args.use_llm}")
    print()

    start_time = time.time()
    stats = rebuild_knowledge_graph(db_path, args.use_llm, args.verbose)
    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Entities extracted: {stats['entities_extracted']}")
    print(f"Relationships extracted: {stats['relationships_extracted']}")
    print(f"Entities stored (new): {stats['entities_stored']}")
    print(f"Relationships stored: {stats['relationships_stored']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total time: {total_time:.1f}s")

    if 'total_entities' in stats:
        print()
        print("Graph Totals:")
        print(f"  Total entities: {stats.get('total_entities', 'N/A')}")
        print(f"  Total relationships: {stats.get('total_relationships', 'N/A')}")

    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
