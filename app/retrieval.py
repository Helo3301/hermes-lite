"""
Retrieval Enhancements for Hermes-Lite

This module implements 6 specs for improving retrieval fidelity:
1. Document Diversity - Prevent single doc from dominating results
2. Intelligent Context Windows - Smart neighbor selection based on similarity
3. Query Expansion - Expand queries with synonyms/acronyms
4. Two-Phase Retrieval - BM25 candidates + vector reranking
5. Recency Weighting - Boost newer documents slightly
6. Metadata Filtering - Filter by date, collection, etc. before ranking
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


# ============================================================================
# SPEC 1: Document Diversity
# ============================================================================

def diversify_results(
    results: List[Dict],
    top_k: int = 10,
    max_per_doc: int = 2
) -> List[Dict]:
    """
    Ensure no single document dominates results (soft constraint).

    Business Rules:
    - FIRST PASS: Respect max_per_doc to give all documents a fair shot
    - BACKFILL: If top_k not reached, pull from overflow regardless of max_per_doc
    - top_k is the HARD constraint, diversity is SOFT

    Args:
        results: Ranked list of chunks (highest score first)
        top_k: Total results to return
        max_per_doc: Maximum chunks per doc in first pass (soft limit)

    Returns:
        Diversified results, backfilled to reach top_k if needed
    """
    if top_k <= 0:
        return []

    diverse = []
    doc_counts = defaultdict(int)
    overflow = []

    # FIRST PASS: Respect max_per_doc (give everyone a fair shot)
    for chunk in results:
        doc_id = chunk.get('doc_id')
        if doc_counts[doc_id] < max_per_doc:
            diverse.append(chunk)
            doc_counts[doc_id] += 1
        else:
            overflow.append(chunk)

        if len(diverse) >= top_k:
            break

    # BACKFILL: If we haven't hit top_k, pull from overflow (ignore max_per_doc)
    for chunk in overflow:
        if len(diverse) >= top_k:
            break
        diverse.append(chunk)

    return diverse


# ============================================================================
# SPEC 2: Intelligent Context Windows
# ============================================================================

def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Returns value between -1 and 1:
    - 1.0 = identical direction (very related)
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite (contradictory)
    """
    if not emb1 or not emb2:
        return 0.0

    dot = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = math.sqrt(sum(a * a for a in emb1))
    norm2 = math.sqrt(sum(b * b for b in emb2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def get_contextual_window(
    matched_chunk: Dict,
    all_chunks: List[Dict],
    window_size: int = 2,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Get neighboring chunks that are contextually related to matched chunk.

    Business Rules:
    - FOR ALL neighbors, inclusion occurs IF AND ONLY IF similarity >= threshold
    - IF a neighbor fails threshold, expansion in that direction STOPS
    - Matched chunk SHALL ALWAYS be included
    - Document order SHALL BE preserved

    Args:
        matched_chunk: The chunk that matched the search query
        all_chunks: All chunks from the same document
        window_size: How far to look in each direction (max)
        similarity_threshold: Minimum similarity to include a neighbor

    Returns:
        List of chunks in order: [...before, matched, ...after]
    """
    doc_id = matched_chunk.get('doc_id')
    chunk_idx = matched_chunk.get('chunk_index')
    matched_embedding = matched_chunk.get('embedding', [])

    # Get chunks from same document only
    same_doc_chunks = [c for c in all_chunks if c.get('doc_id') == doc_id]

    # Sort by chunk index
    same_doc_chunks.sort(key=lambda x: x.get('chunk_index', 0))

    # Find chunks BEFORE the matched chunk
    before_candidates = [
        c for c in same_doc_chunks
        if c.get('chunk_index', 0) < chunk_idx
        and c.get('chunk_index', 0) >= chunk_idx - window_size
    ]
    before_candidates.sort(key=lambda x: x.get('chunk_index', 0), reverse=True)  # Closest first

    # Find chunks AFTER the matched chunk
    after_candidates = [
        c for c in same_doc_chunks
        if c.get('chunk_index', 0) > chunk_idx
        and c.get('chunk_index', 0) <= chunk_idx + window_size
    ]
    after_candidates.sort(key=lambda x: x.get('chunk_index', 0))  # Closest first

    # Check BEFORE chunks (stop at first unrelated one)
    included_before = []
    for neighbor in before_candidates:
        sim = cosine_similarity(matched_embedding, neighbor.get('embedding', []))
        if sim >= similarity_threshold:
            included_before.insert(0, neighbor)  # Maintain document order
        else:
            break  # Stop - hit unrelated content

    # Check AFTER chunks (stop at first unrelated one)
    included_after = []
    for neighbor in after_candidates:
        sim = cosine_similarity(matched_embedding, neighbor.get('embedding', []))
        if sim >= similarity_threshold:
            included_after.append(neighbor)
        else:
            break  # Stop - hit unrelated content

    # Combine: [before...] + [matched] + [after...]
    return included_before + [matched_chunk] + included_after


# ============================================================================
# SPEC 3: Query Expansion
# ============================================================================

# Common expansions for AI/ML domain
EXPANSION_RULES = {
    # Acronyms -> Full forms
    'rag': ['retrieval augmented generation', 'retrieval-augmented generation'],
    'llm': ['large language model', 'large language models'],
    'nlp': ['natural language processing'],
    'ml': ['machine learning'],
    'ai': ['artificial intelligence'],
    'gpt': ['generative pre-trained transformer'],
    'bert': ['bidirectional encoder representations from transformers'],
    'cot': ['chain of thought', 'chain-of-thought'],
    'rlhf': ['reinforcement learning from human feedback'],
    'lora': ['low-rank adaptation'],

    # Full forms -> Acronyms (bidirectional)
    'retrieval augmented generation': ['rag'],
    'large language model': ['llm'],
    'chain of thought': ['cot'],
}


def expand_query(
    query: str,
    expansion_rules: Dict[str, List[str]] = None,
    max_expansions: int = 3
) -> List[str]:
    """
    Expand a query with synonyms and related terms.

    Business Rules:
    - Original query SHALL ALWAYS be first element
    - IF query matches known rule, expansions added up to max_expansions
    - Expansion matching SHALL BE case-insensitive
    - No duplicate queries in result

    Args:
        query: Original search query
        expansion_rules: Dict mapping terms to their expansions
        max_expansions: Maximum number of expansion terms to add

    Returns:
        List of queries: [original, expansion1, expansion2, ...]
    """
    if expansion_rules is None:
        expansion_rules = EXPANSION_RULES

    queries = [query]  # Always include original
    query_lower = query.lower().strip()

    # Direct match
    if query_lower in expansion_rules:
        for expansion in expansion_rules[query_lower][:max_expansions]:
            if expansion not in queries:
                queries.append(expansion)

    # Check if any expansion key is contained in the query
    for term, expansions in expansion_rules.items():
        if term in query_lower and term != query_lower:
            for expansion in expansions[:max_expansions]:
                # Replace the term in the query with its expansion
                expanded = query_lower.replace(term, expansion)
                if expanded not in queries:
                    queries.append(expanded)

    return queries[:max_expansions + 1]  # Original + max_expansions


def merge_search_results(
    result_sets: List[List[Dict]],
    top_k: int = 10
) -> List[Dict]:
    """
    Merge results from multiple queries, handling duplicates.

    Business Rules:
    - Same chunk appearing multiple times -> one and only one in result
    - Higher score SHALL BE retained on duplicate
    - _match_count tracks how many queries returned each chunk

    Args:
        result_sets: List of result lists, one per expanded query
        top_k: Total results to return

    Returns:
        Merged and deduplicated results, sorted by score
    """
    seen_chunks = {}  # chunk_id -> best result dict

    for results in result_sets:
        for result in results:
            chunk_id = result.get('id')

            if chunk_id not in seen_chunks:
                # First time seeing this chunk
                seen_chunks[chunk_id] = result.copy()
                seen_chunks[chunk_id]['_match_count'] = 1
            else:
                # Seen before - keep higher score, track match count
                seen_chunks[chunk_id]['_match_count'] += 1
                if result.get('score', 0) > seen_chunks[chunk_id].get('score', 0):
                    score = result.get('score', 0)
                    match_count = seen_chunks[chunk_id]['_match_count']
                    seen_chunks[chunk_id] = result.copy()
                    seen_chunks[chunk_id]['score'] = score
                    seen_chunks[chunk_id]['_match_count'] = match_count

    # Sort by score descending
    merged = sorted(seen_chunks.values(), key=lambda x: x.get('score', 0), reverse=True)

    return merged[:top_k]


def boost_multi_match_results(
    results: List[Dict],
    boost_factor: float = 0.05
) -> List[Dict]:
    """
    Give a small boost to results that matched multiple query variants.

    Business Rules:
    - IF _match_count > 1, score MAY BE boosted
    - IF _match_count = 1, score SHALL NOT be modified
    - Results re-sorted after boosting

    Args:
        results: Merged results with _match_count field
        boost_factor: Score boost per additional match (0.05 = 5%)

    Returns:
        Results with boosted scores, re-sorted
    """
    for result in results:
        match_count = result.get('_match_count', 1)
        if match_count > 1:
            # Boost for each additional match
            boost = 1 + (boost_factor * (match_count - 1))
            result['score'] = result.get('score', 0) * boost

    # Re-sort after boosting
    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return results


# ============================================================================
# SPEC 4: Two-Phase Retrieval
# ============================================================================

def bm25_search(
    query: str,
    documents: List[Dict],
    top_k: int = 50
) -> List[Dict]:
    """
    Simulated BM25 search - keyword matching.

    In production, this uses SQLite FTS5.
    For testing, we use simple term overlap scoring.

    Business Rules:
    - IF document contains zero query terms, it SHALL NOT appear
    - Rank determined by term overlap count
    """
    query_terms = set(query.lower().split())

    results = []
    for doc in documents:
        content_terms = set(doc.get('content', '').lower().split())

        # Simple term overlap (real BM25 is more sophisticated)
        overlap = len(query_terms & content_terms)
        if overlap > 0:
            # Score based on term frequency (simplified)
            score = overlap / len(query_terms) if query_terms else 0
            result = doc.copy()
            result['bm25_score'] = score
            result['_matched_terms'] = list(query_terms & content_terms)
            results.append(result)

    # Sort by BM25 score
    results.sort(key=lambda x: x.get('bm25_score', 0), reverse=True)

    return results[:top_k]


def vector_rerank(
    query_embedding: List[float],
    candidates: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    """
    Re-rank candidates using vector similarity.

    Business Rules:
    - Rank determined by cosine similarity to query embedding
    - False positives (keyword match, wrong meaning) get demoted
    """
    for candidate in candidates:
        candidate_emb = candidate.get('embedding', [])
        candidate['vector_score'] = cosine_similarity(query_embedding, candidate_emb)

    # Sort by vector score
    candidates.sort(key=lambda x: x.get('vector_score', 0), reverse=True)

    return candidates[:top_k]


def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    vector_results: List[Dict],
    k: int = 60,
    top_k: int = 10
) -> List[Dict]:
    """
    Combine BM25 and vector rankings using Reciprocal Rank Fusion.

    Business Rules:
    - RRF score = sum(1/(k + rank)) for each ranking
    - Document in both rankings SHALL score higher than one in only one
    - Parameterized queries SHALL BE used (k=60 is standard)
    """
    # Build id -> rank mapping for BM25
    bm25_ranks = {r.get('id'): idx + 1 for idx, r in enumerate(bm25_results)}

    # Build id -> rank mapping for vector
    vector_ranks = {r.get('id'): idx + 1 for idx, r in enumerate(vector_results)}

    # Compute RRF scores
    all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

    rrf_scores = {}
    for doc_id in all_ids:
        score = 0

        if doc_id in bm25_ranks:
            score += 1 / (k + bm25_ranks[doc_id])

        if doc_id in vector_ranks:
            score += 1 / (k + vector_ranks[doc_id])

        rrf_scores[doc_id] = score

    # Find the full result objects and add RRF score
    id_to_result = {}
    for r in bm25_results + vector_results:
        if r.get('id') not in id_to_result:
            id_to_result[r.get('id')] = r.copy()

    results = []
    for doc_id, rrf_score in rrf_scores.items():
        if doc_id in id_to_result:
            result = id_to_result[doc_id].copy()
            result['rrf_score'] = rrf_score
            result['_bm25_rank'] = bm25_ranks.get(doc_id, None)
            result['_vector_rank'] = vector_ranks.get(doc_id, None)
            results.append(result)

    # Sort by RRF score
    results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)

    return results[:top_k]


def two_phase_search(
    query: str,
    query_embedding: List[float],
    documents: List[Dict],
    phase1_k: int = 50,
    phase2_k: int = 10,
    fusion_method: str = 'rrf'
) -> List[Dict]:
    """
    Complete two-phase retrieval pipeline.

    Business Rules:
    - Phase 1 (BM25) SHALL EXECUTE before Phase 2 (Vector)
    - Phase 2 SHALL ONLY evaluate Phase 1 candidates
    - IF fusion_method='rerank', final order is vector ranking only
    - IF fusion_method='rrf', scores combined via RRF

    Args:
        query: Text query for BM25
        query_embedding: Embedding vector for semantic search
        documents: All documents to search
        phase1_k: Candidates from BM25
        phase2_k: Final results to return
        fusion_method: 'rrf' for rank fusion, 'rerank' for pure reranking
    """
    # Phase 1: BM25 candidate retrieval
    bm25_candidates = bm25_search(query, documents, top_k=phase1_k)

    if not bm25_candidates:
        return []

    if fusion_method == 'rerank':
        # Pure reranking: just reorder BM25 candidates by vector score
        return vector_rerank(query_embedding, bm25_candidates, top_k=phase2_k)

    elif fusion_method == 'rrf':
        # RRF: combine BM25 and vector rankings
        vector_ranked = vector_rerank(
            query_embedding,
            [c.copy() for c in bm25_candidates],
            top_k=phase1_k
        )
        return reciprocal_rank_fusion(bm25_candidates, vector_ranked, top_k=phase2_k)

    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


# ============================================================================
# SPEC 5: Recency Weighting
# ============================================================================

def apply_recency_weight(
    results: List[Dict],
    reference_date: datetime = None,
    decay_rate: float = 0.03,
    recency_weight: float = 0.15
) -> List[Dict]:
    """
    Apply a score boost to more recent documents.

    Business Rules:
    - Recency boost = e^(-decay_rate * age_months)
    - Final score = original_score * (1 + recency_weight * boost)
    - IF equal base scores, newer document SHALL rank higher
    - Quality still dominates (excellent old beats mediocre new)
    - IF date missing, document treated as old (5 years)

    Args:
        results: Search results with 'score' and 'ingested_at' fields
        reference_date: Date to measure age from (default: now)
        decay_rate: How fast boost decays (0.03 = gradual)
        recency_weight: Maximum boost (0.15 = 15% max)

    Returns:
        Results with adjusted scores, re-sorted
    """
    if reference_date is None:
        reference_date = datetime.now()

    for result in results:
        # Parse the date
        date_str = result.get('ingested_at', '')
        if isinstance(date_str, str):
            try:
                doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                # If date parsing fails, assume old
                doc_date = reference_date - timedelta(days=365*5)
        elif isinstance(date_str, datetime):
            doc_date = date_str
        else:
            doc_date = reference_date - timedelta(days=365*5)

        # Calculate age in months
        try:
            age_days = (reference_date - doc_date.replace(tzinfo=None)).days
        except (AttributeError, TypeError):
            age_days = 365 * 5  # Default to 5 years old
        age_months = age_days / 30

        # Calculate recency boost (exponential decay)
        recency_boost = math.exp(-decay_rate * age_months)

        # Apply boost: score * (1 + weight * boost)
        original_score = result.get('score', 0)
        result['score'] = original_score * (1 + recency_weight * recency_boost)
        result['_recency_boost'] = recency_boost

    # Re-sort by adjusted score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return results


# ============================================================================
# SPEC 6: Metadata Filtering
# ============================================================================

def build_filter_clause(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    collection: Optional[str] = None,
    min_chunks: Optional[int] = None,
    exclude_doc_ids: Optional[List[int]] = None
) -> Tuple[str, List]:
    """
    Build SQL WHERE clause for metadata filtering.

    Business Rules:
    - IF date_from specified, only docs where ingested_at >= date_from
    - IF date_to specified, only docs where ingested_at <= date_to
    - IF collection specified, only docs in that collection
    - IF min_chunks specified, only docs with chunk_count >= min_chunks
    - IF exclude_doc_ids specified, those docs SHALL NOT be included
    - Multiple filters joined with AND
    - No filters = "1=1" (match all)

    Returns:
        Tuple of (where_clause, params) for parameterized queries
    """
    conditions = []
    params = []

    if date_from:
        conditions.append("d.ingested_at >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("d.ingested_at <= ?")
        params.append(date_to)

    if collection:
        conditions.append("c.name = ?")
        params.append(collection)

    if min_chunks is not None:
        conditions.append("""
            (SELECT COUNT(*) FROM chunks WHERE doc_id = d.id) >= ?
        """)
        params.append(min_chunks)

    if exclude_doc_ids:
        placeholders = ','.join('?' * len(exclude_doc_ids))
        conditions.append(f"d.id NOT IN ({placeholders})")
        params.extend(exclude_doc_ids)

    if conditions:
        where_clause = " AND ".join(conditions)
    else:
        where_clause = "1=1"  # No filtering

    return (where_clause, params)


def filter_results_by_metadata(
    results: List[Dict],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    collection: Optional[str] = None,
    min_chunks: Optional[int] = None,
    exclude_doc_ids: Optional[List[int]] = None
) -> List[Dict]:
    """
    Filter results by metadata (in-memory version).

    In production, filtering should happen in the database query.
    This function is for testing and fallback.

    Business Rules:
    - All filter conditions joined with AND
    - Missing metadata field = excluded
    - Original ranking order preserved
    """
    filtered = []

    for result in results:
        # Date from filter
        if date_from:
            doc_date = result.get('ingested_at', '')
            if doc_date < date_from:
                continue

        # Date to filter
        if date_to:
            doc_date = result.get('ingested_at', '')
            if doc_date > date_to:
                continue

        # Collection filter
        if collection:
            doc_collection = result.get('collection', '')
            if doc_collection != collection:
                continue

        # Min chunks filter
        if min_chunks is not None:
            doc_chunks = result.get('chunk_count', 0)
            if doc_chunks < min_chunks:
                continue

        # Exclude specific docs
        if exclude_doc_ids:
            if result.get('doc_id') in exclude_doc_ids:
                continue

        filtered.append(result)

    return filtered


# ============================================================================
# ITERATIVE GAP-FILL RETRIEVAL (SEAL-RAG Style)
# ============================================================================

def iterative_retrieve(
    search_fn,
    query: str,
    budget: int = 10,
    max_iterations: int = 3,
    query_analyzer=None,
    gap_detector=None,
    chunk_scorer=None,
    entity_ranker=None,
    use_entity_first: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Iterative retrieval with gap-fill (SEAL-RAG style).

    Implements fixed-budget evidence assembly: instead of expanding context
    when we find gaps, we REPLACE low-quality chunks with better ones from
    targeted sub-queries.

    Business Rules:
    - Budget is FIXED: always return exactly `budget` chunks (or fewer if not enough)
    - Each iteration MAY replace chunks but SHALL NOT exceed budget
    - Simple queries (complexity 1-2) get 1 iteration max
    - Complex queries (complexity 3+) may iterate up to max_iterations
    - Iteration stops when coverage is good enough or no improvement

    Args:
        search_fn: Function(query, top_k) -> list[dict] for retrieval
        query: The search query
        budget: Fixed number of chunks to return
        max_iterations: Maximum retrieval iterations
        query_analyzer: Optional QueryAnalyzer instance
        gap_detector: Optional GapDetector instance
        chunk_scorer: Optional ChunkScorer instance
        verbose: Print debug info

    Returns:
        Dict with:
        - results: Final selected chunks
        - iterations: Number of iterations performed
        - gap_analyses: Gap analysis from each iteration
        - queries_used: All queries executed
    """
    from .query_analyzer import QueryAnalyzer, QueryType
    from .gap_detector import GapDetector
    from .chunk_scorer import ChunkScorer
    from .entity_ranker import EntityFirstRanker

    # Initialize components if not provided
    if query_analyzer is None:
        query_analyzer = QueryAnalyzer(use_llm=False)
    if gap_detector is None:
        gap_detector = GapDetector(query_analyzer=query_analyzer)
    if chunk_scorer is None:
        chunk_scorer = ChunkScorer()
    if entity_ranker is None and use_entity_first:
        entity_ranker = EntityFirstRanker()

    # Analyze query
    query_analysis = query_analyzer.analyze(query)
    query_entities = query_analysis.entities
    query_type = query_analysis.query_type
    complexity = query_analysis.complexity

    # Simple queries: just do single retrieval
    if complexity <= 2 and query_type == QueryType.SIMPLE:
        results = search_fn(query, top_k=budget)
        return {
            "results": results,
            "iterations": 1,
            "gap_analyses": [],
            "queries_used": [query],
            "query_type": query_type.value,
            "complexity": complexity,
        }

    # Complex queries: iterative retrieval
    all_chunks = []
    queries_used = []
    gap_analyses = []
    previous_coverage = 0.0

    # Start with decomposed sub-queries
    sub_queries = query_analysis.sub_queries

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Determine queries for this iteration
        if iteration == 1:
            # First iteration: use original + sub-queries
            iteration_queries = [sq.query for sq in sub_queries]
            if query not in iteration_queries:
                iteration_queries.insert(0, query)
        else:
            # Subsequent iterations: use gap-fill queries
            if gap_analyses and gap_analyses[-1].suggested_subqueries:
                iteration_queries = gap_analyses[-1].suggested_subqueries
            else:
                break  # No more queries to try

        # Execute searches
        for q in iteration_queries[:3]:  # Limit to 3 queries per iteration
            if q not in queries_used:
                new_results = search_fn(q, top_k=budget)
                all_chunks.extend(new_results)
                queries_used.append(q)
                if verbose:
                    print(f"  Query: {q[:50]}... -> {len(new_results)} chunks")

        # Deduplicate and score all chunks
        unique_chunks = {}
        for chunk in all_chunks:
            chunk_id = chunk.get('id', id(chunk))
            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk
            else:
                # Keep chunk with higher score
                existing_score = unique_chunks[chunk_id].get('score', 0)
                new_score = chunk.get('score', 0)
                if new_score > existing_score:
                    unique_chunks[chunk_id] = chunk

        all_chunks = list(unique_chunks.values())

        # Select best chunks within budget
        # Use entity-first ranking if enabled and we have entities
        if use_entity_first and entity_ranker and query_entities:
            # SEAL-RAG style: rank by entity coverage first
            if iteration == 1:
                # First iteration: rank all by entity coverage
                selected = entity_ranker.rank_by_entity_coverage(
                    all_chunks,
                    query_entities,
                    top_k=budget
                )
            else:
                # Subsequent iterations: use replacement logic
                # Get what we're missing
                missing = gap_analyses[-1].missing_entities if gap_analyses else query_entities
                new_chunks = [c for c in all_chunks if c not in selected]

                selected, still_missing = entity_ranker.replace_chunks(
                    selected,
                    new_chunks,
                    missing,
                    budget=budget
                )
        else:
            # Fallback to standard chunk scoring
            selected = chunk_scorer.select_best_chunks(
                all_chunks,
                query,
                budget=budget,
                query_entities=query_entities
            )

        # Analyze coverage gaps
        gap_analysis = gap_detector.analyze_coverage(
            query,
            selected,
            query_entities=query_entities
        )
        gap_analyses.append(gap_analysis)

        if verbose:
            print(f"  Coverage: {gap_analysis.coverage_score:.2%}")
            print(f"  Missing: {gap_analysis.missing_entities}")

        # Check if we should continue
        if not gap_detector.should_continue_iteration(
            iteration,
            max_iterations,
            gap_analysis,
            previous_coverage
        ):
            if verbose:
                print(f"  Stopping: coverage good enough or no improvement")
            break

        previous_coverage = gap_analysis.coverage_score

    # Final selection with entity-first ranking
    if use_entity_first and entity_ranker and query_entities:
        final_results = entity_ranker.rank_by_entity_coverage(
            all_chunks,
            query_entities,
            top_k=budget
        )
    else:
        final_results = chunk_scorer.select_best_chunks(
            all_chunks,
            query,
            budget=budget,
            query_entities=query_entities
        )

    return {
        "results": final_results,
        "iterations": len(gap_analyses),
        "gap_analyses": gap_analyses,
        "queries_used": queries_used,
        "query_type": query_type.value,
        "complexity": complexity,
        "final_coverage": gap_analyses[-1].coverage_score if gap_analyses else 0.0,
        "entity_first_enabled": use_entity_first and entity_ranker is not None,
    }
