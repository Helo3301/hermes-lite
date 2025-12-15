"""
Test Suite: Query Expansion (Spec 3)

Single-word queries often miss relevant results.
"RAG" might miss papers that say "retrieval augmented generation".

Query expansion adds synonyms and related terms automatically,
then merges results with proper deduplication.

Use cases:
- Acronym expansion: "RAG" -> ["RAG", "retrieval augmented generation"]
- Synonym inclusion: "LLM" -> ["LLM", "large language model"]
- Alternate spellings: "colour" -> ["colour", "color"]
"""

import pytest

# Import from the implementation module
from app.retrieval import (
    expand_query,
    merge_search_results,
    boost_multi_match_results,
    EXPANSION_RULES
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def simple_expansion_rules():
    """Minimal rules for predictable testing."""
    return {
        'rag': ['retrieval augmented generation'],
        'llm': ['large language model'],
        'retrieval augmented generation': ['rag'],
    }


@pytest.fixture
def rag_query_results():
    """
    Simulated results for "RAG" query.
    Papers that use the acronym.
    """
    return [
        {'id': 1, 'doc_id': 'paper-1', 'score': 0.95, 'content': 'RAG improves accuracy'},
        {'id': 2, 'doc_id': 'paper-2', 'score': 0.88, 'content': 'RAG for Q&A systems'},
        {'id': 3, 'doc_id': 'paper-3', 'score': 0.82, 'content': 'RAG vs fine-tuning'},
    ]


@pytest.fixture
def full_form_results():
    """
    Simulated results for "retrieval augmented generation" query.
    Papers that use the full form.
    """
    return [
        {'id': 4, 'doc_id': 'paper-4', 'score': 0.92, 'content': 'Retrieval augmented generation overview'},
        {'id': 1, 'doc_id': 'paper-1', 'score': 0.90, 'content': 'RAG improves accuracy'},  # DUPLICATE
        {'id': 5, 'doc_id': 'paper-5', 'score': 0.85, 'content': 'Survey of retrieval augmented generation'},
    ]


# ============================================================================
# TESTS: Query Expansion
# ============================================================================

class TestQueryExpansion:
    """Test the query expansion logic."""

    def test_expands_known_acronym(self, simple_expansion_rules):
        """Should expand known acronyms to full forms."""
        queries = expand_query('RAG', simple_expansion_rules)

        assert 'RAG' in queries, "Must include original query"
        assert 'retrieval augmented generation' in queries, "Should expand to full form"

    def test_expands_full_form_to_acronym(self, simple_expansion_rules):
        """Should expand full forms back to acronyms."""
        queries = expand_query('retrieval augmented generation', simple_expansion_rules)

        assert 'retrieval augmented generation' in queries
        assert 'rag' in queries, "Should include acronym form"

    def test_case_insensitive(self, simple_expansion_rules):
        """Expansion should work regardless of case."""
        queries_upper = expand_query('RAG', simple_expansion_rules)
        queries_lower = expand_query('rag', simple_expansion_rules)
        queries_mixed = expand_query('Rag', simple_expansion_rules)

        # All should produce expansions
        assert len(queries_upper) > 1
        assert len(queries_lower) > 1
        assert len(queries_mixed) > 1

    def test_unknown_term_returns_original_only(self, simple_expansion_rules):
        """Unknown terms should just return the original."""
        queries = expand_query('quantum computing', simple_expansion_rules)

        assert queries == ['quantum computing']

    def test_respects_max_expansions(self):
        """Should limit number of expansions."""
        # Use rules with many expansions
        many_rules = {
            'test': ['expansion1', 'expansion2', 'expansion3', 'expansion4', 'expansion5']
        }

        queries = expand_query('test', many_rules, max_expansions=2)

        # Original + 2 expansions = 3 total max
        assert len(queries) <= 3

    def test_no_duplicate_queries(self, simple_expansion_rules):
        """Should not return duplicate queries."""
        queries = expand_query('RAG', simple_expansion_rules)

        assert len(queries) == len(set(queries)), "Should have no duplicates"


# ============================================================================
# TESTS: Result Merging
# ============================================================================

class TestResultMerging:
    """Test merging results from multiple expanded queries."""

    def test_merges_without_duplicates(self, rag_query_results, full_form_results):
        """Should merge results and remove duplicates."""
        merged = merge_search_results([rag_query_results, full_form_results])

        # Should have 5 unique chunks (id 1 appears in both)
        ids = [r['id'] for r in merged]
        assert len(ids) == len(set(ids)), "Should have no duplicate IDs"
        assert len(merged) == 5

    def test_keeps_higher_score_on_duplicate(self, rag_query_results, full_form_results):
        """When same chunk appears twice, should keep higher score."""
        merged = merge_search_results([rag_query_results, full_form_results])

        # Chunk 1 has score 0.95 in rag_results and 0.90 in full_form_results
        chunk_1 = next(r for r in merged if r['id'] == 1)
        assert chunk_1['score'] == 0.95, "Should keep the higher score"

    def test_tracks_match_count(self, rag_query_results, full_form_results):
        """Should track how many queries each chunk matched."""
        merged = merge_search_results([rag_query_results, full_form_results])

        # Chunk 1 appears in both result sets
        chunk_1 = next(r for r in merged if r['id'] == 1)
        assert chunk_1['_match_count'] == 2, "Should count matches from both queries"

        # Chunk 4 only appears in full_form_results
        chunk_4 = next(r for r in merged if r['id'] == 4)
        assert chunk_4['_match_count'] == 1

    def test_respects_top_k(self, rag_query_results, full_form_results):
        """Should limit results to top_k."""
        merged = merge_search_results([rag_query_results, full_form_results], top_k=3)

        assert len(merged) == 3

    def test_sorted_by_score(self, rag_query_results, full_form_results):
        """Merged results should be sorted by score descending."""
        merged = merge_search_results([rag_query_results, full_form_results])

        scores = [r['score'] for r in merged]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# TESTS: Multi-Match Boosting
# ============================================================================

class TestMultiMatchBoosting:
    """Test boosting results that match multiple query variants."""

    def test_boosts_multi_match_results(self):
        """Results matching multiple queries should get a boost."""
        results = [
            {'id': 1, 'score': 0.90, '_match_count': 2},  # Matched both forms
            {'id': 2, 'score': 0.92, '_match_count': 1},  # Matched one form only
        ]

        boosted = boost_multi_match_results(results.copy(), boost_factor=0.05)

        # Chunk 1 should be boosted: 0.90 * 1.05 = 0.945
        chunk_1 = next(r for r in boosted if r['id'] == 1)
        assert chunk_1['score'] == pytest.approx(0.945)

    def test_reorders_after_boost(self):
        """Boosting may change the ranking order."""
        results = [
            {'id': 1, 'score': 0.92, '_match_count': 1},  # Higher base score
            {'id': 2, 'score': 0.88, '_match_count': 3},  # Lower score but 3 matches
        ]

        boosted = boost_multi_match_results(results.copy(), boost_factor=0.10)

        # Chunk 2: 0.88 * (1 + 0.10 * 2) = 0.88 * 1.20 = 1.056
        # Chunk 1: 0.92 * 1.00 = 0.92
        # Chunk 2 should now rank first
        assert boosted[0]['id'] == 2

    def test_no_boost_for_single_match(self):
        """Single-match results should not be boosted."""
        results = [
            {'id': 1, 'score': 0.90, '_match_count': 1},
        ]

        boosted = boost_multi_match_results(results.copy(), boost_factor=0.05)

        assert boosted[0]['score'] == 0.90  # Unchanged


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestQueryExpansionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query(self, simple_expansion_rules):
        """Should handle empty query."""
        queries = expand_query('', simple_expansion_rules)
        assert queries == ['']

    def test_whitespace_only_query(self, simple_expansion_rules):
        """Should handle whitespace-only query."""
        queries = expand_query('   ', simple_expansion_rules)
        assert '' in queries or '   ' in queries

    def test_empty_result_sets(self):
        """Should handle empty result sets in merge."""
        merged = merge_search_results([[], []])
        assert merged == []

    def test_single_result_set(self, rag_query_results):
        """Should handle single result set (no expansion used)."""
        merged = merge_search_results([rag_query_results])
        assert len(merged) == 3

    def test_partial_term_in_longer_query(self):
        """Should handle expansion terms within longer queries."""
        rules = {'rag': ['retrieval augmented generation']}

        queries = expand_query('advanced RAG techniques', rules)

        # Should expand "RAG" within the phrase
        assert 'advanced RAG techniques' in queries
        # May also include expanded version
        assert len(queries) >= 1


# ============================================================================
# TESTS: Integration
# ============================================================================

class TestQueryExpansionIntegration:
    """Test full expansion -> search -> merge workflow."""

    def test_full_workflow(self, simple_expansion_rules, rag_query_results, full_form_results):
        """
        Test complete workflow:
        1. Expand query
        2. Search with each variant (simulated)
        3. Merge results
        4. Boost multi-matches
        """
        # Step 1: Expand
        queries = expand_query('RAG', simple_expansion_rules)
        assert len(queries) >= 2

        # Step 2: Simulate search (we'd call search for each query)
        # Here we use our fixtures as if they're search results
        result_sets = [rag_query_results, full_form_results]

        # Step 3: Merge
        merged = merge_search_results(result_sets, top_k=10)

        # Step 4: Boost multi-matches
        final = boost_multi_match_results(merged, boost_factor=0.05)

        # Verify final results
        assert len(final) == 5  # 5 unique chunks

        # Chunk 1 (matched both) should be boosted and likely top
        chunk_1 = next(r for r in final if r['id'] == 1)
        assert chunk_1['_match_count'] == 2
        assert chunk_1['score'] > 0.95  # Boosted from 0.95


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
