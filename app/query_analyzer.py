"""Query Intelligence: Classification, decomposition, and entity extraction."""

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


class QueryType(Enum):
    """Types of queries the system can handle."""
    SIMPLE = "simple"           # Factoid lookup: "What is RAG?"
    MULTI_HOP = "multi_hop"     # Requires combining info: "How does X improve Y?"
    COMPARATIVE = "comparative"  # Comparing entities: "Compare A and B"
    EXPLORATORY = "exploratory"  # Open-ended: "What are the challenges in X?"


class Intent(Enum):
    """User's underlying intent."""
    LOOKUP = "lookup"     # Get a definition or fact
    COMPARE = "compare"   # Compare two or more things
    EXPLAIN = "explain"   # Deep explanation of how something works
    EXPLORE = "explore"   # Broad exploration of a topic


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str
    query_type: QueryType
    target_entity: Optional[str] = None


@dataclass
class QueryAnalysis:
    """Complete analysis of a query."""
    original_query: str
    query_type: QueryType
    intent: Intent
    complexity: int  # 1-5
    entities: list[str]
    sub_queries: list[SubQuery]


# Patterns for classification
COMPARATIVE_PATTERNS = [
    r'\bcompare\b',
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\bdifference(?:s)? between\b',
    r'\bhow (?:do|does) .+ differ\b',
    r'\btradeoffs?\b',
    r'\bpros and cons\b',
    r'\badvantages and disadvantages\b',
    r'\bwhich (?:is|are) better\b',
]

EXPLORATORY_PATTERNS = [
    r'\bwhat are the (?:main|key|major|primary)\b',
    r'\bwhat (?:advances|developments|trends)\b',
    r'\bhow has .+ evolved\b',
    r'\bwhat (?:challenges|problems|issues)\b',
    r'\bwhat (?:techniques|methods|approaches)\b',
    r'\bemerging\b',
    r'\bbest practices\b',
    r'\bopen problems\b',
    r'\brecent (?:research|work|advances)\b',
]

MULTI_HOP_PATTERNS = [
    r'\bhow does .+ (?:address|solve|improve|handle)\b',
    r'\bwhat .+ does .+ use\b',
    r'\bwhich .+ (?:cite|reference|use)\b',
    r'\bhow does .+ relate to\b',
    r'\bwhat (?:role|part) does .+ play\b',
    r'\bwhat evidence\b',
    r'\baccording to\b',
]

# Known entities in the RAG domain
RAG_ENTITIES = {
    # Methods and systems
    "SEAL-RAG", "Self-RAG", "CRAG", "RAG", "DPR", "BM25", "ANCE",
    "ColBERT", "RETRO", "REALM", "ORQA", "FiD", "RAG-Token", "RAG-Sequence",
    "GraphRAG", "LightRAG", "HyDE", "FLARE", "REPLUG",

    # Datasets
    "HotpotQA", "2WikiMultiHopQA", "NQ", "Natural Questions",
    "MS MARCO", "BEIR", "TriviaQA", "SQuAD", "MuSiQue", "StrategyQA",
    "FEVER", "ARC", "MMLU",

    # Concepts
    "context dilution", "multi-hop", "dense retrieval", "sparse retrieval",
    "hybrid search", "reranking", "query expansion", "query decomposition",
    "knowledge graph", "entity extraction", "hallucination",
    "faithfulness", "relevance", "grounding", "retrieval augmented generation",

    # Metrics
    "RAGAS", "F1", "EM", "exact match", "recall", "precision",
}


def _call_llm(prompt: str, model: str = "llama3.2") -> str:
    """Make a call to local Ollama LLM."""
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200}
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return ""


class QueryClassifier:
    """Classify queries into types using pattern matching and LLM."""

    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback
        self._compile_patterns()

    def _compile_patterns(self):
        self.comparative_re = [re.compile(p, re.I) for p in COMPARATIVE_PATTERNS]
        self.exploratory_re = [re.compile(p, re.I) for p in EXPLORATORY_PATTERNS]
        self.multi_hop_re = [re.compile(p, re.I) for p in MULTI_HOP_PATTERNS]

    def classify(self, query: str) -> QueryType:
        """Classify a query into a QueryType."""
        query_lower = query.lower()
        word_count = len(query.split())

        # Simple definitional patterns (check first, highest priority)
        simple_definitional = [
            r'^what is [\w\-]+\??$',  # "What is RAG?" or "What is SEAL-RAG?"
            r'^define [\w\-\s]+$',    # "Define X"
            r'^what (?:is|are) (?:the )?[\w\-]+\??$',  # "What is the X?"
        ]
        for pattern in simple_definitional:
            if re.match(pattern, query_lower):
                return QueryType.SIMPLE

        # Check comparative patterns (explicit comparisons)
        for pattern in self.comparative_re:
            if pattern.search(query):
                return QueryType.COMPARATIVE

        # Check exploratory patterns (broad questions about trends/challenges)
        for pattern in self.exploratory_re:
            if pattern.search(query):
                return QueryType.EXPLORATORY

        # Check multi-hop patterns (how X relates to Y, what X does for Y)
        for pattern in self.multi_hop_re:
            if pattern.search(query):
                return QueryType.MULTI_HOP

        # Additional multi-hop patterns
        multi_hop_additional = [
            r'\bwhat .+ (?:have been|has been) proposed\b',
            r'\bwhat .+ show\b',
            r'\bhow do .+ integrate\b',
            r'\bhow does .+ help\b',
        ]
        for pattern in multi_hop_additional:
            if re.search(pattern, query_lower):
                return QueryType.MULTI_HOP

        # Check for simple question patterns (with looser constraints)
        simple_patterns = [
            r'^what is\b',
            r'^what are\b',
            r'^define\b',
            r'^who is\b',
            r'^when (?:was|did)\b',
            r'^where (?:is|was|did)\b',
        ]
        for pattern in simple_patterns:
            if re.match(pattern, query_lower):
                # Simple if short query (<=10 words)
                if word_count <= 10:
                    return QueryType.SIMPLE

        # Check for multiple distinct entities (suggests multi-hop or comparative)
        entities = self.extract_entities(query)
        # Filter out entities that are substrings of each other
        distinct_entities = []
        for e in entities:
            is_substring = any(
                e.lower() != other.lower() and e.lower() in other.lower()
                for other in entities
            )
            if not is_substring:
                distinct_entities.append(e)

        if len(distinct_entities) >= 2:
            # If we see "and" with multiple entities, likely comparative
            if " and " in query_lower or " vs " in query_lower:
                return QueryType.COMPARATIVE
            # "differ" with multiple entities is comparative
            if "differ" in query_lower:
                return QueryType.COMPARATIVE
            return QueryType.MULTI_HOP

        # Default heuristic: short queries are simple, long ones are multi-hop
        if word_count <= 8:
            return QueryType.SIMPLE

        # For ambiguous cases, use LLM if available
        if self.use_llm_fallback:
            return self._llm_classify(query)

        return QueryType.SIMPLE

    def _llm_classify(self, query: str) -> QueryType:
        """Use LLM for ambiguous classification."""
        prompt = f"""Classify this query into exactly one category:
- SIMPLE: Basic factoid question (e.g., "What is X?")
- MULTI_HOP: Requires combining info from multiple sources (e.g., "How does X solve Y's problem?")
- COMPARATIVE: Compares two or more things (e.g., "Compare X and Y")
- EXPLORATORY: Open-ended exploration (e.g., "What are the challenges in X?")

Query: {query}

Answer with ONLY the category name (SIMPLE, MULTI_HOP, COMPARATIVE, or EXPLORATORY):"""

        response = _call_llm(prompt)

        # Parse response
        response_upper = response.upper().strip()
        if "COMPARATIVE" in response_upper:
            return QueryType.COMPARATIVE
        elif "EXPLORATORY" in response_upper:
            return QueryType.EXPLORATORY
        elif "MULTI_HOP" in response_upper or "MULTI-HOP" in response_upper:
            return QueryType.MULTI_HOP

        return QueryType.SIMPLE

    def extract_entities(self, query: str) -> list[str]:
        """Extract known entities from query."""
        found = []
        query_lower = query.lower()

        for entity in RAG_ENTITIES:
            if entity.lower() in query_lower:
                found.append(entity)

        # Also try to find potential entity patterns (capitalized words/phrases)
        # Look for acronyms (all caps 2+ letters)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        for acr in acronyms:
            if acr not in found:
                found.append(acr)

        # Look for CamelCase or hyphenated terms
        special_terms = re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b', query)
        for term in special_terms:
            if term not in found:
                found.append(term)

        return found


class QueryDecomposer:
    """Decompose complex queries into simpler sub-queries."""

    def __init__(self, classifier: Optional[QueryClassifier] = None):
        self.classifier = classifier or QueryClassifier()

    def decompose(self, query: str) -> list[SubQuery]:
        """
        Decompose a complex query into sub-queries.

        For simple queries, returns just the original.
        For complex queries, breaks down into component questions.
        """
        query_type = self.classifier.classify(query)
        entities = self.classifier.extract_entities(query)

        # Simple queries don't need decomposition
        if query_type == QueryType.SIMPLE:
            return [SubQuery(query=query, query_type=QueryType.SIMPLE)]

        # Comparative: decompose into entity lookups + comparison
        if query_type == QueryType.COMPARATIVE and len(entities) >= 2:
            return self._decompose_comparative(query, entities)

        # Multi-hop: try to identify the chain
        if query_type == QueryType.MULTI_HOP:
            return self._decompose_multi_hop(query, entities)

        # Exploratory: create focused sub-queries
        if query_type == QueryType.EXPLORATORY:
            return self._decompose_exploratory(query, entities)

        # Fallback to LLM decomposition
        return self._llm_decompose(query, query_type)

    def _decompose_comparative(self, query: str, entities: list[str]) -> list[SubQuery]:
        """Decompose comparative query into entity lookups + comparison."""
        sub_queries = []

        # Add lookup for each entity
        for entity in entities[:3]:  # Limit to 3 entities
            sub_queries.append(SubQuery(
                query=f"What is {entity}?",
                query_type=QueryType.SIMPLE,
                target_entity=entity
            ))

        # Add the comparative query itself
        sub_queries.append(SubQuery(
            query=query,
            query_type=QueryType.COMPARATIVE,
        ))

        return sub_queries

    def _decompose_multi_hop(self, query: str, entities: list[str]) -> list[SubQuery]:
        """Decompose multi-hop query into component questions."""
        sub_queries = []

        # Common pattern: "How does X address Y?"
        # -> "What is X?", "What is Y?", original
        if entities:
            for entity in entities[:2]:
                sub_queries.append(SubQuery(
                    query=f"What is {entity}?",
                    query_type=QueryType.SIMPLE,
                    target_entity=entity
                ))

        # Check for specific patterns
        patterns = [
            (r'how does (.+) (?:address|solve|fix|improve) (.+)', 2),
            (r'what (.+) does (.+) use', 2),
            (r'how does (.+) compare to (.+)', 2),
        ]

        for pattern, num_groups in patterns:
            match = re.search(pattern, query, re.I)
            if match:
                for i in range(1, num_groups + 1):
                    group = match.group(i).strip()
                    if group and len(group) < 50:
                        sub_queries.append(SubQuery(
                            query=f"What is {group}?",
                            query_type=QueryType.SIMPLE,
                            target_entity=group
                        ))

        # Always include original
        sub_queries.append(SubQuery(
            query=query,
            query_type=QueryType.MULTI_HOP,
        ))

        # Deduplicate by query text
        seen = set()
        unique = []
        for sq in sub_queries:
            if sq.query.lower() not in seen:
                seen.add(sq.query.lower())
                unique.append(sq)

        return unique

    def _decompose_exploratory(self, query: str, entities: list[str]) -> list[SubQuery]:
        """Decompose exploratory query into focused sub-queries."""
        sub_queries = []

        # For exploratory, we want to search broadly
        # Keep original query
        sub_queries.append(SubQuery(
            query=query,
            query_type=QueryType.EXPLORATORY,
        ))

        # Add entity-specific lookups if present
        for entity in entities[:2]:
            sub_queries.append(SubQuery(
                query=f"What is {entity}?",
                query_type=QueryType.SIMPLE,
                target_entity=entity
            ))

        return sub_queries

    def _llm_decompose(self, query: str, query_type: QueryType) -> list[SubQuery]:
        """Use LLM for complex decomposition."""
        prompt = f"""Decompose this query into 2-4 simpler sub-questions that together answer the original.
Each sub-question should be self-contained and searchable.

Query: {query}

Output format (JSON array of strings):
["sub-question 1", "sub-question 2", ...]

Sub-questions:"""

        response = _call_llm(prompt)

        try:
            # Try to extract JSON array
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                sub_queries_text = json.loads(match.group())
                return [
                    SubQuery(query=sq, query_type=QueryType.SIMPLE)
                    for sq in sub_queries_text if sq
                ]
        except json.JSONDecodeError:
            pass

        # Fallback: return original
        return [SubQuery(query=query, query_type=query_type)]


class QueryAnalyzer:
    """Main query analysis interface."""

    def __init__(self, use_llm: bool = True):
        self.classifier = QueryClassifier(use_llm_fallback=use_llm)
        self.decomposer = QueryDecomposer(classifier=self.classifier)

    def analyze(self, query: str) -> QueryAnalysis:
        """Perform complete analysis of a query."""
        query_type = self.classifier.classify(query)
        entities = self.classifier.extract_entities(query)
        sub_queries = self.decomposer.decompose(query)
        intent = self._detect_intent(query, query_type)
        complexity = self._estimate_complexity(query, query_type, entities)

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            intent=intent,
            complexity=complexity,
            entities=entities,
            sub_queries=sub_queries,
        )

    def _detect_intent(self, query: str, query_type: QueryType) -> Intent:
        """Detect the user's underlying intent."""
        query_lower = query.lower()

        if query_type == QueryType.COMPARATIVE:
            return Intent.COMPARE

        if query_type == QueryType.EXPLORATORY:
            return Intent.EXPLORE

        # Check for explanation intent
        if any(word in query_lower for word in ["how", "why", "explain", "describe"]):
            if query_type == QueryType.MULTI_HOP:
                return Intent.EXPLAIN

        return Intent.LOOKUP

    def _estimate_complexity(
        self,
        query: str,
        query_type: QueryType,
        entities: list[str]
    ) -> int:
        """Estimate query complexity on 1-5 scale."""
        base_complexity = {
            QueryType.SIMPLE: 1,
            QueryType.MULTI_HOP: 3,
            QueryType.COMPARATIVE: 3,
            QueryType.EXPLORATORY: 4,
        }

        complexity = base_complexity.get(query_type, 2)

        # Adjust for number of entities
        complexity += min(len(entities), 2)

        # Adjust for query length
        word_count = len(query.split())
        if word_count > 15:
            complexity += 1

        return min(complexity, 5)

    def classify(self, query: str) -> QueryType:
        """Quick classification without full analysis."""
        return self.classifier.classify(query)

    def extract_entities(self, query: str) -> list[str]:
        """Quick entity extraction without full analysis."""
        return self.classifier.extract_entities(query)

    def decompose(self, query: str) -> list[SubQuery]:
        """Quick decomposition without full analysis."""
        return self.decomposer.decompose(query)
