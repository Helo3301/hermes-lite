"""Entity and Relationship Extraction for Knowledge Graph."""

import re
import json
import logging
from dataclasses import dataclass
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class ExtractedEntity:
    """An extracted named entity."""
    name: str
    entity_type: str  # method, dataset, author, concept, paper
    confidence: float
    span_start: Optional[int] = None
    span_end: Optional[int] = None


@dataclass
class ExtractedRelationship:
    """An extracted relationship between entities."""
    source: str
    target: str
    relationship_type: str  # cites, extends, compares, uses, outperforms
    confidence: float
    evidence: Optional[str] = None


# Known entities in the RAG/ML domain
KNOWN_METHODS = {
    "SEAL-RAG", "Self-RAG", "CRAG", "RAG", "DPR", "BM25", "ANCE",
    "ColBERT", "RETRO", "REALM", "ORQA", "FiD", "RAG-Token", "RAG-Sequence",
    "GraphRAG", "LightRAG", "HyDE", "FLARE", "REPLUG", "SAIL",
    "T5", "BERT", "GPT", "GPT-4", "Claude", "Llama", "Mistral",
    "Gemini", "PaLM", "Falcon", "Vicuna", "Alpaca",
    "LoRA", "QLoRA", "PEFT", "RLHF", "DPO", "PPO",
    "Contriever", "Spider", "BGE", "E5", "GTR",
}

KNOWN_DATASETS = {
    "HotpotQA", "2WikiMultiHopQA", "NQ", "Natural Questions",
    "MS MARCO", "BEIR", "TriviaQA", "SQuAD", "MuSiQue", "StrategyQA",
    "FEVER", "ARC", "MMLU", "TruthfulQA", "HellaSwag", "WinoGrande",
    "KILT", "ELI5", "NarrativeQA", "QuALITY", "Qasper",
    "BioASQ", "PubMedQA", "MedQA",
}

KNOWN_CONCEPTS = {
    "context dilution", "multi-hop", "dense retrieval", "sparse retrieval",
    "hybrid search", "reranking", "query expansion", "query decomposition",
    "knowledge graph", "entity extraction", "hallucination", "grounding",
    "faithfulness", "relevance", "retrieval augmented generation",
    "chain of thought", "few-shot", "zero-shot", "in-context learning",
    "prompt engineering", "instruction tuning", "fine-tuning",
    "embedding", "vector database", "semantic search", "cross-encoder",
    "bi-encoder", "late interaction", "token-level", "passage-level",
}

# Relationship patterns (more flexible matching)
RELATIONSHIP_PATTERNS = {
    "outperforms": [
        r"(\w+(?:-\w+)?)\s+(?:outperforms?|beats?|exceeds?|surpasses?)\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+achieves?\s+(?:better|higher|superior)\s+.*?\s+than\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+(?:achieves?|reaches?)\s+\d+%.*?(?:while|whereas|compared to)\s+(\w+(?:-\w+)?)",
    ],
    "uses": [
        r"(\w+(?:-\w+)?)\s+(?:uses?|employs?|leverages?|utilizes?)\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+is\s+(?:based on|built on)\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+.*?(?:uses?|with)\s+(\w+(?:-\w+)?)\s+for\s+(?:retrieval|search|reranking)",
    ],
    "extends": [
        r"(\w+(?:-\w+)?)\s+(?:extends?|builds? on|improves? on)\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+is\s+an?\s+(?:extension|improvement)\s+of\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+extends?\s+(?:the\s+)?(?:ideas?\s+)?(?:from|of)\s+(\w+(?:-\w+)?)",
    ],
    "compares": [
        r"(?:compare|comparison|comparing)\s+(\w+(?:-\w+)?)\s+(?:and|with|to|vs\.?)\s+(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+vs\.?\s+(\w+(?:-\w+)?)",
        r"(?:compare|comparing)\s+(\w+(?:-\w+)?)\s+with\s+.*?including\s+(\w+(?:-\w+)?)",
    ],
    "cites": [
        r"\(([^)]+\d{4}[^)]*)\)",  # Citations in parentheses
    ],
}


def _call_llm(prompt: str, model: str = "llama3.2") -> str:
    """Make a call to local Ollama LLM."""
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500}
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return ""


class EntityExtractor:
    """Extract named entities from text using patterns and optional LLM."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.method_patterns = self._compile_entity_patterns(KNOWN_METHODS)
        self.dataset_patterns = self._compile_entity_patterns(KNOWN_DATASETS)
        self.concept_patterns = self._compile_entity_patterns(KNOWN_CONCEPTS)

    def _compile_entity_patterns(self, entities: set) -> list:
        """Compile regex patterns for entity matching."""
        patterns = []
        for entity in entities:
            # Escape special regex characters but allow word boundaries
            escaped = re.escape(entity)
            pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
            patterns.append((pattern, entity))
        return patterns

    def extract(self, text: str, doc_id: Optional[int] = None) -> list[ExtractedEntity]:
        """
        Extract entities from text.

        Args:
            text: The text to extract entities from
            doc_id: Optional document ID for context

        Returns:
            List of ExtractedEntity objects
        """
        entities = []
        seen = set()

        # Extract methods
        for pattern, canonical_name in self.method_patterns:
            for match in pattern.finditer(text):
                if canonical_name.lower() not in seen:
                    entities.append(ExtractedEntity(
                        name=canonical_name,
                        entity_type="method",
                        confidence=1.0,
                        span_start=match.start(),
                        span_end=match.end(),
                    ))
                    seen.add(canonical_name.lower())

        # Extract datasets
        for pattern, canonical_name in self.dataset_patterns:
            for match in pattern.finditer(text):
                if canonical_name.lower() not in seen:
                    entities.append(ExtractedEntity(
                        name=canonical_name,
                        entity_type="dataset",
                        confidence=1.0,
                        span_start=match.start(),
                        span_end=match.end(),
                    ))
                    seen.add(canonical_name.lower())

        # Extract concepts
        for pattern, canonical_name in self.concept_patterns:
            for match in pattern.finditer(text):
                if canonical_name.lower() not in seen:
                    entities.append(ExtractedEntity(
                        name=canonical_name,
                        entity_type="concept",
                        confidence=0.9,
                        span_start=match.start(),
                        span_end=match.end(),
                    ))
                    seen.add(canonical_name.lower())

        # Extract additional entities using patterns
        additional = self._extract_additional_entities(text, seen)
        entities.extend(additional)

        # Optionally use LLM for more extraction
        if self.use_llm and len(text) < 5000:
            llm_entities = self._llm_extract(text, seen)
            entities.extend(llm_entities)

        return entities

    def _extract_additional_entities(self, text: str, seen: set) -> list[ExtractedEntity]:
        """Extract additional entities using heuristic patterns."""
        entities = []

        # Look for method-like acronyms (all caps, 2-6 chars, with optional suffix)
        acronym_pattern = r'\b([A-Z]{2,6}(?:-[A-Z]+)?)\b'
        for match in re.finditer(acronym_pattern, text):
            name = match.group(1)
            # Filter out common words and already seen
            if name.lower() not in seen and name not in {"THE", "AND", "FOR", "WITH", "FROM", "INTO", "UPON"}:
                # Check if it looks like a method name (appears with technical context)
                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                if any(word in context.lower() for word in ["method", "model", "approach", "system", "algorithm"]):
                    entities.append(ExtractedEntity(
                        name=name,
                        entity_type="method",
                        confidence=0.7,
                        span_start=match.start(),
                        span_end=match.end(),
                    ))
                    seen.add(name.lower())

        # Look for dataset names (common patterns)
        dataset_pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+(?:QA|Dataset|Corpus|Benchmark)?)\b'
        for match in re.finditer(dataset_pattern, text):
            name = match.group(1)
            if name.lower() not in seen and len(name) > 5:
                # Check if it looks like a dataset
                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                if any(word in context.lower() for word in ["dataset", "benchmark", "corpus", "evaluate", "test"]):
                    entities.append(ExtractedEntity(
                        name=name,
                        entity_type="dataset",
                        confidence=0.6,
                        span_start=match.start(),
                        span_end=match.end(),
                    ))
                    seen.add(name.lower())

        return entities

    def _llm_extract(self, text: str, seen: set) -> list[ExtractedEntity]:
        """Use LLM to extract additional entities."""
        prompt = f"""Extract named entities from this text. Return JSON array.
Entity types: method, dataset, concept

Text: {text[:2000]}

Format: [{{"name": "EntityName", "type": "method|dataset|concept"}}]
Only return entities not in this list: {list(seen)[:20]}

Entities:"""

        response = _call_llm(prompt)

        entities = []
        try:
            # Try to extract JSON array
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for item in parsed:
                    name = item.get("name", "")
                    etype = item.get("type", "concept")
                    if name and name.lower() not in seen:
                        entities.append(ExtractedEntity(
                            name=name,
                            entity_type=etype,
                            confidence=0.5,
                        ))
                        seen.add(name.lower())
        except json.JSONDecodeError:
            pass

        return entities


class RelationshipExtractor:
    """Extract relationships between entities."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.compiled_patterns = {}
        for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
            self.compiled_patterns[rel_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def extract(
        self,
        text: str,
        entities: list[ExtractedEntity]
    ) -> list[ExtractedRelationship]:
        """
        Extract relationships between entities in text.

        Args:
            text: The text to analyze
            entities: Pre-extracted entities

        Returns:
            List of ExtractedRelationship objects
        """
        relationships = []
        entity_names = {e.name.lower() for e in entities}

        # Pattern-based extraction
        for rel_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex >= 2:
                        source = match.group(1).strip()
                        target = match.group(2).strip()

                        # Validate both are known entities
                        source_valid = source.lower() in entity_names
                        target_valid = target.lower() in entity_names

                        if source_valid and target_valid and source != target:
                            # Get context for evidence
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            evidence = text[start:end]

                            relationships.append(ExtractedRelationship(
                                source=source,
                                target=target,
                                relationship_type=rel_type,
                                confidence=0.8,
                                evidence=evidence,
                            ))

        # Use LLM for complex relationship extraction
        if self.use_llm and entities:
            llm_rels = self._llm_extract_relationships(text, entities)
            relationships.extend(llm_rels)

        # Deduplicate
        seen = set()
        unique = []
        for rel in relationships:
            key = (rel.source.lower(), rel.target.lower(), rel.relationship_type)
            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    def _llm_extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity]
    ) -> list[ExtractedRelationship]:
        """Use LLM to extract relationships."""
        entity_names = [e.name for e in entities[:15]]

        prompt = f"""Given these entities: {entity_names}

Identify relationships between them in this text:
{text[:1500]}

Relationship types: outperforms, uses, extends, compares
Format: [{{"source": "A", "target": "B", "type": "relationship_type"}}]

Relationships:"""

        response = _call_llm(prompt)

        relationships = []
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for item in parsed:
                    source = item.get("source", "")
                    target = item.get("target", "")
                    rel_type = item.get("type", "")
                    if source and target and rel_type:
                        relationships.append(ExtractedRelationship(
                            source=source,
                            target=target,
                            relationship_type=rel_type,
                            confidence=0.5,
                        ))
        except json.JSONDecodeError:
            pass

        return relationships


class DocumentEntityExtractor:
    """High-level interface for extracting entities and relationships from documents."""

    def __init__(self, use_llm: bool = False):
        self.entity_extractor = EntityExtractor(use_llm=use_llm)
        self.relationship_extractor = RelationshipExtractor(use_llm=use_llm)

    def process_document(
        self,
        text: str,
        doc_id: int,
        chunk_texts: Optional[list[tuple[int, str]]] = None
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
        """
        Extract all entities and relationships from a document.

        Args:
            text: Full document text
            doc_id: Document ID
            chunk_texts: Optional list of (chunk_id, chunk_content) tuples

        Returns:
            Tuple of (entities, relationships)
        """
        # Extract from full document first
        all_entities = self.entity_extractor.extract(text, doc_id)

        # Also extract from individual chunks if provided
        if chunk_texts:
            seen = {e.name.lower() for e in all_entities}
            for chunk_id, chunk_text in chunk_texts:
                chunk_entities = self.entity_extractor.extract(chunk_text)
                for e in chunk_entities:
                    if e.name.lower() not in seen:
                        all_entities.append(e)
                        seen.add(e.name.lower())

        # Extract relationships
        all_relationships = self.relationship_extractor.extract(text, all_entities)

        return all_entities, all_relationships

    def extract_for_chunk(
        self,
        chunk_text: str,
        chunk_id: int,
        doc_id: int
    ) -> list[ExtractedEntity]:
        """Extract entities from a single chunk."""
        return self.entity_extractor.extract(chunk_text, doc_id)
