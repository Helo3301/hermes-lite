"""Contradiction Detection: Extract claims and detect conflicts between chunks."""

import re
import json
import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import urllib.request

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


class ConflictType(Enum):
    """Types of conflicts between claims."""
    FACTUAL = "factual"           # Different numbers/facts
    METHODOLOGICAL = "methodological"  # Different approaches described
    INTERPRETIVE = "interpretive"   # Different conclusions from same data


class Severity(Enum):
    """Severity of a detected conflict."""
    HIGH = "high"      # Direct contradiction
    MEDIUM = "medium"  # Partial disagreement
    LOW = "low"        # Minor discrepancy


@dataclass
class Claim:
    """A factual claim extracted from a chunk."""
    text: str
    claim_type: str  # statistic, comparison, methodology, result
    subject: Optional[str]  # What the claim is about
    confidence: float
    chunk_id: Optional[int] = None
    source: Optional[str] = None  # Filename or document identifier


@dataclass
class Contradiction:
    """A detected contradiction between two claims."""
    claim_a: Claim
    claim_b: Claim
    conflict_type: ConflictType
    severity: Severity
    explanation: str
    confidence: float


def _call_llm(prompt: str, model: str = "llama3.2") -> str:
    """Make a call to local Ollama LLM."""
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300}
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return ""


class ClaimExtractor:
    """Extract factual claims from text chunks."""

    # Patterns for different claim types
    STATISTIC_PATTERNS = [
        r'(?:achieves?|reaches?|obtains?|gets?)\s+(\d+(?:\.\d+)?%?\s*(?:accuracy|precision|recall|F1|score))',
        r'(\d+(?:\.\d+)?%)\s+(?:accuracy|precision|recall|improvement)',
        r'(?:improves? by|increases? by|reduces? by)\s+(\d+(?:\.\d+)?%?)',
    ]

    COMPARISON_PATTERNS = [
        r'(\w+(?:-\w+)?)\s+(?:outperforms?|beats?|exceeds?)\s+(\w+(?:-\w+)?)',
        r'(\w+(?:-\w+)?)\s+is\s+(?:better|worse|faster|slower)\s+than\s+(\w+(?:-\w+)?)',
        r'(\w+(?:-\w+)?)\s+(?:achieves?|has)\s+(?:higher|lower)\s+.*?\s+than\s+(\w+(?:-\w+)?)',
    ]

    METHODOLOGY_PATTERNS = [
        r'(?:uses?|employs?|applies?|implements?)\s+([\w\s-]+(?:method|approach|technique|algorithm))',
        r'(?:based on|built on|extends?)\s+([\w\s-]+)',
    ]

    RESULT_PATTERNS = [
        r'(?:shows?|demonstrates?|proves?|indicates?)\s+that\s+(.+?)(?:\.|$)',
        r'(?:results? show|experiments? show|we find)\s+(.+?)(?:\.|$)',
    ]

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._compile_patterns()

    def _compile_patterns(self):
        self.stat_re = [re.compile(p, re.I) for p in self.STATISTIC_PATTERNS]
        self.comp_re = [re.compile(p, re.I) for p in self.COMPARISON_PATTERNS]
        self.meth_re = [re.compile(p, re.I) for p in self.METHODOLOGY_PATTERNS]
        self.res_re = [re.compile(p, re.I) for p in self.RESULT_PATTERNS]

    def extract_claims(
        self,
        chunk: dict,
        min_confidence: float = 0.5
    ) -> list[Claim]:
        """
        Extract factual claims from a chunk.

        Args:
            chunk: Dict with 'content', 'id', 'filename' fields
            min_confidence: Minimum confidence to include a claim

        Returns:
            List of extracted Claim objects
        """
        content = chunk.get("content", "")
        chunk_id = chunk.get("id")
        source = chunk.get("filename", "")

        claims = []

        # Extract statistics
        for pattern in self.stat_re:
            for match in pattern.finditer(content):
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end].strip()

                claims.append(Claim(
                    text=context,
                    claim_type="statistic",
                    subject=self._extract_subject(content, match.start()),
                    confidence=0.9,
                    chunk_id=chunk_id,
                    source=source,
                ))

        # Extract comparisons
        for pattern in self.comp_re:
            for match in pattern.finditer(content):
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                context = content[start:end].strip()

                claims.append(Claim(
                    text=context,
                    claim_type="comparison",
                    subject=match.group(1) if match.lastindex else None,
                    confidence=0.8,
                    chunk_id=chunk_id,
                    source=source,
                ))

        # Extract methodology claims
        for pattern in self.meth_re:
            for match in pattern.finditer(content):
                start = max(0, match.start() - 20)
                end = min(len(content), match.end() + 20)
                context = content[start:end].strip()

                claims.append(Claim(
                    text=context,
                    claim_type="methodology",
                    subject=match.group(1) if match.lastindex else None,
                    confidence=0.7,
                    chunk_id=chunk_id,
                    source=source,
                ))

        # Filter by confidence
        claims = [c for c in claims if c.confidence >= min_confidence]

        # Use LLM for additional extraction if enabled
        if self.use_llm and len(claims) < 2:
            llm_claims = self._llm_extract_claims(content, chunk_id, source)
            claims.extend(llm_claims)

        return claims

    def _extract_subject(self, content: str, position: int) -> Optional[str]:
        """Try to identify the subject of a claim from context."""
        # Look for entity-like patterns before the position
        before = content[max(0, position-100):position]

        # Look for capitalized words or known patterns
        patterns = [
            r'(\b[A-Z][\w-]+(?:-[A-Z][\w]+)?)\b',  # CamelCase or hyphenated
            r'\b([A-Z]{2,})\b',  # Acronyms
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, before))
            if matches:
                return matches[-1].group(1)

        return None

    def _llm_extract_claims(
        self,
        content: str,
        chunk_id: Optional[int],
        source: str
    ) -> list[Claim]:
        """Use LLM to extract claims when pattern matching fails."""
        prompt = f"""Extract factual claims from this text. Return JSON array.

Text: {content[:1000]}

Return claims in format:
[{{"text": "claim text", "type": "statistic|comparison|methodology|result"}}]

Claims:"""

        response = _call_llm(prompt)
        claims = []

        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for item in parsed:
                    claims.append(Claim(
                        text=item.get("text", ""),
                        claim_type=item.get("type", "result"),
                        subject=None,
                        confidence=0.5,
                        chunk_id=chunk_id,
                        source=source,
                    ))
        except json.JSONDecodeError:
            pass

        return claims


class ContradictionDetector:
    """Detect contradictions between claims from different chunks."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.claim_extractor = ClaimExtractor(use_llm=use_llm)

    def extract_claims_from_chunks(
        self,
        chunks: list[dict]
    ) -> list[Claim]:
        """Extract all claims from a list of chunks."""
        all_claims = []
        for chunk in chunks:
            claims = self.claim_extractor.extract_claims(chunk)
            all_claims.extend(claims)
        return all_claims

    def detect_contradictions(
        self,
        chunks: list[dict],
        claims: Optional[list[Claim]] = None
    ) -> list[Contradiction]:
        """
        Detect contradictions between retrieved chunks.

        Args:
            chunks: List of retrieved chunk dicts
            claims: Pre-extracted claims (optional)

        Returns:
            List of detected Contradiction objects
        """
        if claims is None:
            claims = self.extract_claims_from_chunks(chunks)

        if len(claims) < 2:
            return []

        contradictions = []

        # Compare each pair of claims
        for i, claim_a in enumerate(claims):
            for claim_b in claims[i+1:]:
                # Skip claims from same chunk
                if claim_a.chunk_id == claim_b.chunk_id:
                    continue

                # Check for potential contradiction
                contradiction = self._check_contradiction(claim_a, claim_b)
                if contradiction:
                    contradictions.append(contradiction)

        return contradictions

    def _check_contradiction(
        self,
        claim_a: Claim,
        claim_b: Claim
    ) -> Optional[Contradiction]:
        """Check if two claims contradict each other."""
        # Quick heuristic checks first

        # 1. Check for conflicting statistics about same subject
        if claim_a.claim_type == "statistic" and claim_b.claim_type == "statistic":
            if claim_a.subject and claim_b.subject:
                if claim_a.subject.lower() == claim_b.subject.lower():
                    # Extract numbers from both claims
                    nums_a = re.findall(r'\d+(?:\.\d+)?%?', claim_a.text)
                    nums_b = re.findall(r'\d+(?:\.\d+)?%?', claim_b.text)

                    if nums_a and nums_b and nums_a[0] != nums_b[0]:
                        return Contradiction(
                            claim_a=claim_a,
                            claim_b=claim_b,
                            conflict_type=ConflictType.FACTUAL,
                            severity=Severity.HIGH,
                            explanation=f"Conflicting statistics for {claim_a.subject}: {nums_a[0]} vs {nums_b[0]}",
                            confidence=0.9,
                        )

        # 2. Check for conflicting comparisons
        if claim_a.claim_type == "comparison" and claim_b.claim_type == "comparison":
            # Look for "X outperforms Y" vs "Y outperforms X"
            if self._are_reversed_comparisons(claim_a.text, claim_b.text):
                return Contradiction(
                    claim_a=claim_a,
                    claim_b=claim_b,
                    conflict_type=ConflictType.FACTUAL,
                    severity=Severity.HIGH,
                    explanation="Reversed comparison claims",
                    confidence=0.85,
                )

        # 3. Use LLM for more nuanced detection
        if self.use_llm:
            return self._llm_check_contradiction(claim_a, claim_b)

        return None

    def _are_reversed_comparisons(self, text_a: str, text_b: str) -> bool:
        """Check if two comparison claims are reversed (A>B vs B>A)."""
        pattern = r'(\w+(?:-\w+)?)\s+(?:outperforms?|beats?|exceeds?|is better than)\s+(\w+(?:-\w+)?)'

        match_a = re.search(pattern, text_a, re.I)
        match_b = re.search(pattern, text_b, re.I)

        if match_a and match_b:
            # Check if A>B in one and B>A in other
            subj_a, obj_a = match_a.group(1).lower(), match_a.group(2).lower()
            subj_b, obj_b = match_b.group(1).lower(), match_b.group(2).lower()

            if subj_a == obj_b and obj_a == subj_b:
                return True

        return False

    def _llm_check_contradiction(
        self,
        claim_a: Claim,
        claim_b: Claim
    ) -> Optional[Contradiction]:
        """Use LLM to detect contradictions between claims."""
        prompt = f"""Do these two claims contradict each other?

Claim 1: {claim_a.text}
Claim 2: {claim_b.text}

If they contradict, respond: CONTRADICTION: [brief explanation]
If they don't contradict, respond: NO CONTRADICTION

Answer:"""

        response = _call_llm(prompt)

        if response.upper().startswith("CONTRADICTION"):
            explanation = response.split(":", 1)[1].strip() if ":" in response else "LLM detected contradiction"
            return Contradiction(
                claim_a=claim_a,
                claim_b=claim_b,
                conflict_type=ConflictType.INTERPRETIVE,
                severity=Severity.MEDIUM,
                explanation=explanation,
                confidence=0.7,
            )

        return None

    def surface_contradictions(
        self,
        contradictions: list[Contradiction],
        strategy: str = "surface"
    ) -> dict:
        """
        Handle detected contradictions.

        Args:
            contradictions: List of detected contradictions
            strategy: How to handle:
                - 'surface': Return info about disagreements
                - 'newest_wins': Prefer more recent source
                - 'most_cited': Prefer more authoritative source

        Returns:
            Dict with contradiction handling results
        """
        if not contradictions:
            return {
                "has_contradictions": False,
                "count": 0,
                "details": [],
            }

        details = []
        for c in contradictions:
            details.append({
                "claim_a": c.claim_a.text[:200],
                "source_a": c.claim_a.source,
                "claim_b": c.claim_b.text[:200],
                "source_b": c.claim_b.source,
                "conflict_type": c.conflict_type.value,
                "severity": c.severity.value,
                "explanation": c.explanation,
            })

        return {
            "has_contradictions": True,
            "count": len(contradictions),
            "strategy": strategy,
            "details": details,
            "warning": "Retrieved information contains conflicting claims. Please verify with primary sources.",
        }
