"""HyDE: Hypothetical Document Embeddings for improved retrieval.

HyDE generates a hypothetical answer to the query, embeds it, and uses
that embedding (or a blend with the original query) for retrieval.
This improves recall by 15-20% on complex queries.

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
"""

import httpx
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"


class HyDEExpander:
    """Generate hypothetical documents for improved retrieval."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        embed_fn=None,
        num_hypotheticals: int = 1,
        blend_weight: float = 0.7,
        timeout: float = 30.0
    ):
        """
        Initialize HyDE expander.

        Args:
            ollama_host: Ollama server URL
            model: LLM model for generating hypotheticals (use small/fast model)
            embed_fn: Function that takes text and returns embedding vector
            num_hypotheticals: Number of hypothetical docs to generate and average
            blend_weight: Weight for hypothetical embedding vs original (0-1)
                         0.7 = 70% hypothetical, 30% original query
            timeout: Timeout for LLM generation
        """
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.embed_fn = embed_fn
        self.num_hypotheticals = num_hypotheticals
        self.blend_weight = blend_weight
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def generate_hypothetical(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: The user's search query

        Returns:
            A hypothetical passage that answers the query
        """
        prompt = f"""You are a helpful research assistant. Write a brief technical passage (2-3 sentences)
that would appear in a research paper or textbook and addresses this topic:

Topic: {query}

Write only the factual content, no preamble or meta-commentary. Begin directly with the information:"""

        try:
            response = self.client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Some creativity helps
                        "num_predict": 150,  # Keep it short
                        "top_p": 0.9
                    }
                }
            )
            response.raise_for_status()
            hypothetical = response.json().get("response", "").strip()

            # Check for refusal patterns
            refusal_patterns = [
                "i can't", "i cannot", "i apologize", "i'm sorry",
                "as an ai", "i don't have", "i am not able"
            ]
            if any(p in hypothetical.lower() for p in refusal_patterns):
                logger.warning(f"HyDE got refusal, falling back to original query")
                return query

            logger.debug(f"HyDE generated: {hypothetical[:100]}...")
            return hypothetical
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}, falling back to original query")
            return query  # Fallback to original query

    def expand_query(
        self,
        query: str,
        return_hypothetical: bool = False
    ) -> Tuple[List[float], Optional[str]]:
        """
        Expand query using HyDE - generate hypothetical and blend embeddings.

        Args:
            query: Original user query
            return_hypothetical: If True, also return the generated text

        Returns:
            Tuple of (blended embedding vector, optional hypothetical text)
        """
        if self.embed_fn is None:
            raise ValueError("embed_fn must be set before calling expand_query")

        # Generate hypothetical document(s)
        hypotheticals = []
        for i in range(self.num_hypotheticals):
            hypo = self.generate_hypothetical(query)
            if hypo and hypo != query:  # Only add if generation succeeded
                hypotheticals.append(hypo)
                logger.info(f"HyDE hypothetical {i+1}: {hypo[:80]}...")

        # If no hypotheticals generated, just return original query embedding
        if not hypotheticals:
            logger.warning("No hypotheticals generated, using original query")
            emb = self.embed_fn(query)
            return (emb, None) if return_hypothetical else emb

        # Embed original query
        query_embedding = np.array(self.embed_fn(query))

        # Embed hypotheticals and average
        hypo_embeddings = [np.array(self.embed_fn(h)) for h in hypotheticals]
        avg_hypo_embedding = np.mean(hypo_embeddings, axis=0)

        # Blend embeddings: weighted combination
        blended = (
            self.blend_weight * avg_hypo_embedding +
            (1 - self.blend_weight) * query_embedding
        )

        # Normalize to unit vector
        blended = blended / (np.linalg.norm(blended) + 1e-8)

        if return_hypothetical:
            return blended.tolist(), hypotheticals[0]
        return blended.tolist()

    def set_embed_fn(self, embed_fn):
        """Set the embedding function after initialization."""
        self.embed_fn = embed_fn


# Pre-defined prompts for different query types
HYDE_PROMPTS = {
    "factual": """Write a short, factual passage (2-3 sentences) that directly answers:
Question: {query}
Passage:""",

    "technical": """Write a technical explanation (2-3 sentences) answering:
Question: {query}
Technical explanation:""",

    "comparison": """Write a balanced comparison (2-3 sentences) addressing:
Question: {query}
Comparison:""",

    "definition": """Provide a clear, precise definition (1-2 sentences) for:
Question: {query}
Definition:""",

    "how_to": """Explain the process or method (2-3 sentences) for:
Question: {query}
Process:""",
}


def detect_query_type(query: str) -> str:
    """
    Simple heuristic to detect query type for prompt selection.

    Args:
        query: The user query

    Returns:
        Query type string: factual, technical, comparison, definition, how_to
    """
    query_lower = query.lower()

    if any(w in query_lower for w in ["what is", "define", "meaning of"]):
        return "definition"
    if any(w in query_lower for w in ["how to", "how do", "how can", "steps to"]):
        return "how_to"
    if any(w in query_lower for w in ["compare", "difference between", "vs", "versus"]):
        return "comparison"
    if any(w in query_lower for w in ["explain", "describe", "how does", "why does"]):
        return "technical"

    return "factual"


def get_hyde_prompt(query: str, query_type: Optional[str] = None) -> str:
    """
    Get appropriate HyDE prompt based on query type.

    Args:
        query: The user query
        query_type: Optional explicit type, otherwise auto-detected

    Returns:
        Formatted prompt string
    """
    if query_type is None:
        query_type = detect_query_type(query)

    template = HYDE_PROMPTS.get(query_type, HYDE_PROMPTS["factual"])
    return template.format(query=query)
