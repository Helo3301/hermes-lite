"""Context compression for RAG pipelines.

Reduces the number of tokens sent to the LLM while preserving semantic meaning.
Research shows 5-10x compression is achievable with minimal quality loss.

References:
- LLMLingua: Compressing Prompts for Accelerated Inference
- Selective Context for Efficient Inference
- ICPC: In-context Prompt Compression
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    strategy: str = "sentence"  # sentence, extractive
    target_ratio: float = 0.3   # Keep 30% of original tokens
    min_sentences: int = 3      # Minimum sentences to keep
    max_tokens: int = 2000      # Hard limit on output tokens
    preserve_top_k: int = 2     # Keep top-k chunks fully uncompressed


@dataclass
class CompressedContext:
    """Result of compression."""
    text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    kept_sentences: int
    total_sentences: int


class ContextCompressor:
    """Compress retrieved context before sending to LLM."""

    def __init__(
        self,
        embed_fn=None,
        config: Optional[CompressionConfig] = None
    ):
        """
        Initialize context compressor.

        Args:
            embed_fn: Function to embed text (for relevance scoring)
            config: Compression configuration
        """
        self.embed_fn = embed_fn
        self.config = config or CompressionConfig()

    def compress(
        self,
        query: str,
        chunks: List[Dict],
        target_tokens: Optional[int] = None
    ) -> CompressedContext:
        """
        Compress retrieved chunks into condensed context.

        Args:
            query: The user query (for relevance scoring)
            chunks: List of retrieved chunks with 'content' field
            target_tokens: Override default max tokens

        Returns:
            CompressedContext with compressed text and stats
        """
        target = target_tokens or self.config.max_tokens

        if not chunks:
            return CompressedContext(
                text="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                kept_sentences=0,
                total_sentences=0
            )

        # Combine chunks, keeping top-k fully uncompressed
        top_chunks = chunks[:self.config.preserve_top_k]
        rest_chunks = chunks[self.config.preserve_top_k:]

        # Full text from top chunks (uncompressed)
        top_text = "\n\n".join(c.get("content", "") for c in top_chunks)
        top_tokens = self._count_tokens(top_text)

        # Combine remaining chunks for compression
        rest_text = "\n\n".join(c.get("content", "") for c in rest_chunks)
        rest_tokens = self._count_tokens(rest_text)

        original_tokens = top_tokens + rest_tokens
        total_sentences = len(self._split_sentences(top_text + rest_text))

        # Check if compression needed
        if original_tokens <= target:
            full_text = top_text + ("\n\n" + rest_text if rest_text else "")
            return CompressedContext(
                text=full_text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                kept_sentences=total_sentences,
                total_sentences=total_sentences
            )

        # Calculate remaining budget after top chunks
        remaining_budget = max(target - top_tokens, 200)

        # Compress the rest
        if self.config.strategy == "extractive":
            compressed_rest = self._extractive_compression(
                query, rest_text, remaining_budget
            )
        else:
            compressed_rest = self._sentence_compression(
                query, rest_text, remaining_budget
            )

        # Combine
        final_text = top_text + ("\n\n" + compressed_rest if compressed_rest else "")
        compressed_tokens = self._count_tokens(final_text)
        kept_sentences = len(self._split_sentences(final_text))

        return CompressedContext(
            text=final_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            kept_sentences=kept_sentences,
            total_sentences=total_sentences
        )

    def _sentence_compression(
        self,
        query: str,
        text: str,
        target_tokens: int
    ) -> str:
        """Compress by selecting most relevant sentences."""
        sentences = self._split_sentences(text)
        if len(sentences) <= self.config.min_sentences:
            return text

        # Score sentences by relevance to query
        if self.embed_fn:
            query_emb = np.array(self.embed_fn(query))
            scored = []
            for i, sent in enumerate(sentences):
                if len(sent.strip()) < 20:  # Skip very short sentences
                    continue
                try:
                    sent_emb = np.array(self.embed_fn(sent))
                    score = self._cosine_sim(query_emb, sent_emb)
                except Exception:
                    score = 0.5  # Default score on error
                scored.append((i, sent, score))
        else:
            # Fallback: keyword matching
            query_words = set(query.lower().split())
            scored = []
            for i, sent in enumerate(sentences):
                if len(sent.strip()) < 20:
                    continue
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                score = overlap / (len(query_words) + 1)
                scored.append((i, sent, score))

        if not scored:
            return text[:target_tokens * 4]  # Rough truncation

        # Sort by score (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        # Select top sentences until budget reached
        selected = []
        current_tokens = 0

        for idx, sent, score in scored:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > target_tokens:
                if len(selected) >= self.config.min_sentences:
                    break
            selected.append((idx, sent, score))
            current_tokens += sent_tokens

        # Sort by original order for coherence
        selected.sort(key=lambda x: x[0])

        return " ".join(sent for _, sent, _ in selected)

    def _extractive_compression(
        self,
        query: str,
        text: str,
        target_tokens: int
    ) -> str:
        """Extract key sentences using TextRank-style scoring."""
        sentences = self._split_sentences(text)
        if len(sentences) <= self.config.min_sentences:
            return text

        valid_sentences = [s for s in sentences if len(s.strip()) >= 20]
        if len(valid_sentences) <= self.config.min_sentences:
            return text

        if not self.embed_fn:
            # Fallback to sentence compression
            return self._sentence_compression(query, text, target_tokens)

        # Embed sentences
        try:
            embeddings = [np.array(self.embed_fn(s)) for s in valid_sentences]
        except Exception as e:
            logger.warning(f"Embedding failed in extractive compression: {e}")
            return self._sentence_compression(query, text, target_tokens)

        n = len(embeddings)
        if n == 0:
            return text

        # Build similarity matrix
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = self._cosine_sim(embeddings[i], embeddings[j])

        # TextRank scoring
        scores = np.ones(n)
        damping = 0.85

        for _ in range(10):
            row_sums = sim_matrix.sum(axis=1, keepdims=True) + 1e-8
            normalized = sim_matrix / row_sums
            new_scores = (1 - damping) + damping * normalized.T @ scores
            if np.allclose(scores, new_scores, atol=1e-4):
                break
            scores = new_scores

        # Also factor in query relevance
        query_emb = np.array(self.embed_fn(query))
        query_scores = np.array([
            self._cosine_sim(query_emb, e) for e in embeddings
        ])

        # Combined: 40% TextRank + 60% query relevance
        combined = 0.4 * (scores / (scores.max() + 1e-8)) + 0.6 * query_scores

        # Select top sentences
        indices = np.argsort(combined)[::-1]
        selected = []
        current_tokens = 0

        for idx in indices:
            sent = valid_sentences[idx]
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > target_tokens:
                if len(selected) >= self.config.min_sentences:
                    break
            selected.append((idx, sent))
            current_tokens += sent_tokens

        # Sort by original order
        selected.sort(key=lambda x: x[0])

        return " ".join(sent for _, sent in selected)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|etc|vs|e\.g|i\.e)\.\s', r'\1<DOT> ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


def compress_for_llm(
    query: str,
    chunks: List[Dict],
    embed_fn=None,
    max_tokens: int = 2000,
    strategy: str = "sentence"
) -> Tuple[str, Dict]:
    """
    Convenience function to compress chunks for LLM input.

    Args:
        query: User query
        chunks: Retrieved chunks
        embed_fn: Embedding function (optional, improves quality)
        max_tokens: Maximum tokens in output
        strategy: Compression strategy

    Returns:
        Tuple of (compressed_text, stats_dict)
    """
    compressor = ContextCompressor(
        embed_fn=embed_fn,
        config=CompressionConfig(
            strategy=strategy,
            max_tokens=max_tokens
        )
    )

    result = compressor.compress(query, chunks)

    stats = {
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_ratio": result.compression_ratio,
        "kept_sentences": result.kept_sentences,
        "total_sentences": result.total_sentences
    }

    return result.text, stats
