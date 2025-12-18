"""Enhanced embedding client with bge-m3 and Matryoshka support.

bge-m3 provides:
- 1024-dimensional embeddings with Matryoshka support
- 100+ language support
- MTEB score of 66.6 (vs 62.4 for nomic-embed-text)

Matryoshka embeddings allow dimension truncation:
- 1024d: Maximum quality (100%)
- 512d: 97% quality
- 256d: 93% quality
- 128d: 87% quality
"""

import httpx
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


# Dimension presets for different use cases
MATRYOSHKA_PRESETS = {
    "fast": 128,      # Fast search, lower quality
    "balanced": 256,  # Good balance (recommended)
    "quality": 512,   # Higher quality
    "max": 1024,      # Maximum quality
}


class EmbeddingClientV2:
    """Embedding client with bge-m3 and Matryoshka dimension support."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "bge-m3",
        full_dim: int = 1024,
        default_dim: int = 1024,  # Store full dim by default
        batch_size: int = 32,
        timeout: float = 120.0
    ):
        """
        Initialize embedding client.

        Args:
            ollama_host: Ollama server URL
            model: Embedding model name
            full_dim: Full dimension of model output
            default_dim: Default dimension for storage/search
            batch_size: Batch size for embedding
            timeout: HTTP timeout
        """
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.full_dim = full_dim
        self.default_dim = default_dim
        self.batch_size = batch_size
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def embed_single(
        self,
        text: str,
        dim: Optional[int] = None
    ) -> List[float]:
        """
        Embed a single text with optional dimension reduction.

        Args:
            text: Text to embed
            dim: Target dimension (None = default_dim)

        Returns:
            Normalized embedding vector
        """
        target_dim = dim or self.default_dim

        # Get full embedding from Ollama
        response = self.client.post(
            f"{self.ollama_host}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        full_emb = np.array(response.json()["embedding"])

        # Apply Matryoshka truncation if needed
        if target_dim < len(full_emb):
            truncated = full_emb[:target_dim]
            # Re-normalize after truncation (important for cosine similarity)
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = truncated / norm
            return truncated.tolist()

        return full_emb.tolist()

    def embed_batch(
        self,
        texts: List[str],
        dim: Optional[int] = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple texts with optional dimension reduction.

        Args:
            texts: List of texts to embed
            dim: Target dimension (None = default_dim)
            show_progress: Log progress

        Returns:
            List of embedding vectors
        """
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                logger.info(f"Embedding batch {batch_num}/{total_batches}")

            for text in batch:
                emb = self.embed_single(text, dim=dim)
                embeddings.append(emb)

        return embeddings

    def embed_for_storage(self, text: str) -> List[float]:
        """Embed for database storage (default dim)."""
        return self.embed_single(text, dim=self.default_dim)

    def embed_for_search(self, text: str) -> List[float]:
        """Embed for search (same dim as storage)."""
        return self.embed_single(text, dim=self.default_dim)

    def embed_for_rerank(self, text: str) -> List[float]:
        """Embed for reranking (full dim for max quality)."""
        return self.embed_single(text, dim=self.full_dim)

    def ensure_model_loaded(self) -> bool:
        """Ensure the embedding model is loaded in Ollama."""
        try:
            # For embedding models, just do a test embedding
            response = self.client.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.model, "prompt": "test"}
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to ensure model loaded: {e}")
            return False

    def get_model_info(self) -> Optional[dict]:
        """Get information about the embedding model."""
        try:
            response = self.client.post(
                f"{self.ollama_host}/api/show",
                json={"name": self.model}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
        return None

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def get_dimension_for_use_case(use_case: str) -> int:
    """Get recommended dimension for use case."""
    return MATRYOSHKA_PRESETS.get(use_case, 1024)


# Backwards compatibility - alias to maintain same interface as EmbeddingClient
EmbeddingClient = EmbeddingClientV2
