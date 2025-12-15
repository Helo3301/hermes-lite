"""Embedding interface via Ollama."""
import httpx
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings via Ollama."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        batch_size: int = 32,
        timeout: float = 60.0
    ):
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = self.client.post(
            f"{self.ollama_host}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, batching as needed."""
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info(f"Embedding batch {i // self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}")

            batch_embeddings = []
            for text in batch:
                emb = self.embed_single(text)
                batch_embeddings.append(emb)

            embeddings.extend(batch_embeddings)

        return embeddings

    def ensure_model_loaded(self) -> bool:
        """Ensure the embedding model is loaded in Ollama."""
        try:
            response = self.client.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": "5m"}
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
