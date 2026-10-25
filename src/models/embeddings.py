"""
Multilingual embedding model using sentence-transformers.

Requirements: NFR-1.3, NFR-1.5, design Section 3.2
"""

import numpy as np

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """Wraps sentence-transformers multilingual model for embeddings."""

    _instance = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        """Lazy-load the model (cold-start optimisation)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a 384-dim embedding vector.

        Args:
            text: Normalised text

        Returns:
            numpy array of shape (384,)
        """
        model = self._load()
        embedding = model.encode(text, show_progress_bar=False)
        return np.array(embedding)

    def encode_batch(self, texts: list) -> np.ndarray:
        """Encode a batch of texts."""
        model = self._load()
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """Singleton accessor for cold-start efficiency."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
