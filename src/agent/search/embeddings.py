from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from agent.config import SearchConfig


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class OpenAIEmbeddingProvider:
    def __init__(self, config: SearchConfig):
        self.model = config.embedding_model
        self.dimensions = config.embedding_dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        vectors = [item.embedding for item in response.data]
        self._validate(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _validate(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
                )


class HashEmbeddingProvider:
    """Deterministic test/dev provider with the same dimensional contract."""

    def __init__(self, dimensions: int = 3072, model: str = "hash-dev"):
        self.dimensions = dimensions
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


def normalize_vectors(vectors: list[list[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def cosine_ip_score(vector_a: list[float], vector_b: list[float]) -> float:
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if math.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(a, b) / denom)
