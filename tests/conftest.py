"""Shared fixtures.

Tests never touch the network or load a real embedding model: `patch_embeddings`
swaps the SentenceTransformer loader for a deterministic fake, and anything that
talks to an LLM is mocked in the individual test modules.
"""

import hashlib

import numpy as np
import pytest

from config.config import Config

EMBED_DIM = 1024  # must match the collection schema in milvus_import_data


class FakeEmbedder:
    """Stand-in for SentenceTransformer that returns stable unit vectors."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True):
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "little")
            vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def patch_embeddings(monkeypatch):
    """Replace the cached model loader so no real model is ever downloaded."""
    from vector_store import embedding

    monkeypatch.setattr(embedding, "_load_model", lambda *args, **kwargs: FakeEmbedder())
    return FakeEmbedder()


@pytest.fixture
def config(tmp_path):
    """A Config pointing at a throwaway db file, with an existing data_dir."""
    return Config(
        db_path=str(tmp_path / "test.db"),
        collection="test_collection",
        data_dir=tmp_path,
        device="cpu",
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": 0,
            "doc_title": "Doc A",
            "heading_2": "Scope",
            "heading_3": "na",
            "heading_4": "na",
            "chunk": "This treaty applies to media services.",
        },
        {
            "chunk_id": 1,
            "doc_title": "Doc A",
            "heading_2": "Obligations",
            "heading_3": "Due diligence",
            "heading_4": "na",
            "chunk": "Providers must exercise due diligence.",
        },
    ]
