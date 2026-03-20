"""
Local embedding service using sentence-transformers.
No API cost. all-MiniLM-L6-v2 is fast, small (80MB), and good enough.
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    log.info("embeddings.loading_model", model=settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Returns list of 384-dim vectors."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
