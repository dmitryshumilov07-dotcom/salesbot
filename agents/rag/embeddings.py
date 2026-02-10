"""
Embedding engine for RAG — converts text to vectors.
Uses sentence-transformers with a multilingual model good for Russian text.
"""
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

# Good multilingual model, ~100MB, 384-dim vectors, fast inference
MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384


class EmbeddingEngine:
    """Singleton embedding engine."""

    def __init__(self):
        logger.info("embedding_loading", model=MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info("embedding_loaded", dim=EMBEDDING_DIM)

    def encode(self, texts: list[str], prefix: str = "passage: ") -> list[list[float]]:
        """
        Encode texts into vectors.

        For E5 models, use prefixes:
        - "passage: " for documents being indexed
        - "query: "   for search queries
        """
        prefixed = [f"{prefix}{t}" for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a search query."""
        return self.encode([query], prefix="query: ")[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode documents for indexing."""
        return self.encode(texts, prefix="passage: ")


_engine: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
