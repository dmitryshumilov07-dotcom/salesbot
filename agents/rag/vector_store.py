"""
Vector store — Qdrant client for storing and searching product embeddings.
"""
import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
    PayloadSchemaType,
)

from agents.rag.embeddings import EMBEDDING_DIM

logger = structlog.get_logger()

COLLECTION_NAME = "etm_catalog"
QDRANT_URL = "http://127.0.0.1:6333"


class VectorStore:
    """Qdrant vector store for ETM product catalog."""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, timeout=30, check_compatibility=False)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if not exists."""
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME not in names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="brand",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="class_code",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="etm_code",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("qdrant_collection_created",
                        name=COLLECTION_NAME, dim=EMBEDDING_DIM)
        else:
            info = self.client.get_collection(COLLECTION_NAME)
            logger.info("qdrant_collection_exists",
                        name=COLLECTION_NAME,
                        points=info.points_count)

    def upsert_batch(self, points: list[PointStruct]):
        """Insert or update a batch of points."""
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        brand_filter: str | None = None,
        class_filter: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Search for similar products.

        Args:
            query_vector: embedding of search query
            limit: max results
            brand_filter: filter by brand name
            class_filter: filter by product class code
            score_threshold: minimum similarity score

        Returns:
            List of {score, etm_code, name, brand, article, class_name, ...}
        """
        conditions = []
        if brand_filter:
            conditions.append(
                FieldCondition(key="brand", match=MatchValue(value=brand_filter))
            )
        if class_filter:
            conditions.append(
                FieldCondition(key="class_code", match=MatchValue(value=class_filter))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
        )

        return [
            {
                "score": round(r.score, 4),
                **r.payload,
            }
            for r in results.points
        ]

    def count(self) -> int:
        """Get total number of indexed products."""
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count

    def delete_collection(self):
        """Drop and recreate collection (for re-indexing)."""
        self.client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()
        logger.info("qdrant_collection_reset", name=COLLECTION_NAME)


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
