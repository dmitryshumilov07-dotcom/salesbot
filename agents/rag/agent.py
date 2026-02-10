"""
RAG Agent — searches product catalog by semantic similarity.
Called by Dispatcher when a task of type ANALYSIS arrives.

Provides:
1. search(query, filters) — semantic search in Qdrant
2. get_status() — collection stats
"""
import structlog

from agents.rag.embeddings import get_embedding_engine
from agents.rag.vector_store import get_vector_store

logger = structlog.get_logger()


class RAGAgent:
    """RAG agent for product catalog search."""

    def __init__(self):
        self.embedder = get_embedding_engine()
        self.store = get_vector_store()

    async def search(
        self,
        query: str,
        limit: int = 10,
        brand: str | None = None,
        category: str | None = None,
        min_score: float = 0.3,
    ) -> dict:
        """
        Semantic search in product catalog.

        Args:
            query: search text (product name, description, specs)
            limit: max results (default 10)
            brand: filter by brand name
            category: filter by class_code
            min_score: minimum similarity threshold

        Returns:
            {
                "query": str,
                "results": [
                    {
                        "score": 0.85,
                        "etm_code": "9536092",
                        "name": "Пост кнопочный ПКУ 15-21",
                        "brand": "Электротехник",
                        "article": "ET054487",
                        "class_name": "Кнопочные посты",
                        ...
                    },
                    ...
                ],
                "total_found": int,
                "collection_size": int,
            }
        """
        logger.info("rag_search", query=query[:100], brand=brand, category=category)

        # Encode query
        query_vector = self.embedder.encode_query(query)

        # Search in Qdrant
        results = self.store.search(
            query_vector=query_vector,
            limit=limit,
            brand_filter=brand,
            class_filter=category,
            score_threshold=min_score,
        )

        collection_size = self.store.count()

        logger.info("rag_search_done",
                     query=query[:50],
                     found=len(results),
                     collection_size=collection_size)

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "collection_size": collection_size,
        }

    async def get_status(self) -> dict:
        """Get RAG system status."""
        count = self.store.count()
        return {
            "collection": "etm_catalog",
            "indexed_products": count,
            "embedding_model": "intfloat/multilingual-e5-small",
            "embedding_dim": 384,
            "status": "ready" if count > 0 else "empty",
        }


_agent: RAGAgent | None = None


def get_rag_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent
