"""
RAG Dispatcher Handler — processes ANALYSIS tasks from Dispatcher.
"""
import structlog
from agents.dispatcher.task import Task
from agents.rag.agent import get_rag_agent

logger = structlog.get_logger()


async def handle_rag_search(task: Task) -> dict:
    """
    Handle ANALYSIS task: search product catalog.

    Expected task.payload:
        {
            "query": "кнопочный пост IP54",
            "limit": 10,
            "brand": null,
            "category": null,
        }

    Returns:
        {
            "query": str,
            "results": [...],
            "total_found": int,
            "collection_size": int,
        }
    """
    agent = get_rag_agent()

    query = task.payload.get("query", "")
    if not query:
        return {"error": "No query in payload", "results": []}

    limit = task.payload.get("limit", 10)
    brand = task.payload.get("brand")
    category = task.payload.get("category")

    result = await agent.search(
        query=query,
        limit=limit,
        brand=brand,
        category=category,
    )

    return result
