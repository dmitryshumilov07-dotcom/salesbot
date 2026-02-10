"""
ETM Catalog Loader — downloads ETM catalog via API and indexes into Qdrant.

Flow:
1. Call ETM API export_catalog() (SgGds) — async job, up to 3 hours
2. Parse JSON catalog items
3. Create text chunks for each product
4. Generate embeddings
5. Upsert into Qdrant

Catalog item structure from ETM:
{
    "id": "9536092",        # ETM code
    "name": "Пост кнопочный ПКУ 15-21",
    "brand": "Электротехник",
    "article": "ET054487",
    "brand_code": "48",
    "cli_code": "",
    "class_code": "503025",
    "class": "Кнопочные посты"
}
"""
import asyncio
import json
import hashlib
import structlog

from qdrant_client.models import PointStruct

from agents.rag.embeddings import get_embedding_engine
from agents.rag.vector_store import get_vector_store
from agents.etm.client import get_etm_client

logger = structlog.get_logger()

BATCH_SIZE = 100  # Points per upsert batch


def _product_to_text(item: dict) -> str:
    """Convert catalog item to searchable text."""
    parts = []

    name = item.get("name", "").strip()
    if name:
        parts.append(name)

    brand = item.get("brand", "").strip()
    if brand:
        parts.append(f"Производитель: {brand}")

    article = item.get("article", "").strip()
    if article:
        parts.append(f"Артикул: {article}")

    class_name = item.get("class", "").strip()
    if class_name:
        parts.append(f"Категория: {class_name}")

    etm_code = str(item.get("id", "")).strip()
    if etm_code:
        parts.append(f"Код ЭТМ: {etm_code}")

    return " | ".join(parts)


def _product_to_payload(item: dict) -> dict:
    """Convert catalog item to Qdrant payload."""
    return {
        "etm_code": str(item.get("id", "")),
        "name": item.get("name", ""),
        "brand": item.get("brand", ""),
        "article": item.get("article", ""),
        "brand_code": item.get("brand_code", ""),
        "class_code": item.get("class_code", ""),
        "class_name": item.get("class", ""),
        "cli_code": item.get("cli_code", ""),
    }


def _generate_point_id(etm_code: str) -> int:
    """Generate deterministic point ID from ETM code."""
    h = hashlib.md5(etm_code.encode()).hexdigest()
    return int(h[:16], 16) % (2**63)


async def load_catalog_from_api(poll_interval: int = 60) -> dict:
    """
    Full pipeline: ETM API -> parse -> embed -> Qdrant.

    Returns:
        {"success": bool, "indexed": int, "errors": int, "message": str}
    """
    logger.info("rag_loader_starting")

    # Step 1: Download catalog from ETM API
    etm = get_etm_client()
    result = await etm.export_catalog(poll_interval=poll_interval)

    if not result.get("success"):
        error = result.get("error", "Unknown error")
        logger.error("rag_loader_etm_failed", error=error)
        return {"success": False, "indexed": 0, "errors": 0, "message": error}

    catalog = result.get("catalog", [])
    if not catalog:
        return {"success": False, "indexed": 0, "errors": 0,
                "message": "Catalog is empty"}

    logger.info("rag_loader_catalog_received", items=len(catalog))

    # Step 2: Index into Qdrant
    return await index_catalog(catalog)


async def index_catalog(catalog: list[dict]) -> dict:
    """
    Index a list of catalog items into Qdrant.
    Can be called with any list of product dicts.
    """
    embedder = get_embedding_engine()
    store = get_vector_store()

    indexed = 0
    errors = 0
    total = len(catalog)

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch = catalog[batch_start:batch_start + BATCH_SIZE]

        # Build texts for embedding
        texts = []
        valid_items = []
        for item in batch:
            etm_code = str(item.get("id", "")).strip()
            if not etm_code:
                errors += 1
                continue

            text = _product_to_text(item)
            if not text:
                errors += 1
                continue

            texts.append(text)
            valid_items.append(item)

        if not texts:
            continue

        # Generate embeddings
        try:
            vectors = embedder.encode_documents(texts)
        except Exception as e:
            logger.error("rag_loader_embed_error",
                         batch=batch_start, error=str(e))
            errors += len(texts)
            continue

        # Build Qdrant points
        points = []
        for item, vector in zip(valid_items, vectors):
            etm_code = str(item.get("id", ""))
            points.append(
                PointStruct(
                    id=_generate_point_id(etm_code),
                    vector=vector,
                    payload=_product_to_payload(item),
                )
            )

        # Upsert to Qdrant
        try:
            store.upsert_batch(points)
            indexed += len(points)
        except Exception as e:
            logger.error("rag_loader_upsert_error",
                         batch=batch_start, error=str(e))
            errors += len(points)

        if batch_start % (BATCH_SIZE * 10) == 0:
            logger.info("rag_loader_progress",
                        indexed=indexed, total=total,
                        pct=round(indexed/total*100, 1))

    logger.info("rag_loader_done",
                indexed=indexed, errors=errors, total=total)

    return {
        "success": True,
        "indexed": indexed,
        "errors": errors,
        "message": f"Indexed {indexed}/{total} products ({errors} errors)",
    }


async def load_catalog_from_file(filepath: str) -> dict:
    """Load catalog from a local JSON file (for testing/manual import)."""
    logger.info("rag_loader_from_file", path=filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if isinstance(catalog, dict):
        catalog = catalog.get("data", catalog.get("rows", [catalog]))

    logger.info("rag_loader_file_loaded", items=len(catalog))
    return await index_catalog(catalog)
