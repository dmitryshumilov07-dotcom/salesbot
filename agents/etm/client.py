"""
ETM API Client - async client for ETM (ipro.etm.ru) product API.

Provides three core functions:
1. get_prices(ids)     - batch price lookup (up to 50 per request)
2. get_remains(id)     - stock/availability check per product
3. export_catalog()    - full catalog export (SgGds) for RAG indexing

Features:
- Automatic session management (login + 8h token caching)
- Rate limiting (1 req/sec for prices/remains, 1 req/2min for auth)
- Retry logic with exponential backoff
- Production server: https://ipro.etm.ru/api/v1
"""
import asyncio
import time
import structlog
import httpx

from config.settings import get_settings

logger = structlog.get_logger()

ETM_API_BASE = "https://ipro.etm.ru/api/v1"


class ETMClient:
    """Async client for ETM iPRO API."""

    def __init__(self):
        self.settings = get_settings()
        self.login = self.settings.etm_login
        self.password = self.settings.etm_password
        self._session_id: str | None = None
        self._session_expires: float = 0
        self._lock = asyncio.Lock()
        self._last_request: float = 0
        self._last_auth: float = 0

    # ==================== Auth ====================

    async def _ensure_session(self) -> str:
        """Get valid session ID, refreshing if expired."""
        async with self._lock:
            now = time.time()
            if self._session_id and now < self._session_expires:
                return self._session_id

            # Rate limit: 1 auth per 2 minutes
            if now - self._last_auth < 120:
                wait = 120 - (now - self._last_auth)
                logger.info("etm_auth_rate_wait", wait=wait)
                await asyncio.sleep(wait)

            logger.info("etm_authenticating", login=self.login)
            self._last_auth = time.time()

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{ETM_API_BASE}/user/login",
                    params={"log": self.login, "pwd": self.password},
                )
                data = resp.json()

                if data.get("status", {}).get("code") != 200:
                    msg = data.get("status", {}).get("message", "Unknown error")
                    logger.error("etm_auth_failed", message=msg)
                    raise Exception(f"ETM auth failed: {msg}")

                self._session_id = data["data"]["session"]
                # Token valid for 8 hours, refresh 30 min early
                self._session_expires = time.time() + (8 * 3600) - 1800

                logger.info("etm_authenticated", expires_in_hours=7.5)
                return self._session_id

    async def _rate_limit(self):
        """Enforce 1 request per second for data endpoints."""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        self._last_request = time.time()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """Make authenticated GET request with rate limiting and retry."""
        session = await self._ensure_session()
        await self._rate_limit()

        if params is None:
            params = {}
        params["session-id"] = session

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(f"{ETM_API_BASE}{path}", params=params)
                    data = resp.json()

                    code = data.get("status", {}).get("code", 0)

                    if code == 200:
                        return data

                    if code == 403:
                        # Session expired, force re-auth
                        logger.warning("etm_session_expired", attempt=attempt)
                        self._session_id = None
                        self._session_expires = 0
                        session = await self._ensure_session()
                        params["session-id"] = session
                        continue

                    msg = data.get("status", {}).get("message", "")
                    logger.error("etm_api_error", code=code, message=msg, path=path)
                    return data

            except Exception as e:
                logger.error("etm_request_error", path=path, attempt=attempt, error=str(e))
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        return {"status": {"code": 500, "message": "Max retries exceeded"}}

    async def _post(self, path: str, params: dict | None = None, json_body: dict | None = None) -> dict:
        """Make authenticated POST request."""
        session = await self._ensure_session()
        await self._rate_limit()

        if params is None:
            params = {}
        params["session-id"] = session

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{ETM_API_BASE}{path}",
                params=params,
                json=json_body,
            )
            return resp.json()

    # ==================== 1. Prices ====================

    async def get_prices(self, product_ids: list[str], id_type: str = "etm") -> list[dict]:
        """
        Get prices for products by IDs.

        Args:
            product_ids: List of product codes (up to 50)
            id_type: 'etm' (ETM codes), 'mnf' (manufacturer article), 'cli' (client codes)

        Returns:
            List of price dicts: [{gdscode, price, pricewnds, price_tarif, price_retail}, ...]
        """
        if not product_ids:
            return []

        results = []
        # API supports up to 50 IDs per request, joined by %2C (comma)
        for i in range(0, len(product_ids), 50):
            batch = product_ids[i:i + 50]
            ids_str = ",".join(str(pid) for pid in batch)

            logger.info("etm_get_prices", count=len(batch), type=id_type)

            data = await self._get(
                f"/goods/{ids_str}/price",
                params={"type": id_type},
            )

            if data.get("status", {}).get("code") == 200:
                raw = data.get("data", {})
                # Response can be {rows: [...]} or direct dict
                if isinstance(raw, dict) and "rows" in raw:
                    results.extend(raw["rows"])
                elif isinstance(raw, list):
                    results.extend(raw)
                elif isinstance(raw, dict):
                    results.append(raw)
            else:
                logger.warning("etm_prices_error",
                               ids=ids_str[:50],
                               error=data.get("status", {}).get("message", ""))

        return results

    # ==================== 2. Remains ====================

    async def get_remains(self, product_id: str, id_type: str = "etm") -> dict:
        """
        Get stock/availability for a single product.

        Args:
            product_id: Product code
            id_type: 'etm', 'mnf', or 'cli'

        Returns:
            Dict with stores, forecasts, supplier info, delivery times
        """
        logger.info("etm_get_remains", id=product_id, type=id_type)

        data = await self._get(
            f"/goods/{product_id}/remains",
            params={"type": id_type},
        )

        if data.get("status", {}).get("code") == 200:
            return data.get("data", {})
        else:
            msg = data.get("status", {}).get("message", "")
            logger.warning("etm_remains_error", id=product_id, error=msg)
            return {"error": msg}

    # ==================== 3. Catalog Export ====================

    async def export_catalog(self, poll_interval: int = 60, max_wait: int = 10800) -> dict:
        """
        Export full product catalog (SgGds) for RAG indexing.

        This is a 2-step process:
        Step 1: POST /job/create/40029846 - create export job
        Step 2: GET /job/{uuid} - poll until file is ready (up to 3 hours)

        Args:
            poll_interval: Seconds between status polls (default 60)
            max_wait: Maximum wait time in seconds (default 3 hours)

        Returns:
            Dict with download URL and file content or error
        """
        logger.info("etm_catalog_export_starting")

        # Step 1: Create job
        data = await self._post("/job/create/40029846")

        if data.get("status", {}).get("code") != 200:
            msg = data.get("status", {}).get("message", "")
            logger.error("etm_catalog_create_failed", error=msg)
            return {"success": False, "error": f"Failed to create catalog job: {msg}"}

        job_uuid = data.get("data", {}).get("uuid", "")
        if not job_uuid:
            return {"success": False, "error": "No UUID in job create response"}

        logger.info("etm_catalog_job_created", uuid=job_uuid)

        # Step 2: Poll for completion
        start = time.time()
        while time.time() - start < max_wait:
            await asyncio.sleep(poll_interval)

            status_data = await self._get(f"/job/{job_uuid}")

            if status_data.get("status", {}).get("code") != 200:
                continue

            rows = status_data.get("data", {}).get("rows", [])
            if not rows:
                continue

            job = rows[0]
            state = str(job.get("state", "0"))

            logger.info("etm_catalog_poll",
                        state=state,
                        elapsed=int(time.time() - start))

            if state == "1":  # Successfully completed
                urls = job.get("urls", [])
                if urls:
                    download_url = urls[0].get("url", "")
                    logger.info("etm_catalog_ready", url=download_url)

                    # Download the catalog file
                    catalog = await self._download_catalog(download_url)
                    return {
                        "success": True,
                        "url": download_url,
                        "catalog": catalog,
                        "elapsed_seconds": int(time.time() - start),
                    }
                return {"success": False, "error": "Job completed but no URL"}

            elif state == "2":  # Completed with error
                msg = job.get("msg", "Unknown error")
                logger.error("etm_catalog_job_failed", error=msg)
                return {"success": False, "error": f"Catalog job failed: {msg}"}

            # state 0 (created) or 3 (waiting) - keep polling

        return {"success": False, "error": f"Catalog job timed out after {max_wait}s"}

    async def _download_catalog(self, url: str) -> list[dict]:
        """Download and parse catalog JSON file."""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        logger.info("etm_catalog_downloaded", items=len(data))
                        return data
                    elif isinstance(data, dict):
                        items = data.get("data", data.get("rows", [data]))
                        logger.info("etm_catalog_downloaded", items=len(items))
                        return items
                else:
                    logger.error("etm_catalog_download_failed", status=resp.status_code)
                    return []
        except Exception as e:
            logger.error("etm_catalog_download_error", error=str(e))
            return []

    # ==================== Utility ====================

    async def get_product_info(self, product_id: str, id_type: str = "etm") -> dict:
        """Get full product characteristics (bonus method for enrichment)."""
        data = await self._get(
            f"/goods/{product_id}",
            params={"type": id_type},
        )
        if data.get("status", {}).get("code") == 200:
            rows = data.get("data", {}).get("rows", [])
            return rows[0] if rows else {}
        return {}

    async def get_manufacturers(self) -> list[dict]:
        """Get manufacturer directory."""
        data = await self._get("/info/search/r-manuf/")
        if data.get("status", {}).get("code") == 200:
            return data.get("data", {}).get("rows", [])
        return []


# Singleton
_client: ETMClient | None = None


def get_etm_client() -> ETMClient:
    global _client
    if _client is None:
        _client = ETMClient()
    return _client
