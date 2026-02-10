"""
GigaChat API client with token caching and auto-refresh.
Sber GigaChat REST API integration.
"""
import ssl
import time
import uuid
import asyncio
import structlog
import httpx

from config.settings import get_settings

logger = structlog.get_logger()


class GigaChatClient:
    """Async client for GigaChat API with automatic OAuth token management."""

    def __init__(self):
        self.settings = get_settings()
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        self._lock = asyncio.Lock()

        # GigaChat requires TLS without strict cert verification (self-signed Sber certs)
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    async def _get_token(self) -> str:
        """Get or refresh OAuth token. Thread-safe with async lock."""
        async with self._lock:
            # Return cached token if still valid (with 60s buffer)
            if self._access_token and time.time() < (self._token_expires_at - 60):
                return self._access_token

            logger.info("gigachat_auth", status="requesting_new_token")

            async with httpx.AsyncClient(verify=self._ssl_context) as client:
                response = await client.post(
                    self.settings.gigachat_auth_url,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "RqUID": str(uuid.uuid4()),
                        "Authorization": f"Basic {self.settings.gigachat_auth_key}",
                    },
                    data={"scope": self.settings.gigachat_scope},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

            self._access_token = data["access_token"]
            # Token expires_at is in milliseconds
            self._token_expires_at = data["expires_at"] / 1000
            logger.info("gigachat_auth", status="token_received",
                        expires_in=int(self._token_expires_at - time.time()))
            return self._access_token

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        model: str | None = None,
    ) -> str:
        """
        Send chat completion request to GigaChat.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max response tokens
            model: Model name override

        Returns:
            Assistant message content string

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        token = await self._get_token()
        model = model or self.settings.gigachat_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        async with httpx.AsyncClient(verify=self._ssl_context) as client:
            response = await client.post(
                f"{self.settings.gigachat_api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        logger.info("gigachat_chat", model=model,
                    prompt_tokens=data.get("usage", {}).get("prompt_tokens"),
                    completion_tokens=data.get("usage", {}).get("completion_tokens"))
        return content

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        model: str | None = None,
    ):
        """
        Stream chat completion from GigaChat. Yields content chunks.

        Yields:
            str: Content delta chunks
        """
        token = await self._get_token()
        model = model or self.settings.gigachat_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(verify=self._ssl_context) as client:
            async with client.stream(
                "POST",
                f"{self.settings.gigachat_api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]
                        if chunk_data.strip() == "[DONE]":
                            break
                        import json
                        chunk = json.loads(chunk_data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content


# Singleton instance
_client: GigaChatClient | None = None


def get_gigachat_client() -> GigaChatClient:
    global _client
    if _client is None:
        _client = GigaChatClient()
    return _client
