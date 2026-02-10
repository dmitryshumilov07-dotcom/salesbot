"""
Session management via Redis. Stores conversation history per user.
"""
import json
import structlog
from redis.asyncio import Redis
from config.settings import get_settings

logger = structlog.get_logger()

SESSION_TTL = 86400  # 24 hours
SESSION_PREFIX = "session:"


class SessionManager:
    def __init__(self):
        settings = get_settings()
        self._redis = Redis.from_url(settings.redis_url, decode_responses=True)

    async def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        key = f"{SESSION_PREFIX}{session_id}"
        data = await self._redis.get(key)
        if data is None:
            return []
        return json.loads(data)

    async def save_message(self, session_id: str, role: str, content: str):
        """Append a message to session history."""
        key = f"{SESSION_PREFIX}{session_id}"
        history = await self.get_history(session_id)
        history.append({"role": role, "content": content})

        # Keep last 50 messages to avoid token overflow
        if len(history) > 50:
            history = history[-50:]

        await self._redis.set(key, json.dumps(history, ensure_ascii=False), ex=SESSION_TTL)
        logger.debug("session_save", session_id=session_id, messages=len(history))

    async def clear(self, session_id: str):
        """Clear session history."""
        key = f"{SESSION_PREFIX}{session_id}"
        await self._redis.delete(key)

    async def close(self):
        await self._redis.close()


_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
