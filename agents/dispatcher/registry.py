"""
Agent Registry — maintains a live registry of all agents in the system.
Tracks status, health, capabilities. No LLM — pure state management.
"""
import time
import json
import asyncio
import structlog
from enum import Enum
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from config.settings import get_settings

logger = structlog.get_logger()

REGISTRY_KEY = "dispatcher:agents"
HEALTH_PREFIX = "dispatcher:health:"
HEALTH_CHECK_INTERVAL = 30  # seconds


class AgentStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"       # Работает, но с ошибками
    MAINTENANCE = "maintenance"  # На обслуживании


class AgentInfo(BaseModel):
    """Descriptor of a registered agent."""
    name: str                    # Уникальное имя: pricing_agent, search_agent, etc.
    task_types: list[str]        # Какие TaskType обрабатывает
    endpoint: str = ""           # URL или internal path
    status: AgentStatus = AgentStatus.OFFLINE
    last_heartbeat: float = 0
    error_count: int = 0
    max_errors: int = 5          # После этого — OFFLINE
    description: str = ""
    version: str = "0.1.0"


class AgentRegistry:
    """
    Central registry of all agents. State stored in Redis for persistence.
    """

    def __init__(self):
        settings = get_settings()
        self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self._local_cache: dict[str, AgentInfo] = {}

    async def register(self, agent: AgentInfo) -> None:
        """Register an agent or update its info."""
        agent.last_heartbeat = time.time()
        agent.status = AgentStatus.ONLINE
        self._local_cache[agent.name] = agent
        await self._redis.hset(REGISTRY_KEY, agent.name, agent.model_dump_json())
        logger.info("agent_registered", agent=agent.name,
                    task_types=agent.task_types, status=agent.status)

    async def unregister(self, agent_name: str) -> None:
        """Remove an agent from registry."""
        self._local_cache.pop(agent_name, None)
        await self._redis.hdel(REGISTRY_KEY, agent_name)
        logger.info("agent_unregistered", agent=agent_name)

    async def heartbeat(self, agent_name: str) -> None:
        """Agent reports it is alive. Resets error count."""
        agent = await self.get(agent_name)
        if agent:
            agent.last_heartbeat = time.time()
            agent.status = AgentStatus.ONLINE
            agent.error_count = 0
            self._local_cache[agent_name] = agent
            await self._redis.hset(REGISTRY_KEY, agent_name, agent.model_dump_json())

    async def report_error(self, agent_name: str, error: str) -> None:
        """Report an error for an agent. Auto-degrade/offline if too many."""
        agent = await self.get(agent_name)
        if not agent:
            return
        agent.error_count += 1
        if agent.error_count >= agent.max_errors:
            agent.status = AgentStatus.OFFLINE
            logger.error("agent_offline_errors", agent=agent_name,
                         error_count=agent.error_count, last_error=error)
        elif agent.error_count >= agent.max_errors // 2:
            agent.status = AgentStatus.DEGRADED
            logger.warning("agent_degraded", agent=agent_name,
                           error_count=agent.error_count)
        self._local_cache[agent_name] = agent
        await self._redis.hset(REGISTRY_KEY, agent_name, agent.model_dump_json())

    async def get(self, agent_name: str) -> AgentInfo | None:
        """Get agent info. Uses local cache first, then Redis."""
        if agent_name in self._local_cache:
            return self._local_cache[agent_name]
        data = await self._redis.hget(REGISTRY_KEY, agent_name)
        if data:
            agent = AgentInfo.model_validate_json(data)
            self._local_cache[agent_name] = agent
            return agent
        return None

    async def find_agent_for_task(self, task_type: str) -> AgentInfo | None:
        """Find an ONLINE agent capable of handling this task type."""
        all_agents = await self.get_all()
        candidates = [
            a for a in all_agents
            if task_type in a.task_types and a.status == AgentStatus.ONLINE
        ]
        if not candidates:
            # Try DEGRADED as fallback
            candidates = [
                a for a in all_agents
                if task_type in a.task_types and a.status == AgentStatus.DEGRADED
            ]
        return candidates[0] if candidates else None

    async def get_all(self) -> list[AgentInfo]:
        """Get all registered agents."""
        raw = await self._redis.hgetall(REGISTRY_KEY)
        agents = []
        for name, data in raw.items():
            agent = AgentInfo.model_validate_json(data)
            self._local_cache[name] = agent
            agents.append(agent)
        return agents

    async def get_status_report(self) -> dict:
        """Generate a full status report of all agents."""
        agents = await self.get_all()
        return {
            "total": len(agents),
            "online": sum(1 for a in agents if a.status == AgentStatus.ONLINE),
            "offline": sum(1 for a in agents if a.status == AgentStatus.OFFLINE),
            "degraded": sum(1 for a in agents if a.status == AgentStatus.DEGRADED),
            "agents": {a.name: {
                "status": a.status,
                "task_types": a.task_types,
                "error_count": a.error_count,
                "last_heartbeat": a.last_heartbeat,
            } for a in agents},
        }

    async def close(self):
        await self._redis.close()
