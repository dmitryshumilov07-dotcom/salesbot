"""
Dispatcher — the central rule-based router of the agent system.
NO LLM. Pure deterministic routing based on task_type → agent mapping.

Responsibilities:
1. Receive tasks from Chat Agent or other sources
2. Look up capable agent in registry
3. If agent ONLINE — dispatch task
4. If agent OFFLINE — return rejection with clear message
5. Track task lifecycle in Redis
6. Provide status API for monitoring
"""
import json
import time
import structlog
from redis.asyncio import Redis

from config.settings import get_settings
from agents.dispatcher.registry import AgentRegistry, AgentInfo, AgentStatus
from agents.dispatcher.task import Task, TaskResult, TaskStatus, TaskType

logger = structlog.get_logger()

TASK_LOG_PREFIX = "dispatcher:tasks:"
TASK_QUEUE_PREFIX = "dispatcher:queue:"


class Dispatcher:
    """
    Deterministic task dispatcher. Zero LLM, zero guessing.
    Routes by task_type → registered agent.
    """

    def __init__(self):
        settings = get_settings()
        self.registry = AgentRegistry()
        self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self._handlers: dict[str, callable] = {}  # task_type → async handler

    # ==================== Core Dispatch ====================

    async def dispatch(self, task: Task) -> TaskResult:
        """
        Main dispatch method. Routes task to appropriate agent.

        Flow:
        1. Validate task_type is known
        2. Find agent in registry
        3. If agent online → execute
        4. If no agent → reject with message
        """
        logger.info("dispatcher_received",
                    task_id=task.task_id,
                    task_type=task.task_type,
                    source=task.source)

        # Save task to Redis for tracking
        await self._save_task(task)

        # Validate task type
        if task.task_type == TaskType.UNKNOWN:
            return await self._reject(task, "Неизвестный тип задачи")

        # Find capable agent
        agent = await self.registry.find_agent_for_task(task.task_type.value)

        if agent is None:
            return await self._reject(
                task,
                f"Обработка запроса типа '{task.task_type.value}' "
                f"в данный момент недоступна. Попробуйте позже."
            )

        # Agent found — dispatch
        task.status = TaskStatus.DISPATCHED
        task.assigned_to = agent.name
        await self._save_task(task)

        logger.info("dispatcher_routed",
                    task_id=task.task_id,
                    agent=agent.name,
                    agent_status=agent.status)

        # Execute via registered handler or queue
        result = await self._execute(task, agent)
        return result

    async def _execute(self, task: Task, agent: AgentInfo) -> TaskResult:
        """Execute task via registered handler or push to queue."""

        handler = self._handlers.get(task.task_type.value)

        if handler:
            # Direct handler (in-process agent)
            try:
                task.status = TaskStatus.IN_PROGRESS
                await self._save_task(task)

                result_data = await handler(task)

                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    agent_name=agent.name,
                    result=result_data if isinstance(result_data, dict) else {"data": result_data},
                )

                # Report success heartbeat
                await self.registry.heartbeat(agent.name)

            except Exception as e:
                logger.error("dispatcher_execute_error",
                             task_id=task.task_id,
                             agent=agent.name,
                             error=str(e))

                await self.registry.report_error(agent.name, str(e))

                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    agent_name=agent.name,
                    error=f"Ошибка при обработке: {str(e)}",
                )
        else:
            # No handler — push to task queue (for future async workers)
            await self._enqueue(task)
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.DISPATCHED,
                agent_name=agent.name,
                result={"message": "Задача поставлена в очередь на обработку."},
            )

        await self._save_result(result)
        return result

    async def _reject(self, task: Task, reason: str) -> TaskResult:
        """Reject a task — agent unavailable or unknown type."""
        task.status = TaskStatus.REJECTED
        await self._save_task(task)

        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.REJECTED,
            agent_name="dispatcher",
            error=reason,
        )
        await self._save_result(result)

        logger.warning("dispatcher_rejected",
                       task_id=task.task_id,
                       reason=reason)
        return result

    # ==================== Handler Registration ====================

    def register_handler(self, task_type: str, handler: callable):
        """
        Register an in-process handler for a task type.
        Handler signature: async def handler(task: Task) -> dict
        """
        self._handlers[task_type] = handler
        logger.info("dispatcher_handler_registered", task_type=task_type)

    # ==================== Task Queue (Redis) ====================

    async def _enqueue(self, task: Task):
        """Push task to Redis queue for async processing."""
        queue_key = f"{TASK_QUEUE_PREFIX}{task.task_type.value}"
        await self._redis.rpush(queue_key, task.model_dump_json())
        logger.info("dispatcher_enqueued",
                    task_id=task.task_id,
                    queue=queue_key)

    async def dequeue(self, task_type: str, timeout: int = 0) -> Task | None:
        """Pop a task from queue. For worker agents."""
        queue_key = f"{TASK_QUEUE_PREFIX}{task_type}"
        if timeout > 0:
            data = await self._redis.blpop(queue_key, timeout=timeout)
            if data:
                return Task.model_validate_json(data[1])
        else:
            data = await self._redis.lpop(queue_key)
            if data:
                return Task.model_validate_json(data)
        return None

    # ==================== Task Persistence ====================

    async def _save_task(self, task: Task):
        key = f"{TASK_LOG_PREFIX}{task.task_id}"
        await self._redis.set(key, task.model_dump_json(), ex=86400)

    async def _save_result(self, result: TaskResult):
        key = f"{TASK_LOG_PREFIX}{result.task_id}:result"
        await self._redis.set(key, result.model_dump_json(), ex=86400)

    async def get_task(self, task_id: str) -> Task | None:
        key = f"{TASK_LOG_PREFIX}{task_id}"
        data = await self._redis.get(key)
        return Task.model_validate_json(data) if data else None

    async def get_result(self, task_id: str) -> TaskResult | None:
        key = f"{TASK_LOG_PREFIX}{task_id}:result"
        data = await self._redis.get(key)
        return TaskResult.model_validate_json(data) if data else None

    # ==================== Status ====================

    async def get_status(self) -> dict:
        """Full system status report."""
        registry_report = await self.registry.get_status_report()
        return {
            "dispatcher": "online",
            "handlers_registered": list(self._handlers.keys()),
            "registry": registry_report,
        }

    async def close(self):
        await self.registry.close()
        await self._redis.close()


# ==================== Singleton ====================

_dispatcher: Dispatcher | None = None


def get_dispatcher() -> Dispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = Dispatcher()
    return _dispatcher
