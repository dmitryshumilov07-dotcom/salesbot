"""
Chat Agent — front-desk agent that collects product requirements from clients.
Uses GigaChat LLM for conversation.
Detects order confirmation → sends Task to Dispatcher.
"""
import json
import re
import uuid
import structlog

from agents.llm.gigachat_client import get_gigachat_client
from agents.dispatcher import get_dispatcher, Task, TaskType, TaskStatus
from config.settings import get_settings

logger = structlog.get_logger()

SYSTEM_PROMPT_PATH = "config/prompts/chat_agent.txt"


def _load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT = _load_system_prompt()

# Regex to extract JSON block from LLM response
JSON_BLOCK_PATTERN = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
# Fallback: raw JSON line
JSON_LINE_PATTERN = re.compile(r'(\{"action"\s*:\s*"submit_order".*?\})', re.DOTALL)


def _extract_order(response: str) -> dict | None:
    """Try to extract submit_order JSON from LLM response."""
    # Try ```json ... ``` block first
    match = JSON_BLOCK_PATTERN.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            if data.get("action") == "submit_order":
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: raw JSON
    match = JSON_LINE_PATTERN.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            if data.get("action") == "submit_order":
                return data
        except json.JSONDecodeError:
            pass

    return None


def _clean_response(response: str) -> str:
    """Remove JSON block from response text shown to user."""
    cleaned = JSON_BLOCK_PATTERN.sub('', response)
    cleaned = JSON_LINE_PATTERN.sub('', cleaned)
    return cleaned.strip()


class ChatAgent:
    """Chat agent with LLM + Dispatcher integration."""

    def __init__(self):
        self.gigachat = get_gigachat_client()

    async def respond(
        self,
        user_message: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ) -> str:
        session_id = session_id or str(uuid.uuid4())
        history = history or []

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        logger.info("chat_agent_request",
                     session_id=session_id,
                     user_message_len=len(user_message),
                     history_len=len(history))

        try:
            raw_response = await self.gigachat.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )

            # Check if LLM included a submit_order JSON
            order_data = _extract_order(raw_response)
            clean_text = _clean_response(raw_response)

            if order_data:
                # Order confirmed! Send to Dispatcher
                dispatch_result = await self._dispatch_order(
                    order_data, session_id
                )
                # Append dispatcher feedback to response
                clean_text += "\n\n" + dispatch_result

            logger.info("chat_agent_response",
                         session_id=session_id,
                         response_len=len(clean_text),
                         order_detected=order_data is not None)
            return clean_text

        except Exception as e:
            logger.error("chat_agent_error", session_id=session_id, error=str(e))
            return "Извините, произошла техническая ошибка. Попробуйте повторить запрос через минуту."

    async def _dispatch_order(self, order_data: dict, session_id: str) -> str:
        """Send order to Dispatcher and return status message."""
        try:
            dispatcher = get_dispatcher()

            task = Task(
                task_type=TaskType.PRICING,
                source="chat_agent",
                session_id=session_id,
                payload={
                    "items": order_data.get("items", []),
                    "raw_order": order_data,
                },
            )

            result = await dispatcher.dispatch(task)

            if result.status == TaskStatus.REJECTED:
                logger.warning("chat_agent_dispatch_rejected",
                               session_id=session_id,
                               error=result.error)
                return (
                    "⚠ К сожалению, система расценки сейчас недоступна. "
                    "Ваш запрос сохранён и будет обработан, как только "
                    "сервис восстановится. Мы свяжемся с вами."
                )

            if result.status == TaskStatus.FAILED:
                logger.error("chat_agent_dispatch_failed",
                             session_id=session_id,
                             error=result.error)
                return (
                    "⚠ Произошла ошибка при обработке. "
                    "Запрос сохранён, специалисты разберутся и свяжутся с вами."
                )

            if result.status in (TaskStatus.DISPATCHED, TaskStatus.COMPLETED):
                logger.info("chat_agent_dispatch_ok",
                            session_id=session_id,
                            task_id=task.task_id)
                return (
                    f"✓ Запрос #{task.task_id[:8]} принят в обработку."
                )

            return "Запрос передан в систему."

        except Exception as e:
            logger.error("chat_agent_dispatch_error",
                         session_id=session_id, error=str(e))
            return (
                "⚠ Не удалось передать запрос в систему. "
                "Попробуйте позже или свяжитесь с менеджером."
            )

    async def respond_stream(
        self,
        user_message: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ):
        """Stream response. Note: dispatch happens only in non-stream mode."""
        session_id = session_id or str(uuid.uuid4())
        history = history or []

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            async for chunk in self.gigachat.chat_stream(
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            ):
                yield chunk
        except Exception as e:
            logger.error("chat_agent_stream_error",
                         session_id=session_id, error=str(e))
            yield "Извините, произошла техническая ошибка."


_agent: ChatAgent | None = None


def get_chat_agent() -> ChatAgent:
    global _agent
    if _agent is None:
        _agent = ChatAgent()
    return _agent
