"""
Chat Agent — front-desk agent that collects product requirements from clients.
Uses GigaChat LLM for conversation.
Detects order confirmation → sends Task to Dispatcher.
Detects ETM IDs → queries ETM prices + stock via Dispatcher → returns to user.
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

# ETM price query JSON: {"action":"etm_price","ids":["9536092","1037375"]}
ETM_PRICE_BLOCK = re.compile(r'```json\s*(\{.*?"action"\s*:\s*"etm_price".*?\})\s*```', re.DOTALL)
ETM_PRICE_LINE = re.compile(r'(\{"action"\s*:\s*"etm_price".*?\})', re.DOTALL)

# Direct ETM ID detection from user message (bypass LLM)
# Matches 6-8 digit numbers, optionally prefixed with ETM/etm/Etm
ETM_CODE_PATTERN = re.compile(r'(?:ETM|etm|Etm|ЭТМ|этм)?\s*(\d{6,8})', re.IGNORECASE)
# Keywords that signal a price/stock query
ETM_KEYWORDS = re.compile(
    r'(?:цен[аыу]|price|стоимость|остат[коки]|наличие|проверь|узнай|запроси|покажи|по коду|по кодам|по id)',
    re.IGNORECASE,
)
# Pattern to check if message is ONLY ETM codes (numbers, commas, spaces, ETM prefix)
ETM_ONLY_PATTERN = re.compile(
    r'^[\s,;.]*(?:(?:ETM|ЭТМ|etm)?\s*\d{6,8}[\s,;.]*)+$',
    re.IGNORECASE,
)


# WebUI system requests that should NOT be intercepted
WEBUI_SYSTEM_PATTERN = re.compile(
    r'(?:### Task:|Generate a concise|Suggest \d|summarizing the chat|categorizing the main themes)',
    re.IGNORECASE,
)


def _detect_etm_ids_from_user(message: str) -> list[str] | None:
    """
    Detect ETM product IDs directly from user message.
    Returns list of IDs if user is asking for ETM prices, None otherwise.

    Two modes:
    1. Message with keyword: "цена 9536092" / "проверь 9536092, 1037375"
    2. Message is ONLY codes: "9536092" / "9536092, 1037375" / "ETM9536092"

    Excludes WebUI system requests (title/tags/suggestions generation).
    """
    msg = message.strip()

    # Skip WebUI internal requests (title generation, tags, suggestions)
    if WEBUI_SYSTEM_PATTERN.search(msg):
        return None

    # Skip messages longer than 200 chars — likely not a simple ETM query
    if len(msg) > 200:
        return None

    # Mode 1: message contains only ETM codes (no other text)
    if ETM_ONLY_PATTERN.match(msg):
        codes = ETM_CODE_PATTERN.findall(msg)
        if codes:
            seen = set()
            return [c for c in codes if c not in seen and not seen.add(c)]

    # Mode 2: message has keyword + codes
    if ETM_KEYWORDS.search(msg):
        codes = ETM_CODE_PATTERN.findall(msg)
        if codes:
            seen = set()
            return [c for c in codes if c not in seen and not seen.add(c)]

    return None


def _extract_order(response: str) -> dict | None:
    """Try to extract submit_order JSON from LLM response."""
    match = JSON_BLOCK_PATTERN.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            if data.get("action") == "submit_order":
                return data
        except json.JSONDecodeError:
            pass

    match = JSON_LINE_PATTERN.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            if data.get("action") == "submit_order":
                return data
        except json.JSONDecodeError:
            pass

    return None


def _extract_etm_price(response: str) -> dict | None:
    """Try to extract etm_price JSON from LLM response."""
    for pattern in [ETM_PRICE_BLOCK, ETM_PRICE_LINE]:
        match = pattern.search(response)
        if match:
            try:
                data = json.loads(match.group(1))
                if data.get("action") == "etm_price":
                    return data
            except json.JSONDecodeError:
                pass
    return None


def _clean_response(response: str) -> str:
    """Remove JSON blocks from response text shown to user."""
    cleaned = JSON_BLOCK_PATTERN.sub('', response)
    cleaned = JSON_LINE_PATTERN.sub('', cleaned)
    return cleaned.strip()


def _format_etm_result(result_data: dict) -> str:
    """Format ETM price+stock result for user-friendly display."""
    products = result_data.get("products", [])
    if not products:
        return "К сожалению, данные по указанным кодам не найдены."

    lines = []
    for p in products:
        code = p.get("gdscode", "?")
        price = p.get("price", 0)
        pricewnds = p.get("pricewnds", 0)
        price_tarif = p.get("price_tarif", 0)
        price_retail = p.get("price_retail", 0)
        remains = p.get("remains", {})
        total_stock = remains.get("total_stock", 0)
        unit = remains.get("unit", "шт")
        delivery = remains.get("delivery_days", "")

        lines.append(f"📦 ETM {code}")
        lines.append(f"  Цена без НДС:      {price} руб.")
        lines.append(f"  Цена с НДС:        {pricewnds} руб.")
        lines.append(f"  Тариф производителя: {price_tarif} руб.")
        lines.append(f"  Розничная цена:     {price_retail} руб.")

        # Stock info
        stores = remains.get("stores", [])
        if stores:
            lines.append(f"  Остатки (всего {total_stock} {unit}):")
            for s in stores[:5]:
                lines.append(f"    • {s['name']}: {s['quantity']} {unit}")
            if len(stores) > 5:
                lines.append(f"    ... и ещё {len(stores) - 5} складов")
        else:
            lines.append(f"  Остатки: нет данных")

        if delivery:
            lines.append(f"  Срок поставки: {delivery}")

        lines.append("")  # blank line between products

    return "\n".join(lines).strip()


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

        logger.info("chat_agent_request",
                     session_id=session_id,
                     user_message_len=len(user_message),
                     history_len=len(history))

        # === FAST PATH: detect ETM IDs directly from user message ===
        # This bypasses LLM entirely — no risk of LLM formatting issues
        etm_ids = _detect_etm_ids_from_user(user_message)
        if etm_ids:
            logger.info("chat_agent_etm_direct",
                        session_id=session_id,
                        ids=etm_ids)
            etm_result = await self._dispatch_etm_price(
                {"ids": etm_ids, "type": "etm"}, session_id
            )
            return etm_result

        # === NORMAL PATH: LLM conversation ===
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            raw_response = await self.gigachat.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )

            # Check for etm_price action from LLM (fallback)
            etm_data = _extract_etm_price(raw_response)
            if etm_data:
                clean_text = _clean_response(raw_response)
                etm_result = await self._dispatch_etm_price(etm_data, session_id)
                if clean_text:
                    return clean_text + "\n\n" + etm_result
                return etm_result

            # Check for submit_order action
            order_data = _extract_order(raw_response)
            clean_text = _clean_response(raw_response)

            if order_data:
                dispatch_result = await self._dispatch_order(
                    order_data, session_id
                )
                clean_text += "\n\n" + dispatch_result

            logger.info("chat_agent_response",
                         session_id=session_id,
                         response_len=len(clean_text),
                         order_detected=order_data is not None)
            return clean_text

        except Exception as e:
            logger.error("chat_agent_error", session_id=session_id, error=str(e))
            return "Извините, произошла техническая ошибка. Попробуйте повторить запрос через минуту."

    async def _dispatch_etm_price(self, etm_data: dict, session_id: str) -> str:
        """Send ETM price request to Dispatcher → ETM Agent → format result."""
        try:
            dispatcher = get_dispatcher()

            ids = etm_data.get("ids", [])
            id_type = etm_data.get("type", "etm")

            if not ids:
                return "Не указаны коды товаров для запроса."

            task = Task(
                task_type=TaskType.ETM_PRICE,
                source="chat_agent",
                session_id=session_id,
                payload={
                    "product_ids": ids,
                    "id_type": id_type,
                },
            )

            logger.info("chat_agent_etm_dispatch",
                        session_id=session_id,
                        ids=ids)

            result = await dispatcher.dispatch(task)

            if result.status == TaskStatus.COMPLETED:
                return _format_etm_result(result.result)
            elif result.status == TaskStatus.REJECTED:
                return (
                    "⚠ Сервис запроса цен ЭТМ сейчас недоступен. "
                    "Попробуйте позже."
                )
            else:
                return (
                    f"Запрос #{task.task_id[:8]} передан в обработку. "
                    "Результат будет готов в ближайшее время."
                )

        except Exception as e:
            logger.error("chat_agent_etm_error",
                         session_id=session_id, error=str(e))
            return "⚠ Ошибка при запросе цен ЭТМ. Попробуйте позже."

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
        """Stream response with ETM fast-path support."""
        session_id = session_id or str(uuid.uuid4())
        history = history or []

        # === FAST PATH: detect ETM IDs — return result as single chunk ===
        etm_ids = _detect_etm_ids_from_user(user_message)
        if etm_ids:
            logger.info("chat_agent_etm_direct_stream",
                        session_id=session_id, ids=etm_ids)
            etm_result = await self._dispatch_etm_price(
                {"ids": etm_ids, "type": "etm"}, session_id
            )
            yield etm_result
            return

        # === NORMAL PATH: LLM streaming ===
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
