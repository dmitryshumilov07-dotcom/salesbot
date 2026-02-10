"""
Telegram Bot — front interface for SalesBot.
Receives user messages, sends to Gateway API, returns responses.
Uses aiogram 3 with long polling.
"""
import asyncio
import structlog
import httpx
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode, ChatAction

from config.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()

GATEWAY_URL = f"http://127.0.0.1:{settings.gateway_port}/api/chat"

bot = Bot(token=settings.telegram_bot_token)
dp = Dispatcher()


def _session_id(user_id: int) -> str:
    """Generate consistent session ID from Telegram user ID."""
    return f"tg_{user_id}"


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start — send greeting via agent."""
    session_id = _session_id(message.from_user.id)

    logger.info("tg_start", user_id=message.from_user.id,
                username=message.from_user.username)

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(GATEWAY_URL, json={
                "session_id": session_id,
                "message": "Привет",
                "channel": "telegram",
            })
            resp.raise_for_status()
            data = resp.json()
            await message.answer(data["reply"])
    except Exception as e:
        logger.error("tg_start_error", error=str(e))
        await message.answer(
            "Добро пожаловать! Напишите, какая продукция вам нужна, "
            "и я помогу оформить запрос на расценку."
        )


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    """Handle /clear — reset conversation."""
    session_id = _session_id(message.from_user.id)

    try:
        clear_url = f"http://127.0.0.1:{settings.gateway_port}/api/chat/clear"
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(clear_url, params={"session_id": session_id})
    except Exception as e:
        logger.error("tg_clear_error", error=str(e))

    logger.info("tg_clear", user_id=message.from_user.id)
    await message.answer("Диалог сброшен. Напишите, какая продукция вам нужна.")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Handle /help."""
    await message.answer(
        "Я помогу собрать информацию о нужной продукции и передать на расценку.\n\n"
        "Просто напишите, что вам нужно. Например:\n"
        "— «Нужны трубы стальные»\n"
        "— «Нужен швеллер и лист горячекатаный»\n\n"
        "Команды:\n"
        "/start — начать заново\n"
        "/clear — сбросить диалог\n"
        "/help — помощь"
    )


@dp.message(F.text)
async def handle_message(message: types.Message):
    """Handle any text message — send to Chat Agent via Gateway."""
    session_id = _session_id(message.from_user.id)
    user_text = message.text.strip()

    if not user_text:
        return

    logger.info("tg_message", user_id=message.from_user.id,
                text_len=len(user_text))

    # Show "typing..." indicator
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(GATEWAY_URL, json={
                "session_id": session_id,
                "message": user_text,
                "channel": "telegram",
            })
            resp.raise_for_status()
            data = resp.json()

        reply = data["reply"]

        # Telegram message limit is 4096 chars
        if len(reply) > 4000:
            # Split into chunks
            for i in range(0, len(reply), 4000):
                await message.answer(reply[i:i+4000])
                if i + 4000 < len(reply):
                    await asyncio.sleep(0.3)
        else:
            await message.answer(reply)

        logger.info("tg_reply", user_id=message.from_user.id,
                    reply_len=len(reply))

    except httpx.TimeoutException:
        logger.error("tg_timeout", user_id=message.from_user.id)
        await message.answer(
            "Запрос обрабатывается дольше обычного. Попробуйте через минуту."
        )
    except Exception as e:
        logger.error("tg_error", user_id=message.from_user.id, error=str(e))
        await message.answer(
            "Произошла техническая ошибка. Попробуйте позже или напишите /start."
        )


@dp.message()
async def handle_non_text(message: types.Message):
    """Handle non-text messages (photos, stickers, etc.)."""
    await message.answer("Я работаю только с текстовыми сообщениями. Напишите, какая продукция вам нужна.")


async def main():
    """Start bot with long polling."""
    logger.info("telegram_bot_starting")

    # Delete webhook if any (to use polling)
    await bot.delete_webhook(drop_pending_updates=True)

    logger.info("telegram_bot_started", bot_id=bot.id)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
