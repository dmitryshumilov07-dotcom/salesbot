"""
Telegram notifier for monitoring alerts.
Sends messages ONLY to hardcoded admin (chat_id 160217558).
"""
import httpx
import structlog

from config.settings import get_settings

logger = structlog.get_logger()

TG_API = "https://api.telegram.org/bot"
ADMIN_CHAT_ID = "160217558"


class TelegramNotifier:
    def __init__(self):
        self.settings = get_settings()
        self.token = self.settings.monitoring_bot_token

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send alert message to admin."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{TG_API}{self.token}/sendMessage",
                    json={
                        "chat_id": ADMIN_CHAT_ID,
                        "text": message,
                        "parse_mode": parse_mode,
                    },
                )
                data = resp.json()
                if data.get("ok"):
                    return True
                logger.error("monitoring_send_fail", response=data)
                return False
        except Exception as e:
            logger.error("monitoring_send_error", error=str(e))
            return False

    async def send_alert(self, level: str, title: str, details: str = ""):
        """Send formatted alert."""
        icons = {"critical": "🔴", "warning": "🟡", "info": "🟢", "recovery": "✅"}
        icon = icons.get(level, "⚪")
        msg = f"{icon} <b>[{level.upper()}] {title}</b>"
        if details:
            msg += f"\n\n{details}"
        await self.send(msg)
