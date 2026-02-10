"""
Repair Approval - Telegram inline keyboard for admin to approve/reject repairs.

Uses the monitoring bot to send approval requests.
Admin approves or rejects via inline buttons.
Timeout: 30 minutes (after which the request is re-sent as escalation).
"""
import asyncio
import json
import uuid
import structlog
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from config.settings import get_settings

logger = structlog.get_logger()

ADMIN_CHAT_ID = 160217558
APPROVAL_TIMEOUT = 1800  # 30 minutes


class ApprovalManager:
    """Manages repair approval requests via Telegram."""

    def __init__(self, bot: Bot):
        self.bot = bot
        self._pending: dict[str, asyncio.Future] = {}

    async def request_approval(
        self,
        level: int,
        description: str,
        action: str,
        details: str = "",
    ) -> bool:
        """
        Send approval request to admin. Block until approved/rejected/timeout.

        Args:
            level: RepairLevel (2=config, 3=cursor)
            description: What the problem is
            action: What repair action is proposed
            details: Additional details

        Returns:
            True if approved, False if rejected or timeout
        """
        request_id = str(uuid.uuid4())[:8]
        level_name = {2: "CONFIG", 3: "CURSOR/CODE"}.get(level, f"L{level}")

        # Format message
        message = (
            f"<b>REPAIR REQUEST [{level_name}]</b>\n\n"
            f"<b>Problem:</b> {description}\n"
            f"<b>Action:</b> {action}\n"
        )
        if details:
            message += f"\n<b>Details:</b>\n<pre>{details[:500]}</pre>\n"

        message += f"\n<i>ID: {request_id} | Timeout: {APPROVAL_TIMEOUT // 60} min</i>"

        # Create inline keyboard
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Approve",
                    callback_data=f"repair_approve:{request_id}",
                ),
                InlineKeyboardButton(
                    text="Reject",
                    callback_data=f"repair_reject:{request_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Details",
                    callback_data=f"repair_details:{request_id}",
                ),
            ],
        ])

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            await self.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=message,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
            logger.info("repair_approval_sent",
                        request_id=request_id,
                        level=level_name)

            # Wait for response with timeout
            approved = await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
            return approved

        except asyncio.TimeoutError:
            logger.warning("repair_approval_timeout", request_id=request_id)
            await self.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=f"Repair request {request_id} timed out (no response in {APPROVAL_TIMEOUT // 60} min).",
                parse_mode="HTML",
            )
            return False

        finally:
            self._pending.pop(request_id, None)

    def handle_callback(self, request_id: str, approved: bool):
        """Called when admin clicks Approve/Reject."""
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(approved)
            logger.info("repair_approval_response",
                        request_id=request_id,
                        approved=approved)
        else:
            logger.warning("repair_approval_no_pending", request_id=request_id)

    def get_pending_request_ids(self) -> list[str]:
        """Get list of pending approval request IDs."""
        return list(self._pending.keys())
