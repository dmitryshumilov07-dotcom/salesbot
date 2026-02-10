"""
Monitoring Agent - watches over the entire system.
Checks every 5 minutes. Alerts instantly on problems.
Full report to admin every 3 hours. Bot responds ONLY to admin 160217558.

Integration with Repair Agent:
- On critical/warning status change, pushes task to Redis queue 'repair:tasks'
- Handles Repair Agent's approval callbacks (inline buttons)
"""
import asyncio
import json
import time
from datetime import datetime
import structlog

from config.settings import get_settings
from agents.monitoring.health_checks import HealthChecks
from agents.monitoring.notifier import TelegramNotifier

logger = structlog.get_logger()

# FIX: Use settings instead of hardcoded admin ID for security
# The admin user ID should be configured via ADMIN_USER_ID environment variable
_settings = get_settings()
ADMIN_USER_ID = _settings.admin_user_id
REPORT_INTERVAL = 10800  # 3 hours in seconds
REPAIR_QUEUE_KEY = "repair:tasks"
REPAIR_APPROVAL_QUEUE = "repair:approvals"


class MonitoringAgent:
    def __init__(self):
        self.settings = get_settings()
        self.checks = HealthChecks()
        self.notifier = TelegramNotifier()
        self.check_interval = self.settings.monitoring_interval  # 300s = 5 min
        self._previous_states: dict[str, str] = {}
        self._alert_cooldown: dict[str, float] = {}
        self._last_report_time: float = 0
        self.COOLDOWN_SECONDS = 300
        self._redis = None

    async def _get_redis(self):
        """Lazy init Redis connection."""
        if self._redis is None:
            from redis.asyncio import Redis
            self._redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        return self._redis

    async def _push_repair_task(self, check_result: dict):
        """Push a failed health check to the repair queue."""
        try:
            r = await self._get_redis()
            task_data = json.dumps({
                "name": check_result["name"],
                "status": check_result["status"],
                "value": check_result.get("value", ""),
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False)
            await r.rpush(REPAIR_QUEUE_KEY, task_data)
            logger.info("monitoring_repair_task_pushed",
                        check=check_result["name"],
                        status=check_result["status"])
        except Exception as e:
            logger.error("monitoring_repair_push_error", error=str(e))

    async def handle_repair_approval(self, request_id: str, approved: bool):
        """Write approval result to Redis so Repair Agent can pick it up."""
        try:
            r = await self._get_redis()
            key = f"{REPAIR_APPROVAL_QUEUE}:{request_id}"
            value = "approved" if approved else "rejected"
            await r.set(key, value, ex=3600)  # Expire in 1 hour
            logger.info("monitoring_repair_approval",
                        request_id=request_id,
                        approved=approved)
        except Exception as e:
            logger.error("monitoring_repair_approval_error", error=str(e))

    async def start(self):
        """Main monitoring loop."""
        logger.info("monitoring_agent_starting",
                    check_interval=self.check_interval,
                    report_interval=REPORT_INTERVAL)

        await self.notifier.send_alert(
            "info", "Monitoring Agent started",
            f"Checks: every {self.check_interval // 60} min.\n"
            f"Reports: every {REPORT_INTERVAL // 3600} h.\n"
            f"Repair integration: ACTIVE\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # First check + first report immediately
        await self._run_cycle(force_report=True)

        while True:
            await asyncio.sleep(self.check_interval)
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error("monitoring_cycle_error", error=str(e))
                await self.notifier.send_alert(
                    "critical", "Monitoring cycle error", str(e)
                )

    async def _run_cycle(self, force_report: bool = False):
        """Single monitoring cycle. Alerts on problems, report every 3h."""
        results = await self.checks.run_all()
        now = time.time()
        problems = []

        for check in results:
            name = check["name"]
            status = check["status"]
            value = check.get("value", "")
            prev_status = self._previous_states.get(name)

            if prev_status != status:
                # Problem detected -> instant alert + repair task
                if status in ("critical", "warning"):
                    last_alert = self._alert_cooldown.get(name, 0)
                    if now - last_alert > self.COOLDOWN_SECONDS:
                        await self.notifier.send_alert(
                            status, name,
                            f"Status: {prev_status or 'unknown'} -> {status}\n"
                            f"Value: {value}\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}"
                        )
                        self._alert_cooldown[name] = now

                    # Push to repair queue
                    await self._push_repair_task(check)

                # Recovery -> notify
                elif prev_status in ("critical", "warning") and status == "ok":
                    await self.notifier.send_alert(
                        "recovery", f"{name} recovered",
                        f"Status: {prev_status} -> ok\n"
                        f"Value: {value}\n"
                        f"Time: {datetime.now().strftime('%H:%M:%S')}"
                    )

                self._previous_states[name] = status

            if status in ("critical", "warning"):
                problems.append(f"{name}: {value}")
                logger.warning("health_check", name=name, status=status, value=value)

        # Periodic report every 3 hours (or forced on startup)
        if force_report or (now - self._last_report_time >= REPORT_INTERVAL):
            report = await self.get_report()
            await self.notifier.send(report)
            self._last_report_time = now
            logger.info("monitoring_report_sent")

        logger.info("monitoring_cycle_complete", problems=len(problems))

    async def get_report(self) -> str:
        """Generate full text report."""
        results = await self.checks.run_all()
        icons = {"ok": "\U0001f7e2", "warning": "\U0001f7e1", "critical": "\U0001f534", "info": "\u26aa"}
        lines = [f"<b>\U0001f4ca System Status Report</b>",
                 f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", ""]

        sections = {"Server": [], "Services": [], "Docker": [], "Agents": []}
        for r in results:
            name = r["name"]
            if name in ("CPU", "RAM", "Disk"):
                sections["Server"].append(r)
            elif name.startswith("Service:"):
                sections["Services"].append(r)
            elif name.startswith("Docker:"):
                sections["Docker"].append(r)
            elif name.startswith("Agent:"):
                sections["Agents"].append(r)
            else:
                sections["Services"].append(r)

        for section, checks in sections.items():
            if checks:
                lines.append(f"<b>{section}:</b>")
                for c in checks:
                    icon = icons.get(c["status"], "\u26aa")
                    lines.append(f"  {icon} {c['name']}: {c.get('value', '')}")
                lines.append("")

        return "\n".join(lines)


async def main():
    """Entry point - monitoring agent + admin-only TG bot + repair callbacks."""
    from aiogram import Bot, Dispatcher as TgDispatcher, types, F
    from aiogram.filters import CommandStart, Command

    settings = get_settings()
    agent = MonitoringAgent()

    bot = Bot(token=settings.monitoring_bot_token)
    dp = TgDispatcher()

    @dp.message(CommandStart())
    async def cmd_start(message: types.Message):
        if message.from_user.id != ADMIN_USER_ID:
            await message.answer("\u26d4 Access denied.")
            return
        await message.answer(
            "\u2705 Monitoring Agent connected.\n\n"
            f"Checks: every {agent.check_interval // 60} min.\n"
            f"Reports: every {REPORT_INTERVAL // 3600} h.\n"
            f"Repair integration: ACTIVE\n\n"
            "Commands:\n"
            "/status - full report\n"
            "/check - run check now\n"
            "/repair_status - repair agent status",
            parse_mode="HTML",
        )

    @dp.message(Command("status"))
    async def cmd_status(message: types.Message):
        if message.from_user.id != ADMIN_USER_ID:
            return
        report = await agent.get_report()
        await message.answer(report, parse_mode="HTML")

    @dp.message(Command("check"))
    async def cmd_check(message: types.Message):
        if message.from_user.id != ADMIN_USER_ID:
            return
        await message.answer("\u23f3 Running check...")
        await agent._run_cycle(force_report=True)

    @dp.message(Command("repair_status"))
    async def cmd_repair_status(message: types.Message):
        if message.from_user.id != ADMIN_USER_ID:
            return
        # Read repair log from Redis
        try:
            from redis.asyncio import Redis
            r = Redis.from_url(settings.redis_url, decode_responses=True)
            logs = await r.lrange("repair:log", -10, -1)
            await r.close()
            if logs:
                lines = ["<b>Repair Agent - Recent Actions:</b>", ""]
                for log_str in reversed(logs):
                    log = json.loads(log_str)
                    success = log.get("result", {}).get("success", False)
                    icon = "\u2705" if success else "\u274c"
                    lines.append(
                        f"{icon} {log.get('check', '?')} - L{log.get('level', '?')} "
                        f"{log.get('action', '?')}\n"
                        f"   {log.get('started_at', '')[:19]}"
                    )
                await message.answer("\n".join(lines), parse_mode="HTML")
            else:
                await message.answer("No repair actions recorded yet.")
        except Exception as e:
            await message.answer(f"Error reading repair log: {e}")

    @dp.message(Command("repair_test"))
    async def cmd_repair_test(message: types.Message):
        """Test repair by simulating a problem in the queue."""
        if message.from_user.id != ADMIN_USER_ID:
            return
        from redis.asyncio import Redis
        r = Redis.from_url(settings.redis_url, decode_responses=True)
        test_task = {
            "name": "Service:salesbot-gateway",
            "status": "critical",
            "value": "TEST - port 8000 not responding",
        }
        await r.rpush(REPAIR_QUEUE_KEY, json.dumps(test_task))
        await r.close()
        await message.answer(
            "Test repair task added to queue.\n"
            "Repair Agent will process it shortly.",
            parse_mode="HTML",
        )

    # Handle Repair Agent's approval callbacks
    @dp.callback_query(F.data.startswith("repair_"))
    async def handle_repair_callback(callback: types.CallbackQuery):
        if callback.from_user.id != ADMIN_USER_ID:
            await callback.answer("Access denied", show_alert=True)
            return

        parts = callback.data.split(":")
        if len(parts) != 2:
            await callback.answer("Invalid callback", show_alert=True)
            return

        action_type, request_id = parts

        if action_type == "repair_approve":
            await agent.handle_repair_approval(request_id, True)
            await callback.answer("Approved!")
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n<b>\u2705 APPROVED by admin</b>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
        elif action_type == "repair_reject":
            await agent.handle_repair_approval(request_id, False)
            await callback.answer("Rejected!")
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n<b>\u274c REJECTED by admin</b>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
        else:
            await callback.answer("Unknown action", show_alert=True)

    @dp.message()
    async def ignore_all(message: types.Message):
        if message.from_user.id != ADMIN_USER_ID:
            await message.answer("\u26d4 Access denied.")

    async def monitoring_loop():
        await asyncio.sleep(3)
        await agent.start()

    await bot.delete_webhook(drop_pending_updates=True)
    monitor_task = asyncio.create_task(monitoring_loop())

    logger.info("monitoring_agent_started", admin_id=ADMIN_USER_ID)
    try:
        await dp.start_polling(bot)
    finally:
        monitor_task.cancel()
        if agent._redis:
            await agent._redis.close()


if __name__ == "__main__":
    asyncio.run(main())
