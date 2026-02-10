"""
Repair Agent - the main orchestrator for automated system repair.

Listens to Redis queue 'repair:tasks' for problems detected by Monitoring Agent.
Classifies problems, executes repairs, verifies results.

NOTE: This agent does NOT run its own Telegram polling.
It uses Bot API for sending messages/keyboards directly.
Callback handling for approvals is done via Monitoring Agent's dispatcher.
"""
import asyncio
import json
import time
from datetime import datetime
import structlog
import httpx

from config.settings import get_settings
from agents.repair.classifier import RepairClassifier, RepairLevel, RepairPlan
from agents.repair.executor import RepairExecutor
from agents.repair.verifier import RepairVerifier
from agents.repair.cursor_engine import CursorEngine

logger = structlog.get_logger()

ADMIN_CHAT_ID = "160217558"
QUEUE_KEY = "repair:tasks"
REPAIR_LOG_KEY = "repair:log"
APPROVAL_QUEUE = "repair:approvals"  # Redis queue for approval responses
TG_API = "https://api.telegram.org/bot"


class RepairAgent:
    """Main Repair Agent - orchestrates problem detection -> fix -> verify cycle."""

    def __init__(self):
        self.settings = get_settings()
        self.token = self.settings.monitoring_bot_token
        self.classifier = RepairClassifier()
        self.executor = RepairExecutor()
        self.verifier = RepairVerifier()
        self.cursor = CursorEngine()
        self._running = True
        self._repair_history: list[dict] = []
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            from redis.asyncio import Redis
            self._redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        return self._redis

    async def start(self):
        """Start listening to repair queue."""
        logger.info("repair_agent_starting")

        await self._notify(
            "Repair Agent started.\n"
            f"Queue: {QUEUE_KEY}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        while self._running:
            try:
                task_data = await self._dequeue()
                if task_data:
                    await self._process_task(task_data)
                else:
                    await asyncio.sleep(5)  # Poll every 5 seconds
            except Exception as e:
                logger.error("repair_agent_error", error=str(e))
                await asyncio.sleep(10)

    async def _dequeue(self) -> dict | None:
        """Pop next task from Redis repair queue."""
        try:
            r = await self._get_redis()
            data = await r.lpop(QUEUE_KEY)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("repair_dequeue_error", error=str(e))
            return None

    async def _process_task(self, task_data: dict):
        """Process a single repair task."""
        check_name = task_data.get("name", "unknown")
        check_status = task_data.get("status", "unknown")
        check_value = task_data.get("value", "")

        logger.info("repair_processing",
                     check=check_name,
                     status=check_status,
                     value=check_value)

        # Classify the problem
        plan = self.classifier.classify(check_name, check_status, check_value)
        if plan is None:
            logger.info("repair_no_action_needed", check=check_name)
            return

        # Record start
        repair_record = {
            "check": check_name,
            "status": check_status,
            "level": plan.level.value,
            "action": plan.action,
            "started_at": datetime.now().isoformat(),
            "result": None,
        }

        # Get pre-repair state for verification
        pre_state = await self.verifier.get_current_state()

        # Execute based on level
        if plan.level == RepairLevel.AUTO:
            result = await self._handle_auto(plan)
        elif plan.level == RepairLevel.APPROVE:
            result = await self._handle_approve(plan)
        elif plan.level == RepairLevel.CURSOR:
            result = await self._handle_cursor(plan)
        else:
            result = {"success": False, "message": f"Unknown level: {plan.level}"}

        # Verify repair (only for actions that actually did something)
        if result.get("success") and plan.action not in ("log_and_wait", "notify_admin"):
            verification = await self.verifier.verify(check_name, pre_state)

            if verification["success"]:
                self.classifier.reset_failure_count(check_name)
                await self._notify(
                    f"Repair SUCCESS\n"
                    f"Check: {check_name}\n"
                    f"Action: {plan.action}\n"
                    f"Verification: {verification['message']}"
                )
            elif verification.get("new_problems"):
                await self._notify(
                    f"Repair WARNING\n"
                    f"Check: {check_name} - action: {plan.action}\n"
                    f"New problems: {', '.join(verification['new_problems'])}\n"
                    f"Manual investigation required!"
                )
            else:
                await self._notify(
                    f"Repair FAILED\n"
                    f"Check: {check_name}\n"
                    f"Action: {plan.action} did not resolve the issue.\n"
                    f"Will escalate on next cycle."
                )
        elif not result.get("success"):
            await self._notify(
                f"Repair FAILED\n"
                f"Check: {check_name}\n"
                f"Action: {plan.action}\n"
                f"Error: {result.get('message', 'unknown')}"
            )

        # Log repair
        repair_record["result"] = result
        repair_record["completed_at"] = datetime.now().isoformat()
        self._repair_history.append(repair_record)

        if len(self._repair_history) > 100:
            self._repair_history = self._repair_history[-100:]

        await self._save_log(repair_record)

    async def _handle_auto(self, plan: RepairPlan) -> dict:
        """Handle Level 1 - automatic repair."""
        logger.info("repair_auto", action=plan.action, check=plan.source_check)

        result = await self.executor.execute(plan.action, plan.action_details)

        status_icon = "OK" if result["success"] else "FAIL"
        await self._notify(
            f"Auto-repair [{status_icon}]\n"
            f"Check: {plan.source_check}\n"
            f"Action: {plan.action}\n"
            f"Result: {result['message']}"
        )
        return result

    async def _handle_approve(self, plan: RepairPlan) -> dict:
        """Handle Level 2 - needs admin approval via inline keyboard."""
        logger.info("repair_approval_requested",
                     action=plan.action,
                     check=plan.source_check)

        import uuid
        request_id = str(uuid.uuid4())[:8]
        level_name = {2: "CONFIG", 3: "CURSOR/CODE"}.get(plan.level.value, f"L{plan.level.value}")

        message = (
            f"<b>REPAIR REQUEST [{level_name}]</b>\n\n"
            f"<b>Problem:</b> {plan.description}\n"
            f"<b>Action:</b> {plan.action}\n"
            f"\n<i>ID: {request_id} | Timeout: 30 min</i>"
        )

        # Send with inline keyboard via Bot API
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"repair_approve:{request_id}"},
                    {"text": "Reject", "callback_data": f"repair_reject:{request_id}"},
                ],
            ]
        }

        await self._send_tg(message, reply_markup=keyboard)

        # Wait for approval via Redis
        approved = await self._wait_approval(request_id, timeout=1800)

        if not approved:
            logger.info("repair_approval_rejected", check=plan.source_check)
            return {"success": False, "message": "Admin rejected or timeout"}

        result = await self.executor.execute(plan.action, plan.action_details)
        return result

    async def _handle_cursor(self, plan: RepairPlan) -> dict:
        """Handle Level 3 - Cursor code fix with approval."""
        logger.info("repair_cursor_requested",
                     action=plan.action,
                     check=plan.source_check)

        logs = await self._get_service_logs(plan.source_check)

        import uuid
        request_id = str(uuid.uuid4())[:8]

        cursor_desc = (
            f"<b>CURSOR REPAIR REQUEST</b>\n\n"
            f"<b>Problem:</b> {plan.description}\n"
            f"<b>Action:</b> Cursor Background Agent - code analysis & fix\n"
            f"<b>Failures:</b> {self.classifier._failure_counts.get(plan.source_check, 0)}\n"
            f"\n<i>ID: {request_id} | Timeout: 30 min</i>"
        )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Approve Cursor Fix", "callback_data": f"repair_approve:{request_id}"},
                    {"text": "Reject", "callback_data": f"repair_reject:{request_id}"},
                ],
            ]
        }

        await self._send_tg(cursor_desc, reply_markup=keyboard)

        approved = await self._wait_approval(request_id, timeout=1800)

        if not approved:
            return {"success": False, "message": "Admin rejected Cursor repair"}

        result = await self.cursor.launch_repair(
            problem_description=plan.description,
            logs=logs,
            context=plan.action_details.get("context", ""),
        )

        if result.get("success"):
            agent_id = result.get("agent_id", "unknown")
            await self._notify(
                f"Cursor Agent launched: {agent_id}\n"
                f"Problem: {plan.source_check}\n"
                f"Waiting for completion..."
            )

            final = await self.cursor.wait_for_completion(agent_id, timeout=600)

            if final.get("success"):
                await self._notify(
                    f"Cursor Agent completed: {agent_id}\n"
                    f"PR should be available in GitHub.\n"
                    f"After review, pull changes on server."
                )
                return {"success": True, "message": f"Cursor agent {agent_id} completed"}
            else:
                await self._notify(
                    f"Cursor Agent FAILED: {agent_id}\n"
                    f"Error: {final.get('error', 'unknown')}"
                )
                return {"success": False, "message": f"Cursor agent failed: {final.get('error')}"}
        else:
            return {"success": False, "message": f"Failed to launch Cursor: {result.get('error')}"}

    async def _wait_approval(self, request_id: str, timeout: int = 1800) -> bool:
        """Wait for admin approval via Redis queue."""
        r = await self._get_redis()
        key = f"{APPROVAL_QUEUE}:{request_id}"
        start = time.time()

        while time.time() - start < timeout:
            result = await r.get(key)
            if result is not None:
                await r.delete(key)
                return result == "approved"
            await asyncio.sleep(2)

        await self._notify(f"Approval timeout for request {request_id}")
        return False

    async def _get_service_logs(self, check_name: str) -> str:
        """Get recent logs for a service."""
        service_map = {
            "Service:salesbot-gateway": "salesbot-gateway",
            "Service:salesbot-telegram": "salesbot-telegram",
            "Service:salesbot-monitoring": "salesbot-monitoring",
            "Gateway API": "salesbot-gateway",
        }
        service = service_map.get(check_name, "")
        if not service:
            return ""

        try:
            proc = await asyncio.create_subprocess_exec(
                "journalctl", "-u", service, "-n", "100", "--no-pager",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            return stdout.decode()
        except Exception as e:
            logger.error("repair_get_logs_error", service=service, error=str(e))
            return f"Failed to get logs: {e}"

    async def _send_tg(self, text: str, reply_markup: dict | None = None):
        """Send message via Telegram Bot API (no polling needed)."""
        try:
            payload = {
                "chat_id": ADMIN_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            }
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{TG_API}{self.token}/sendMessage",
                    json=payload,
                )
                data = resp.json()
                if not data.get("ok"):
                    logger.error("repair_tg_send_fail", response=data)
        except Exception as e:
            logger.error("repair_tg_send_error", error=str(e))

    async def _notify(self, message: str):
        """Send notification to admin."""
        await self._send_tg(f"<b>Repair Agent</b>\n\n{message}")

    async def _save_log(self, record: dict):
        """Save repair log to Redis."""
        try:
            r = await self._get_redis()
            await r.rpush(REPAIR_LOG_KEY, json.dumps(record, ensure_ascii=False))
            await r.ltrim(REPAIR_LOG_KEY, -500, -1)
        except Exception as e:
            logger.error("repair_save_log_error", error=str(e))

    async def get_status(self) -> str:
        """Get repair agent status report."""
        recent = self._repair_history[-10:] if self._repair_history else []
        lines = [
            "<b>Repair Agent Status</b>",
            f"Queue: {QUEUE_KEY}",
            f"Total repairs: {len(self._repair_history)}",
            f"Rate limit: {len(self.executor._action_log)}/3 per hour",
            "",
        ]
        if recent:
            lines.append("<b>Recent repairs:</b>")
            for r in reversed(recent):
                success = r.get("result", {}).get("success", False)
                icon = "OK" if success else "FAIL"
                lines.append(
                    f"  [{icon}] {r['check']} - L{r['level']} {r['action']} "
                    f"({r.get('started_at', '')[:16]})"
                )
        else:
            lines.append("No repairs performed yet.")
        return "\n".join(lines)

    async def close(self):
        self._running = False
        if self._redis:
            await self._redis.close()


async def main():
    """Entry point - Repair Agent (no TG polling, only queue listener)."""
    agent = RepairAgent()
    logger.info("repair_agent_started")
    try:
        await agent.start()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
