"""
Cursor Engine - client for Cursor Background Agents API.

Uses Cursor's cloud infrastructure to analyze code, diagnose issues,
and create fixes via Pull Requests.

CRITICAL RULES embedded in every prompt to Cursor:
- NEVER delete existing functionality
- NEVER simplify by removing features
- Make MINIMAL, targeted changes
- EXPAND and IMPROVE, never cut corners
- Preserve all existing tests and interfaces
"""
import asyncio
import json
import structlog
import httpx

from config.settings import get_settings

logger = structlog.get_logger()

CURSOR_API_BASE = "https://api.cursor.com/v0"

# System instructions appended to EVERY Cursor prompt
CURSOR_SYSTEM_RULES = """
CRITICAL RULES FOR CODE CHANGES:
1. NEVER delete existing functionality. Every existing feature must be preserved.
2. NEVER simplify by removing code. If refactoring, move code to appropriate modules.
3. Make MINIMAL, targeted changes. Fix only what is broken.
4. EXPAND and IMPROVE, never reduce capabilities.
5. Preserve ALL existing interfaces (API endpoints, function signatures, class interfaces).
6. Preserve ALL existing error handling and logging.
7. Add comments explaining WHY changes were made.
8. If a service keeps crashing, investigate ROOT CAUSE in logs before changing code.
9. Test changes mentally before applying - will this break anything else?
10. If unsure, add defensive code (try/except, fallbacks) rather than removing problematic code.

PROJECT STRUCTURE:
- /opt/salesbot/ - root project directory
- agents/ - all agent modules (chat, dispatcher, monitoring, repair, etc.)
- config/ - settings and prompts
- gateway/ - FastAPI API gateway
- interfaces/ - Telegram bot, WebUI connectors
- docker-compose.yml - PostgreSQL, Redis, WebUI containers

STACK: Python 3.10+, FastAPI, aiogram 3, Redis, PostgreSQL, Docker, systemd
"""


class CursorEngine:
    """Client for Cursor Background Agents API."""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.cursor_api_key
        # FIX: Validate API key presence during initialization
        # This prevents runtime errors when methods requiring authorization are called
        if not self.api_key:
            logger.warning("cursor_api_key_not_configured",
                          message="Cursor API key is not set. Repair operations will fail.")
        self.repo_url = self.settings.github_repo_url
        self.repo_branch = getattr(self.settings, "github_repo_branch", "main")

    async def launch_repair(
        self,
        problem_description: str,
        logs: str = "",
        affected_files: list[str] | None = None,
        context: str = "",
    ) -> dict:
        """
        Launch a Cursor Background Agent to fix a problem.

        Args:
            problem_description: What went wrong
            logs: Recent log output from the failing service
            affected_files: List of file paths likely involved
            context: Additional context from classifier

        Returns:
            dict with agent_id, status, url
        """
        # FIX: Check API key before making request to fail early with clear error
        if not self.api_key:
            logger.error("cursor_launch_no_api_key",
                        message="Cannot launch repair - Cursor API key is not configured")
            return {
                "success": False,
                "error": "Cursor API key is not configured. Set CURSOR_API_KEY environment variable.",
            }
        
        prompt = self._build_prompt(problem_description, logs, affected_files, context)

        logger.info("cursor_launching_agent",
                     problem=problem_description[:100],
                     repo=self.repo_url)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{CURSOR_API_BASE}/agents",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": {"text": prompt},
                        "source": {
                            "repository": self.repo_url,
                            "ref": self.repo_branch,
                        },
                    },
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    agent_id = data.get("id", data.get("agent_id", "unknown"))
                    logger.info("cursor_agent_launched", agent_id=agent_id)
                    return {
                        "success": True,
                        "agent_id": agent_id,
                        "status": "launched",
                        "data": data,
                    }
                else:
                    error_text = resp.text[:500]
                    logger.error("cursor_launch_failed",
                                 status=resp.status_code,
                                 error=error_text)
                    return {
                        "success": False,
                        "error": f"Cursor API returned {resp.status_code}: {error_text}",
                    }

        except Exception as e:
            logger.error("cursor_launch_error", error=str(e))
            return {
                "success": False,
                "error": f"Failed to reach Cursor API: {str(e)}",
            }

    async def get_agent_status(self, agent_id: str) -> dict:
        """Poll Cursor agent status."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{CURSOR_API_BASE}/agents/{agent_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code == 200:
                    return {"success": True, "data": resp.json()}
                return {"success": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_followup(self, agent_id: str, message: str) -> dict:
        """Send additional instructions to a running agent."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{CURSOR_API_BASE}/agents/{agent_id}/followup",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"prompt": {"text": message}},
                )
                if resp.status_code in (200, 201):
                    return {"success": True, "data": resp.json()}
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_completion(
        self,
        agent_id: str,
        timeout: int = 600,
        poll_interval: int = 15,
    ) -> dict:
        """
        Poll agent until completion or timeout.
        Returns final status.
        """
        start = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                return {
                    "success": False,
                    "error": f"Cursor agent {agent_id} timed out after {timeout}s",
                    "agent_id": agent_id,
                }

            status = await self.get_agent_status(agent_id)
            if not status.get("success"):
                await asyncio.sleep(poll_interval)
                continue

            data = status.get("data", {})
            agent_status = data.get("status", "unknown")

            logger.info("cursor_agent_poll",
                        agent_id=agent_id,
                        status=agent_status,
                        elapsed=int(elapsed))

            if agent_status in ("completed", "succeeded", "done"):
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "status": agent_status,
                    "data": data,
                }
            elif agent_status in ("failed", "error", "cancelled"):
                return {
                    "success": False,
                    "agent_id": agent_id,
                    "status": agent_status,
                    "error": data.get("error", "Agent failed"),
                    "data": data,
                }

            await asyncio.sleep(poll_interval)

    def _build_prompt(
        self,
        problem: str,
        logs: str,
        affected_files: list[str] | None,
        context: str,
    ) -> str:
        """Build a detailed prompt for Cursor with safety rules."""
        parts = [CURSOR_SYSTEM_RULES]

        parts.append(f"\n\nPROBLEM:\n{problem}")

        if context:
            parts.append(f"\n\nCONTEXT:\n{context}")

        if logs:
            # Truncate logs to avoid exceeding context limits
            truncated = logs[-3000:] if len(logs) > 3000 else logs
            parts.append(f"\n\nRECENT LOGS:\n```\n{truncated}\n```")

        if affected_files:
            parts.append(f"\n\nLIKELY AFFECTED FILES:\n" +
                         "\n".join(f"- {f}" for f in affected_files))

        parts.append(
            "\n\nINSTRUCTIONS:\n"
            "1. Read the logs carefully to identify the ROOT CAUSE.\n"
            "2. Check the affected files for bugs or misconfigurations.\n"
            "3. Apply a MINIMAL fix that resolves the issue.\n"
            "4. DO NOT remove any existing functionality.\n"
            "5. DO NOT simplify code by cutting features.\n"
            "6. Add error handling if the crash was unhandled.\n"
            "7. Create a PR with a clear description of what was fixed and why."
        )

        return "\n".join(parts)


# Singleton
_engine: CursorEngine | None = None


def get_cursor_engine() -> CursorEngine:
    global _engine
    if _engine is None:
        _engine = CursorEngine()
    return _engine
