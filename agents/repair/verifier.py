"""
Repair Verifier - checks system health after a repair action.

Flow:
1. Wait VERIFY_DELAY seconds after repair
2. Run the same health checks as Monitoring Agent
3. If the original problem is resolved -> SUCCESS
4. If new problems appeared -> ROLLBACK + alert
5. If problem persists -> report as failed repair
"""
import asyncio
import structlog

from agents.monitoring.health_checks import HealthChecks

logger = structlog.get_logger()

VERIFY_DELAY = 30  # seconds to wait after repair before checking


class RepairVerifier:
    """Verifies system health after repair actions."""

    def __init__(self):
        self.checks = HealthChecks()

    async def verify(
        self,
        original_check_name: str,
        pre_repair_results: list[dict] | None = None,
    ) -> dict:
        """
        Verify that repair was successful.

        Args:
            original_check_name: The health check that triggered the repair
            pre_repair_results: Health check results before repair (for comparison)

        Returns:
            dict with:
            - success: bool
            - original_resolved: bool
            - new_problems: list of new issues
            - message: str
        """
        logger.info("repair_verify_waiting",
                     delay=VERIFY_DELAY,
                     check=original_check_name)

        await asyncio.sleep(VERIFY_DELAY)

        # Run all checks
        post_results = await self.checks.run_all()

        # Check if original problem is resolved
        original_resolved = True
        for r in post_results:
            if r["name"] == original_check_name:
                if r["status"] in ("critical", "warning"):
                    original_resolved = False
                break

        # Check for new problems
        new_problems = []
        if pre_repair_results:
            pre_critical = {
                r["name"] for r in pre_repair_results
                if r["status"] in ("critical", "warning")
            }
            post_critical = {
                r["name"] for r in post_results
                if r["status"] in ("critical", "warning")
            }
            new_problems = list(post_critical - pre_critical)

        # Determine overall success
        if original_resolved and not new_problems:
            message = f"Repair verified: {original_check_name} resolved, no new issues"
            logger.info("repair_verify_success", check=original_check_name)
            return {
                "success": True,
                "original_resolved": True,
                "new_problems": [],
                "message": message,
                "post_results": post_results,
            }
        elif original_resolved and new_problems:
            message = (
                f"Repair of {original_check_name} succeeded but "
                f"NEW PROBLEMS detected: {', '.join(new_problems)}"
            )
            logger.warning("repair_verify_new_problems",
                           check=original_check_name,
                           new_problems=new_problems)
            return {
                "success": False,
                "original_resolved": True,
                "new_problems": new_problems,
                "message": message,
                "post_results": post_results,
            }
        else:
            message = f"Repair FAILED: {original_check_name} still in error state"
            logger.warning("repair_verify_failed", check=original_check_name)
            return {
                "success": False,
                "original_resolved": False,
                "new_problems": new_problems,
                "message": message,
                "post_results": post_results,
            }

    async def get_current_state(self) -> list[dict]:
        """Snapshot current health state (for pre-repair comparison)."""
        return await self.checks.run_all()

    async def rollback_systemd(self, service_name: str) -> dict:
        """Emergency stop of a service that's causing more problems."""
        logger.warning("repair_rollback_systemd", service=service_name)
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "stop", service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = (stdout.decode() + stderr.decode()).strip()
            return {"success": proc.returncode == 0, "output": output}
        except Exception as e:
            return {"success": False, "output": str(e)}

    async def rollback_container(self, container_name: str) -> dict:
        """Emergency stop of a container that's causing more problems."""
        logger.warning("repair_rollback_container", container=container_name)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = (stdout.decode() + stderr.decode()).strip()
            return {"success": proc.returncode == 0, "output": output}
        except Exception as e:
            return {"success": False, "output": str(e)}
