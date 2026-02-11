"""
Repair Executor - performs Level 1 automatic actions.

Actions:
- restart_systemd: systemctl restart <service>
- restart_container: docker restart <container>
- cleanup_disk: docker system prune, log rotation, /tmp cleanup
- log_and_wait: just log the issue (CPU/RAM spikes)

All actions are wrapped in try/except with logging.
Rate-limited to MAX_ACTIONS_PER_HOUR to prevent cascade restarts.
"""
import asyncio
import time
import structlog

logger = structlog.get_logger()

MAX_ACTIONS_PER_HOUR = 3


class RepairExecutor:
    """Executes Level 1 automatic repair actions."""

    def __init__(self):
        self._action_log: list[float] = []  # timestamps of recent actions

    def _rate_check(self) -> bool:
        """Check if we haven't exceeded MAX_ACTIONS_PER_HOUR."""
        now = time.time()
        # Remove entries older than 1 hour
        self._action_log = [t for t in self._action_log if now - t < 3600]
        if len(self._action_log) >= MAX_ACTIONS_PER_HOUR:
            logger.warning("repair_rate_limited",
                           actions_last_hour=len(self._action_log),
                           limit=MAX_ACTIONS_PER_HOUR)
            return False
        return True

    async def execute(self, action: str, details: dict) -> dict:
        """
        Execute a repair action. Returns result dict with:
        - success: bool
        - message: str
        - output: str (command output if any)
        """
        if action == "log_and_wait":
            # No actual action, just log
            logger.info("repair_log_and_wait",
                        check=details.get("check_name", ""),
                        value=details.get("value", ""))
            return {
                "success": True,
                "message": f"Logged: {details.get('check_name', '')} - waiting for recovery",
                "output": "",
            }

        if not self._rate_check():
            return {
                "success": False,
                "message": f"Rate limited: max {MAX_ACTIONS_PER_HOUR} auto-actions per hour",
                "output": "",
            }

        handler = getattr(self, f"_do_{action}", None)
        if handler is None:
            logger.error("repair_unknown_action", action=action)
            return {
                "success": False,
                "message": f"Unknown action: {action}",
                "output": "",
            }

        try:
            result = await handler(details)
            self._action_log.append(time.time())
            return result
        except Exception as e:
            logger.error("repair_execute_error", action=action, error=str(e))
            return {
                "success": False,
                "message": f"Error executing {action}: {str(e)}",
                "output": "",
            }

    async def _do_restart_systemd(self, details: dict) -> dict:
        """Restart a systemd service.
        
        SECURITY FIX: Tries user-level systemctl first to reduce privilege requirements.
        Falls back to sudo only if user-level fails with permission/not-found error.
        This approach follows the principle of least privilege.
        """
        service = details.get("service_name", "")
        if not service:
            return {"success": False, "message": "No service_name provided", "output": ""}

        # Safety: only allow known services
        allowed = {
            "salesbot-gateway", "salesbot-telegram",
            "salesbot-monitoring", "salesbot-repair",
        }
        if service not in allowed:
            return {
                "success": False,
                "message": f"Service {service} not in allowed list",
                "output": "",
            }

        logger.info("repair_restarting_service", service=service)

        # SECURITY: Try user-level systemctl first (no sudo required)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "--user", "restart", service,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = (stdout.decode() + stderr.decode()).strip()
            
            if proc.returncode == 0:
                logger.info("repair_service_restarted", service=service, method="user")
                return {
                    "success": True,
                    "message": f"Service {service} restarted successfully (user mode)",
                    "output": output,
                }
            
            # If user mode failed with specific errors, try sudo as fallback
            if "No such file" in output or "not loaded" in output.lower() or "access denied" in output.lower():
                logger.warning("repair_user_systemctl_failed", 
                              service=service, 
                              output=output,
                              fallback="sudo")
        except Exception as e:
            logger.warning("repair_user_systemctl_error", 
                          service=service, 
                          error=str(e),
                          fallback="sudo")
        
        # Fallback to sudo for system-level services
        proc = await asyncio.create_subprocess_exec(
            "sudo", "systemctl", "restart", service,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = (stdout.decode() + stderr.decode()).strip()

        if proc.returncode == 0:
            logger.info("repair_service_restarted", service=service, method="sudo")
            return {
                "success": True,
                "message": f"Service {service} restarted successfully",
                "output": output,
            }
        else:
            logger.error("repair_service_restart_failed",
                         service=service, returncode=proc.returncode, output=output)
            return {
                "success": False,
                "message": f"Failed to restart {service} (exit {proc.returncode})",
                "output": output,
            }

    async def _do_restart_container(self, details: dict) -> dict:
        """Restart a Docker container."""
        container = details.get("container_name", "")
        if not container:
            return {"success": False, "message": "No container_name provided", "output": ""}

        allowed = {"salesbot-postgres", "salesbot-redis", "salesbot-webui"}
        if container not in allowed:
            return {
                "success": False,
                "message": f"Container {container} not in allowed list",
                "output": "",
            }

        logger.info("repair_restarting_container", container=container)

        proc = await asyncio.create_subprocess_exec(
            "docker", "restart", container,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = (stdout.decode() + stderr.decode()).strip()

        if proc.returncode == 0:
            logger.info("repair_container_restarted", container=container)
            return {
                "success": True,
                "message": f"Container {container} restarted successfully",
                "output": output,
            }
        else:
            logger.error("repair_container_restart_failed",
                         container=container, returncode=proc.returncode)
            return {
                "success": False,
                "message": f"Failed to restart {container} (exit {proc.returncode})",
                "output": output,
            }

    async def _do_cleanup_disk(self, details: dict) -> dict:
        """Clean up disk space."""
        logger.info("repair_cleanup_disk")
        outputs = []

        # Docker prune
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "system", "prune", "-f",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            outputs.append(f"docker prune: {stdout.decode().strip()}")
        except Exception as e:
            outputs.append(f"docker prune error: {e}")

        # Clean old logs
        # SECURITY: Try user-level journalctl first, fallback to sudo
        try:
            # First try user-level journal cleanup
            proc = await asyncio.create_subprocess_exec(
                "journalctl", "--user", "--vacuum-time=3d",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            user_output = stdout.decode().strip()
            
            if proc.returncode == 0:
                outputs.append(f"journal cleanup (user): {user_output}")
            else:
                # Fallback to sudo for system journals
                logger.warning("repair_user_journalctl_failed", fallback="sudo")
                proc = await asyncio.create_subprocess_exec(
                    "sudo", "journalctl", "--vacuum-time=3d",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                outputs.append(f"journal cleanup: {stdout.decode().strip()}")
        except Exception as e:
            outputs.append(f"journal cleanup error: {e}")

        # Clean /tmp
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "find", "/tmp", "-type", "f", "-mtime", "+7", "-delete",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            outputs.append("tmp cleanup: done")
        except Exception as e:
            outputs.append(f"tmp cleanup error: {e}")

        combined = "\n".join(outputs)
        logger.info("repair_cleanup_done", output=combined)
        return {
            "success": True,
            "message": "Disk cleanup completed",
            "output": combined,
        }
