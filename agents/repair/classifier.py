"""
Repair Classifier - determines problem severity and repair action level.

Level 1 (AUTO):    Restart services, containers, cleanup disk
Level 2 (APPROVE): Config changes (.env, docker-compose, prompts, nginx)
Level 3 (CURSOR):  Python code fixes, new features, refactoring

No LLM. Pure rule-based classification.
"""
import structlog
from enum import Enum
from dataclasses import dataclass

logger = structlog.get_logger()


class RepairLevel(int, Enum):
    AUTO = 1       # Automatic - restart, cleanup
    APPROVE = 2    # Needs admin approval - config changes
    CURSOR = 3     # Needs Cursor + approval - code fixes


class ProblemType(str, Enum):
    SERVICE_DOWN = "service_down"
    CONTAINER_DOWN = "container_down"
    DISK_FULL = "disk_full"
    HIGH_CPU = "high_cpu"
    HIGH_RAM = "high_ram"
    REDIS_DOWN = "redis_down"
    POSTGRES_DOWN = "postgres_down"
    GATEWAY_ERROR = "gateway_error"
    AGENT_ERROR = "agent_error"
    CODE_ERROR = "code_error"
    CONFIG_ERROR = "config_error"
    UNKNOWN = "unknown"


@dataclass
class RepairPlan:
    """What the Repair Agent should do."""
    problem_type: ProblemType
    level: RepairLevel
    source_check: str          # Name of the health check that triggered
    description: str           # Human-readable description
    action: str                # What to do
    action_details: dict       # Parameters for the action


# Mapping: health check name pattern -> (ProblemType, RepairLevel, action)
REPAIR_RULES: list[tuple[str, ProblemType, RepairLevel, str]] = [
    # Service restarts - Level 1
    ("Service:salesbot-gateway", ProblemType.SERVICE_DOWN, RepairLevel.AUTO, "restart_systemd"),
    ("Service:salesbot-telegram", ProblemType.SERVICE_DOWN, RepairLevel.AUTO, "restart_systemd"),
    ("Service:salesbot-monitoring", ProblemType.SERVICE_DOWN, RepairLevel.AUTO, "restart_systemd"),
    ("Service:salesbot-repair", ProblemType.SERVICE_DOWN, RepairLevel.AUTO, "restart_systemd"),

    # Container restarts - Level 1
    ("Docker:salesbot-postgres", ProblemType.CONTAINER_DOWN, RepairLevel.AUTO, "restart_container"),
    ("Docker:salesbot-redis", ProblemType.CONTAINER_DOWN, RepairLevel.AUTO, "restart_container"),
    ("Docker:salesbot-webui", ProblemType.CONTAINER_DOWN, RepairLevel.AUTO, "restart_container"),

    # Core services - Level 1 first, escalate if persists
    ("Redis", ProblemType.REDIS_DOWN, RepairLevel.AUTO, "restart_container"),
    ("PostgreSQL", ProblemType.POSTGRES_DOWN, RepairLevel.AUTO, "restart_container"),
    ("Gateway API", ProblemType.GATEWAY_ERROR, RepairLevel.AUTO, "restart_systemd"),

    # Resources - Level 1
    ("Disk", ProblemType.DISK_FULL, RepairLevel.AUTO, "cleanup_disk"),
    ("CPU", ProblemType.HIGH_CPU, RepairLevel.AUTO, "log_and_wait"),
    ("RAM", ProblemType.HIGH_RAM, RepairLevel.AUTO, "log_and_wait"),
]


SERVICE_MAP = {
    "Service:salesbot-gateway": "salesbot-gateway",
    "Service:salesbot-telegram": "salesbot-telegram",
    "Service:salesbot-monitoring": "salesbot-monitoring",
    "Service:salesbot-repair": "salesbot-repair",
    "Gateway API": "salesbot-gateway",
}

CONTAINER_MAP = {
    "Docker:salesbot-postgres": "salesbot-postgres",
    "Docker:salesbot-redis": "salesbot-redis",
    "Docker:salesbot-webui": "salesbot-webui",
    "Redis": "salesbot-redis",
    "PostgreSQL": "salesbot-postgres",
}


class RepairClassifier:
    """Classifies health check failures into repair plans."""

    def __init__(self):
        self._failure_counts: dict[str, int] = {}  # Track repeated failures
        self.ESCALATION_THRESHOLD = 3  # After 3 auto-fixes fail, escalate

    def classify(self, check_name: str, check_status: str, check_value: str) -> RepairPlan | None:
        """
        Classify a failed health check into a repair plan.
        Returns None if no action needed (e.g., status is 'ok').
        """
        if check_status not in ("critical", "warning"):
            # Clear failure count on recovery
            self._failure_counts.pop(check_name, None)
            return None

        # Find matching rule
        for pattern, problem_type, level, action in REPAIR_RULES:
            if check_name == pattern or check_name.startswith(pattern.rstrip("*")):
                # Track failures for escalation
                count = self._failure_counts.get(check_name, 0) + 1
                self._failure_counts[check_name] = count

                # BUG FIX: Initialize effective_level before the escalation check
                # This ensures the variable is always defined before use,
                # avoiding potential NameError if conditions change
                effective_level = level
                
                # Escalate if auto-fix keeps failing
                if count > self.ESCALATION_THRESHOLD and level == RepairLevel.AUTO:
                    effective_level = RepairLevel.CURSOR
                    action = "cursor_diagnose"
                    logger.warning("repair_escalated",
                                   check=check_name,
                                   failures=count,
                                   new_level="CURSOR")

                # Build action details
                action_details = self._build_action_details(
                    check_name, action, check_value
                )

                plan = RepairPlan(
                    problem_type=problem_type,
                    level=effective_level,
                    source_check=check_name,
                    description=f"{check_name}: {check_value}",
                    action=action,
                    action_details=action_details,
                )

                logger.info("repair_classified",
                            check=check_name,
                            problem=problem_type.value,
                            level=effective_level.value,
                            action=action,
                            failure_count=count)
                return plan

        # Agent errors - check if it's a dispatcher agent issue
        if check_name.startswith("Agent:"):
            return RepairPlan(
                problem_type=ProblemType.AGENT_ERROR,
                level=RepairLevel.APPROVE,
                source_check=check_name,
                description=f"{check_name}: {check_value}",
                action="investigate_agent",
                action_details={"agent_name": check_name.replace("Agent:", "")},
            )

        # Unknown problem
        logger.warning("repair_unknown_check", check=check_name, status=check_status)
        return RepairPlan(
            problem_type=ProblemType.UNKNOWN,
            level=RepairLevel.APPROVE,
            source_check=check_name,
            description=f"Unknown issue: {check_name}: {check_value}",
            action="notify_admin",
            action_details={"check_name": check_name, "value": check_value},
        )

    def _build_action_details(self, check_name: str, action: str, value: str) -> dict:
        """Build specific action parameters."""
        details: dict = {"check_name": check_name, "value": value}

        if action == "restart_systemd":
            details["service_name"] = SERVICE_MAP.get(check_name, "")
        elif action == "restart_container":
            details["container_name"] = CONTAINER_MAP.get(check_name, "")
        elif action == "cleanup_disk":
            details["actions"] = ["docker_prune", "log_rotate", "tmp_cleanup"]
        elif action == "cursor_diagnose":
            details["context"] = (
                f"Service {check_name} has failed {self._failure_counts.get(check_name, 0)} times. "
                f"Auto-restart did not help. Last error: {value}. "
                f"Investigate logs and fix the root cause."
            )

        return details

    def reset_failure_count(self, check_name: str):
        """Reset failure counter after successful repair."""
        self._failure_counts.pop(check_name, None)
