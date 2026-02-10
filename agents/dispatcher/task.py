"""
Task and TaskResult models - the universal contract between all agents.
No LLM, pure data structures.
"""
import uuid
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """All known task types in the system."""
    PRICING = "pricing"           # Расценка продукции
    SEARCH = "search"             # Поиск закупок/тендеров
    ANALYSIS = "analysis"         # Анализ документов (RAG)
    CRM_WRITE = "crm_write"      # Запись в CRM
    DB_WRITE = "db_write"         # Запись в БД
    MONITORING = "monitoring"     # Проверка здоровья системы
    REPAIR = "repair"             # Автоматическое восстановление системы
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"     # Отправлена агенту
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"         # Агент недоступен


class Task(BaseModel):
    """Universal task object passed between agents."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    source: str = ""              # Кто создал (chat_agent, web, telegram, etc.)
    session_id: str = ""
    payload: dict = Field(default_factory=dict)  # Данные задачи (product_spec, etc.)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: str | None = None  # Имя агента-исполнителя
    priority: int = 0             # 0 = обычный, 1+ = повышенный
    metadata: dict = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result returned by an agent after processing a task."""
    task_id: str
    status: TaskStatus
    agent_name: str
    result: dict = Field(default_factory=dict)
    error: str | None = None
    completed_at: datetime = Field(default_factory=datetime.utcnow)
