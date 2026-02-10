# SalesBot System - Cursor Rules

## CRITICAL: Quality Preservation Rules

When making changes to this codebase, you MUST follow these rules:

### 1. NEVER Delete Functionality
- Do NOT remove existing features, endpoints, or capabilities
- Do NOT simplify code by cutting features
- If refactoring, move code to appropriate modules, never delete it

### 2. Minimal Changes Only
- Fix ONLY what is broken
- Make the smallest possible change that resolves the issue
- Do NOT refactor unrelated code
- Do NOT change code style unless directly fixing the bug

### 3. Expand and Improve
- Add error handling where missing
- Add logging for better debugging
- Add defensive code (try/except, fallbacks) rather than removing problematic code
- Preserve ALL existing interfaces (API endpoints, function signatures)

### 4. Safety First
- Always check logs BEFORE changing code
- Test changes mentally - will this break anything else?
- Preserve ALL existing tests
- Add comments explaining WHY changes were made

## Project Structure

```
/opt/salesbot/
  agents/
    dispatcher/    - Rule-based task router (NO LLM)
    monitoring/    - System health monitoring
    repair/        - Automated repair agent
    orchestrator/  - Chat agent with GigaChat
    llm/           - GigaChat client
    pricing/       - (future) Pricing agent
    search/        - (future) Search agent
    analysis/      - (future) RAG analysis
  config/
    settings.py    - Pydantic settings from .env
    prompts/       - Agent system prompts
  gateway/
    main.py        - FastAPI API gateway
    sessions.py    - Redis session manager
  interfaces/
    telegram/      - Aiogram 3 Telegram bot
  docker-compose.yml - PostgreSQL, Redis, WebUI
```

## Tech Stack
- Python 3.10+, FastAPI, aiogram 3
- PostgreSQL (Docker), Redis (Docker)
- GigaChat LLM, Cursor Background Agents API
- systemd for service management
- structlog for logging

## Services (systemd)
- salesbot-gateway (port 8000)
- salesbot-telegram
- salesbot-monitoring
- salesbot-repair

## Important Notes
- .env file contains ALL secrets - never commit it
- Settings loaded via Pydantic BaseSettings
- All agents communicate through Redis queues and HTTP
- Dispatcher is pure rule-based, NO LLM
- Admin chat ID: 160217558 (hardcoded for security)
