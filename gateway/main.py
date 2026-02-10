"""
Gateway API - OpenAI-compatible endpoint for Open WebUI and Telegram.
Receives messages, routes to Chat Agent, returns responses.
Dispatcher endpoints for agent system management.
"""
import json
import time
import uuid
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from config.settings import get_settings
from agents.orchestrator.chat_agent import get_chat_agent
from agents.dispatcher import (
    Dispatcher, get_dispatcher,
    AgentRegistry, AgentInfo, AgentStatus,
    Task, TaskResult, TaskStatus, TaskType,
)
from gateway.sessions import get_session_manager
from agents.etm.handler import handle_etm_price, handle_etm_remains

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("gateway_startup", host=settings.gateway_host, port=settings.gateway_port)

    # Initialize dispatcher and register known agents (initially offline)
    dispatcher = get_dispatcher()
    await _register_known_agents(dispatcher)
    logger.info("dispatcher_initialized")

    # Register ETM handlers (in-process, synchronous dispatch)
    dispatcher.register_handler("etm_price", handle_etm_price)
    dispatcher.register_handler("etm_remains", handle_etm_remains)
    logger.info("etm_handlers_registered")

    # Set etm_agent as ONLINE since handler is in-process
    from redis.asyncio import Redis as _Redis
    _r = _Redis.from_url(settings.redis_url, decode_responses=True)
    _agent_data = await _r.hget("dispatcher:agents", "etm_agent")
    if _agent_data:
        import json as _json
        _ad = _json.loads(_agent_data)
        _ad["status"] = "online"
        import time as _time
        _ad["last_heartbeat"] = _time.time()
        await _r.hset("dispatcher:agents", "etm_agent", _json.dumps(_ad))
    await _r.close()
    logger.info("etm_agent_set_online")

    yield

    dispatcher = get_dispatcher()
    await dispatcher.close()
    sm = get_session_manager()
    await sm.close()
    logger.info("gateway_shutdown")


async def _register_known_agents(dispatcher: Dispatcher):
    """Pre-register all known agents. They start OFFLINE until they send heartbeat."""
    known_agents = [
        AgentInfo(
            name="pricing_agent",
            task_types=[TaskType.PRICING.value],
            status=AgentStatus.OFFLINE,
            description="Pricing products on client request",
        ),
        AgentInfo(
            name="search_agent",
            task_types=[TaskType.SEARCH.value],
            status=AgentStatus.OFFLINE,
            description="Search for tenders and procurements",
        ),
        AgentInfo(
            name="analysis_agent",
            task_types=[TaskType.ANALYSIS.value],
            status=AgentStatus.OFFLINE,
            description="RAG document analysis",
        ),
        AgentInfo(
            name="crm_agent",
            task_types=[TaskType.CRM_WRITE.value],
            status=AgentStatus.OFFLINE,
            description="Write data to CRM",
        ),
        AgentInfo(
            name="db_agent",
            task_types=[TaskType.DB_WRITE.value],
            status=AgentStatus.OFFLINE,
            description="Write data to database",
        ),
        AgentInfo(
            name="monitoring_agent",
            task_types=[TaskType.MONITORING.value],
            status=AgentStatus.OFFLINE,
            description="System health monitoring",
        ),
        AgentInfo(
            name="repair_agent",
            task_types=[TaskType.REPAIR.value],
            status=AgentStatus.OFFLINE,
            description="Automated system repair via Cursor API",
        ),
        AgentInfo(
            name="etm_agent",
            task_types=[TaskType.ETM_PRICE.value, TaskType.ETM_REMAINS.value, TaskType.ETM_CATALOG.value],
            status=AgentStatus.OFFLINE,
            description="ETM API agent: prices, stock, catalog",
        ),
    ]
    for agent in known_agents:
        await dispatcher.registry.register(agent)
        # Set them offline (register sets online, we override)
        agent.status = AgentStatus.OFFLINE
        agent.last_heartbeat = 0
        from redis.asyncio import Redis
        r = Redis.from_url(settings.redis_url, decode_responses=True)
        await r.hset("dispatcher:agents", agent.name, agent.model_dump_json())
        await r.close()


app = FastAPI(
    title="SalesBot Gateway",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health ====================

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ==================== Dispatcher API ====================

@app.get("/api/dispatcher/status")
async def dispatcher_status():
    """Full system status - all agents, their health, capabilities."""
    dispatcher = get_dispatcher()
    return await dispatcher.get_status()


@app.post("/api/dispatcher/dispatch")
async def dispatch_task(task: Task):
    """Submit a task to the dispatcher for routing."""
    dispatcher = get_dispatcher()
    result = await dispatcher.dispatch(task)
    return result.model_dump()


@app.post("/api/dispatcher/heartbeat/{agent_name}")
async def agent_heartbeat(agent_name: str):
    """Agent reports it is alive."""
    dispatcher = get_dispatcher()
    await dispatcher.registry.heartbeat(agent_name)
    return {"status": "ok", "agent": agent_name}


@app.post("/api/dispatcher/register")
async def register_agent(agent: AgentInfo):
    """Register a new agent or update existing."""
    dispatcher = get_dispatcher()
    await dispatcher.registry.register(agent)
    return {"status": "registered", "agent": agent.name}


@app.get("/api/dispatcher/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result."""
    dispatcher = get_dispatcher()
    task = await dispatcher.get_task(task_id)
    result = await dispatcher.get_result(task_id)
    return {
        "task": task.model_dump() if task else None,
        "result": result.model_dump() if result else None,
    }


# ==================== OpenAI-Compatible API ====================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "salesbot-consultant"
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = 500
    stream: bool = False


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list."""
    return {
        "object": "list",
        "data": [
            {
                "id": "salesbot-consultant",
                "object": "model",
                "created": 1700000000,
                "owned_by": "salesbot",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint for Open WebUI."""
    agent = get_chat_agent()

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_msg = request.messages[-1]
    if user_msg.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    history = []
    for msg in request.messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.content})

    session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(len(history)) + user_msg.content[:50]))
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_response(agent, user_msg.content, history, session_id, completion_id, created),
            media_type="text/event-stream",
        )

    response_text = await agent.respond(
        user_message=user_msg.content,
        history=history,
        session_id=session_id,
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": "salesbot-consultant",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_response(agent, user_message, history, session_id, completion_id, created):
    """SSE stream in OpenAI format for Open WebUI."""
    try:
        async for chunk in agent.respond_stream(user_message=user_message, history=history, session_id=session_id):
            data = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": "salesbot-consultant",
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        final = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                 "model": "salesbot-consultant",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("stream_error", error=str(e))
        err = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
               "model": "salesbot-consultant",
               "choices": [{"index": 0, "delta": {"content": "Error. Try again."}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


# ==================== Direct API (Telegram) ====================

@app.post("/api/chat/clear")
async def clear_session(session_id: str):
    """Clear session history."""
    session_mgr = get_session_manager()
    await session_mgr.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


class DirectChatRequest(BaseModel):
    session_id: str
    message: str
    channel: str = "api"


@app.post("/api/chat")
async def direct_chat(request: DirectChatRequest):
    """Direct chat endpoint for Telegram bot."""
    agent = get_chat_agent()
    session_mgr = get_session_manager()
    history = await session_mgr.get_history(request.session_id)

    response_text = await agent.respond(
        user_message=request.message,
        history=history,
        session_id=request.session_id,
    )

    await session_mgr.save_message(request.session_id, "user", request.message)
    await session_mgr.save_message(request.session_id, "assistant", response_text)

    return {"session_id": request.session_id, "reply": response_text}
