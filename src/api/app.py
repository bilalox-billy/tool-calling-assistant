"""
FastAPI application — Tool-Calling Assistant API
"""
import uuid
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.adapters.ollama import OllamaAdapter
from src.tool_assistant.config import AppConfig
from src.tool_assistant.tools.registry import ProductionToolRegistry
from src.tool_assistant.tools.tools import CalculatorTool, WeatherTool, DateTool
from src.tool_assistant.orchestration.loop import ProductionAgentLoop, AgentResult

logger = logging.getLogger(__name__)

# ── App state (initialised once at startup) ───────────────────────────────────

class AppState:
    config: AppConfig
    adapter: OllamaAdapter
    registry: ProductionToolRegistry


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy objects once; tear them down on shutdown."""
    state.config = AppConfig()

    state.adapter = OllamaAdapter(
        base_url=state.config.ollama_base_url,
        model=state.config.ollama_model,
    )

    state.registry = ProductionToolRegistry(
        allowed_tools=state.config.get_allowed_tools_set(),
        log_calls=state.config.log_tool_calls,
    )
    for tool in [CalculatorTool(), WeatherTool(), DateTool()]:
        state.registry.register_tool(tool)

    logger.info("App started — model: %s  tools: %s",
                state.config.ollama_model,
                list(state.registry._tools))
    yield

    # Shutdown cleanup
    state.adapter._client.close()
    logger.info("App shutdown — HTTP client closed")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tool-Calling Assistant",
    version="0.1.0",
    description="Agent loop powered by Ollama with calculator, weather, and date tools.",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="User message")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional session ID; auto-generated if omitted",
    )
    max_iterations: int = Field(default=10, ge=1, le=50)


class ChatResponse(BaseModel):
    session_id: str
    final_answer: str
    tool_calls_count: int
    iterations: int
    termination_reason: str
    prompt_tokens: int
    completion_tokens: int
    errors: list[str]


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    ollama_reachable: bool
    model: str
    tools_registered: list[str]


class ToolInfo(BaseModel):
    name: str
    schema: dict


class ToolLogEntry(BaseModel):
    timestamp: str
    session_id: str
    tool_name: str
    arguments: dict
    result: str
    success: bool
    latency_ms: float
    iteration: int


# ── Dependency ────────────────────────────────────────────────────────────────

def get_loop() -> ProductionAgentLoop:
    return ProductionAgentLoop(
        adapter=state.adapter,
        registry=state.registry,
        max_iterations=state.config.max_iterations,
        max_repeat_tool_calls=state.config.max_tool_calls_per_tool,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check whether the API and Ollama backend are reachable."""
    reachable = await run_in_threadpool(state.adapter.health_check)
    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama_reachable=reachable,
        model=state.config.ollama_model,
        tools_registered=list(state.registry._tools),
    )


@app.get("/tools", response_model=list[ToolInfo], tags=["Tools"])
async def list_tools():
    """Return all registered tools and their Ollama-compatible schemas."""
    return [
        ToolInfo(name=name, schema=tool.get_ollama_schema())
        for name, tool in state.registry._tools.items()
    ]


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(
    req: ChatRequest,
    loop: Annotated[ProductionAgentLoop, Depends(get_loop)],
):
    """
    Send a message to the agent.
    The loop runs tool calls automatically and returns a final answer.
    """
    # Run the synchronous agent loop in a thread pool so it doesn't block
    result: AgentResult = await run_in_threadpool(
        loop.run,
        user_message=req.message,
        session_id=req.session_id,
    )
    return ChatResponse(
        session_id=result.session_id,
        final_answer=result.final_answer,
        tool_calls_count=result.tool_calls_count,
        iterations=result.iterations,
        termination_reason=result.termination_reason,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        errors=result.errors,
    )


@app.get("/sessions/{session_id}/logs", response_model=list[ToolLogEntry], tags=["Logs"])
async def session_logs(session_id: str):
    """Return all tool-call logs for a given session ID."""
    logs = [
        ToolLogEntry(**asdict(entry))
        for entry in state.registry.call_logs
        if entry.session_id == session_id
    ]
    if not logs:
        raise HTTPException(status_code=404, detail=f"No logs found for session '{session_id}'")
    return logs


@app.get("/logs", response_model=list[ToolLogEntry], tags=["Logs"])
async def all_logs():
    """Return all tool-call logs across every session (useful for debugging)."""
    return [ToolLogEntry(**asdict(entry)) for entry in state.registry.call_logs]
