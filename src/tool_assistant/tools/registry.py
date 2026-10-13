import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pydantic import ValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolCallLog:
    timestamp: str
    session_id: str
    tool_name: str
    arguments: dict
    result: str
    success: bool
    latency_ms: float
    iteration: int


class ProductionToolRegistry:
    """
    Full production registry:
    - Allowlisting
    - Validation
    - Execution with logging
    - Structured call log
    """

    def __init__(self, allowed_tools: set[str] | None = None, log_calls: bool = True):

        self._tools: dict = {}
        self.allowed_tools = allowed_tools or set()
        self.log_calls = log_calls
        self.call_logs: list[ToolCallLog] = []

    
    def register_tool(self, tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name!r}")
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def get(self, name: str):
        return self._tools.get(name)
    
    def get_schema(self)-> list[dict]:
        schemas = []
        for name, tool in self._tools.items():
            if self.allowed_tools is None or name in self.allowed_tools:
                schemas.append(tool.get_ollama_schema())
        return schemas
    
    def execute(
        self,
        tool_name: str,
        arguments: dict,
        session_id: str,
        iteration: int = 0,
    ) -> tuple[str, bool]:
        import time
        start = time.perf_counter()

        # Allowlist check
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            result = f"Error: Tool '{tool_name}' is not in the allowed tools list"
            self._log(session_id, tool_name, arguments, result, False, 0.0, iteration)
            return result, False
        

        # Tool Exist checks
        if tool_name not in self._tools:
            result = f"Error: Unknown tool '{tool_name}'. Available: {list(self._tools)}"
            self._log(session_id, tool_name, arguments, result, False, 0.0, iteration)
            return result, False
        
        tool = self._tools[tool_name]

        # Validation
        try:
            validated = tool.input_schema.model_validate(arguments)
        except ValidationError as e:
            errors = "; ".join(
                f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            result = f"Error: Invalid arguments for {tool_name}: {errors}"
            self._log(session_id, tool_name, arguments, result, False,
                      (time.perf_counter() - start) * 1000, iteration)
            return result, False
    
        # Execution
        try:
            result = tool.execute(validated)
            success = not str(result).startswith("Error:")
        except Exception as e:
            result = f"Error: Tool execution failed: {type(e).__name__}: {e}"
            success = False
        
        latelatency = (time.perf_counter() - start) * 1000
        self._log(session_id, tool_name, arguments, result, success, latency, iteration)
        return result, success

    def _log(self, session_id, tool_name, arguments, result, success, latency_ms, iteration):
        if not self.log_calls:
            return
        log_entry = ToolCallLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            result=str(result)[:500],
            success=success,
            latency_ms=round(latency_ms, 2),
            iteration=iteration,
        )
        self.call_log.append(log_entry)
        logger.info(
            f"[{session_id}][iter {iteration}] {tool_name} → "
            f"{'OK' if success else 'ERR'} ({latency_ms:.1f}ms)"
        )






        
