# src/tool_assistant/orchestration/loop.py
"""
Production agent loop with all safety features.
"""
import logging
from dataclasses import dataclass, field
from ..adapters.base import ModelAdapter
from ..tools.registry import ProductionToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    session_id: str
    final_answer: str
    tool_calls_count: int = 0
    iterations: int = 0
    termination_reason: str = ""   # answer | max_iterations | error | loop_detected
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)


class ProductionAgentLoop:
    """
    Full production agent loop:
    - Bounded iterations
    - Loop detection (same args repeated)
    - Structured logging
    - Graceful degradation
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant with access to tools. "
        "Use tools when you need to calculate, look up dates, or search documents. "
        "Always use the calculator for arithmetic — do not compute mentally. "
        "After receiving tool results, synthesize a clear answer for the user."
    )

    def __init__(
        self,
        adapter: ModelAdapter,
        registry: ProductionToolRegistry,
        max_iterations: int = 10,
        max_repeat_tool_calls: int = 3,
    ):
        self.adapter = adapter
        self.registry = registry
        self.max_iterations = max_iterations
        self.max_repeat_tool_calls = max_repeat_tool_calls

    def run(self, user_message: str, session_id: str) -> AgentResult:
        result = AgentResult(session_id=session_id, final_answer="")
        history = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        recent_calls: list[str] = []   # for loop detection

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1

            # Call model
            try:
                model_response = self.adapter.chat(history, self.registry.get_schemas())
            except ConnectionError as e:
                result.final_answer = f"Error: Cannot reach model service: {e}"
                result.termination_reason = "error"
                return result

            result.prompt_tokens += model_response.prompt_tokens
            result.completion_tokens += model_response.completion_tokens

            if model_response.is_final_answer:
                result.final_answer = model_response.content
                result.termination_reason = "answer"
                return result

            if model_response.has_tool_call:
                tc = model_response.tool_calls[0]

                # Loop detection
                call_sig = f"{tc.name}:{str(sorted(tc.arguments.items()))}"
                if recent_calls.count(call_sig) >= self.max_repeat_tool_calls:
                    logger.warning(f"[{session_id}] Loop detected: {tc.name} with same args")
                    result.errors.append(f"Loop: {tc.name} called repeatedly")
                    result.termination_reason = "loop_detected"
                    result.final_answer = (
                        "I got stuck in a loop. I was repeatedly calling the same tool "
                        "without making progress. Please rephrase your request."
                    )
                    return result

                recent_calls.append(call_sig)

                # Execute tool
                tool_result, success = self.registry.execute(
                    tc.name, tc.arguments,
                    session_id=session_id,
                    iteration=iteration + 1
                )
                result.tool_calls_count += 1

                # Update history
                history.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": tc.name, "arguments": tc.arguments}}],
                })
                history.append({"role": "tool", "content": tool_result})

            else:
                # No tool call, no content
                logger.warning(f"[{session_id}] Empty response at iteration {iteration + 1}")
                result.errors.append(f"Empty response at iteration {iteration + 1}")

        result.final_answer = (
            f"I could not complete the task within {self.max_iterations} steps. "
            f"I made {result.tool_calls_count} tool calls."
        )
        result.termination_reason = "max_iterations"
        return result
