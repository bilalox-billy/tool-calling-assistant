from dataclasses import dataclass
from src.adapters.base import ModelResponse, ToolCallRequest

@dataclass
class ScriptedTurn:
    tool_name: str | None = None
    arguments: dict | None = None
    content: str | None = None

    def to_response(self) -> ModelResponse:
        if self.tool_name:
            return ModelResponse(
                content=None,
                tool_calls=[ToolCallRequest(name=self.tool_name, arguments=self.arguments or {})]
            )
        
        return ModelResponse(
            content=self.content or "",
            tool_calls=[],
        )


class MockAdapter:
    """Scripted adapter for testing — no Ollama required."""

    def __init__(self, turns: list[ScriptedTurn]):
        self._turns = turns
        self._index = 0
        self.calls: list[tuple[str, dict]] = []


    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.calls.append((messages, tools))
        if self._index >= len(self._turns):
            return ModelResponse(content="Mock: no more scripted turns", tool_calls=[])
        turn = self._turns[self._index]
        self._index += 1
        return turn.to_response()

    def health_check(self) -> bool:
        return True




