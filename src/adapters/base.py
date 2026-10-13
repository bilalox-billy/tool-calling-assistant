from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCallRequest:
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def has_tool_calls(self)-> bool:
        return bool(self.tool_calls)
    
    @property
    def is_final_answer(self) -> bool:
        return bool(self.content) and not self.tool_calls


class ModelAdapter(Protocol):
    """Swap Ollama for another model provider"""
    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...
    def health_check(self) -> bool: ...

    

