import httpx
import logging
from src.adapters.base import ModelResponse, ToolCallRequest


logger = logging.getLogger(__name__)

class OllamaAdapter:
    """
    Adapter for Ollama's /api/chat endpoint.
    Satisfies ModelAdapter Protocol without inheriting.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))

    
    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Ollama: {e}")
            raise ConnectionError("Ollama is not reachable")
        
        data = resp.json()
        msg = data.get("message", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})

            tool_calls.append(ToolCallRequest(
                name=fn.get("name"),
                arguments=fn.get("arguments", {}),
            ))
        
        return ModelResponse(
            content=msg.get("content", "") or None,
            tool_calls=tool_calls,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
    
    def health_check(self) -> bool:
        try:
            resp= self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._client.close()

        