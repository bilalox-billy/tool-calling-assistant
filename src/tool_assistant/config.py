from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppConfig(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    use_native_tool_calling: bool = True


    # Loop Control
    max_iterations: int = Field(default=10, ge=1, le=50)
    max_tool_calls_per_tool: int = Field(default=5, ge=1)
    max_total_tool_calls: int = Field(default=20, ge=1)
    tool_timeout_seconds: float = Field(default=10.0, ge=0.5)


    # Security
    allowed_tools: str = "calculate,get_date,search_docs"
    log_tool_calls: bool = True
    sanitize_tool_results: bool = True
    max_history_messages: int = Field(default=50, ge=5)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TOOL_ASSISTANT_",
        case_sensitive=False,
    )

    def get_allowed_tools_set(self)-> set[str]:
        return {t.strip() for t in self.allowed_tools.split(",") if t.strip()}


