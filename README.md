# Tool-Calling Assistant

A production-ready AI agent that uses **Ollama** as its LLM backend and automatically calls tools to answer user questions. Exposed as a **FastAPI** REST API and packaged in Docker.

---

## How it works

```
User message
     │
     ▼
ProductionAgentLoop
     │
     ├── OllamaAdapter  ──►  Ollama (qwen2.5:7b)
     │         ◄── tool_calls / final answer ──
     │
     └── ProductionToolRegistry
              ├── CalculatorTool   (arithmetic expressions)
              ├── WeatherTool      (simulated weather data)
              └── DateTool         (date operations)
```

The agent loop:
1. Sends the user message + tool schemas to the model
2. If the model returns a tool call → validates, executes, appends the result to history, repeats
3. If the model returns a final answer → returns it
4. Safety guards: **max iterations**, **loop detection** (same tool + same args repeated), **allowlist**

---

## Project structure

```
tool-calling-assistant/
├── src/
│   ├── adapters/
│   │   ├── base.py          # ModelAdapter protocol, ModelResponse, ToolCallRequest
│   │   ├── ollama.py        # OllamaAdapter (httpx → /api/chat)
│   │   └── mock.py          # MockAdapter for tests (scripted turns)
│   ├── tool_assistant/
│   │   ├── config.py        # AppConfig (pydantic-settings, env prefix TOOL_ASSISTANT_)
│   │   ├── tools/
│   │   │   ├── tools.py     # CalculatorTool, WeatherTool, DateTool
│   │   │   └── registry.py  # ProductionToolRegistry (allowlist, validation, logging)
│   │   └── orchestration/
│   │       └── loop.py      # ProductionAgentLoop
│   └── api/
│       └── app.py           # FastAPI app (lifespan, routes)
├── notebooks/
│   └── test.ipynb           # Interactive exploration / manual tests
├── main.py                  # Uvicorn entrypoint
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── .env
```

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally with the model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```
- Docker Desktop (optional, for containerised runs)

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo-url>
cd tool-calling-assistant

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

### 2. Configure environment

```bash
cp .env .env.local   # optional — edit as needed
```

`.env` file (all vars use the `TOOL_ASSISTANT_` prefix):

```env
TOOL_ASSISTANT_OLLAMA_BASE_URL=http://localhost:11434
TOOL_ASSISTANT_OLLAMA_MODEL=qwen2.5:7b
```

### 3. Run the server

```bash
python main.py
```

Server starts at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Ollama reachability + registered tools |
| `GET` | `/tools` | All tool schemas (Ollama format) |
| `POST` | `/chat` | Send a message; agent handles tool calls automatically |
| `GET` | `/sessions/{id}/logs` | Tool-call logs for a specific session |
| `GET` | `/logs` | All tool-call logs |

### `POST /chat` — request body

```json
{
  "message": "What is 123 * 456 and what is the weather in Paris?",
  "session_id": "optional-custom-id",
  "max_iterations": 10
}
```

### `POST /chat` — response

```json
{
  "session_id": "abc-123",
  "final_answer": "123 × 456 = 56088. In Paris it is 22°C and partly cloudy.",
  "tool_calls_count": 2,
  "iterations": 3,
  "termination_reason": "answer",
  "prompt_tokens": 412,
  "completion_tokens": 38,
  "errors": []
}
```

`termination_reason` is one of: `answer` · `max_iterations` · `loop_detected` · `error`

---

## Docker

### Build

```bash
docker build -t tool-calling-assistant .
```

### Run

```bash
# host.docker.internal resolves to your PC from inside the container
docker run --rm -p 8000:8000 --env-file .env \
  -e TOOL_ASSISTANT_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  tool-calling-assistant
```

---

## Configuration reference

All variables are prefixed with `TOOL_ASSISTANT_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model to use |
| `MAX_ITERATIONS` | `10` | Max agent loop iterations |
| `MAX_TOOL_CALLS_PER_TOOL` | `5` | Loop detection threshold |
| `ALLOWED_TOOLS` | `calculate,get_date,get_weather` | Comma-separated allowlist |
| `LOG_TOOL_CALLS` | `true` | Write tool-call logs |

---

## Running tests

```bash
pytest
```

Tests use `MockAdapter` (scripted turns) — no live Ollama required.

---

## Available tools

| Tool | Name | What it does |
|------|------|-------------|
| `CalculatorTool` | `calculate` | Evaluates safe arithmetic expressions (`12 * 8`, `(3+4)/2`) |
| `WeatherTool` | `get_weather` | Returns simulated weather for a city (celsius / fahrenheit) |
| `DateTool` | `get_date` | Today's date, days until/since a date, day of week |

---

## Swapping the model provider

`OllamaAdapter` implements the `ModelAdapter` protocol defined in `src/adapters/base.py`. To use a different provider (OpenAI, Anthropic, etc.) create a new adapter that implements:

```python
def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...
def health_check(self) -> bool: ...
```

Then pass it to `ProductionAgentLoop` — no other changes needed.
