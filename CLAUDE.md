# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3dSt_Coder is a locally-hosted AI coding assistant designed for law firms, providing privacy-first code generation and development tools with multi-engine LLM support (Ollama, OpenAI API, vLLM) and a ReAct-pattern agent.

## Common Development Commands

### Windows Setup
```bash
# Install dependencies using Python 3.12
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings

# Set encoding for Windows
set PYTHONIOENCODING=utf-8
```

### Running Services

#### With Ollama (Recommended)
```bash
# Start with auto-detection of available models
/c/Python312/python.exe start_with_ollama.py

# Start with specific models
/c/Python312/python.exe start_mistral.py  # Uses mistral:7b-instruct-q4_K_M
/c/Python312/python.exe start_saul.py     # Uses adrienbrault/saul-instruct-v1:q4_k_m

# Pull Ollama models if needed
ollama pull codellama:7b          # 4GB - Great for coding
ollama pull deepseek-coder:6.7b   # 4GB - Excellent for code
ollama pull mistral:7b-instruct-q4_K_M
```

#### Direct API Server
```bash
# Set engine type and model
set LLM_ENGINE_TYPE=ollama
set LLM_MODEL_PATH=codellama:7b

# Start API server
/c/Python312/python.exe -m api.main
```

### Testing
```bash
# Run basic functionality tests
/c/Python312/python.exe run_tests.py

# Run pytest if available
/c/Python312/python.exe -m pytest llm_server/tests/ -v
/c/Python312/python.exe -m pytest agent/tests/ -v
/c/Python312/python.exe -m pytest api/tests/ -v
```

### Checking Ollama Status
```bash
# Check if Ollama server is running
curl http://localhost:11434/api/version

# List available models
ollama list
```

## Architecture

### LLM Engines

The project supports multiple LLM backends through a factory pattern (`llm_server/engine_factory.py`):

1. **OllamaEngine** (`ollama_engine.py`) - Ollama HTTP API integration for local models
2. **OpenAIEngine** (`openai_engine.py`) - OpenAI API compatible endpoints
3. **VLLMEngine** (`inference.py`) - vLLM CUDA-accelerated inference for MoE models
4. **MultiEngine** (`multi_engine.py`) - Task-based routing to different models

Engine selection is controlled by the `LLM_ENGINE_TYPE` environment variable.

### Agent System

The **CodingAgent** (`agent/core.py`) implements a ReAct (Reasoning + Acting) pattern:
- Tool invocation syntax: `{{tool:name:args}}`
- Tool result format: `{{result:content}}`
- Available tools:
  - `file_read` - Read file contents with path protection
  - `file_write` - Write/create files in project scope
  - `git_status`, `git_diff`, `git_commit` - Git operations
  - `shell` - Sandboxed command execution
  - `test_runner` - Execute project tests
- Conversation persistence via SQLite (`agent/memory.py`)
- Token-budget aware context management

### API Layer

FastAPI server (`api/main.py`, `api/router.py`) with OpenAPI spec:
- `/api/v1/chat` - SSE streaming chat endpoint
- `/api/v1/chat/complete` - Synchronous completion
- `/api/v1/health` - Service health with LLM status
- `/api/v1/conversations` - Conversation history management
- `/api/v1/tools` - Tool discovery endpoint
- `/static/` - Web chat interface (single-page app)

### Tool Security

All tools implement path validation and sandboxing:
- File operations restricted to project directory
- Shell commands execute in controlled environment
- Path traversal protection via `os.path.commonpath`
- Command filtering for dangerous operations

### Configuration

Environment variables:
```bash
LLM_ENGINE_TYPE=ollama|openai|vllm|multi
LLM_MODEL_PATH=model_name_or_path
LLM_HOST=127.0.0.1
LLM_PORT=8000
OPENAI_API_KEY=your_key  # For OpenAI engine
OPENAI_BASE_URL=endpoint  # Custom endpoint
PYTHONIOENCODING=utf-8  # Windows encoding fix
```

## Project Structure

```
├── llm_server/          # LLM engine implementations
│   ├── engine_factory.py  # Engine selection logic
│   ├── ollama_engine.py   # Ollama HTTP client
│   ├── openai_engine.py   # OpenAI API client
│   ├── inference.py       # vLLM CUDA engine
│   └── config.py         # Pydantic settings
├── agent/               # ReAct agent implementation
│   ├── core.py          # Main agent loop
│   ├── memory.py        # Conversation persistence
│   └── tools/           # Tool implementations
│       ├── file.py      # File I/O operations
│       ├── git.py       # Git integration
│       ├── shell.py     # Command execution
│       └── test_runner.py # Test execution
├── api/                 # FastAPI server
│   ├── main.py         # Application entry
│   └── router.py       # API endpoints
├── static/             # Web interface
│   └── index.html      # Single-page chat app
├── data/conversations/ # SQLite storage
├── start_*.py         # Model-specific launchers
└── run_tests.py       # Test suite runner
```

## Development Workflow

1. **Before making changes:** Read existing code to understand conventions
2. **When adding features:** Write tests first (TDD approach)
3. **Tool development:** Inherit from `BaseTool` protocol
4. **API changes:** Update OpenAPI schema automatically via FastAPI
5. **Testing:** Run `run_tests.py` for basic validation