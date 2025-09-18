# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3dSt is a comprehensive AI application platform designed for building intelligent systems with multi-engine LLM support (Ollama, OpenAI, vLLM), implementing a ReAct-pattern agent with parallel execution capabilities, intelligent task detection, file processing, and enterprise authentication.

## Common Development Commands

### Setup & Dependencies
```bash
# Install all dependencies (Windows Python 3.12)
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings "passlib[bcrypt]" "python-jose[cryptography]" python-multipart email-validator pytest pytest-asyncio psutil pyyaml

# Alternative: Poetry dependency management (if available)
poetry install
poetry install --extras vllm  # Include vLLM support

# Windows encoding fix (run before any Python commands)
set PYTHONIOENCODING=utf-8

# Create admin user (first time only)
/c/Python312/python.exe scripts/create_admin.py create

# List existing users
/c/Python312/python.exe scripts/create_admin.py list
```

### Running the Service
```bash
# Auto-detect and start with Ollama (recommended)
/c/Python312/python.exe start_with_ollama.py

# Start with specific Ollama models
/c/Python312/python.exe start_mistral.py    # mistral:7b-instruct-q4_K_M
/c/Python312/python.exe start_saul.py       # adrienbrault/saul-instruct-v1:q4_k_m

# Direct API server with environment variables
set LLM_ENGINE_TYPE=ollama
set LLM_MODEL_PATH=codellama:7b
/c/Python312/python.exe -m api.main

# Pull recommended Ollama models
ollama pull codellama:7b
ollama pull deepseek-coder:6.7b
ollama pull mistral:7b-instruct-q4_K_M
```

### Testing
```bash
# Quick functionality check (no pytest needed)
/c/Python312/python.exe run_tests.py

# Full test suite with comprehensive reporting
/c/Python312/python.exe test_all.py

# All tests with short traceback and limited failures
/c/Python312/python.exe -m pytest --tb=short --maxfail=3

# Module-specific tests
/c/Python312/python.exe -m pytest agent/tests/ -v
/c/Python312/python.exe -m pytest api/tests/ -v
/c/Python312/python.exe -m pytest auth/tests/ -v
/c/Python312/python.exe -m pytest llm_server/tests/ -v

# Parallel execution tests
/c/Python312/python.exe -m pytest agent/tests/test_parallel.py -v
/c/Python312/python.exe -m pytest agent/tests/test_parallel_integration.py -v
/c/Python312/python.exe -m pytest agent/tests/test_parallel_performance.py -v

# Run all parallel-related tests together
/c/Python312/python.exe -m pytest agent/tests/test_parallel*.py -v

# Single test execution
/c/Python312/python.exe -m pytest agent/tests/test_core.py::TestCodingAgent::test_concurrent_chat_sessions -v

# Quick test run with minimal output
/c/Python312/python.exe -m pytest --tb=short -q

# Test with coverage reporting
/c/Python312/python.exe -m pytest --cov=llm_server --cov=agent --cov=api --cov=auth --cov-report=html
```

### Code Quality
```bash
# Linting
/c/Python312/python.exe -m ruff check .
/c/Python312/python.exe -m ruff check . --fix

# Formatting
/c/Python312/python.exe -m black .

# Type checking
/c/Python312/python.exe -m mypy llm_server agent api auth
```

## Architecture

### Multi-Engine LLM Support

The `EngineFactory` (`llm_server/engine_factory.py`) creates engines based on `LLM_ENGINE_TYPE`:
- **ollama**: Local Ollama models via HTTP API
- **openai**: OpenAI-compatible endpoints
- **vllm**: CUDA-accelerated inference
- **multi**: Task-based routing between engines
- **transformers**: HuggingFace transformers (optional)
- **llama_cpp**: GGML/GGUF models (optional)

### ReAct Agent Pattern

`CodingAgent` (`agent/core.py`) uses tool invocation syntax:
- Single tool: `{{tool:name:args}}`
- Parallel execution: `{{parallel:[tool1, tool2]}}`
- Results: `{{result:content}}`

Memory persists via SQLite (`agent/memory.py`) with token-aware context management.

### Parallel Execution System

The `Orchestrator` (`agent/orchestrator.py`) and `SubAgent` (`agent/subagent.py`) enable:
- Task decomposition with dependency graphs
- Concurrent execution (default limit: 5)
- Performance metrics collection
- Specialized parallel tools (`agent/tools/parallel.py`)

### Hybrid Subagents

Configure specialized agents in `.claude/agents/` with YAML frontmatter:
```yaml
---
name: code-reviewer
tools: file_read, git_diff  # Optional tool restrictions
model: codellama            # Optional model override
---
```

Test hybrid subagent functionality:
```bash
# Demo the hybrid subagent system
/c/Python312/python.exe demo_hybrid_subagents.py

# Test subagent integration
/c/Python312/python.exe test_hybrid_subagents.py
/c/Python312/python.exe test_subagent_integration.py
```

### Authentication & Security

JWT-based auth (`auth/`) with:
- User roles and permissions (`auth/models.py`)
- Network access control (`auth/network.py`)
- Session management (`auth/database.py`)
- Path traversal protection in all file tools
- Command sandboxing in shell tool

### API Endpoints

FastAPI server (`api/main.py`):
- `/api/v1/chat` - SSE streaming chat (auth required)
- `/api/v1/health` - Service health check
- `/auth/login` - User authentication
- `/auth/status` - Auth & network status
- `/static/` - Web interface

## Environment Variables

```bash
# LLM Configuration
LLM_ENGINE_TYPE=ollama|openai|vllm|multi
LLM_MODEL_PATH=model_name
LLM_HOST=127.0.0.1
LLM_PORT=8000

# OpenAI (if using)
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=custom_endpoint

# Authentication
AUTH_SECRET_KEY=generate-unique-key
AUTH_TOKEN_EXPIRE_MINUTES=480
AUTH_REQUIRE_LOCAL_NETWORK=true
AUTH_ALLOWED_NETWORKS=10.0.0.0/8,192.168.0.0/16

# Windows
PYTHONIOENCODING=utf-8
```

## Key Development Patterns

### Adding Tools
Implement `BaseTool` protocol in `agent/tools/`:
```python
class NewTool:
    name = "tool_name"
    description = "Tool description"

    async def run(self, *args, **kwargs) -> str:
        # Implementation with path validation
        return result
```
Register in `get_available_tools()` (`agent/tools/__init__.py`)

### Adding LLM Engines
Extend `BaseLLMEngine` in `llm_server/`:
- Implement `generate()` and `generate_stream()`
- Register in `EngineFactory.create_engine()`
- Add config to `LLMConfig`

### Testing New Features
- Unit tests: Individual component logic
- Integration tests: Tool interactions
- E2E tests: Complete workflows
- Use pytest fixtures for test data
- Mock external dependencies

### Code Standards
- Python 3.12+ features (match, union types, walrus)
- Type hints on all functions
- Async/await for I/O operations
- Pydantic for validation
- Path validation for file operations

## Performance Testing

Performance benchmarks and optimization:
```bash
# Run performance tests
/c/Python312/python.exe test_performance.py

# Memory and resource usage analysis
/c/Python312/python.exe -m pytest agent/tests/test_parallel_performance.py -v
```

## Debugging and Development

```bash
# Check system dependencies
/c/Python312/python.exe -c "import fastapi, uvicorn, aiosqlite; print('Dependencies available')"

# Verify Python version
/c/Python312/python.exe --version

# Check Ollama connectivity
curl http://localhost:11434/api/tags

# Network configuration check
ipconfig
netsh advfirewall firewall show rule name="Python"
```