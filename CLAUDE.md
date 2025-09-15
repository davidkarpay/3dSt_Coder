# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalLLM is a locally-hosted coding agent that wraps an open-source LLM with Mixture-of-Experts (MoE) backbone and developer-oriented tools. The system is designed using Test-Driven Development (TDD), Clean Code principles, and SOLID patterns.

## Technology Stack

- **LLM Inference**: vLLM (CUDA-accelerated) with DeepSeek-MoE-Coder or similar MoE models
- **Agent Orchestration**: LangChain-Core for tool-calling and ReAct patterns
- **Backend API**: FastAPI (Python 3.12) with async support
- **Desktop GUI**: Tauri (Rust + React/TypeScript)
- **Testing**: pytest with pytest-asyncio for async tests
- **CI/CD**: GitHub Actions with Poetry for dependency management
- **Containerization**: Docker with multi-stage builds

## Common Development Commands

### Environment Setup
```bash
# Install dependencies with Poetry
poetry install --with dev

# Create and activate virtual environment (if not using Poetry)
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate    # Unix/macOS
```

### Running Services
```bash
# Start LLM server (replace with actual model path)
poetry run python -m llm_server.main --model /models/deepseek-moe-coder

# Start API server
poetry run uvicorn api.main:app --reload

# Start UI development server
cd ui
npm install
npm run tauri dev
```

### Testing
```bash
# Run all tests with coverage
poetry run pytest -n auto --cov=.

# Run specific test file
poetry run pytest agent/tests/test_core.py

# Run tests with verbose output
poetry run pytest -v

# Run async tests
poetry run pytest -m asyncio
```

### Code Quality
```bash
# Lint code with ruff
poetry run ruff check .

# Format code (if configured)
poetry run black .
```

### Docker Operations
```bash
# Build Docker image
docker build -t coding-agent:latest .

# Run container
docker run -p 8000:8000 coding-agent:latest
```

## Project Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Core Components

1. **LLM Server** (`llm_server/`)
   - `inference.py`: VLLMEngine wrapper for token streaming
   - `config.py`: Pydantic settings for model configuration
   - `main.py`: FastAPI entry point for LLM service

2. **Agent System** (`agent/`)
   - `core.py`: ReAct loop implementation
   - `memory.py`: LangChain memory wrapper for conversation persistence
   - `tools/`: Tool implementations (file, git, shell, test_runner)
   - Each tool follows BaseTool protocol with async run() method

3. **API Layer** (`api/`)
   - `router.py`: FastAPI endpoints for /chat, /tools, /history
   - `main.py`: Uvicorn entry point
   - Supports streaming responses via EventSourceResponse

4. **UI** (`ui/`)
   - Tauri-based desktop application
   - React/TypeScript frontend in `src/`
   - Rust backend bridge in `src-tauri/`

## Development Workflow

The project strictly follows Test-Driven Development:

1. Write failing test first (red phase)
2. Implement minimal code to pass test (green phase)
3. Refactor while keeping tests green
4. Ensure ≥90% code coverage before committing

All tools are sandboxed - shell/Python execution happens inside Docker containers with read-only mounts and resource limits.

## Key Design Patterns

- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Components receive dependencies via constructors
- **Protocol-based Interfaces**: Tools implement BaseTool protocol
- **Async-first**: All I/O operations are async using asyncio
- **Stream Processing**: LLM responses stream tokens in real-time

## Testing Strategy

- Unit tests for individual components
- Integration tests for tool interactions
- End-to-end tests for full agent workflows
- Mock LLM responses for deterministic testing
- Cypress tests for UI components