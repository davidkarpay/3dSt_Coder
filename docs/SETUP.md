# 🔧 Setup & Installation Guide

Comprehensive installation and configuration guide for all deployment scenarios.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Backend Configuration](#backend-configuration)
- [Model Setup](#model-setup)
- [Environment Configuration](#environment-configuration)
- [Verification](#verification)
- [Performance Tuning](#performance-tuning)

## System Requirements

### Minimum Requirements
- **OS:** Windows 10+, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python:** 3.12 or higher
- **RAM:** 4GB available (8GB recommended)
- **Storage:** 10GB free space (for models and data)
- **Network:** Internet connection for initial model download

### Recommended Hardware
- **CPU:** 8+ cores for faster inference
- **RAM:** 16GB+ for multiple models
- **Storage:** SSD with 50GB+ free space
- **GPU:** NVIDIA GPU with 8GB+ VRAM (for vLLM backend)

## Installation Methods

### Method 1: Ollama Backend (Recommended)

**Best for:** Most users, easiest setup, great model selection

#### Step 1: Install Ollama

**Windows:**
```bash
# Download installer from https://ollama.ai
# Or use Chocolatey:
choco install ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Step 2: Install Python Dependencies
```bash
cd F:\GitHub\localLLM
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings
set PYTHONIOENCODING=utf-8
```

#### Step 3: Start Ollama Service
```bash
ollama serve
# Keep running in background
```

#### Step 4: Launch 3dSt_Coder
```bash
/c/Python312/python.exe start_with_ollama.py
```

### Method 2: OpenAI API Backend

**Best for:** Users with OpenAI API access, no local compute needed

#### Step 1: Get API Key
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Save it securely

#### Step 2: Install Dependencies
```bash
cd F:\GitHub\localLLM
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings
```

#### Step 3: Configure API Key
```bash
set OPENAI_API_KEY=your_api_key_here
set LLM_ENGINE_TYPE=openai
set LLM_MODEL_PATH=gpt-4
```

#### Step 4: Start Server
```bash
/c/Python312/python.exe -m api.main
```

### Method 3: vLLM Backend (Advanced)

**Best for:** Users with powerful GPUs, maximum performance

#### Step 1: Install CUDA
1. Install NVIDIA drivers (latest)
2. Install CUDA Toolkit 12.1+
3. Install cuDNN

#### Step 2: Install vLLM
```bash
pip install vllm
# May take some time to compile CUDA kernels
```

#### Step 3: Download Model
```bash
# Download a compatible model (e.g., CodeLlama)
huggingface-cli download codellama/CodeLlama-7b-Instruct-hf --local-dir models/codellama-7b
```

#### Step 4: Configure and Start
```bash
set LLM_ENGINE_TYPE=vllm
set LLM_MODEL_PATH=models/codellama-7b
/c/Python312/python.exe -m api.main
```

## Backend Configuration

### Ollama Configuration

#### Available Models
```bash
# List all available models
ollama list

# Popular coding models:
ollama pull codellama:7b         # 4GB - Meta's code model
ollama pull deepseek-coder:6.7b  # 4GB - Excellent coding performance
ollama pull codegemma:7b         # 4GB - Google's code model
ollama pull llama3:8b            # 5GB - General purpose
ollama pull mistral:7b           # 4GB - Efficient and capable
```

#### Model Performance Comparison
| Model | Size | Strengths | Best For |
|-------|------|-----------|----------|
| codellama:7b | 4GB | Code completion, debugging | Python, JS, general coding |
| deepseek-coder:6.7b | 4GB | Code understanding, refactoring | Complex codebases |
| codegemma:7b | 4GB | Code generation, explanation | Documentation, tutorials |
| llama3:8b | 5GB | General knowledge, reasoning | Mixed coding & text tasks |

#### Switch Models
```bash
# Stop current server (Ctrl+C)
export LLM_MODEL_PATH=deepseek-coder:6.7b
/c/Python312/python.exe start_with_ollama.py
```

### OpenAI Configuration

#### Supported Models
- **GPT-4** - Best quality, slower, more expensive
- **GPT-3.5-turbo** - Fast, good quality, cost-effective
- **Custom endpoints** - Azure OpenAI, other compatible APIs

```bash
# Use different model
set LLM_MODEL_PATH=gpt-3.5-turbo

# Custom endpoint (e.g., Azure OpenAI)
set OPENAI_BASE_URL=https://your-resource.openai.azure.com/
set OPENAI_API_VERSION=2024-02-01
```

### vLLM Configuration

#### Memory Management
```bash
# GPU memory fraction (0.0 to 1.0)
export VLLM_GPU_MEMORY_UTILIZATION=0.9

# Maximum model length
export VLLM_MAX_MODEL_LEN=4096

# Number of GPUs
export VLLM_TENSOR_PARALLEL_SIZE=1
```

## Model Setup

### Choosing the Right Model

#### For Code Generation:
- **codellama:7b** - Excellent Python, JavaScript, general coding
- **deepseek-coder:6.7b** - Superior code understanding and refactoring

#### For Legal Work:
- **llama3:8b** - Better reasoning for legal analysis
- **mixtral:8x7b** - Advanced reasoning (requires more RAM)

#### For General Use:
- **mistral:7b** - Good balance of capability and efficiency

### Model Storage

#### Ollama Models Location:
- **Windows:** `C:\Users\{user}\.ollama\models`
- **macOS:** `~/.ollama/models`
- **Linux:** `/usr/share/ollama/.ollama/models`

#### Disk Space Management:
```bash
# Check model sizes
ollama list

# Remove unused models
ollama rm old-model:tag

# Clean up space
ollama prune
```

## Environment Configuration

### Environment Variables

Create a `.env` file or set these variables:

```bash
# Core Configuration
LLM_ENGINE_TYPE=ollama          # ollama|openai|vllm|multi
LLM_MODEL_PATH=codellama:7b     # Model identifier
LLM_HOST=127.0.0.1             # Server host
LLM_PORT=8000                  # Server port

# Generation Parameters
LLM_TEMPERATURE=0.7            # Creativity (0.0-2.0)
LLM_MAX_TOKENS=2048           # Max response length
LLM_TOP_P=0.95                # Nucleus sampling
LLM_MAX_CONTEXT=4096          # Context window size

# OpenAI Specific
OPENAI_API_KEY=your_key_here   # API key
OPENAI_BASE_URL=custom_url     # Custom endpoint (optional)

# Performance
PYTHONIOENCODING=utf-8         # Windows encoding fix
```

### Project Configuration

#### Data Directories
3dSt_Coder automatically creates:
- `data/conversations/` - Chat history storage
- `logs/` - Application logs

#### Conversation Isolation
Each project gets its own conversation history:
```bash
# Different projects maintain separate contexts
http://localhost:8000?project=project-alpha
http://localhost:8000?project=project-beta
```

## Verification

### Health Checks

#### Test Basic Functionality:
```bash
# Check if server is running
curl http://localhost:8000/ping

# Test health endpoint
curl http://localhost:8000/api/v1/health

# Verify tools are loaded
curl http://localhost:8000/api/v1/tools
```

#### Run Test Suite:
```bash
# Basic functionality tests
/c/Python312/python.exe run_tests.py

# Full test suite (if pytest available)
/c/Python312/python.exe -m pytest llm_server/tests/ -v
/c/Python312/python.exe -m pytest agent/tests/ -v
/c/Python312/python.exe -m pytest api/tests/ -v
```

### Performance Testing

#### Measure Response Time:
```bash
# Time a simple request
time curl -X POST http://localhost:8000/api/v1/chat/complete \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, write a simple Python function"}'
```

#### Monitor Resource Usage:
```bash
# Check memory usage
tasklist | findstr python  # Windows
ps aux | grep python       # Linux/macOS

# Monitor GPU usage (if using vLLM)
nvidia-smi -l 1
```

## Performance Tuning

### Ollama Optimization

#### Memory Management:
```bash
# Set Ollama memory usage
set OLLAMA_NUM_PARALLEL=1      # Concurrent requests
set OLLAMA_MAX_LOADED_MODELS=1 # Models in memory
```

#### Model Quantization:
```bash
# Use quantized models for better performance
ollama pull codellama:7b-code-q4_K_M  # 4-bit quantization
ollama pull codellama:7b-code-q8_0     # 8-bit quantization
```

### System Optimization

#### Windows Specific:
```bash
# Increase virtual memory
# Set environment variables in System Properties

# Use Windows Performance Toolkit for profiling
```

#### Hardware Monitoring:
```bash
# Monitor CPU/memory usage
perfmon  # Windows Performance Monitor

# Check disk I/O
resmon   # Resource Monitor
```

### Scaling Considerations

#### Multiple Models:
```bash
# Use multi-engine for task-specific models
set LLM_ENGINE_TYPE=multi

# Configure model routing in llm_server/multi_engine.py
```

#### Load Balancing:
```bash
# Run multiple instances on different ports
LLM_PORT=8001 python -m api.main &
LLM_PORT=8002 python -m api.main &
```

---

## 🆘 Next Steps

- **✅ Installation Complete?** → Try the [Quick Start Guide](QUICK_START.md)
- **❓ Having Issues?** → Check [Troubleshooting](TROUBLESHOOTING.md)
- **📖 Ready to Use?** → Read the [User Guide](USER_GUIDE.md)