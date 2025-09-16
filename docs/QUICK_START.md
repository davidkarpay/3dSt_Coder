# 🚀 Quick Start Guide

Get 3dSt_Coder running in **5 minutes** with this step-by-step guide.

## Prerequisites

✅ **Windows machine** with Python 3.12
✅ **4GB+ RAM** for model inference
✅ **10GB+ free disk space** for models

## Step 1: Install Ollama (2 minutes)

Ollama provides the local LLM backend for 3dSt_Coder.

### Windows Installation:
```bash
# Option A: Download installer
# Visit https://ollama.ai and download the Windows installer

# Option B: Use Chocolatey (if installed)
choco install ollama
```

### Verify Installation:
```bash
ollama --version
# Should show: ollama version 0.x.x
```

### Start Ollama Server:
```bash
ollama serve
# Keep this terminal open - server runs on http://localhost:11434
```

## Step 2: Install Python Dependencies (1 minute)

Open a **new terminal** and install the required packages:

```bash
# Navigate to the project directory
cd F:\GitHub\localLLM

# Install dependencies
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings

# Set encoding for Windows
set PYTHONIOENCODING=utf-8
```

**✅ Verification:** All packages install without errors.

## Step 3: Download a Model (1 minute)

Download a coding-focused model (this will take a moment):

```bash
# Choose one model (codellama recommended for coding):
ollama pull codellama:7b         # 4GB - Great for coding
ollama pull deepseek-coder:6.7b  # 4GB - Excellent for code
ollama pull codegemma:7b         # 4GB - Google's coding model
```

**✅ Verification:** Check available models:
```bash
ollama list
# Should show your downloaded model
```

## Step 4: Launch 3dSt_Coder (1 minute)

Start the 3dSt_Coder web interface:

```bash
cd F:\GitHub\localLLM

# Auto-detect and start with available model
/c/Python312/python.exe start_with_ollama.py
```

**✅ Expected Output:**
```
3dSt_Coder + Ollama Startup
========================================
[OK] Ollama server is running
[OK] Model 'codellama:7b' is available
[START] Starting 3dSt_Coder with model: codellama:7b
[WEB] Web interface will be available at: http://localhost:8000
```

## Step 5: Open Web Interface (30 seconds)

1. **Open your browser**
2. **Navigate to:** `http://localhost:8000`
3. **Test with a simple prompt:** "Hello, can you help me with coding?"

**🎉 Success!** You should see:
- Dark-themed chat interface
- Streaming AI responses
- Tool execution capabilities

## 🧪 Test Drive - Try These Prompts

**File Operations:**
```
"Read the README.md file and summarize it"
```

**Git Integration:**
```
"Check the git status of this repository"
```

**Code Generation:**
```
"Write a Python function to calculate the fibonacci sequence"
```

**Testing:**
```
"Run the project tests and show me the results"
```

## ⚡ What's Next?

**Explore More Features:**
- 📖 **[User Guide](USER_GUIDE.md)** - Complete usage manual
- 🔧 **[Setup Options](SETUP.md)** - Configure different models and backends
- 🚨 **[Troubleshooting](TROUBLESHOOTING.md)** - Fix common issues

**Different Models:**
Try switching models for different tasks:
```bash
# Stop current server (Ctrl+C)
export LLM_MODEL_PATH=deepseek-coder:6.7b
/c/Python312/python.exe start_with_ollama.py
```

## 🚨 Quick Troubleshooting

**Server won't start?**
- Check if port 8000 is free: `netstat -an | findstr :8000`
- Verify Ollama is running: `curl http://localhost:11434/api/version`

**Model not found?**
- List available models: `ollama list`
- Pull missing model: `ollama pull codellama:7b`

**Dependencies missing?**
- Re-run pip install with the full command from Step 2

---

**🎉 You're all set!** 3dSt_Coder is now running locally and privately on your machine.