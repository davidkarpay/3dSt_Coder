# 🌐 Browser Setup: 3dSt_Coder + Ollama

Run 3dSt_Coder as a **web application** in your browser with Ollama as the backend!

## 🚀 Quick Start (5 minutes)

### 1. **Install Ollama** (if not already installed)
```bash
# Windows: Download from https://ollama.ai
# Or use chocolatey:
choco install ollama

# macOS:
brew install ollama

# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. **Start Ollama Server**
```bash
ollama serve
# Should start on http://localhost:11434
```

### 3. **Pull a Coding Model**
```bash
# Choose one (recommended: codellama:7b for coding tasks)
ollama pull codellama:7b         # 4GB - Great for coding
ollama pull deepseek-coder:6.7b  # 4GB - Excellent for code
ollama pull codegemma:7b         # 4GB - Google's coding model
ollama pull llama3:8b            # 5GB - General purpose
```

### 4. **Start 3dSt_Coder Web Interface**
```bash
cd F:\GitHub\localLLM

# Easy way - use the startup script:
/c/Python312/python.exe start_with_ollama.py

# Manual way - set environment and run:
export LLM_ENGINE_TYPE=ollama
export LLM_MODEL_PATH=codellama:7b
/c/Python312/python.exe -m api.main
```

### 5. **Open in Browser**
```
🌐 Web Interface: http://localhost:8000
📖 API Docs:      http://localhost:8000/docs
💓 Health Check:  http://localhost:8000/ping
```

## 🎯 **What You Get**

### ✅ **Full Web Interface**
- **Real-time streaming** responses
- **Tool execution** (file operations, git commands, shell, tests)
- **Code syntax highlighting**
- **Example prompts** for quick testing
- **Live status** indicators

### ✅ **Available Tools in Browser**
- 📁 **File Tools**: Read/write files in your project
- 🔄 **Git Tools**: Status, diff, commit operations
- 🐚 **Shell Tools**: Execute commands safely
- 🧪 **Test Tools**: Run pytest, unittest, etc.

### ✅ **Example Prompts to Try**
1. **"Check the git status of this repository"**
   - Uses `{{tool:git_status}}` automatically

2. **"Read the README.md file and summarize it"**
   - Uses `{{tool:file_read:path=README.md}}`

3. **"Write a Python function to calculate fibonacci numbers"**
   - Pure code generation (no tools needed)

4. **"Run the tests in this project"**
   - Uses `{{tool:test_runner}}` to execute tests

5. **"Create a backup script and save it to backup.sh"**
   - Code generation + `{{tool:file_write}}`

## 🔧 **Configuration Options**

### **Different Models**
```bash
# Switch models instantly:
export LLM_MODEL_PATH=deepseek-coder:6.7b
export LLM_MODEL_PATH=codegemma:7b
export LLM_MODEL_PATH=llama3:8b
```

### **Performance Tuning**
```bash
# Adjust generation parameters:
export LLM_TEMPERATURE=0.7    # Creativity (0.0-2.0)
export LLM_MAX_TOKENS=1024    # Response length
export LLM_TOP_P=0.95         # Nucleus sampling
```

### **Memory Management**
```bash
# Conversation memory settings:
export LLM_MAX_CONTEXT=4096   # Context window
```

## 🔒 **Security Features**

### ✅ **Built-in Safety**
- **Path traversal protection**: Can't access files outside project
- **Command filtering**: Dangerous shell commands blocked
- **Resource limits**: File size and execution time limits
- **Sandboxed execution**: Tools run in controlled environment

### ✅ **Local-First**
- **No data leaves your machine**: Everything runs locally
- **No API keys required**: Uses your local Ollama server
- **Offline capable**: Works without internet connection

## 🚨 **Troubleshooting**

### **Ollama Not Found**
```bash
# Check if Ollama is running:
curl http://localhost:11434/api/version

# If not, start it:
ollama serve
```

### **Model Not Available**
```bash
# List available models:
ollama list

# Pull a model if needed:
ollama pull codellama:7b
```

### **3dSt_Coder Won't Start**
```bash
# Check Python dependencies:
/c/Python312/python.exe -m pip install fastapi uvicorn aiosqlite requests

# Check if port 8000 is available:
netstat -an | findstr :8000
```

### **Tools Not Working**
```bash
# Verify git is available:
git --version

# Verify Python is available:
python --version

# Check project permissions
```

## 🎉 **Success! You Now Have:**

✅ **Local AI coding assistant** running in your browser
✅ **Real-time streaming** responses with tool execution
✅ **Complete privacy** - nothing leaves your machine
✅ **Production-ready** ReAct agent with memory
✅ **Plug-and-play** model swapping

**Perfect for**: Code generation, debugging, documentation, git operations, testing, and general coding assistance - all running locally with your own models! 🚀