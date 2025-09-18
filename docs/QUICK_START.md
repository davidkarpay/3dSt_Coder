# 🚀 Quick Start Guide

Get 3dSt Platform running in **5 minutes** with this step-by-step guide.

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

### Check if Ollama is Already Running:
```bash
# Test if Ollama server is running
curl http://localhost:11434/api/version
# If this returns JSON with version info, Ollama is already running - skip to Step 2
```

### Start Ollama Server (if not running):
```bash
ollama serve
# Keep this terminal open - server runs on http://localhost:11434
# If you get "bind: Only one usage..." error, Ollama is already running - continue to Step 2
```

## Step 2: Install Python Dependencies (1 minute)

Open a **new terminal** and install the required packages:

```bash
# Navigate to the project directory
cd F:\GitHub\localLLM

# Install dependencies
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings "passlib[bcrypt]" "python-jose[cryptography]" python-multipart email-validator

# Set encoding for Windows
set PYTHONIOENCODING=utf-8
```

**✅ Verification:** All packages install without errors.

**Test installation:**
```bash
/c/Python312/python.exe -c "import fastapi, uvicorn, aiosqlite; print('Dependencies installed successfully')"
```

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
# Should show your downloaded model with size info (e.g., "codellama:7b    8fdf8f752f6e    3.8 GB")

# Test model availability
curl -X POST http://localhost:11434/api/show -d '{"name":"codellama:7b"}' -H "Content-Type: application/json"
# Should return model information JSON (not an error)
```

## Step 4: Create Admin User (1 minute)

Create your administrator account for secure access:

```bash
cd F:\GitHub\localLLM

# Create admin user interactively
/c/Python312/python.exe scripts/create_admin.py create
```

**✅ Follow the prompts:**
- Enter a **username** (3-50 characters)
- Enter an **email** (optional but recommended)
- Create a **strong password** (8+ characters with uppercase, lowercase, numbers, symbols)
- Confirm your password

**✅ Expected Output:**
```
3dSt_Coder Admin User Creation
========================================
Enter admin username (3-50 characters): admin
Enter admin email (optional): admin@yourfirm.com
Enter admin password: [hidden]
Confirm admin password: [hidden]
✅ Password strength: strong

✅ Admin user 'admin' created successfully!
```

**✅ Verification:** List created users:
```bash
/c/Python312/python.exe scripts/create_admin.py list
# Should show your admin user with "👑" icon
```

## Step 5: Launch 3dSt_Coder (1 minute)

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

**✅ Verification:** Test the server is running:
```bash
# Open another terminal and test the API
curl http://localhost:8000/ping
# Should return: {"status": "pong", "message": "Coding agent API is running"}
```

## Step 6: Login and Test (1 minute)

1. **Open your browser**
2. **Navigate to:** `http://localhost:8000`
3. **Login with your admin credentials:**
   - Enter your username and password
   - Click "Login"
4. **Test with a simple prompt:** "Hello, can you help me with coding?"

**🎉 Success!** You should see:
- ✅ Network access validation (green checkmark)
- 🔐 Secure login interface
- 💬 Dark-themed authenticated chat interface
- ⚡ Streaming AI responses with tool execution
- 👤 User session info displayed

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
set LLM_MODEL_PATH=deepseek-coder:6.7b
/c/Python312/python.exe start_with_ollama.py
```

## 🚨 Quick Troubleshooting

**Ollama "bind" error when running `ollama serve`?**
- **Error message:** `"Error: listen tcp 127.0.0.1:11434: bind: Only one usage..."`
- **Solution:** Ollama is already running! This is normal - just continue to Step 2
- **Verify:** Run `curl http://localhost:11434/api/version` to confirm it's working

**Server won't start?**
- Check if port 8000 is free: `netstat -an | findstr :8000`
- Verify Ollama is running: `curl http://localhost:11434/api/version`
- Kill existing server: Find process ID with `netstat -ano | findstr :8000` then `taskkill /PID <id> /F`

**Model not found?**
- List available models: `ollama list`
- Pull missing model: `ollama pull codellama:7b`
- Verify model works: `curl -X POST http://localhost:11434/api/show -d '{"name":"codellama:7b"}' -H "Content-Type: application/json"`

**Login issues?**
- Reset admin user: `/c/Python312/python.exe scripts/create_admin.py create`
- Check network access: Try from `localhost` (127.0.0.1)
- Verify password strength requirements (8+ chars, upper/lower/number/symbol)
- List existing users: `/c/Python312/python.exe scripts/create_admin.py list`

**Network access denied?**
- Ensure you're accessing from `localhost` or local network
- Check if your IP is in allowed ranges (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Test network status: `curl http://localhost:8000/auth/status`

**Dependencies missing?**
- Re-run pip install with the full command from Step 2
- Test installation: `/c/Python312/python.exe -c "import fastapi, uvicorn, aiosqlite; print('OK')"`

**Environment variable issues?**
- Use `set` not `export` on Windows: `set PYTHONIOENCODING=utf-8`
- Check current value: `echo %PYTHONIOENCODING%`

---

**🎉 You're all set!** 3dSt_Coder is now running locally and privately on your machine.