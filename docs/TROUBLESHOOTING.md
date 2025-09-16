# 🚨 Troubleshooting Guide

Solutions to common issues when using 3dSt_Coder.

## Table of Contents

- [Quick Diagnosis](#quick-diagnosis)
- [Installation Issues](#installation-issues)
- [Server Won't Start](#server-wont-start)
- [Model Problems](#model-problems)
- [Performance Issues](#performance-issues)
- [Tool Failures](#tool-failures)
- [Connection Problems](#connection-problems)
- [Windows-Specific Issues](#windows-specific-issues)
- [Getting Help](#getting-help)

## Quick Diagnosis

### System Health Check

Run these commands to quickly identify issues:

```bash
# 1. Check Python installation
/c/Python312/python.exe --version
# Expected: Python 3.12.x

# 2. Check if Ollama is running
curl http://localhost:11434/api/version
# Expected: JSON response with version info

# 3. Test 3dSt_Coder server
curl http://localhost:8000/ping
# Expected: {"status": "pong", "message": "Coding agent API is running"}

# 4. Verify available models
ollama list
# Expected: List of installed models

# 5. Check dependencies
/c/Python312/python.exe -c "import fastapi, uvicorn, aiosqlite; print('Dependencies OK')"
# Expected: "Dependencies OK"
```

## Installation Issues

### Python 3.12 Not Found

**Problem:** `python` command not recognized or wrong version

**Solutions:**
```bash
# Check if Python 3.12 is installed
where python
python --version

# If not installed, download from python.org
# Or use specific path
/c/Python312/python.exe --version

# Add to PATH (Windows)
setx PATH "%PATH%;C:\Python312;C:\Python312\Scripts"
```

### Pip Install Failures

**Problem:** Package installation fails with errors

**Solutions:**
```bash
# Upgrade pip first
/c/Python312/python.exe -m pip install --upgrade pip

# Install with verbose output to see errors
/c/Python312/python.exe -m pip install -v fastapi

# Use different index if needed
/c/Python312/python.exe -m pip install --index-url https://pypi.org/simple/ fastapi

# Clear pip cache
/c/Python312/python.exe -m pip cache purge
```

### Permission Errors

**Problem:** Access denied during installation

**Solutions:**
```bash
# Run as administrator (Windows)
# Right-click Command Prompt → "Run as administrator"

# Install for user only
/c/Python312/python.exe -m pip install --user fastapi

# Check directory permissions
icacls F:\GitHub\localLLM
```

## Server Won't Start

### Port Already in Use

**Problem:** `Address already in use: port 8000`

**Diagnosis:**
```bash
# Check what's using port 8000
netstat -ano | findstr :8000
# Kill the process if needed
taskkill /PID <process_id> /F
```

**Solutions:**
```bash
# Use a different port
set LLM_PORT=8001
/c/Python312/python.exe -m api.main

# Or find and stop the conflicting service
```

### Module Import Errors

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solutions:**
```bash
# Install missing dependencies
/c/Python312/python.exe -m pip install pydantic fastapi uvicorn aiosqlite gitpython requests aiofiles sse-starlette httpx pydantic-settings

# Check Python path
/c/Python312/python.exe -c "import sys; print(sys.path)"

# Verify installation
/c/Python312/python.exe -c "import fastapi; print(fastapi.__version__)"
```

### Configuration Errors

**Problem:** Server starts but crashes immediately

**Check logs:**
```bash
# Look for error messages in console output
/c/Python312/python.exe -m api.main

# Check if configuration is valid
/c/Python312/python.exe -c "from llm_server.config import LLMConfig; print(LLMConfig())"
```

## Model Problems

### Ollama Not Running

**Problem:** `Connection refused` to Ollama server

**Diagnosis:**
```bash
# Check if Ollama process is running
tasklist | findstr ollama

# Test Ollama API
curl http://localhost:11434/api/version
```

**Solutions:**
```bash
# Start Ollama server
ollama serve

# Check Ollama installation
ollama --version

# Restart Ollama service
# On Windows, restart the Ollama service from Services.msc
```

### Model Not Found

**Problem:** `Model 'codellama:7b' not found`

**Diagnosis:**
```bash
# List available models
ollama list

# Check model status
ollama show codellama:7b
```

**Solutions:**
```bash
# Pull the missing model
ollama pull codellama:7b

# Wait for download to complete (may take several minutes)
# Verify the model is available
ollama list
```

### Model Loading Errors

**Problem:** Model fails to load or generates errors

**Solutions:**
```bash
# Try a different model
ollama pull llama3:8b
export LLM_MODEL_PATH=llama3:8b

# Check available memory
# Close other applications to free RAM

# Use a smaller model
ollama pull codellama:7b-code-q4_K_M  # Quantized version
```

### Slow Model Performance

**Problem:** Responses are very slow

**Optimizations:**
```bash
# Use quantized models
ollama pull codellama:7b-code-q4_K_M

# Reduce context length
set LLM_MAX_CONTEXT=2048

# Close other applications using RAM
# Increase virtual memory (Windows)
```

## Performance Issues

### High Memory Usage

**Problem:** System running out of memory

**Solutions:**
```bash
# Monitor memory usage
tasklist | findstr python
tasklist | findstr ollama

# Use lighter models
ollama pull phi3:3.8b  # Smaller model

# Limit concurrent requests
set OLLAMA_NUM_PARALLEL=1

# Increase system virtual memory
```

### Slow Response Times

**Problem:** AI responses take too long

**Diagnosis:**
```bash
# Test response time
time curl -X POST http://localhost:8000/api/v1/chat/complete -H "Content-Type: application/json" -d '{"message": "Hello"}'

# Check system resources
perfmon  # Windows Performance Monitor
```

**Solutions:**
```bash
# Use faster models
ollama pull phi3:3.8b

# Reduce max tokens
set LLM_MAX_TOKENS=1024

# Enable GPU acceleration (if available)
# Install CUDA and use vLLM backend
```

## Tool Failures

### Git Commands Fail

**Problem:** Git tools return errors

**Diagnosis:**
```bash
# Check if git is installed
git --version

# Test git in project directory
cd F:\GitHub\localLLM
git status
```

**Solutions:**
```bash
# Install Git for Windows
# Download from git-scm.com

# Add Git to PATH
setx PATH "%PATH%;C:\Program Files\Git\bin"

# Initialize repository if needed
git init
```

### File Access Errors

**Problem:** Cannot read/write files

**Diagnosis:**
```bash
# Check file permissions
icacls filename.txt

# Test file access
/c/Python312/python.exe -c "open('test.txt', 'w').write('test')"
```

**Solutions:**
```bash
# Run as administrator if needed
# Change file permissions
icacls filename.txt /grant Everyone:F

# Check if file is in use by another program
```

### Shell Commands Blocked

**Problem:** Shell tool won't execute commands

**Diagnosis:**
```bash
# Check if command exists
where python
where npm
where git
```

**Solutions:**
```bash
# Install missing tools
# Add tools to system PATH
# Use full paths in commands
```

## Connection Problems

### Web Interface Won't Load

**Problem:** Browser shows "This site can't be reached"

**Diagnosis:**
```bash
# Check if server is running
curl http://localhost:8000/ping

# Test with different browser
# Try http://127.0.0.1:8000
```

**Solutions:**
```bash
# Check Windows Firewall settings
# Allow Python through firewall
netsh advfirewall firewall add rule name="Python" program="/c/Python312/python.exe" action=allow dir=in

# Try different port
set LLM_PORT=8001
```

### API Requests Fail

**Problem:** API returns errors or timeouts

**Check server logs:**
```bash
# Look for errors in console where you started the server
# Common errors: CORS issues, timeout errors, model errors
```

**Solutions:**
```bash
# Check request format
curl -X POST http://localhost:8000/api/v1/chat/complete \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Increase timeout settings
set LLM_TIMEOUT=120
```

## Windows-Specific Issues

### PowerShell Execution Policy

**Problem:** Scripts won't run due to execution policy

**Solution:**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy to allow scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Path Length Limitations

**Problem:** File paths too long for Windows

**Solution:**
```bash
# Enable long paths in Windows
# Run as admin:
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

# Or use shorter directory names
```

### Encoding Issues

**Problem:** Unicode/encoding errors

**Solutions:**
```bash
# Set encoding environment variable
set PYTHONIOENCODING=utf-8

# Use UTF-8 codepage
chcp 65001

# Add to system environment variables permanently
```

### Antivirus Interference

**Problem:** Antivirus blocking Python or Ollama

**Solutions:**
```bash
# Add exclusions to your antivirus:
# - F:\GitHub\localLLM\
# - C:\Python312\
# - Ollama installation directory
# - %USERPROFILE%\.ollama\
```

## Getting Help

### Collect Diagnostic Information

Before asking for help, gather this information:

```bash
# System information
systeminfo | findstr /C:"OS Name" /C:"Total Physical Memory"

# Python environment
/c/Python312/python.exe --version
/c/Python312/python.exe -m pip list

# Ollama status
ollama --version
ollama list
curl http://localhost:11434/api/version

# 3dSt_Coder status
curl http://localhost:8000/api/v1/health

# Error logs
# Copy any error messages from console output
```

### Create a Support Request

Include this information when asking for help:

1. **Operating System:** Windows version, architecture
2. **Python Version:** Output of `python --version`
3. **Installation Method:** Which setup method you used
4. **Error Message:** Exact error text
5. **Steps to Reproduce:** What you did before the error
6. **Expected vs Actual:** What should happen vs what actually happened

### Common Solutions Summary

| Problem | Quick Fix |
|---------|-----------|
| Server won't start | Check if port 8000 is free |
| Model not found | Run `ollama pull codellama:7b` |
| Slow responses | Try smaller model or reduce max_tokens |
| Import errors | Reinstall dependencies |
| Git tools fail | Install Git and add to PATH |
| Web UI won't load | Check firewall settings |
| Memory issues | Use quantized models |
| Encoding errors | Set `PYTHONIOENCODING=utf-8` |

---

**Still having issues?** Check the other documentation:
- 📖 [User Guide](USER_GUIDE.md) - Usage instructions
- 🔧 [Setup Guide](SETUP.md) - Installation options
- 🚀 [Quick Start](QUICK_START.md) - Basic setup