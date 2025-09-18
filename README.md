# 🚀 3dSt Platform

**A comprehensive AI application platform for building intelligent systems**

3dSt is a privacy-first AI platform that runs entirely on your local machine, providing a foundation for building sophisticated AI applications with multi-engine LLM support, intelligent task detection, file processing, and extensible agent capabilities.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Version](https://img.shields.io/badge/version-0.1.0-blue) ![Python](https://img.shields.io/badge/python-3.12-blue)

## ✨ Key Features

- 🔒 **100% Local & Private** - No data leaves your machine
- 🔐 **Enterprise Security** - JWT authentication + network access control
- 🧠 **Multi-Engine LLM Support** - Ollama, OpenAI API, vLLM backends with task routing
- 🛠️ **Integrated Tool Ecosystem** - Git, file operations, shell, testing, and custom tools
- 📁 **Intelligent File Processing** - Drag-and-drop uploads with content extraction
- 🎯 **Smart Task Detection** - Automatic task classification and agent selection
- 💬 **Real-time Streaming** - Live responses with parallel tool execution
- 🌐 **Modern Web Interface** - Authenticated browser-based platform
- 🔧 **Extensible Architecture** - Plugin system for custom agents and tools
- ⚡ **Fast Setup** - Running in 5 minutes with Ollama

## 🚀 Quick Start

**New to 3dSt Platform?** Follow our 5-minute setup guide:

→ **[📖 Quick Start Guide](docs/QUICK_START.md)**

**Already have it installed?** Launch the platform:

```bash
# Create admin user (first time only)
/c/Python312/python.exe scripts/create_admin.py create

# Launch the platform
/c/Python312/python.exe start_with_ollama.py
# Open http://localhost:8000 and login with your admin credentials
```

## 📚 Documentation

### For Users
- **[🚀 Quick Start](docs/QUICK_START.md)** - Get up and running in 5 minutes
- **[📖 User Guide](docs/USER_GUIDE.md)** - Complete usage manual with examples
- **[🔧 Setup & Installation](docs/SETUP.md)** - Comprehensive installation options
- **[🔐 Security Guide](docs/SECURITY.md)** - Authentication and network security
- **[🚨 Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### For Developers
- **[👩‍💻 Developer Guide](CLAUDE.md)** - Development environment setup
- **[🏗️ Architecture](First.txt)** - Technical architecture and design
- **[📋 Implementation Status](IMPLEMENTATION_SUMMARY.md)** - Current development status

## 💬 Example Interactions

```
You: "Check the git status of this repository"
🤖: I'll check the git status for you.
    {{tool:git_status}}
    → On branch master
    → Changes not staged for commit: ...
```

```
You: [Drag and drop a Python file]
🤖: File uploaded successfully! I've detected this is a code review task.
    Analyzing code structure and identifying potential improvements...
```

```
You: "Create a new REST API endpoint for user management"
🤖: I'll create a comprehensive API endpoint with proper validation:
    {{tool:file_write}}
    → Generated endpoint with authentication, validation, and tests
```

## 🎯 Platform Applications

- **Software Development** - Intelligent coding assistance with context awareness
- **Document Processing** - Multi-format file analysis and content extraction
- **Research & Analysis** - Automated information gathering and synthesis
- **Process Automation** - Custom workflows with tool integration
- **Knowledge Management** - Intelligent content organization and retrieval
- **Quality Assurance** - Automated testing and validation workflows

## 🛡️ Security & Privacy

- **Air-gapped Operation** - Works completely offline
- **No Data Transmission** - Everything stays on your local machine
- **JWT Authentication** - Secure user authentication with role-based access
- **Network Access Control** - Local network/VPN access only
- **Sandboxed Execution** - Tools run in controlled environments
- **Path Protection** - File operations restricted to project directories
- **Command Filtering** - Dangerous operations automatically blocked
- **User Isolation** - Conversation history scoped per user

## 🆘 Need Help?

- **❓ Questions?** Check our [User Guide](docs/USER_GUIDE.md)
- **🐛 Issues?** See [Troubleshooting](docs/TROUBLESHOOTING.md)
- **💻 Development?** Read the [Developer Guide](CLAUDE.md)
- **🏗️ Architecture?** Review the [Technical Specs](First.txt)

## 📊 System Requirements

- **Python 3.12+** - Core runtime
- **4GB+ RAM** - For model inference
- **10GB+ Disk** - For models and data
- **Optional GPU** - For faster inference with vLLM

---

**Ready to build with 3dSt?** 👉 **[Begin with Quick Start](docs/QUICK_START.md)**