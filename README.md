# 🤖 3dSt_Coder

**A locally-hosted AI coding assistant designed for law firms**

3dSt_Coder is a privacy-first AI coding agent that runs entirely on your local machine, wrapping powerful open-source LLMs with developer tools and legal workflow automation.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Version](https://img.shields.io/badge/version-0.1.0-blue) ![Python](https://img.shields.io/badge/python-3.12-blue)

## ✨ Key Features

- 🔒 **100% Local & Private** - No data leaves your machine
- 🧠 **Multiple LLM Backends** - Ollama, OpenAI API, vLLM support
- 🛠️ **Integrated Developer Tools** - Git, file operations, shell, testing
- 💬 **Real-time Streaming** - Live responses with tool execution
- 🌐 **Web Interface** - Browser-based chat with dark theme
- ⚡ **Fast Setup** - Running in 5 minutes with Ollama

## 🚀 Quick Start

**New to 3dSt_Coder?** Follow our 5-minute setup guide:

→ **[📖 Quick Start Guide](docs/QUICK_START.md)**

**Already have it installed?** Launch the web interface:

```bash
/c/Python312/python.exe start_with_ollama.py
# Open http://localhost:8000 in your browser
```

## 📚 Documentation

### For Users
- **[🚀 Quick Start](docs/QUICK_START.md)** - Get up and running in 5 minutes
- **[📖 User Guide](docs/USER_GUIDE.md)** - Complete usage manual with examples
- **[🔧 Setup & Installation](docs/SETUP.md)** - Comprehensive installation options
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
You: "Write a Python function to validate email addresses"
🤖: I'll create an email validation function for you:

    def validate_email(email: str) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
```

```
You: "Run the tests and fix any failures"
🤖: I'll run the test suite first to see what needs fixing.
    {{tool:test_runner}}
    → Found 3 failing tests. Let me analyze and fix them...
```

## 🎯 Perfect For Law Firms

- **Document Processing** - Extract case information and citations
- **Legal Research** - Automated case law analysis
- **Filing Generation** - Court document templates and automation
- **Code Review** - Ensure legal tech solutions meet compliance standards
- **Privacy Compliance** - 100% local processing for confidential matters

## 🛡️ Security & Privacy

- **Air-gapped Operation** - Works completely offline
- **No Data Transmission** - Everything stays on your local machine
- **Sandboxed Execution** - Tools run in controlled environments
- **Path Protection** - File operations restricted to project directories
- **Command Filtering** - Dangerous operations automatically blocked

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

**Ready to get started?** 👉 **[Begin with Quick Start](docs/QUICK_START.md)**