# 📖 3dSt_Coder User Guide

Complete guide to using 3dSt_Coder effectively for coding tasks and legal workflows.

## Table of Contents

- [Getting Started](#getting-started)
- [Understanding the Interface](#understanding-the-interface)
- [Core Features](#core-features)
- [Tool System](#tool-system)
- [Example Workflows](#example-workflows)
- [Legal Use Cases](#legal-use-cases)
- [Best Practices](#best-practices)
- [Advanced Usage](#advanced-usage)

## Getting Started

### What is 3dSt_Coder?

3dSt_Coder is your **local AI coding assistant** that combines the power of large language models with practical developer tools. Unlike cloud-based AI assistants, everything runs on your machine for complete privacy.

**Key Benefits:**
- 🔒 **100% Private** - No data ever leaves your machine
- ⚡ **Real-time** - Streaming responses with live tool execution
- 🛠️ **Integrated Tools** - Git, file operations, shell commands, testing
- 🧠 **Smart Context** - Remembers your project conversations
- 🎯 **Task-focused** - Designed specifically for coding and automation

### First Steps

1. **Launch the Interface**
   ```bash
   /c/Python312/python.exe start_with_ollama.py
   ```

2. **Open Your Browser**
   Navigate to `http://localhost:8000`

3. **Start Your First Conversation**
   Try: *"Hello! Can you help me understand this codebase?"*

## Understanding the Interface

### Web Chat Interface

The 3dSt_Coder interface includes:

- **Message Area** - Shows conversation history with syntax highlighting
- **Input Box** - Where you type your requests and questions
- **Status Indicators** - Shows when tools are running
- **Tool Execution** - Live display of tool results
- **Example Prompts** - Built-in suggestions for common tasks

### Message Types

**Your Messages** (blue, right-aligned)
- Questions, requests, and instructions you send

**AI Responses** (gray, left-aligned)
- Generated text, explanations, and code

**Tool Execution** (highlighted)
- Live display when AI uses tools like `{{tool:git_status}}`

**Tool Results** (formatted)
- Output from executed tools with syntax highlighting

## Core Features

### 1. Code Generation

Ask for any programming task:

```
"Write a Python function to parse CSV files"
"Create a REST API endpoint for user authentication"
"Generate a shell script to backup my project"
```

**What you get:**
- Complete, functional code
- Proper error handling
- Clear documentation
- Best practices applied

### 2. Code Analysis

Upload or reference existing code:

```
"Review this function for potential bugs"
"Explain how this algorithm works"
"Suggest improvements for better performance"
```

### 3. Project Management

Manage your entire project:

```
"What's the current git status?"
"Run the test suite and show results"
"Create a new feature branch"
"Commit these changes with a good message"
```

### 4. Real-time Tool Integration

The AI automatically uses tools when needed:
- **File operations** - Read, write, create files
- **Git commands** - Status, diff, commit, branch
- **Shell execution** - Run build scripts, tests, utilities
- **Testing** - Execute test suites and analyze results

## Tool System

### Available Tools

#### File Tools
- **`file_read`** - Read file contents
- **`file_write`** - Create or modify files

```
"Read the config.py file"
"Create a new README with installation instructions"
```

#### Git Tools
- **`git_status`** - Check repository status
- **`git_diff`** - Show changes
- **`git_commit`** - Commit changes

```
"Show me what files have changed"
"Commit these updates with a descriptive message"
```

#### Shell Tools
- **`shell`** - Execute command-line operations

```
"Install the required Python packages"
"Run the build script"
```

#### Test Tools
- **`test_runner`** - Execute test suites

```
"Run all the unit tests"
"Test just the authentication module"
```

### Tool Safety

All tools operate within **security boundaries**:
- **Path restrictions** - Cannot access files outside your project
- **Command filtering** - Dangerous operations are blocked
- **Resource limits** - Execution time and memory constraints
- **Sandboxed execution** - Isolated from your system

## Example Workflows

### Workflow 1: New Feature Development

```
You: "I need to add user authentication to my Flask app"

AI: I'll help you implement user authentication. Let me start by examining your current Flask structure.
    {{tool:file_read:path=app.py}}

    Based on your current setup, I'll create:
    1. User model with password hashing
    2. Login/logout routes
    3. Session management
    4. Authentication decorators

    {{tool:file_write:path=auth.py}}
    [Creates complete authentication module]

    {{tool:file_write:path=templates/login.html}}
    [Creates login template]

    Now let's test the authentication:
    {{tool:test_runner:command=pytest tests/test_auth.py}}
```

### Workflow 2: Code Review and Debugging

```
You: "My API is returning 500 errors. Can you help debug it?"

AI: I'll help debug your API. Let me examine the error logs and code structure.

    {{tool:shell:command=tail -50 logs/api.log}}

    I see the issue - there's a database connection problem. Let me check your database configuration:

    {{tool:file_read:path=config/database.py}}

    The issue is in line 23 - missing connection timeout. Here's the fix:

    {{tool:file_write:path=config/database.py}}
    [Updates configuration with proper timeout]

    Let's test the fix:
    {{tool:shell:command=python -m pytest tests/test_api.py::test_database_connection}}
```

### Workflow 3: Project Setup

```
You: "Set up a new Python project with testing and CI"

AI: I'll set up a complete Python project structure for you.

    {{tool:file_write:path=pyproject.toml}}
    [Creates Poetry configuration]

    {{tool:file_write:path=src/main.py}}
    [Creates main application file]

    {{tool:file_write:path=tests/test_main.py}}
    [Creates initial tests]

    {{tool:file_write:path=.github/workflows/ci.yml}}
    [Creates GitHub Actions CI]

    {{tool:shell:command=python -m pytest}}
    ✅ All tests passing!

    Your project is ready with:
    - Poetry dependency management
    - Pytest testing framework
    - GitHub Actions CI/CD
    - Proper project structure
```

## Legal Use Cases

### Document Processing

```
"Create a Python script to extract case citations from legal documents"
"Build a tool to validate court filing formats"
"Generate templates for common legal motions"
```

### Compliance Automation

```
"Write a script to audit our codebase for PII handling"
"Create a tool to check document retention policies"
"Build compliance reports from our logs"
```

### Workflow Automation

```
"Automate the process of generating discovery responses"
"Create a pipeline for legal document review"
"Build a system to track case deadlines"
```

### Research Tools

```
"Parse legal databases and extract relevant case law"
"Create a citation checker for legal briefs"
"Build a tool to analyze contract terms"
```

## Best Practices

### Effective Prompting

**Be Specific:**
```
✅ "Create a FastAPI endpoint that validates email addresses and returns JSON"
❌ "Make an API endpoint"
```

**Provide Context:**
```
✅ "In my Flask app, add OAuth2 authentication using the existing User model"
❌ "Add authentication"
```

**Ask for Testing:**
```
✅ "Create a user registration function and include comprehensive tests"
❌ "Create a user registration function"
```

### Project Organization

**Use Descriptive Names:**
- Keep file and function names clear
- Use consistent naming conventions
- Add comments for complex logic

**Leverage Tools:**
- Let the AI read existing code before making changes
- Use git tools to track changes
- Run tests frequently during development

**Security Awareness:**
- Never include API keys or secrets in prompts
- Review generated code before running
- Use the AI's security suggestions

### Conversation Management

**Keep Context:**
- Reference previous work: *"Based on the auth module we just created..."*
- Build incrementally: *"Now let's add error handling to that function"*

**Be Clear About Scope:**
- *"Just update the user model, don't change the API"*
- *"Create a new file, don't modify existing ones"*

## Advanced Usage

### Custom Tool Integration

You can extend 3dSt_Coder with custom tools by modifying the `agent/tools/` directory.

### Multiple Models

Switch between different models for different tasks:

```bash
# Use a code-focused model
export LLM_MODEL_PATH=codellama:7b

# Use a general-purpose model
export LLM_MODEL_PATH=llama3:8b

# Use a specialized model
export LLM_MODEL_PATH=deepseek-coder:6.7b
```

### API Integration

Access 3dSt_Coder programmatically using the REST API:

```python
import requests

response = requests.post('http://localhost:8000/api/v1/chat', json={
    'message': 'Create a Python class for user management',
    'project_id': 'my-project'
})
```

### Configuration

Customize behavior with environment variables:

```bash
# Adjust response creativity
export LLM_TEMPERATURE=0.7

# Set maximum response length
export LLM_MAX_TOKENS=2048

# Configure context window
export LLM_MAX_CONTEXT=4096
```

---

## 🆘 Getting Help

- **🐛 Issues?** Check [Troubleshooting](TROUBLESHOOTING.md)
- **⚙️ Setup?** See [Setup Guide](SETUP.md)
- **🚀 Quick Start?** Review [Quick Start](QUICK_START.md)

**Happy coding with 3dSt_Coder!** 🤖