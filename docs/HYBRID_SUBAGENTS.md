# Hybrid Subagent Architecture

## Overview

3dSt_Coder's hybrid subagent system combines the best of two approaches:
- **Configuration-based subagents** (like Claude Code) for easy management and reusability
- **Parallel execution architecture** (our original design) for performance and efficiency

This creates a powerful, flexible system that's both user-friendly and performant.

## Key Features

### 🎯 Configuration-Based Management
- Define subagents using Markdown files with YAML frontmatter
- Store in `.claude/agents/` (project) or `~/.claude/agents/` (user)
- Reusable across sessions and shareable with teams
- Version control friendly

### ⚡ Parallel Execution
- Execute multiple subagents concurrently
- Dependency graph management with topological sorting
- Performance metrics and efficiency tracking
- Time savings through parallelization

### 🔧 Flexible Tool Permissions
- Granular tool access control per subagent
- Inherit all tools or specify allowed subset
- Security through least-privilege principle

### 🧠 Model Selection
- Per-subagent model configuration
- Support for inherit, specific models, or aliases
- Mix different models for different tasks

### 💬 Natural Language Invocation
- Invoke subagents by name in commands
- Automatic subagent selection based on task
- Proactive subagent activation

## Quick Start

### 1. Creating a Subagent

Create a file in `.claude/agents/my-agent.md`:

```markdown
---
name: my-custom-agent
description: Description of when this agent should be used
tools: file_read, file_write, shell  # Optional - inherits all if omitted
model: inherit  # or codellama, sonnet, etc.
---

Your subagent's system prompt goes here.
This defines the agent's behavior and expertise.
```

### 2. Using Subagents

#### Explicit Invocation
```python
# In your code or command
"Use the my-custom-agent to analyze the codebase"
"Have the code-reviewer check my recent changes"
"Ask the debugger to fix the failing tests"
```

#### Automatic Selection
```python
# The system automatically selects appropriate subagents
"Review my code for security issues"  # → security-scanner
"Run all tests and fix failures"      # → test-runner
"Optimize database queries"           # → performance-analyzer
```

#### Programmatic Usage
```python
from agent.subagent_config import SubAgentRegistry
from agent.configured_subagent import SubAgentOrchestrator

# Initialize
registry = SubAgentRegistry()
orchestrator = SubAgentOrchestrator(registry)

# Delegate to specific subagent
result = await orchestrator.delegate_to_subagent(
    "code-reviewer",
    "Review the authentication module"
)

# Auto-delegate based on task
async for update in orchestrator.auto_delegate(
    "Analyze all Python files for performance issues"
):
    print(update)
```

## Configuration Format

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Unique identifier (lowercase, hyphens) | `code-reviewer` |
| `description` | When to use this subagent | `Review code for quality and security` |
| System Prompt | Agent's behavior definition | See examples below |

### Optional Fields

| Field | Description | Default |
|-------|-------------|---------|
| `tools` | Comma-separated tool names | All tools inherited |
| `model` | Model to use (inherit/codellama/etc.) | `inherit` |

### Model Options

- `inherit` - Use the main conversation's model
- `codellama` - CodeLlama 7B (local)
- `deepseek` - DeepSeek Coder 6.7B (local)
- `mistral` - Mistral 7B Instruct (local)
- `sonnet` - Claude 3 Sonnet (API)
- `opus` - Claude 3 Opus (API)
- `haiku` - Claude 3 Haiku (API)

## Built-in Subagents

### code-reviewer
Reviews code for quality, security, and best practices. Activated after code modifications.

### test-runner
Executes tests and fixes failures. Uses parallel test execution for speed.

### debugger
Specializes in root cause analysis and fixing errors.

### performance-analyzer
Analyzes code performance and suggests optimizations.

### security-scanner
Scans for security vulnerabilities before deployment.

## Example Subagents

### Data Analyst
```yaml
---
name: data-analyst
description: Analyze data and create visualizations. Use for data science tasks.
tools: file_read, file_write, shell, parallel_file_analyzer
model: codellama
---

You are a data science expert specializing in analysis and visualization.

When analyzing data:
1. Load and explore the dataset
2. Check for missing values and outliers
3. Generate descriptive statistics
4. Create appropriate visualizations
5. Provide insights and recommendations

Use pandas for data manipulation and matplotlib/seaborn for visualization.
```

### API Designer
```yaml
---
name: api-designer
description: Design and implement REST APIs. Use proactively for API tasks.
tools: file_read, file_write, grep
model: inherit
---

You are an API design expert focused on RESTful best practices.

When designing APIs:
1. Follow REST principles (resources, verbs, stateless)
2. Use proper HTTP status codes
3. Implement consistent error handling
4. Design clear, intuitive endpoints
5. Include proper validation and documentation

Always consider:
- Versioning strategy
- Authentication/authorization
- Rate limiting
- Caching strategies
- CORS configuration
```

### Database Optimizer
```yaml
---
name: db-optimizer
description: Optimize database queries and schemas. Must be used for database performance.
tools: file_read, shell, parallel_searcher
model: deepseek
---

You are a database optimization specialist.

Focus areas:
1. Query performance analysis
2. Index optimization
3. Schema design improvements
4. Connection pooling configuration
5. Query caching strategies

For each optimization:
- Explain the current issue
- Show the improved solution
- Provide performance metrics
- Document any trade-offs
```

## Parallel Execution

### How It Works

1. **Task Decomposition**: Complex tasks are automatically broken into subtasks
2. **Dependency Analysis**: Build execution graph with dependencies
3. **Layer Execution**: Execute independent tasks in parallel layers
4. **Result Aggregation**: Combine results with performance metrics

### Example Parallel Workflow

```python
# User request
"Analyze all modules, run tests, and generate documentation"

# System creates execution plan:
Layer 1 (Parallel):
  - analyze-core (file analysis)
  - analyze-api (API analysis)
  - analyze-auth (auth analysis)

Layer 2 (Parallel):
  - test-agent (run agent tests)
  - test-api (run API tests)
  - test-auth (run auth tests)

Layer 3 (Sequential):
  - generate-docs (uses results from layers 1 & 2)

# Performance metrics:
Total time: 15s (vs 45s sequential)
Parallel efficiency: 3.0x speedup
```

## Advanced Features

### Dependency Management
```python
from agent.orchestrator import ExecutionPlan, Task

tasks = [
    Task(id="analyze", description="Analyze code", prompt="..."),
    Task(id="test", description="Run tests", prompt="..."),
    Task(id="report", description="Generate report", prompt="...",
         dependencies=["analyze", "test"])
]

plan = ExecutionPlan(tasks)
# Automatically determines optimal execution order
```

### Performance Metrics
```python
orchestrator = Orchestrator(llm, max_parallel=5)
metrics = orchestrator.get_metrics()

print(f"Tasks completed: {metrics['tasks_completed']}")
print(f"Tasks failed: {metrics['tasks_failed']}")
print(f"Total duration: {metrics['total_duration']}s")
print(f"Parallel efficiency: {metrics['parallel_efficiency']}x")
```

### Custom Tool Sets
```python
config = SubAgentConfig(
    name="limited-agent",
    description="Agent with limited tools",
    system_prompt="...",
    tools=["file_read", "grep"]  # Only these tools available
)
```

## Best Practices

### 1. Subagent Design
- **Single Responsibility**: Each subagent should have one clear purpose
- **Clear Descriptions**: Use descriptive names and descriptions
- **Detailed Prompts**: Provide comprehensive system prompts
- **Tool Minimization**: Only grant necessary tools

### 2. Performance Optimization
- **Identify Parallelizable Tasks**: Look for independent operations
- **Set Appropriate Timeouts**: Balance thoroughness with speed
- **Use Caching**: Leverage result caching for repeated operations
- **Monitor Metrics**: Track efficiency to identify bottlenecks

### 3. Security
- **Least Privilege**: Grant minimum required tool access
- **Path Validation**: Ensure file operations stay within project
- **Command Filtering**: Block dangerous shell commands
- **Audit Logging**: Track subagent activities

### 4. Organization
- **Version Control**: Check project subagents into git
- **Documentation**: Document each subagent's purpose
- **Naming Convention**: Use consistent, descriptive names
- **Regular Review**: Periodically review and update subagents

## Troubleshooting

### Subagent Not Found
```bash
# Check available subagents
from agent.subagent_config import SubAgentRegistry
registry = SubAgentRegistry()
print(registry.list_all())
```

### Tool Permission Errors
```bash
# Verify tool availability
from agent.tools import get_available_tools
print(get_available_tools().keys())
```

### Performance Issues
```bash
# Adjust parallelism
orchestrator = Orchestrator(llm, max_parallel=3)  # Reduce parallel agents
orchestrator.timeout_per_task = 120  # Increase timeout
```

## API Reference

### SubAgentConfig
```python
@dataclass
class SubAgentConfig:
    name: str                    # Unique identifier
    description: str            # When to use
    system_prompt: str          # Behavior definition
    tools: Optional[List[str]]  # Allowed tools
    model: str = "inherit"      # Model selection
    proactive: bool = False     # Auto-activate
```

### SubAgentRegistry
```python
class SubAgentRegistry:
    def get(name: str) -> Optional[SubAgentConfig]
    def list_all() -> List[SubAgentConfig]
    def find_matching(task: str) -> List[SubAgentConfig]
    def create_subagent(config: SubAgentConfig, location: str) -> Path
    def delete_subagent(name: str) -> bool
    def update_subagent(name: str, updates: Dict) -> bool
```

### SubAgentOrchestrator
```python
class SubAgentOrchestrator:
    async def delegate_to_subagent(name: str, prompt: str) -> str
    async def auto_delegate(prompt: str) -> AsyncIterator[str]
    def get_metrics() -> Dict[str, Any]
```

## Migration Guide

### From Original Parallel System
Your existing parallel tools and orchestration continue to work. The new system adds:
1. Configuration layer on top
2. Natural language invocation
3. Tool permission management
4. Model selection per subagent

### From Claude Code Style
Your Markdown configurations work with minor adjustments:
1. Place files in `.claude/agents/`
2. Tools are validated against available set
3. Parallel execution is automatic when beneficial

## Future Enhancements

- [ ] Web UI for subagent management
- [ ] Subagent marketplace/sharing
- [ ] Learning from execution history
- [ ] Dynamic subagent generation
- [ ] Cross-project subagent libraries
- [ ] Performance profiling per subagent
- [ ] A/B testing different subagents
- [ ] Collaborative subagent editing

## Summary

The hybrid subagent system provides:
- **Ease of Use**: Simple Markdown configuration
- **Performance**: Parallel execution with metrics
- **Flexibility**: Per-agent tools and models
- **Security**: Granular permission control
- **Scalability**: Efficient resource utilization

This architecture enables both simple single-agent tasks and complex multi-agent workflows, adapting to your needs while maintaining high performance and security standards.