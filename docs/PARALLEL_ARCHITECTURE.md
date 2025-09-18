# Parallel Subagent Architecture

## Overview

The 3dSt_Coder parallel subagent architecture enables concurrent execution of independent tasks, significantly improving performance for complex operations. This system allows the AI agent to decompose large tasks into smaller, parallelizable components and execute them simultaneously.

**Note**: We now offer a [Hybrid Subagent System](HYBRID_SUBAGENTS.md) that combines this parallel execution architecture with configuration-based subagent management similar to Claude Code, providing both performance and ease of use.

## Architecture Components

### 1. SubAgent Framework (`agent/subagent.py`)

The **SubAgent** extends the base CodingAgent with task decomposition and parallel execution capabilities:

- **Task Decomposition**: Automatically identifies parallelizable components in complex tasks
- **Task Execution**: Manages individual task lifecycle (pending → running → completed/failed)
- **Result Aggregation**: Combines outputs from parallel tasks into coherent responses
- **Parent-Child Relationships**: Maintains hierarchy for nested parallel execution

#### Key Classes:

```python
class Task:
    id: str                    # Unique task identifier
    description: str           # Human-readable description
    prompt: str               # Task prompt for LLM
    dependencies: List[str]   # Task IDs that must complete first
    status: TaskStatus        # Current execution state
    result: Optional[str]     # Task output
```

### 2. Orchestrator (`agent/orchestrator.py`)

The **Orchestrator** manages the parallel execution of multiple subagents:

- **Execution Planning**: Builds dependency graph and determines execution layers
- **Concurrency Control**: Limits parallel agents via semaphore (default: 5)
- **Timeout Management**: Enforces per-task timeouts (default: 60s)
- **Metric Collection**: Tracks performance and parallel efficiency

#### Execution Flow:

1. **Task Decomposition**: Master agent analyzes prompt and creates subtasks
2. **Dependency Resolution**: Builds execution plan with parallel layers
3. **Layer Execution**: Runs independent tasks in each layer concurrently
4. **Result Aggregation**: Combines outputs and calculates metrics

### 3. Enhanced CodingAgent (`agent/core.py`)

Extended with parallel tool execution capabilities:

- **Parallel Tool Pattern**: `{{parallel:[tool1, tool2, tool3]}}`
- **Concurrent Execution**: Uses `asyncio.gather()` for parallel tool calls
- **Result Synchronization**: Waits for all tools before continuing

### 4. Parallel Tools (`agent/tools/parallel.py`)

Specialized tools optimized for parallel execution:

#### ParallelFileAnalyzer
Analyzes multiple files simultaneously for:
- Structure analysis (classes, functions, imports)
- Pattern detection (async/await, dataclasses, type hints)
- Dependency mapping

#### ParallelTestRunner
Executes multiple test suites concurrently:
- Runs pytest on different test directories in parallel
- Aggregates pass/fail statistics
- Provides unified test report

#### ParallelCodeGenerator
Generates multiple code components simultaneously:
- Class templates
- Function templates
- Test templates

#### ParallelSearcher
Searches for patterns across multiple paths concurrently:
- Uses grep for fast pattern matching
- Supports file type filtering
- Aggregates match counts

## Usage Examples

### 1. Simple Parallel Tool Execution

```python
# In the agent's response:
"I'll check the status and read the config. {{parallel:[git_status, file_read]}}"
```

### 2. Complex Task Orchestration

```python
from agent.orchestrator import Orchestrator

orchestrator = Orchestrator(llm, max_parallel=5)

async for update in orchestrator.execute_parallel(
    "Analyze all Python files, run tests, and generate documentation"
):
    print(update)
```

### 3. Task Decomposition Patterns

The system automatically decomposes tasks based on keywords:

- **"analyze ... files"** → Parallel file analysis tasks
- **"run all tests"** → Parallel test suite execution
- **"refactor modules"** → Parallel refactoring with dependency analysis
- **"document codebase"** → Parallel documentation generation

### 4. Using ParallelExecutor

High-level interface for parallel execution:

```python
from agent.orchestrator import ParallelExecutor

executor = ParallelExecutor(llm, max_parallel=5)

result = await executor.execute(
    prompt="Analyze the entire codebase",
    parallel_hints=["Check all modules", "Analyze dependencies"]
)
```

## Performance Benefits

### Parallel Efficiency Metrics

The system calculates and reports:
- **Sequential Time**: Total time if tasks ran sequentially
- **Parallel Time**: Actual execution time with parallelization
- **Speedup Factor**: Sequential time / Parallel time
- **Time Saved**: Sequential time - Parallel time

### Example Performance Gains

For a task with 4 independent subtasks:
- Task 1: 2.5s
- Task 2: 3.0s
- Task 3: 1.5s
- Task 4: 2.0s

**Sequential**: 9.0s total
**Parallel**: 3.0s total (limited by longest task)
**Speedup**: 3.0x
**Time Saved**: 6.0s

## Configuration

### Environment Variables

```bash
# Maximum parallel subagents (default: 5)
MAX_PARALLEL_AGENTS=5

# Task timeout in seconds (default: 60)
TASK_TIMEOUT=60

# Enable parallel execution logging
PARALLEL_DEBUG=true
```

### Orchestrator Configuration

```python
orchestrator = Orchestrator(
    llm=llm_engine,
    max_parallel=10,        # Maximum concurrent subagents
    timeout_per_task=120.0  # Timeout in seconds
)
```

## Best Practices

### 1. Task Design
- Keep tasks focused and independent
- Minimize inter-task dependencies
- Provide clear task descriptions

### 2. Dependency Management
- Explicitly declare task dependencies
- Avoid circular dependencies
- Use dependency results as context

### 3. Resource Management
- Set appropriate parallelism limits
- Monitor memory usage with many agents
- Use timeouts to prevent hanging tasks

### 4. Error Handling
- Tasks fail independently
- Failed tasks don't block other tasks
- Aggregator handles partial results

## Testing

Run parallel execution tests:

```bash
# Run all parallel tests
/c/Python312/python.exe -m pytest agent/tests/test_parallel.py -v

# Test specific components
/c/Python312/python.exe -m pytest agent/tests/test_parallel.py::TestOrchestrator -v
/c/Python312/python.exe -m pytest agent/tests/test_parallel.py::TestParallelTools -v
```

## Limitations

1. **Shared State**: Subagents don't share memory during execution
2. **Context Size**: Each subagent has limited context window
3. **Dependency Chains**: Deep dependencies reduce parallelization benefits
4. **Resource Constraints**: System resources limit maximum parallelism

## Future Enhancements

- **Dynamic Task Splitting**: Automatically split large tasks based on complexity
- **Adaptive Parallelism**: Adjust parallelism based on system load
- **Inter-Agent Communication**: Allow subagents to share results during execution
- **Distributed Execution**: Support for multi-machine agent clusters
- **ML-Based Decomposition**: Use machine learning to optimize task decomposition