"""SubAgent framework for parallel task execution."""

import asyncio
import logging
from typing import Dict, Any, AsyncIterator, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .core import CodingAgent
from .memory import ConversationMemory
from .tools import BaseTool

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a parallel task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a task for parallel execution."""

    id: str
    description: str
    prompt: str
    dependencies: List[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskResult:
    """Result from a completed task."""

    task_id: str
    success: bool
    output: str
    duration: float
    error: Optional[str] = None


class SubAgent(CodingAgent):
    """Extended agent capable of running as a subagent."""

    def __init__(
        self,
        llm: Any,
        tools: Dict[str, BaseTool],
        memory: ConversationMemory,
        parent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Initialize subagent with parent reference."""
        super().__init__(llm, tools, memory)
        self.parent_id = parent_id
        self.task_id = task_id
        self.subtasks: List[Task] = []

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task and return result."""
        start_time = datetime.now()
        task.status = TaskStatus.RUNNING
        task.started_at = start_time

        try:
            # Execute the task
            logger.info(f"SubAgent {self.task_id} executing task {task.id}: {task.description}")

            # Generate response for the task
            response = ""
            async for chunk in self.chat(task.prompt):
                response += chunk

            task.status = TaskStatus.COMPLETED
            task.result = response
            task.completed_at = datetime.now()

            duration = (task.completed_at - start_time).total_seconds()

            return TaskResult(
                task_id=task.id,
                success=True,
                output=response,
                duration=duration
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            duration = (task.completed_at - start_time).total_seconds()

            logger.error(f"Task {task.id} failed: {e}")

            return TaskResult(
                task_id=task.id,
                success=False,
                output="",
                duration=duration,
                error=str(e)
            )

    def decompose_task(self, main_prompt: str) -> List[Task]:
        """Decompose a complex task into parallel subtasks."""
        subtasks = []

        # Analyze prompt for parallelizable components
        prompt_lower = main_prompt.lower()

        # Example decomposition patterns
        if "analyze" in prompt_lower and ("files" in prompt_lower or "codebase" in prompt_lower):
            # File analysis can be parallelized
            subtasks.extend(self._create_file_analysis_tasks(main_prompt))

        elif "test" in prompt_lower and ("all" in prompt_lower or "suite" in prompt_lower):
            # Test execution can be parallelized
            subtasks.extend(self._create_test_tasks(main_prompt))

        elif "refactor" in prompt_lower and "module" in prompt_lower:
            # Module refactoring can be parallelized
            subtasks.extend(self._create_refactor_tasks(main_prompt))

        elif "document" in prompt_lower and "codebase" in prompt_lower:
            # Documentation generation can be parallelized
            subtasks.extend(self._create_documentation_tasks(main_prompt))

        # If no specific pattern matched, create a single task
        if not subtasks:
            subtasks.append(Task(
                id="main",
                description="Main task",
                prompt=main_prompt
            ))

        self.subtasks = subtasks
        return subtasks

    def _create_file_analysis_tasks(self, prompt: str) -> List[Task]:
        """Create parallel tasks for file analysis."""
        tasks = []

        # This would normally extract file paths from the prompt
        # For now, using example structure
        tasks.append(Task(
            id="analyze_core",
            description="Analyze core modules",
            prompt="Analyze the structure and patterns in agent/core.py"
        ))

        tasks.append(Task(
            id="analyze_tools",
            description="Analyze tool implementations",
            prompt="Analyze the tool implementations in agent/tools/"
        ))

        tasks.append(Task(
            id="analyze_api",
            description="Analyze API layer",
            prompt="Analyze the API endpoints and routing in api/"
        ))

        # Add dependency - summary depends on individual analyses
        tasks.append(Task(
            id="create_summary",
            description="Create analysis summary",
            prompt="Summarize the analysis results from all components",
            dependencies=["analyze_core", "analyze_tools", "analyze_api"]
        ))

        return tasks

    def _create_test_tasks(self, prompt: str) -> List[Task]:
        """Create parallel tasks for test execution."""
        tasks = []

        tasks.append(Task(
            id="test_agent",
            description="Run agent tests",
            prompt="Execute tests in agent/tests/"
        ))

        tasks.append(Task(
            id="test_llm",
            description="Run LLM server tests",
            prompt="Execute tests in llm_server/tests/"
        ))

        tasks.append(Task(
            id="test_api",
            description="Run API tests",
            prompt="Execute tests in api/tests/"
        ))

        tasks.append(Task(
            id="test_auth",
            description="Run authentication tests",
            prompt="Execute tests in auth/tests/"
        ))

        # Test report depends on all test results
        tasks.append(Task(
            id="generate_report",
            description="Generate test report",
            prompt="Create a comprehensive test report from all test results",
            dependencies=["test_agent", "test_llm", "test_api", "test_auth"]
        ))

        return tasks

    def _create_refactor_tasks(self, prompt: str) -> List[Task]:
        """Create parallel tasks for refactoring."""
        tasks = []

        tasks.append(Task(
            id="identify_patterns",
            description="Identify refactoring patterns",
            prompt="Identify code patterns that need refactoring"
        ))

        tasks.append(Task(
            id="analyze_dependencies",
            description="Analyze dependencies",
            prompt="Analyze module dependencies for refactoring impact"
        ))

        tasks.append(Task(
            id="create_plan",
            description="Create refactoring plan",
            prompt="Create a detailed refactoring plan",
            dependencies=["identify_patterns", "analyze_dependencies"]
        ))

        return tasks

    def _create_documentation_tasks(self, prompt: str) -> List[Task]:
        """Create parallel tasks for documentation."""
        tasks = []

        tasks.append(Task(
            id="doc_api",
            description="Document API endpoints",
            prompt="Generate documentation for API endpoints"
        ))

        tasks.append(Task(
            id="doc_tools",
            description="Document available tools",
            prompt="Generate documentation for agent tools"
        ))

        tasks.append(Task(
            id="doc_architecture",
            description="Document architecture",
            prompt="Document the system architecture"
        ))

        tasks.append(Task(
            id="create_readme",
            description="Update README",
            prompt="Update README with comprehensive documentation",
            dependencies=["doc_api", "doc_tools", "doc_architecture"]
        ))

        return tasks

    async def aggregate_results(self, results: List[TaskResult]) -> str:
        """Aggregate results from parallel task execution."""
        if not results:
            return "No results to aggregate"

        # Build aggregated response
        aggregated = []

        # Group by success/failure
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            aggregated.append("## Completed Tasks\n")
            for result in successful:
                task = next((t for t in self.subtasks if t.id == result.task_id), None)
                if task:
                    aggregated.append(f"### {task.description}")
                    aggregated.append(result.output)
                    aggregated.append(f"*Completed in {result.duration:.2f}s*\n")

        if failed:
            aggregated.append("\n## Failed Tasks\n")
            for result in failed:
                task = next((t for t in self.subtasks if t.id == result.task_id), None)
                if task:
                    aggregated.append(f"### {task.description}")
                    aggregated.append(f"Error: {result.error}\n")

        # Add summary statistics
        aggregated.append("\n## Summary")
        aggregated.append(f"- Total tasks: {len(results)}")
        aggregated.append(f"- Successful: {len(successful)}")
        aggregated.append(f"- Failed: {len(failed)}")

        total_duration = sum(r.duration for r in results)
        aggregated.append(f"- Total execution time: {total_duration:.2f}s")

        # Calculate time saved by parallel execution
        sequential_time = sum(r.duration for r in results)
        parallel_time = max((r.duration for r in results), default=0)
        time_saved = sequential_time - parallel_time

        if time_saved > 0:
            aggregated.append(f"- Time saved by parallelization: {time_saved:.2f}s")

        return "\n".join(aggregated)