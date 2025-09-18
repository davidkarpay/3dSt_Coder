"""Orchestrator for managing parallel subagent execution."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncIterator, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .subagent import SubAgent, Task, TaskStatus, TaskResult
from .memory import ConversationMemory
from .tools import get_available_tools

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Execution plan for parallel tasks."""

    tasks: List[Task]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[List[str]] = field(default_factory=list)

    def __post_init__(self):
        """Build execution order based on dependencies."""
        self._build_execution_order()

    def _build_execution_order(self):
        """Build layers of tasks that can execute in parallel."""
        # Build dependency graph
        self.dependencies = {
            task.id: task.dependencies or []
            for task in self.tasks
        }

        # Topological sort to determine execution layers
        layers = []
        completed = set()
        remaining = {task.id for task in self.tasks}

        while remaining:
            # Find tasks with no pending dependencies
            current_layer = []
            for task_id in remaining:
                deps = self.dependencies[task_id]
                if all(dep in completed for dep in deps):
                    current_layer.append(task_id)

            if not current_layer:
                # Circular dependency detected
                logger.error(f"Circular dependency detected in tasks: {remaining}")
                break

            layers.append(current_layer)
            completed.update(current_layer)
            remaining.difference_update(current_layer)

        self.execution_order = layers


class Orchestrator:
    """Orchestrates parallel execution of subagents."""

    def __init__(
        self,
        llm: Any,
        max_parallel: int = 5,
        timeout_per_task: float = 60.0
    ):
        """Initialize orchestrator.

        Args:
            llm: Language model engine
            max_parallel: Maximum number of parallel subagents
            timeout_per_task: Timeout in seconds for each task
        """
        self.llm = llm
        self.max_parallel = max_parallel
        self.timeout_per_task = timeout_per_task
        self.tools = get_available_tools()
        self.active_agents: Dict[str, SubAgent] = {}
        self.results: Dict[str, TaskResult] = {}
        self.execution_metrics = {
            "tasks_started": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_duration": 0.0,
            "parallel_efficiency": 0.0
        }

    async def execute_parallel(
        self,
        main_prompt: str,
        context: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Execute a complex task using parallel subagents.

        Args:
            main_prompt: The main task prompt
            context: Optional context for all subagents

        Yields:
            Progress updates and final results
        """
        start_time = datetime.now()

        # Create master agent for task decomposition
        master_memory = ConversationMemory(
            max_tokens=4096,
            project_id="orchestrator_master"
        )
        master_agent = SubAgent(
            self.llm,
            self.tools,
            master_memory,
            parent_id="orchestrator",
            task_id="master"
        )

        # Decompose task
        yield "Analyzing task and creating execution plan...\n"
        tasks = master_agent.decompose_task(main_prompt)

        if len(tasks) == 1:
            # Single task, no parallelization needed
            yield "Task doesn't require parallelization. Executing directly...\n"
            result = await master_agent.execute_task(tasks[0])
            yield result.output
            return

        # Create execution plan
        plan = ExecutionPlan(tasks)
        yield f"Created execution plan with {len(tasks)} tasks in {len(plan.execution_order)} layers\n"

        # Execute tasks layer by layer
        for layer_idx, task_ids in enumerate(plan.execution_order):
            yield f"\n## Executing Layer {layer_idx + 1} ({len(task_ids)} parallel tasks)\n"

            # Execute tasks in this layer in parallel
            layer_results = await self._execute_layer(
                task_ids,
                tasks,
                context
            )

            # Update results
            self.results.update({
                result.task_id: result
                for result in layer_results
            })

            # Yield progress for each completed task
            for result in layer_results:
                task = next((t for t in tasks if t.id == result.task_id), None)
                if task:
                    status = "✓" if result.success else "✗"
                    yield f"{status} {task.description} ({result.duration:.2f}s)\n"

        # Aggregate all results
        yield "\n## Aggregating Results...\n"
        all_results = list(self.results.values())
        final_output = await master_agent.aggregate_results(all_results)

        # Calculate metrics
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        self.execution_metrics["total_duration"] = total_duration

        # Calculate parallel efficiency
        sequential_time = sum(r.duration for r in all_results)
        parallel_efficiency = (sequential_time / total_duration) if total_duration > 0 else 1.0
        self.execution_metrics["parallel_efficiency"] = parallel_efficiency

        yield "\n" + final_output
        yield f"\n\nParallel efficiency: {parallel_efficiency:.1f}x speedup\n"

    async def _execute_layer(
        self,
        task_ids: List[str],
        all_tasks: List[Task],
        context: Optional[str]
    ) -> List[TaskResult]:
        """Execute a layer of tasks in parallel.

        Args:
            task_ids: IDs of tasks to execute
            all_tasks: All tasks in the plan
            context: Optional context

        Returns:
            List of task results
        """
        # Create semaphore to limit parallelism
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_limit(task: Task) -> TaskResult:
            """Execute task with concurrency limit."""
            async with semaphore:
                return await self._execute_single_task(task, context)

        # Get tasks for this layer
        layer_tasks = [
            task for task in all_tasks
            if task.id in task_ids
        ]

        # Execute all tasks in parallel with timeout
        tasks_with_timeout = [
            asyncio.wait_for(
                execute_with_limit(task),
                timeout=self.timeout_per_task
            )
            for task in layer_tasks
        ]

        # Gather results, handling timeouts
        results = []
        completed_tasks = await asyncio.gather(
            *tasks_with_timeout,
            return_exceptions=True
        )

        for task, result_or_error in zip(layer_tasks, completed_tasks):
            if isinstance(result_or_error, asyncio.TimeoutError):
                # Task timed out
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    output="",
                    duration=self.timeout_per_task,
                    error=f"Task timed out after {self.timeout_per_task}s"
                ))
                self.execution_metrics["tasks_failed"] += 1
            elif isinstance(result_or_error, Exception):
                # Task failed with error
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    output="",
                    duration=0.0,
                    error=str(result_or_error)
                ))
                self.execution_metrics["tasks_failed"] += 1
            else:
                # Task completed successfully
                results.append(result_or_error)
                if result_or_error.success:
                    self.execution_metrics["tasks_completed"] += 1
                else:
                    self.execution_metrics["tasks_failed"] += 1

        return results

    async def _execute_single_task(
        self,
        task: Task,
        context: Optional[str]
    ) -> TaskResult:
        """Execute a single task with a dedicated subagent.

        Args:
            task: Task to execute
            context: Optional context

        Returns:
            Task result
        """
        self.execution_metrics["tasks_started"] += 1

        # Create subagent for this task
        memory = ConversationMemory(
            max_tokens=2048,
            project_id=f"subagent_{task.id}"
        )

        # Add context if provided
        if context:
            await memory.add_system_message(context)

        # Add dependency results as context
        if task.dependencies:
            dep_context = self._build_dependency_context(task.dependencies)
            if dep_context:
                await memory.add_system_message(dep_context)

        subagent = SubAgent(
            self.llm,
            self.tools,
            memory,
            parent_id="orchestrator",
            task_id=task.id
        )

        # Store active agent
        self.active_agents[task.id] = subagent

        try:
            # Execute task
            result = await subagent.execute_task(task)
            return result
        finally:
            # Clean up
            del self.active_agents[task.id]

    def _build_dependency_context(self, dependencies: List[str]) -> Optional[str]:
        """Build context from dependency results.

        Args:
            dependencies: List of dependency task IDs

        Returns:
            Context string or None
        """
        if not dependencies:
            return None

        context_parts = ["Previous task results:"]

        for dep_id in dependencies:
            if dep_id in self.results:
                result = self.results[dep_id]
                if result.success:
                    context_parts.append(f"\n{dep_id}: {result.output[:500]}")

        if len(context_parts) > 1:
            return "\n".join(context_parts)
        return None

    async def cancel_all(self):
        """Cancel all active subagents."""
        for agent_id, agent in self.active_agents.items():
            logger.info(f"Cancelling subagent {agent_id}")
            # Would need to implement cancellation in SubAgent
            # For now, just clear the dict
        self.active_agents.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        return self.execution_metrics.copy()


class ParallelExecutor:
    """High-level interface for parallel execution."""

    def __init__(self, llm: Any, max_parallel: int = 5):
        """Initialize parallel executor."""
        self.llm = llm
        self.max_parallel = max_parallel

    async def execute(
        self,
        prompt: str,
        parallel_hints: Optional[List[str]] = None
    ) -> str:
        """Execute a task with automatic parallelization.

        Args:
            prompt: Task prompt
            parallel_hints: Optional hints for parallelization

        Returns:
            Aggregated result
        """
        orchestrator = Orchestrator(
            self.llm,
            max_parallel=self.max_parallel
        )

        # Build context from hints
        context = None
        if parallel_hints:
            context = "Parallelization hints:\n" + "\n".join(
                f"- {hint}" for hint in parallel_hints
            )

        # Execute and collect results
        full_result = []
        async for chunk in orchestrator.execute_parallel(prompt, context):
            full_result.append(chunk)

        return "".join(full_result)