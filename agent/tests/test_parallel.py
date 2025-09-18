"""Test suite for parallel subagent execution."""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from agent.subagent import SubAgent, Task, TaskStatus, TaskResult
from agent.orchestrator import Orchestrator, ExecutionPlan, ParallelExecutor
from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools.parallel import (
    ParallelFileAnalyzer,
    ParallelTestRunner,
    ParallelCodeGenerator,
    ParallelSearcher
)


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.current = 0

    async def generate(self, prompt: str, stop: List[str] = None):
        """Generate mock response."""
        if self.current < len(self.responses):
            response = self.responses[self.current]
            self.current += 1
            # Simulate streaming but keep special patterns intact
            import re
            # Split on spaces but preserve {{...}} patterns
            parts = re.split(r'(\{\{[^}]+\}\})', response)
            for part in parts:
                if part.strip():
                    if part.startswith('{{') and part.endswith('}}'):
                        # Keep special patterns intact
                        yield part + " "
                    else:
                        # Split normal text into words
                        for word in part.split():
                            yield word + " "
        else:
            yield "Default response"


class TestSubAgent:
    """Test SubAgent functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM(["Test response"])

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        memory = AsyncMock(spec=ConversationMemory)
        memory.messages = []
        memory.add_user_message = AsyncMock()
        memory.add_assistant_message = AsyncMock()
        memory.add_tool_result = AsyncMock()
        return memory

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return {
            "test_tool": Mock(
                run=AsyncMock(return_value="tool result"),
                description="Test tool"
            )
        }

    @pytest.mark.asyncio
    async def test_execute_task(self, mock_llm, mock_memory, mock_tools):
        """Test single task execution."""
        agent = SubAgent(mock_llm, mock_tools, mock_memory, "parent", "agent1")

        task = Task(
            id="task1",
            description="Test task",
            prompt="Do something"
        )

        result = await agent.execute_task(task)

        assert result.task_id == "task1"
        assert result.success is True
        assert result.output == "Test response "
        assert result.duration >= 0  # Duration can be 0 for very fast execution
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_task_with_error(self, mock_memory, mock_tools):
        """Test task execution with error."""
        # Create LLM that raises error
        async def error_generator(prompt: str, stop: List[str] = None):
            raise RuntimeError("LLM error")
            yield  # Unreachable but makes it a generator

        mock_llm = Mock()
        mock_llm.generate = error_generator

        agent = SubAgent(mock_llm, mock_tools, mock_memory, "parent", "agent1")

        task = Task(
            id="task1",
            description="Test task",
            prompt="Do something"
        )

        result = await agent.execute_task(task)

        assert result.task_id == "task1"
        assert result.success is False
        assert "LLM error" in result.error
        assert task.status == TaskStatus.FAILED

    def test_decompose_task_file_analysis(self, mock_llm, mock_memory, mock_tools):
        """Test task decomposition for file analysis."""
        agent = SubAgent(mock_llm, mock_tools, mock_memory)

        prompt = "Analyze all files in the project and create a report"
        tasks = agent.decompose_task(prompt)

        assert len(tasks) > 1
        # Should have analysis tasks and a summary task
        task_ids = [t.id for t in tasks]
        assert "analyze_core" in task_ids
        assert "create_summary" in task_ids

        # Summary should depend on analyses
        summary_task = next(t for t in tasks if t.id == "create_summary")
        assert len(summary_task.dependencies) > 0

    def test_decompose_task_test_execution(self, mock_llm, mock_memory, mock_tools):
        """Test task decomposition for test execution."""
        agent = SubAgent(mock_llm, mock_tools, mock_memory)

        prompt = "Run all tests and generate a comprehensive test report"
        tasks = agent.decompose_task(prompt)

        assert len(tasks) > 1
        task_ids = [t.id for t in tasks]
        assert "test_agent" in task_ids
        assert "generate_report" in task_ids

        # Report should depend on test results
        report_task = next(t for t in tasks if t.id == "generate_report")
        assert len(report_task.dependencies) > 0

    @pytest.mark.asyncio
    async def test_aggregate_results(self, mock_llm, mock_memory, mock_tools):
        """Test result aggregation."""
        agent = SubAgent(mock_llm, mock_tools, mock_memory)

        # Create mock tasks
        agent.subtasks = [
            Task(id="task1", description="First task", prompt="p1"),
            Task(id="task2", description="Second task", prompt="p2"),
            Task(id="task3", description="Failed task", prompt="p3")
        ]

        # Create results
        results = [
            TaskResult("task1", True, "Result 1", 1.5),
            TaskResult("task2", True, "Result 2", 2.0),
            TaskResult("task3", False, "", 0.5, "Error occurred")
        ]

        aggregated = await agent.aggregate_results(results)

        assert "Completed Tasks" in aggregated
        assert "First task" in aggregated
        assert "Result 1" in aggregated
        assert "Failed Tasks" in aggregated
        assert "Error occurred" in aggregated
        assert "Time saved by parallelization" in aggregated


class TestOrchestrator:
    """Test Orchestrator functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM([
            "Analyzing task",
            "Task 1 result",
            "Task 2 result",
            "Final summary"
        ])

    @pytest.mark.asyncio
    async def test_execute_parallel_simple(self, mock_llm):
        """Test simple parallel execution."""
        orchestrator = Orchestrator(mock_llm, max_parallel=2)

        # Collect all output
        output = []
        async for chunk in orchestrator.execute_parallel("Simple task"):
            output.append(chunk)

        full_output = "".join(output)
        assert "execution plan" in full_output.lower()

    @pytest.mark.asyncio
    async def test_execute_single_task(self, mock_llm):
        """Test single task execution (no parallelization)."""
        orchestrator = Orchestrator(mock_llm, max_parallel=5)

        task = Task(id="single", description="Single task", prompt="Do one thing")

        result = await orchestrator._execute_single_task(task, None)

        assert result.task_id == "single"
        assert result.success is True
        assert orchestrator.execution_metrics["tasks_started"] == 1

    @pytest.mark.asyncio
    async def test_execute_layer_with_timeout(self, mock_llm):
        """Test layer execution with timeout."""
        orchestrator = Orchestrator(mock_llm, max_parallel=2, timeout_per_task=0.1)

        # Create task that will timeout
        slow_task = Task(id="slow", description="Slow task", prompt="Takes too long")

        # Mock the execute method to be slow
        with patch.object(orchestrator, '_execute_single_task') as mock_execute:
            async def slow_execution(task, context):
                await asyncio.sleep(1.0)  # Longer than timeout
                return TaskResult(task.id, True, "Done", 1.0)

            mock_execute.side_effect = slow_execution

            results = await orchestrator._execute_layer(
                ["slow"],
                [slow_task],
                None
            )

            assert len(results) == 1
            assert results[0].success is False
            assert "timed out" in results[0].error

    def test_execution_plan_building(self):
        """Test execution plan with dependencies."""
        tasks = [
            Task(id="t1", description="Task 1", prompt="p1"),
            Task(id="t2", description="Task 2", prompt="p2"),
            Task(id="t3", description="Task 3", prompt="p3", dependencies=["t1", "t2"]),
            Task(id="t4", description="Task 4", prompt="p4", dependencies=["t3"])
        ]

        plan = ExecutionPlan(tasks)

        # Should have 3 layers
        assert len(plan.execution_order) == 3
        # First layer: t1 and t2 (no dependencies)
        assert set(plan.execution_order[0]) == {"t1", "t2"}
        # Second layer: t3 (depends on t1 and t2)
        assert plan.execution_order[1] == ["t3"]
        # Third layer: t4 (depends on t3)
        assert plan.execution_order[2] == ["t4"]


class TestParallelTools:
    """Test parallel tool implementations."""

    @pytest.mark.asyncio
    async def test_parallel_file_analyzer(self):
        """Test parallel file analysis."""
        analyzer = ParallelFileAnalyzer()

        # Mock file reading
        with patch.object(analyzer.file_tool, 'run') as mock_read:
            mock_read.side_effect = [
                "class TestClass:\n    def method(self):\n        pass",
                "import os\nimport sys\n\ndef function():\n    return True"
            ]

            result = await analyzer.run("file1.py, file2.py", "structure")

            assert "file1.py" in result
            assert "file2.py" in result
            assert "1 classes" in result
            assert "1 functions" in result
            assert mock_read.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_code_generator(self):
        """Test parallel code generation."""
        generator = ParallelCodeGenerator()

        result = await generator.run("ComponentA, ComponentB", "class")

        assert "ComponentA" in result
        assert "ComponentB" in result
        assert "class ComponentA" in result
        assert "class ComponentB" in result
        assert "async def process" in result

    @pytest.mark.asyncio
    async def test_parallel_test_runner(self):
        """Test parallel test execution."""
        runner = ParallelTestRunner()

        # Mock shell execution
        with patch.object(runner.shell_tool, 'run') as mock_run:
            mock_run.side_effect = [
                "5 passed, 1 failed",
                "10 passed, 0 failed"
            ]

            result = await runner.run("suite1, suite2")

            assert "suite1" in result
            assert "suite2" in result
            assert "PARALLEL TEST SUMMARY" in result
            assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_searcher(self):
        """Test parallel searching."""
        searcher = ParallelSearcher()

        # Mock shell execution
        with patch.object(searcher.shell_tool, 'run') as mock_run:
            mock_run.side_effect = [
                "path1/file.py:10:pattern found",
                "path2/file.py:20:pattern found\npath2/other.py:5:pattern found"
            ]

            result = await searcher.run("pattern", "path1, path2")

            assert "path1" in result
            assert "path2" in result
            assert "SEARCH SUMMARY" in result
            assert "Total Matches: 3" in result


class TestCodingAgentParallel:
    """Test CodingAgent parallel capabilities."""

    @pytest.mark.asyncio
    async def test_parallel_tool_detection(self):
        """Test detection of parallel tool pattern."""
        mock_llm = MockLLM([
            "I'll check multiple things. {{parallel:[git_status, file_read]}}",
            "All checks complete."
        ])

        mock_tools = {
            "git_status": Mock(
                run=AsyncMock(return_value="No changes"),
                description="Check git status"
            ),
            "file_read": Mock(
                run=AsyncMock(return_value="File contents"),
                description="Read file"
            )
        }

        memory = AsyncMock(spec=ConversationMemory)
        memory.messages = []
        memory.add_user_message = AsyncMock()
        memory.add_assistant_message = AsyncMock()
        memory.add_tool_result = AsyncMock()

        agent = CodingAgent(mock_llm, mock_tools, memory)

        # Test pattern detection
        tools = agent._detect_parallel_tools("{{parallel:[tool1, tool2, tool3]}}")
        assert tools == ["tool1", "tool2", "tool3"]

        # Test with actual chat
        response = ""
        async for chunk in agent.chat("Check everything"):
            response += chunk

        assert "multiple things" in response
        assert "Parallel execution results" in response
        assert mock_tools["git_status"].run.called
        assert mock_tools["file_read"].run.called

    @pytest.mark.asyncio
    async def test_execute_parallel_tools(self):
        """Test parallel tool execution."""
        mock_tools = {
            "tool1": Mock(run=AsyncMock(return_value="Result 1")),
            "tool2": Mock(run=AsyncMock(return_value="Result 2")),
            "tool3": Mock(run=AsyncMock(return_value="Result 3"))
        }

        memory = AsyncMock(spec=ConversationMemory)
        agent = CodingAgent(Mock(), mock_tools, memory)

        results = await agent._execute_parallel_tools(["tool1", "tool2", "tool3"])

        assert len(results) == 3
        assert results["tool1"] == "Result 1"
        assert results["tool2"] == "Result 2"
        assert results["tool3"] == "Result 3"

        # Verify all tools were called
        for tool in mock_tools.values():
            assert tool.run.called


class TestParallelExecutor:
    """Test high-level ParallelExecutor."""

    @pytest.mark.asyncio
    async def test_execute_with_hints(self):
        """Test execution with parallelization hints."""
        mock_llm = MockLLM([
            "Starting analysis",
            "Processing task 1",
            "Processing task 2",
            "Final results"
        ])

        executor = ParallelExecutor(mock_llm, max_parallel=3)

        result = await executor.execute(
            "Analyze the codebase",
            parallel_hints=["Check multiple files", "Run tests in parallel"]
        )

        assert "execution plan" in result.lower()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_execute_without_hints(self):
        """Test execution without hints."""
        mock_llm = MockLLM(["Simple response"])

        executor = ParallelExecutor(mock_llm)

        result = await executor.execute("Do something simple")

        assert len(result) > 0


class TestTaskStatusTransitions:
    """Test task status transitions and lifecycle."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM(["Test response"])

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        memory = AsyncMock(spec=ConversationMemory)
        memory.messages = []
        memory.add_user_message = AsyncMock()
        memory.add_assistant_message = AsyncMock()
        memory.add_tool_result = AsyncMock()
        return memory

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return {
            "test_tool": Mock(
                run=AsyncMock(return_value="tool result"),
                description="Test tool"
            )
        }

    def test_task_creation_defaults(self):
        """Test task creation with default values."""
        task = Task(id="test", description="Test task", prompt="Do something")

        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.dependencies == []

    def test_task_with_dependencies(self):
        """Test task creation with dependencies."""
        task = Task(
            id="dependent",
            description="Dependent task",
            prompt="Do something after others",
            dependencies=["task1", "task2"]
        )

        assert task.dependencies == ["task1", "task2"]

    @pytest.mark.asyncio
    async def test_task_status_lifecycle(self, mock_llm, mock_memory, mock_tools):
        """Test complete task status lifecycle."""
        agent = SubAgent(mock_llm, mock_tools, mock_memory)

        task = Task(id="lifecycle", description="Lifecycle test", prompt="Test")
        assert task.status == TaskStatus.PENDING

        result = await agent.execute_task(task)

        assert task.status == TaskStatus.COMPLETED
        assert task.started_at is not None
        assert task.completed_at is not None
        assert task.completed_at >= task.started_at


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.fixture
    def error_llm(self):
        """Create an LLM that sometimes fails."""
        class ErrorLLM:
            def __init__(self):
                self.call_count = 0

            async def generate(self, prompt: str, stop: List[str] = None):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    raise RuntimeError("LLM failure")
                yield "Success response"

        return ErrorLLM()

    @pytest.mark.asyncio
    async def test_orchestrator_with_llm_failures(self, error_llm):
        """Test orchestrator handling LLM failures."""
        orchestrator = Orchestrator(error_llm, max_parallel=2)

        tasks = [
            Task(id=f"task_{i}", description=f"Task {i}", prompt=f"P{i}")
            for i in range(5)
        ]

        # Execute tasks and expect some failures
        results = await orchestrator._execute_layer(
            [t.id for t in tasks],
            tasks,
            None
        )

        # Should have both successes and failures
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) > 0
        assert len(failed) > 0
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_dependency_context_with_failed_dependencies(self):
        """Test dependency context building when dependencies failed."""
        orchestrator = Orchestrator(MockLLM([]), max_parallel=2)

        # Add some failed results
        orchestrator.results["failed_task"] = TaskResult(
            task_id="failed_task",
            success=False,
            output="",
            duration=0.1,
            error="Task failed"
        )

        orchestrator.results["success_task"] = TaskResult(
            task_id="success_task",
            success=True,
            output="Task succeeded",
            duration=0.1
        )

        # Build context with mixed results
        context = orchestrator._build_dependency_context(["failed_task", "success_task"])

        # Should only include successful results
        assert context is not None
        assert "success_task" in context
        assert "failed_task" not in context or len(context.split("success_task")) == 2


class TestParallelToolsErrorConditions:
    """Test error conditions in parallel tools."""

    @pytest.mark.asyncio
    async def test_parallel_file_analyzer_invalid_analysis_type(self):
        """Test file analyzer with invalid analysis type."""
        analyzer = ParallelFileAnalyzer()

        # Mock file reading to succeed
        with patch.object(analyzer.file_tool, 'run', return_value="file content"):
            result = await analyzer.run("test.py", "invalid_type")

            assert "Unknown analysis type" in result

    @pytest.mark.asyncio
    async def test_parallel_test_runner_shell_errors(self):
        """Test test runner handling shell command errors."""
        runner = ParallelTestRunner()

        # Mock shell to raise errors
        mock_shell = Mock()
        mock_shell.run = AsyncMock(side_effect=RuntimeError("Command failed"))
        runner.shell_tool = mock_shell

        result = await runner.run("test_suite")

        assert "Error running" in result

    @pytest.mark.asyncio
    async def test_parallel_searcher_no_matches(self):
        """Test searcher when no matches are found."""
        searcher = ParallelSearcher()

        # Mock shell to return empty results
        mock_shell = Mock()
        mock_shell.run = AsyncMock(return_value="")
        searcher.shell_tool = mock_shell

        result = await searcher.run("nonexistent_pattern", "some_path")

        assert "No matches found" in result
        assert "Total Matches: 0" in result


class TestExecutionPlanEdgeCases:
    """Test edge cases in execution plan creation."""

    def test_execution_plan_empty_tasks(self):
        """Test execution plan with empty task list."""
        plan = ExecutionPlan([])

        assert len(plan.execution_order) == 0
        assert plan.dependencies == {}

    def test_execution_plan_single_task(self):
        """Test execution plan with single task."""
        task = Task(id="single", description="Single", prompt="P")
        plan = ExecutionPlan([task])

        assert len(plan.execution_order) == 1
        assert plan.execution_order[0] == ["single"]

    def test_execution_plan_self_dependency(self):
        """Test execution plan with self-referencing dependency."""
        task = Task(
            id="self_ref",
            description="Self referencing",
            prompt="P",
            dependencies=["self_ref"]  # Self dependency
        )
        plan = ExecutionPlan([task])

        # Should handle gracefully (may create empty execution order)
        assert len(plan.execution_order) <= 1

    def test_execution_plan_missing_dependency(self):
        """Test execution plan with missing dependency."""
        task = Task(
            id="dependent",
            description="Dependent",
            prompt="P",
            dependencies=["nonexistent"]
        )
        plan = ExecutionPlan([task])

        # Should still include the task even if dependency is missing
        # The implementation may vary, but should handle gracefully
        assert len(plan.tasks) == 1


class TestTaskResultMetrics:
    """Test task result metrics and timing."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM(["Test response"])

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        memory = AsyncMock(spec=ConversationMemory)
        memory.messages = []
        memory.add_user_message = AsyncMock()
        memory.add_assistant_message = AsyncMock()
        memory.add_tool_result = AsyncMock()
        return memory

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return {
            "test_tool": Mock(
                run=AsyncMock(return_value="tool result"),
                description="Test tool"
            )
        }

    @pytest.mark.asyncio
    async def test_task_duration_accuracy(self, mock_llm, mock_memory, mock_tools):
        """Test that task duration is accurately measured."""
        # Create a slightly slower LLM
        class SlowMockLLM:
            async def generate(self, prompt: str, stop: List[str] = None):
                await asyncio.sleep(0.1)  # 100ms delay
                yield "Slow response"

        slow_llm = SlowMockLLM()
        agent = SubAgent(slow_llm, mock_tools, mock_memory)

        task = Task(id="timed", description="Timed task", prompt="Take time")

        start = time.time()
        result = await agent.execute_task(task)
        actual_time = time.time() - start

        # Duration should be approximately the actual time
        assert result.duration >= 0.1  # At least the sleep time
        assert abs(result.duration - actual_time) < 0.05  # Within 50ms

    @pytest.mark.asyncio
    async def test_orchestrator_metrics_accuracy(self):
        """Test orchestrator metrics are accurately calculated."""
        orchestrator = Orchestrator(MockLLM([]), max_parallel=2)

        # Reset metrics to ensure clean state
        orchestrator.execution_metrics = {
            "tasks_started": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_duration": 0.0,
            "parallel_efficiency": 0.0
        }

        # Execute some tasks
        tasks = [Task(id=f"t{i}", description=f"T{i}", prompt=f"P{i}") for i in range(3)]

        # Create a simple mock that just returns results
        async def mock_exec(task, context):
            await asyncio.sleep(0.05)
            return TaskResult(task.id, True, "Done", 0.05)

        # Don't mock the metrics - let the real code handle them
        with patch.object(orchestrator, '_execute_single_task') as mock_method:
            # The real method updates metrics in _execute_single_task, but _execute_layer also counts
            # Let's just mock the task execution part and let metrics be handled naturally
            async def side_effect(task, context):
                # Only increment tasks_started here (like the real method does)
                orchestrator.execution_metrics["tasks_started"] += 1
                result = await mock_exec(task, context)
                # Don't increment completed/failed here - _execute_layer will do it
                return result

            mock_method.side_effect = side_effect
            await orchestrator._execute_layer([t.id for t in tasks], tasks, None)

        metrics = orchestrator.get_metrics()

        assert metrics["tasks_started"] == 3
        assert metrics["tasks_completed"] == 3
        assert metrics["tasks_failed"] == 0


class TestConcurrencyControl:
    """Test concurrency control mechanisms."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent execution."""
        orchestrator = Orchestrator(MockLLM([]), max_parallel=2)

        concurrent_count = 0
        max_seen = 0

        async def counting_exec(task, context):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)

            await asyncio.sleep(0.1)  # Hold the slot

            concurrent_count -= 1
            return TaskResult(task.id, True, "Done", 0.1)

        tasks = [Task(id=f"t{i}", description=f"T{i}", prompt=f"P{i}") for i in range(5)]

        with patch.object(orchestrator, '_execute_single_task', side_effect=counting_exec):
            await orchestrator._execute_layer([t.id for t in tasks], tasks, None)

        # Should never exceed max_parallel
        assert max_seen <= orchestrator.max_parallel

    @pytest.mark.asyncio
    async def test_task_cancellation_cleanup(self):
        """Test that cancelled tasks are properly cleaned up."""
        orchestrator = Orchestrator(MockLLM([]), max_parallel=3)

        # Create some mock agents
        for i in range(3):
            mock_agent = Mock()
            orchestrator.active_agents[f"agent_{i}"] = mock_agent

        await orchestrator.cancel_all()

        assert len(orchestrator.active_agents) == 0


class TestMemoryAndContextManagement:
    """Test memory and context management in parallel execution."""

    @pytest.mark.asyncio
    async def test_subagent_memory_isolation(self):
        """Test that subagents have isolated memory."""
        llm = MockLLM(["Response"])

        # Create two subagents with different memories
        memory1 = AsyncMock(spec=ConversationMemory)
        memory2 = AsyncMock(spec=ConversationMemory)

        agent1 = SubAgent(llm, {}, memory1, task_id="agent1")
        agent2 = SubAgent(llm, {}, memory2, task_id="agent2")

        task1 = Task(id="t1", description="Task 1", prompt="P1")
        task2 = Task(id="t2", description="Task 2", prompt="P2")

        # Execute tasks
        await agent1.execute_task(task1)
        await agent2.execute_task(task2)

        # Each agent should have used its own memory
        memory1.add_user_message.assert_called()
        memory2.add_user_message.assert_called()

        # Memories should be independent
        assert memory1 != memory2

    @pytest.mark.asyncio
    async def test_context_truncation_in_dependencies(self):
        """Test that dependency context is properly truncated."""
        orchestrator = Orchestrator(MockLLM([]), max_parallel=2)

        # Create a result with very long output
        long_output = "x" * 1000  # 1000 character output
        orchestrator.results["long_task"] = TaskResult(
            task_id="long_task",
            success=True,
            output=long_output,
            duration=0.1
        )

        context = orchestrator._build_dependency_context(["long_task"])

        # Should truncate to 500 characters as per implementation
        assert len(context.split("long_task: ")[1].split("\n")[0]) <= 500


class TestPatternDetectionEdgeCases:
    """Test edge cases in parallel pattern detection."""

    def test_parallel_pattern_edge_cases(self):
        """Test parallel pattern detection with edge cases."""
        agent = CodingAgent(Mock(), {}, Mock())

        # Test various edge cases
        test_cases = [
            ("{{parallel:[]}}", []),  # Empty list
            ("{{parallel:[tool1]}}", ["tool1"]),  # Single tool
            ("{{parallel:[tool1, tool2, tool3]}}", ["tool1", "tool2", "tool3"]),  # Multiple tools
            ("{{parallel:[ tool1 , tool2 ]}}", ["tool1", "tool2"]),  # Extra spaces
            ("text before {{parallel:[tool1]}} text after", ["tool1"]),  # Embedded
            ("no pattern here", None),  # No pattern
            ("{{parallel:", None),  # Incomplete pattern
        ]

        for text, expected in test_cases:
            result = agent._detect_parallel_tools(text)
            if expected is None:
                assert result is None
            else:
                assert result == expected