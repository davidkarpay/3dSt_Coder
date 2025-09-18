"""Integration tests for parallel subagent architecture."""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from agent.subagent import SubAgent, Task, TaskStatus, TaskResult
from agent.orchestrator import Orchestrator, ExecutionPlan, ParallelExecutor
from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools import get_available_tools
from agent.tools.parallel import (
    ParallelFileAnalyzer,
    ParallelTestRunner,
    ParallelCodeGenerator,
    ParallelSearcher
)


class TestOrchestratorIntegration:
    """Integration tests for Orchestrator with real SubAgents."""

    @pytest.fixture
    def real_memory(self):
        """Create real memory instance."""
        return ConversationMemory(max_tokens=4096, project_id="test_integration")

    @pytest.fixture
    def mock_llm_with_responses(self):
        """Create mock LLM with realistic responses."""
        class MockLLM:
            def __init__(self):
                self.response_index = 0
                self.responses = [
                    "Analyzing the task and breaking it down into subtasks...",
                    "Processing subtask 1: Analyzing code structure",
                    "Processing subtask 2: Running tests",
                    "Processing subtask 3: Generating documentation",
                    "Aggregating all results into final output"
                ]

            async def generate(self, prompt: str, stop: List[str] = None):
                if self.response_index < len(self.responses):
                    response = self.responses[self.response_index]
                    self.response_index += 1
                    # Simulate streaming
                    for word in response.split():
                        yield word + " "
                        await asyncio.sleep(0.01)  # Small delay to simulate streaming
                else:
                    yield "Task completed successfully."

        return MockLLM()

    @pytest.mark.asyncio
    async def test_orchestrator_with_dependencies(self, mock_llm_with_responses):
        """Test orchestrator handling tasks with dependencies."""
        orchestrator = Orchestrator(
            llm=mock_llm_with_responses,
            max_parallel=3,
            timeout_per_task=10.0
        )

        # Create tasks with dependencies
        tasks = [
            Task(id="analyze", description="Analyze codebase", prompt="Analyze the code structure"),
            Task(id="test", description="Run tests", prompt="Execute all tests"),
            Task(id="report", description="Generate report", prompt="Create summary report",
                 dependencies=["analyze", "test"]),
            Task(id="deploy", description="Deploy changes", prompt="Deploy to staging",
                 dependencies=["report"])
        ]

        # Execute using the orchestrator's internal method
        plan = ExecutionPlan(tasks)

        # Verify execution order
        assert len(plan.execution_order) == 3
        assert set(plan.execution_order[0]) == {"analyze", "test"}  # First layer - parallel
        assert plan.execution_order[1] == ["report"]  # Second layer - depends on first
        assert plan.execution_order[2] == ["deploy"]  # Third layer - depends on report

        # Execute first layer
        results = await orchestrator._execute_layer(
            plan.execution_order[0],
            tasks,
            "Integration test context"
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.duration > 0 for r in results)

    @pytest.mark.asyncio
    async def test_orchestrator_parallel_execution_timing(self, mock_llm_with_responses):
        """Test that parallel execution is faster than sequential."""
        orchestrator = Orchestrator(
            llm=mock_llm_with_responses,
            max_parallel=5
        )

        # Create multiple independent tasks
        tasks = [
            Task(id=f"task_{i}", description=f"Task {i}", prompt=f"Execute task {i}")
            for i in range(5)
        ]

        # Mock execute_single_task to simulate work
        async def mock_execute(task, context):
            await asyncio.sleep(0.1)  # Simulate 100ms of work
            return TaskResult(
                task_id=task.id,
                success=True,
                output=f"Completed {task.id}",
                duration=0.1
            )

        with patch.object(orchestrator, '_execute_single_task', side_effect=mock_execute):
            # Time parallel execution
            start = time.time()
            parallel_results = await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )
            parallel_time = time.time() - start

            # All tasks should complete in roughly the same time (parallel)
            # With 5 tasks of 100ms each, parallel should be ~100ms
            assert parallel_time < 0.3  # Allow some overhead
            assert len(parallel_results) == 5
            assert all(r.success for r in parallel_results)

    @pytest.mark.asyncio
    async def test_orchestrator_with_failing_tasks(self, mock_llm_with_responses):
        """Test orchestrator handling partial failures."""
        orchestrator = Orchestrator(llm=mock_llm_with_responses)

        tasks = [
            Task(id="success1", description="Successful task 1", prompt="Do something"),
            Task(id="failure", description="Failing task", prompt="This will fail"),
            Task(id="success2", description="Successful task 2", prompt="Do something else")
        ]

        # Mock to make one task fail
        async def mock_execute(task, context):
            if task.id == "failure":
                raise RuntimeError("Task intentionally failed")
            return TaskResult(
                task_id=task.id,
                success=True,
                output=f"Completed {task.id}",
                duration=0.1
            )

        with patch.object(orchestrator, '_execute_single_task', side_effect=mock_execute):
            results = await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )

            assert len(results) == 3
            success_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            assert len(success_results) == 2
            assert len(failed_results) == 1
            assert "intentionally failed" in failed_results[0].error

    @pytest.mark.asyncio
    async def test_context_passing_between_layers(self, mock_llm_with_responses, real_memory):
        """Test that context is properly passed between dependent tasks."""
        orchestrator = Orchestrator(llm=mock_llm_with_responses)

        # First layer task
        task1 = Task(id="task1", description="First task", prompt="Generate data")

        # Create subagent for first task
        subagent1 = SubAgent(
            mock_llm_with_responses,
            get_available_tools(),
            real_memory,
            parent_id="orchestrator",
            task_id="task1"
        )

        # Execute first task
        result1 = await subagent1.execute_task(task1)
        orchestrator.results[task1.id] = result1

        # Test dependency context building
        context = orchestrator._build_dependency_context(["task1"])

        assert context is not None
        assert "Previous task results:" in context
        assert "task1:" in context

    @pytest.mark.asyncio
    async def test_orchestrator_cancellation(self, mock_llm_with_responses):
        """Test cancelling all active subagents."""
        orchestrator = Orchestrator(llm=mock_llm_with_responses)

        # Create mock active agents
        for i in range(3):
            mock_agent = Mock(spec=SubAgent)
            orchestrator.active_agents[f"agent_{i}"] = mock_agent

        # Cancel all
        await orchestrator.cancel_all()

        assert len(orchestrator.active_agents) == 0


class TestParallelToolsIntegration:
    """Integration tests for parallel tools working together."""

    @pytest.mark.asyncio
    async def test_parallel_file_analyzer_with_real_files(self, tmp_path):
        """Test parallel file analyzer with actual file operations."""
        # Create test files
        file1 = tmp_path / "test1.py"
        file1.write_text("""
class TestClass:
    def method1(self):
        pass

    def method2(self):
        return True

def standalone_function():
    import os
    return os.path.exists('.')
""")

        file2 = tmp_path / "test2.py"
        file2.write_text("""
import sys
import asyncio
from typing import List

async def async_function():
    await asyncio.sleep(0.1)

class AnotherClass:
    pass
""")

        analyzer = ParallelFileAnalyzer(base_path=tmp_path)

        # Mock the file tool to return the actual content
        file_contents = {
            str(file1): file1.read_text(),
            str(file2): file2.read_text()
        }

        async def mock_file_read(path: str):
            return file_contents.get(path, f"File not found: {path}")

        with patch.object(analyzer.file_tool, 'run', side_effect=mock_file_read):
            # Analyze both files
            result = await analyzer.run(
                files=f"{file1.name}, {file2.name}",
                analysis_type="structure"
            )

        assert "test1.py" in result
        assert "test2.py" in result
        assert "1 classes" in result  # test1.py has 1 class
        assert "3 functions" in result  # test1.py has 3 functions (method1, method2, standalone_function)
        assert "imports" in result

    @pytest.mark.asyncio
    async def test_parallel_code_generator_consistency(self):
        """Test that parallel code generation produces consistent results."""
        generator = ParallelCodeGenerator()

        # Generate multiple components
        result = await generator.run(
            components="UserService, AuthService, DataService",
            template="class"
        )

        # Check all components were generated
        assert "UserService" in result
        assert "AuthService" in result
        assert "DataService" in result

        # Check consistent structure
        assert result.count("@dataclass") == 3
        assert result.count("async def process") == 3
        assert result.count("def validate") == 3

    @pytest.mark.asyncio
    async def test_parallel_tools_error_handling(self):
        """Test error handling in parallel tools."""
        analyzer = ParallelFileAnalyzer()

        # Mock the file tool to raise errors for non-existent files
        async def mock_file_read_error(path: str):
            raise FileNotFoundError(f"File '{path}' does not exist")

        with patch.object(analyzer.file_tool, 'run', side_effect=mock_file_read_error):
            # Try to analyze non-existent files
            result = await analyzer.run(
                files="nonexistent1.py, nonexistent2.py",
                analysis_type="structure"
            )

        assert "Error analyzing" in result
        assert "nonexistent1.py" in result
        assert "nonexistent2.py" in result


class TestCodingAgentParallelIntegration:
    """Integration tests for CodingAgent with parallel capabilities."""

    @pytest.fixture
    def agent_with_tools(self):
        """Create agent with all tools including parallel ones."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock()

        memory = ConversationMemory(max_tokens=2048)
        tools = get_available_tools()

        return CodingAgent(mock_llm, tools, memory)

    @pytest.mark.asyncio
    async def test_parallel_tool_registration(self, agent_with_tools):
        """Test that parallel tools are properly registered."""
        tools = agent_with_tools.tools

        # Check that parallel tools are available
        assert "parallel_file_analyzer" in tools
        assert "parallel_test_runner" in tools
        assert "parallel_code_generator" in tools
        assert "parallel_searcher" in tools

    @pytest.mark.asyncio
    async def test_parallel_pattern_detection_integration(self, agent_with_tools):
        """Test detection and execution of parallel tools in agent."""
        # Mock LLM to generate parallel tool pattern
        responses = [
            "Let me check multiple things {{parallel:[git_status, file_read]}}",
            "Done checking."
        ]
        response_index = 0

        async def mock_generate(prompt: str, stop: List[str] = None):
            nonlocal response_index
            if response_index < len(responses):
                text = responses[response_index]
                response_index += 1
                async for chunk in self._async_generator(text):
                    yield chunk
            else:
                yield "Default response"

        agent_with_tools.llm.generate = mock_generate

        # Mock tool execution
        with patch.object(agent_with_tools.tools["git_status"], 'run',
                         new_callable=AsyncMock, return_value="No changes"):
            with patch.object(agent_with_tools.tools["file_read"], 'run',
                             new_callable=AsyncMock, return_value="File content"):

                response = ""
                async for chunk in agent_with_tools.chat("Check everything"):
                    response += chunk

                assert "multiple things" in response
                assert "Parallel execution results" in response

    async def _async_generator(self, text):
        """Helper to create async generator from text."""
        import re
        # Split on spaces but preserve {{...}} patterns
        parts = re.split(r'(\{\{[^}]+\}\})', text)
        for part in parts:
            if part.strip():
                if part.startswith('{{') and part.endswith('}}'):
                    # Keep special patterns intact
                    yield part + " "
                else:
                    # Split normal text into words
                    for word in part.split():
                        yield word + " "


class TestExecutionPlanIntegration:
    """Integration tests for ExecutionPlan with complex dependencies."""

    def test_execution_plan_with_complex_dependencies(self):
        """Test execution plan with complex dependency graph."""
        tasks = [
            # Layer 1: Independent tasks
            Task(id="fetch_data", description="Fetch data", prompt="Get data from source"),
            Task(id="load_config", description="Load config", prompt="Load configuration"),

            # Layer 2: Depends on layer 1
            Task(id="process_data", description="Process data", prompt="Process fetched data",
                 dependencies=["fetch_data", "load_config"]),
            Task(id="validate_config", description="Validate config", prompt="Validate configuration",
                 dependencies=["load_config"]),

            # Layer 3: Depends on layer 2
            Task(id="analyze", description="Analyze", prompt="Analyze processed data",
                 dependencies=["process_data"]),
            Task(id="optimize", description="Optimize", prompt="Optimize based on validation",
                 dependencies=["validate_config", "process_data"]),

            # Layer 4: Final aggregation
            Task(id="report", description="Generate report", prompt="Create final report",
                 dependencies=["analyze", "optimize"])
        ]

        plan = ExecutionPlan(tasks)

        # Verify correct layering
        assert len(plan.execution_order) == 4

        # Layer 1: Independent tasks
        assert set(plan.execution_order[0]) == {"fetch_data", "load_config"}

        # Layer 2: Tasks depending on layer 1
        assert set(plan.execution_order[1]) == {"process_data", "validate_config"}

        # Layer 3: Tasks depending on layer 2
        assert set(plan.execution_order[2]) == {"analyze", "optimize"}

        # Layer 4: Final task
        assert plan.execution_order[3] == ["report"]

    def test_execution_plan_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        tasks = [
            Task(id="task1", description="Task 1", prompt="Do task 1", dependencies=["task2"]),
            Task(id="task2", description="Task 2", prompt="Do task 2", dependencies=["task3"]),
            Task(id="task3", description="Task 3", prompt="Do task 3", dependencies=["task1"])  # Circular!
        ]

        plan = ExecutionPlan(tasks)

        # Should detect circular dependency and have incomplete execution order
        assert len(plan.execution_order) == 0  # No valid execution order possible

    def test_execution_plan_with_isolated_groups(self):
        """Test execution plan with multiple isolated task groups."""
        tasks = [
            # Group 1
            Task(id="group1_task1", description="G1T1", prompt="p1"),
            Task(id="group1_task2", description="G1T2", prompt="p2", dependencies=["group1_task1"]),

            # Group 2 (isolated from group 1)
            Task(id="group2_task1", description="G2T1", prompt="p3"),
            Task(id="group2_task2", description="G2T2", prompt="p4", dependencies=["group2_task1"]),

            # Group 3 (isolated from others)
            Task(id="group3_task1", description="G3T1", prompt="p5")
        ]

        plan = ExecutionPlan(tasks)

        # Should have 2 layers
        assert len(plan.execution_order) == 2

        # First layer: all independent tasks
        assert set(plan.execution_order[0]) == {"group1_task1", "group2_task1", "group3_task1"}

        # Second layer: dependent tasks
        assert set(plan.execution_order[1]) == {"group1_task2", "group2_task2"}


class TestParallelExecutorIntegration:
    """Integration tests for high-level ParallelExecutor."""

    @pytest.mark.asyncio
    async def test_parallel_executor_end_to_end(self):
        """Test ParallelExecutor with full workflow."""
        class MockLLM:
            def __init__(self):
                self.call_count = 0

            async def generate(self, prompt: str, stop: List[str] = None):
                self.call_count += 1
                responses = [
                    "Breaking down the task into parallel components...",
                    "Executing component 1...",
                    "Executing component 2...",
                    "Aggregating results..."
                ]

                if self.call_count <= len(responses):
                    response = responses[self.call_count - 1]
                else:
                    response = "Task completed."

                for word in response.split():
                    yield word + " "

        mock_llm = MockLLM()
        executor = ParallelExecutor(llm=mock_llm, max_parallel=3)

        # Execute with hints
        result = await executor.execute(
            prompt="Analyze the entire codebase and generate documentation",
            parallel_hints=[
                "Analyze multiple modules in parallel",
                "Generate docs for each component simultaneously"
            ]
        )

        assert len(result) > 0
        assert "execution plan" in result.lower() or "breaking down" in result.lower()

    @pytest.mark.asyncio
    async def test_parallel_executor_resource_limits(self):
        """Test ParallelExecutor respects resource limits."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock()

        executor = ParallelExecutor(llm=mock_llm, max_parallel=2)

        # Create orchestrator through executor
        with patch('agent.orchestrator.Orchestrator') as MockOrchestrator:
            mock_instance = MockOrchestrator.return_value

            # Mock execute_parallel as async generator
            async def mock_execute_parallel(prompt, context=None):
                yield "Mocked execution result"

            mock_instance.execute_parallel = mock_execute_parallel

            await executor.execute("Test task")

            # Verify max_parallel was passed correctly
            MockOrchestrator.assert_called_with(mock_llm, max_parallel=2)


@pytest.mark.asyncio
async def test_full_integration_workflow():
    """Test complete workflow from task decomposition to result aggregation."""

    # Create a mock LLM that simulates realistic responses
    class WorkflowMockLLM:
        def __init__(self):
            self.stage = 0

        async def generate(self, prompt: str, stop: List[str] = None):
            self.stage += 1

            if "decompose" in prompt.lower() or self.stage == 1:
                response = "I'll analyze files, run tests, and generate docs in parallel"
            elif "analyze" in prompt.lower():
                response = "Found 10 classes and 50 functions"
            elif "test" in prompt.lower():
                response = "All 25 tests passed"
            elif "doc" in prompt.lower():
                response = "Generated documentation for 15 modules"
            elif "aggregate" in prompt.lower() or "summary" in prompt.lower():
                response = "Summary: Analyzed 10 classes, 50 functions. All tests passed. Documented 15 modules."
            else:
                response = "Task processing..."

            for word in response.split():
                yield word + " "
                await asyncio.sleep(0.001)

    llm = WorkflowMockLLM()
    orchestrator = Orchestrator(llm, max_parallel=3)

    # Execute a complex task
    full_output = []
    async for chunk in orchestrator.execute_parallel(
        "Analyze the codebase, run all tests, and generate documentation",
        context="This is a Python project with multiple modules"
    ):
        full_output.append(chunk)

    result = "".join(full_output)

    # Verify the workflow executed correctly
    assert "execution plan" in result.lower() or "analyzing" in result.lower()
    assert len(result) > 0

    # Check metrics were updated
    metrics = orchestrator.get_metrics()
    assert metrics["tasks_started"] > 0
    assert metrics["total_duration"] >= 0