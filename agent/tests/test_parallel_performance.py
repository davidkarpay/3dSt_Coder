"""Performance and stress tests for parallel subagent architecture."""

import pytest
import asyncio
import time
import random
import psutil
import gc
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from dataclasses import dataclass

from agent.subagent import SubAgent, Task, TaskStatus, TaskResult
from agent.orchestrator import Orchestrator, ExecutionPlan, ParallelExecutor
from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools import get_available_tools


@dataclass
class PerformanceMetrics:
    """Performance metrics for test analysis."""
    execution_time: float
    memory_usage_mb: float
    tasks_per_second: float
    parallel_efficiency: float
    peak_memory_mb: float
    cpu_usage_percent: float


class TestParallelPerformance:
    """Performance tests for parallel execution."""

    @pytest.fixture
    def fast_mock_llm(self):
        """Create a fast mock LLM for performance testing."""
        class FastMockLLM:
            async def generate(self, prompt: str, stop: List[str] = None):
                # Minimal delay to simulate fast processing
                await asyncio.sleep(0.001)
                yield "Quick response"

        return FastMockLLM()

    @pytest.fixture
    def slow_mock_llm(self):
        """Create a slow mock LLM for performance testing."""
        class SlowMockLLM:
            async def generate(self, prompt: str, stop: List[str] = None):
                # Simulate slower processing
                await asyncio.sleep(0.1)
                yield "Slow response after processing"

        return SlowMockLLM()

    def measure_memory(self):
        """Measure current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def measure_cpu(self):
        """Measure current CPU usage."""
        return psutil.cpu_percent(interval=0.1)

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, fast_mock_llm):
        """Compare parallel vs sequential execution performance."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=5)

        # Create 10 independent tasks
        tasks = [
            Task(id=f"task_{i}", description=f"Task {i}", prompt=f"Process item {i}")
            for i in range(10)
        ]

        # Mock the actual task execution with controlled timing
        async def mock_task_execution(task, context):
            await asyncio.sleep(0.05)  # 50ms per task
            return TaskResult(
                task_id=task.id,
                success=True,
                output=f"Completed {task.id}",
                duration=0.05
            )

        with patch.object(orchestrator, '_execute_single_task', side_effect=mock_task_execution):
            # Measure parallel execution
            start_parallel = time.time()
            parallel_results = await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )
            parallel_time = time.time() - start_parallel

            # Calculate sequential time (would be sum of all task times)
            sequential_time = len(tasks) * 0.05

            # Calculate efficiency
            parallel_efficiency = sequential_time / parallel_time

            # Parallel should be significantly faster
            assert parallel_time < sequential_time
            assert parallel_efficiency > 1.5  # At least 1.5x speedup
            assert len(parallel_results) == 10
            assert all(r.success for r in parallel_results)

            print(f"\nPerformance Results:")
            print(f"Sequential time (theoretical): {sequential_time:.2f}s")
            print(f"Parallel time (actual): {parallel_time:.2f}s")
            print(f"Speedup: {parallel_efficiency:.2f}x")

    @pytest.mark.asyncio
    async def test_max_parallel_limit_enforcement(self, fast_mock_llm):
        """Test that max_parallel limit is properly enforced."""
        max_parallel = 3
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=max_parallel)

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_task_with_tracking(task, context):
            nonlocal concurrent_count, max_concurrent

            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.1)  # Simulate work

            async with lock:
                concurrent_count -= 1

            return TaskResult(task.id, True, "Done", 0.1)

        with patch.object(orchestrator, '_execute_single_task', side_effect=mock_task_with_tracking):
            # Create more tasks than max_parallel
            tasks = [Task(id=f"task_{i}", description=f"T{i}", prompt=f"P{i}") for i in range(10)]

            await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )

            # Max concurrent should not exceed limit
            assert max_concurrent <= max_parallel
            print(f"\nMax concurrent tasks: {max_concurrent} (limit: {max_parallel})")

    @pytest.mark.asyncio
    async def test_memory_usage_with_many_agents(self, fast_mock_llm):
        """Test memory usage with many concurrent agents."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=10)

        # Measure initial memory
        gc.collect()
        initial_memory = self.measure_memory()

        # Create many tasks
        num_tasks = 50
        tasks = [
            Task(id=f"task_{i}", description=f"Task {i}", prompt=f"Process {i}")
            for i in range(num_tasks)
        ]

        # Track peak memory
        peak_memory = initial_memory
        memory_samples = []

        async def mock_task_with_memory(task, context):
            nonlocal peak_memory

            # Simulate memory allocation
            data = [random.random() for _ in range(1000)]  # Allocate some memory

            current_memory = self.measure_memory()
            memory_samples.append(current_memory)
            peak_memory = max(peak_memory, current_memory)

            await asyncio.sleep(0.01)
            return TaskResult(task.id, True, str(data[:10]), 0.01)

        with patch.object(orchestrator, '_execute_single_task', side_effect=mock_task_with_memory):
            await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )

        # Measure final memory
        gc.collect()
        final_memory = self.measure_memory()

        memory_increase = peak_memory - initial_memory
        memory_leaked = final_memory - initial_memory

        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.2f} MB")
        print(f"Peak: {peak_memory:.2f} MB")
        print(f"Final: {final_memory:.2f} MB")
        print(f"Peak increase: {memory_increase:.2f} MB")
        print(f"Potential leak: {memory_leaked:.2f} MB")

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB for 50 tasks
        # Memory should be mostly released after completion
        assert memory_leaked < 20  # Less than 20MB retained

    @pytest.mark.asyncio
    async def test_task_timeout_performance(self, slow_mock_llm):
        """Test performance impact of task timeouts."""
        orchestrator = Orchestrator(slow_mock_llm, max_parallel=5, timeout_per_task=0.05)

        # Create mix of fast and slow tasks
        async def mixed_speed_execution(task, context):
            if "slow" in task.id:
                await asyncio.sleep(0.2)  # Will timeout
            else:
                await asyncio.sleep(0.01)  # Will complete

            return TaskResult(task.id, True, "Done", 0.01)

        tasks = []
        for i in range(10):
            task_type = "slow" if i % 3 == 0 else "fast"
            tasks.append(Task(id=f"{task_type}_{i}", description=f"Task {i}", prompt=f"P{i}"))

        with patch.object(orchestrator, '_execute_single_task', side_effect=mixed_speed_execution):
            start_time = time.time()
            results = await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )
            execution_time = time.time() - start_time

        # Check timeout handling
        timed_out = [r for r in results if not r.success and "timed out" in str(r.error)]
        successful = [r for r in results if r.success]

        print(f"\nTimeout Performance:")
        print(f"Total tasks: {len(tasks)}")
        print(f"Successful: {len(successful)}")
        print(f"Timed out: {len(timed_out)}")
        print(f"Execution time: {execution_time:.2f}s")

        # Execution should not wait for all slow tasks
        assert execution_time < 0.3  # Should finish quickly due to timeout
        assert len(timed_out) > 0  # Some tasks should timeout

    @pytest.mark.asyncio
    async def test_scalability_with_increasing_tasks(self, fast_mock_llm):
        """Test how performance scales with increasing number of tasks."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=10)

        task_counts = [10, 20, 50, 100]
        results = []

        for count in task_counts:
            tasks = [
                Task(id=f"task_{i}", description=f"T{i}", prompt=f"P{i}")
                for i in range(count)
            ]

            # Simple mock execution
            async def mock_exec(task, context):
                await asyncio.sleep(0.01)
                return TaskResult(task.id, True, "Done", 0.01)

            with patch.object(orchestrator, '_execute_single_task', side_effect=mock_exec):
                start = time.time()
                await orchestrator._execute_layer(
                    [t.id for t in tasks],
                    tasks,
                    None
                )
                elapsed = time.time() - start

                tasks_per_second = count / elapsed
                results.append({
                    'count': count,
                    'time': elapsed,
                    'tps': tasks_per_second
                })

        print("\nScalability Results:")
        for r in results:
            print(f"{r['count']} tasks: {r['time']:.2f}s ({r['tps']:.1f} tasks/sec)")

        # Performance should scale reasonably
        # Tasks per second should not degrade significantly
        assert all(r['tps'] > 50 for r in results)  # At least 50 tasks/sec


class TestStressConditions:
    """Stress tests for extreme conditions."""

    @pytest.fixture
    def fast_mock_llm(self):
        """Create a fast mock LLM for performance testing."""
        class FastMockLLM:
            async def generate(self, prompt: str, stop: List[str] = None):
                # Minimal delay to simulate fast processing
                await asyncio.sleep(0.001)
                yield "Quick response"

        return FastMockLLM()

    @pytest.mark.asyncio
    async def test_many_layers_of_dependencies(self):
        """Test with deep dependency chains."""
        # Create a chain of dependent tasks
        tasks = []
        for i in range(20):  # 20 layers deep
            if i == 0:
                task = Task(id=f"task_{i}", description=f"T{i}", prompt=f"P{i}")
            else:
                task = Task(
                    id=f"task_{i}",
                    description=f"T{i}",
                    prompt=f"P{i}",
                    dependencies=[f"task_{i-1}"]
                )
            tasks.append(task)

        plan = ExecutionPlan(tasks)

        # Should create 20 layers
        assert len(plan.execution_order) == 20
        # Each layer should have exactly one task
        assert all(len(layer) == 1 for layer in plan.execution_order)

    @pytest.mark.asyncio
    async def test_wide_parallel_execution(self, fast_mock_llm):
        """Test with many tasks executing in parallel."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=50)

        # Create 100 independent tasks
        tasks = [
            Task(id=f"task_{i}", description=f"T{i}", prompt=f"P{i}")
            for i in range(100)
        ]

        async def quick_exec(task, context):
            await asyncio.sleep(0.001)
            return TaskResult(task.id, True, "Done", 0.001)

        with patch.object(orchestrator, '_execute_single_task', side_effect=quick_exec):
            start = time.time()
            results = await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )
            elapsed = time.time() - start

        assert len(results) == 100
        assert all(r.success for r in results)
        print(f"\n100 parallel tasks completed in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_mixed_success_failure_performance(self, fast_mock_llm):
        """Test performance with mixed success and failure scenarios."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=10)

        # Create tasks that will have mixed results
        tasks = [Task(id=f"task_{i}", description=f"T{i}", prompt=f"P{i}") for i in range(50)]

        async def mixed_results(task, context):
            task_num = int(task.id.split('_')[1])

            # 20% fail quickly
            if task_num % 5 == 0:
                raise RuntimeError(f"Task {task.id} failed")

            # 20% are slow
            if task_num % 5 == 1:
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)

            return TaskResult(task.id, True, "Success", 0.01)

        with patch.object(orchestrator, '_execute_single_task', side_effect=mixed_results):
            start = time.time()
            results = await orchestrator._execute_layer(
                [t.id for t in tasks],
                tasks,
                None
            )
            elapsed = time.time() - start

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\nMixed Results Performance:")
        print(f"Total: {len(results)}, Success: {len(successful)}, Failed: {len(failed)}")
        print(f"Completed in {elapsed:.2f}s")

        assert len(results) == 50
        assert len(failed) == 10  # 20% should fail
        assert elapsed < 1.0  # Should still complete quickly

    @pytest.mark.asyncio
    async def test_resource_cleanup_after_cancellation(self, fast_mock_llm):
        """Test that resources are properly cleaned up after cancellation."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=5)

        # Track resource allocation
        allocated_resources = []

        async def task_with_resources(task, context):
            # Simulate resource allocation
            resource = f"Resource_{task.id}"
            allocated_resources.append(resource)

            try:
                await asyncio.sleep(0.5)  # Long running task
                return TaskResult(task.id, True, "Done", 0.5)
            finally:
                # Cleanup should happen
                if resource in allocated_resources:
                    allocated_resources.remove(resource)

        tasks = [Task(id=f"task_{i}", description=f"T{i}", prompt=f"P{i}") for i in range(10)]

        with patch.object(orchestrator, '_execute_single_task', side_effect=task_with_resources):
            # Start execution in background
            execution_task = asyncio.create_task(
                orchestrator._execute_layer([t.id for t in tasks], tasks, None)
            )

            # Let some tasks start
            await asyncio.sleep(0.1)

            # Cancel execution
            execution_task.cancel()

            try:
                await execution_task
            except asyncio.CancelledError:
                pass

            # Give time for cleanup
            await asyncio.sleep(0.1)

        # Resources should be cleaned up
        assert len(allocated_resources) == 0
        print(f"\nResource cleanup successful after cancellation")


class TestConcurrencyPatterns:
    """Test various concurrency patterns and edge cases."""

    @pytest.fixture
    def fast_mock_llm(self):
        """Create a fast mock LLM for performance testing."""
        class FastMockLLM:
            async def generate(self, prompt: str, stop: List[str] = None):
                # Minimal delay to simulate fast processing
                await asyncio.sleep(0.001)
                yield "Quick response"

        return FastMockLLM()

    @pytest.mark.asyncio
    async def test_fan_out_fan_in_pattern(self):
        """Test fan-out/fan-in dependency pattern."""
        # Create a fan-out/fan-in pattern:
        # task1 -> [task2, task3, task4] -> task5

        tasks = [
            Task(id="source", description="Source task", prompt="Start"),
            Task(id="parallel1", description="P1", prompt="P1", dependencies=["source"]),
            Task(id="parallel2", description="P2", prompt="P2", dependencies=["source"]),
            Task(id="parallel3", description="P3", prompt="P3", dependencies=["source"]),
            Task(id="sink", description="Sink", prompt="End",
                 dependencies=["parallel1", "parallel2", "parallel3"])
        ]

        plan = ExecutionPlan(tasks)

        # Should have 3 layers
        assert len(plan.execution_order) == 3
        assert plan.execution_order[0] == ["source"]
        assert set(plan.execution_order[1]) == {"parallel1", "parallel2", "parallel3"}
        assert plan.execution_order[2] == ["sink"]

    @pytest.mark.asyncio
    async def test_diamond_dependency_pattern(self):
        """Test diamond dependency pattern."""
        # Create diamond pattern:
        #     task1
        #    /      \
        #  task2   task3
        #    \      /
        #     task4

        tasks = [
            Task(id="top", description="Top", prompt="Start"),
            Task(id="left", description="Left", prompt="L", dependencies=["top"]),
            Task(id="right", description="Right", prompt="R", dependencies=["top"]),
            Task(id="bottom", description="Bottom", prompt="End", dependencies=["left", "right"])
        ]

        plan = ExecutionPlan(tasks)

        assert len(plan.execution_order) == 3
        assert plan.execution_order[0] == ["top"]
        assert set(plan.execution_order[1]) == {"left", "right"}
        assert plan.execution_order[2] == ["bottom"]

    @pytest.mark.asyncio
    async def test_complex_dag_execution(self, fast_mock_llm):
        """Test execution of a complex directed acyclic graph."""
        orchestrator = Orchestrator(fast_mock_llm, max_parallel=10)

        # Create a complex DAG
        tasks = [
            # Layer 0
            Task(id="init", description="Initialize", prompt="Start"),

            # Layer 1
            Task(id="load_data", description="Load", prompt="Load", dependencies=["init"]),
            Task(id="load_config", description="Config", prompt="Config", dependencies=["init"]),

            # Layer 2
            Task(id="validate_data", description="Validate", prompt="V",
                 dependencies=["load_data"]),
            Task(id="preprocess", description="Preprocess", prompt="P",
                 dependencies=["load_data", "load_config"]),

            # Layer 3
            Task(id="analyze", description="Analyze", prompt="A",
                 dependencies=["validate_data", "preprocess"]),
            Task(id="optimize", description="Optimize", prompt="O",
                 dependencies=["preprocess"]),

            # Layer 4
            Task(id="generate_report", description="Report", prompt="R",
                 dependencies=["analyze", "optimize"]),
        ]

        plan = ExecutionPlan(tasks)

        # Track execution order
        execution_log = []

        async def logging_execution(task, context):
            execution_log.append(f"START:{task.id}")
            await asyncio.sleep(0.01)
            execution_log.append(f"END:{task.id}")
            return TaskResult(task.id, True, "Done", 0.01)

        with patch.object(orchestrator, '_execute_single_task', side_effect=logging_execution):
            # Execute all layers
            for layer in plan.execution_order:
                await orchestrator._execute_layer(layer, tasks, None)

        # Verify dependencies were respected
        def task_position(task_id, event_type):
            for i, log in enumerate(execution_log):
                if log == f"{event_type}:{task_id}":
                    return i
            return -1

        # Check critical dependencies
        assert task_position("init", "END") < task_position("load_data", "START")
        assert task_position("load_data", "END") < task_position("validate_data", "START")
        assert task_position("validate_data", "END") < task_position("analyze", "START")
        assert task_position("analyze", "END") < task_position("generate_report", "START")

        print(f"\nComplex DAG execution completed with {len(execution_log)} events")


def benchmark_parallel_performance():
    """Standalone benchmark function for parallel performance."""
    import asyncio

    async def run_benchmark():
        class BenchmarkLLM:
            async def generate(self, prompt: str, stop: List[str] = None):
                await asyncio.sleep(0.01)
                yield "Benchmark response"

        llm = BenchmarkLLM()
        orchestrator = Orchestrator(llm, max_parallel=10)

        # Different task configurations
        configurations = [
            (10, 0),   # 10 independent tasks
            (20, 10),  # 20 tasks with 10 dependencies
            (50, 25),  # 50 tasks with 25 dependencies
            (100, 50), # 100 tasks with 50 dependencies
        ]

        print("\n=== Parallel Performance Benchmark ===\n")

        for total_tasks, dependent_tasks in configurations:
            tasks = []

            # Create independent tasks
            for i in range(total_tasks - dependent_tasks):
                tasks.append(Task(id=f"ind_{i}", description=f"Independent {i}", prompt=f"P{i}"))

            # Create dependent tasks
            for i in range(dependent_tasks):
                deps = [f"ind_{i % (total_tasks - dependent_tasks)}"]
                tasks.append(Task(
                    id=f"dep_{i}",
                    description=f"Dependent {i}",
                    prompt=f"P{i}",
                    dependencies=deps
                ))

            plan = ExecutionPlan(tasks)

            async def mock_exec(task, context):
                await asyncio.sleep(0.01)
                return TaskResult(task.id, True, "Done", 0.01)

            with patch.object(orchestrator, '_execute_single_task', side_effect=mock_exec):
                start = time.time()

                for layer in plan.execution_order:
                    await orchestrator._execute_layer(layer, tasks, None)

                elapsed = time.time() - start

            theoretical_sequential = total_tasks * 0.01
            speedup = theoretical_sequential / elapsed

            print(f"Tasks: {total_tasks} (deps: {dependent_tasks})")
            print(f"  Layers: {len(plan.execution_order)}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print()

    asyncio.run(run_benchmark())


if __name__ == "__main__":
    # Run standalone benchmark
    benchmark_parallel_performance()