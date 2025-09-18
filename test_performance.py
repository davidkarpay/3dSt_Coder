#!/usr/bin/env python3
"""Performance and stress tests for 3dSt_Coder."""

import asyncio
import time
import sys
import psutil
import concurrent.futures
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class PerformanceMonitor:
    """Monitor system performance during tests."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        self.start_cpu = psutil.cpu_percent()

    def stop(self):
        """Stop monitoring and return metrics."""
        if self.start_time is None:
            return {}

        duration = time.time() - self.start_time
        end_memory = psutil.Process().memory_info().rss
        memory_delta = end_memory - self.start_memory

        return {
            "duration": duration,
            "memory_start_mb": self.start_memory / 1024 / 1024,
            "memory_end_mb": end_memory / 1024 / 1024,
            "memory_delta_mb": memory_delta / 1024 / 1024,
            "cpu_usage": psutil.cpu_percent()
        }


async def test_agent_response_time():
    """Test agent response time under various loads."""
    print("\n" + "="*50)
    print("Testing Agent Response Time")
    print("="*50)

    try:
        from agent import CodingAgent, ConversationMemory
        from agent.tools import get_available_tools
        from llm_server import VLLMEngine, LLMConfig

        # Initialize components
        config = LLMConfig()
        llm_engine = VLLMEngine(config)
        tools = get_available_tools()
        memory = ConversationMemory(max_tokens=1024, project_id="perf_test")
        agent = CodingAgent(llm_engine, tools, memory)

        # Test messages of varying complexity
        test_messages = [
            "Hello",
            "What is the current time?",
            "Can you read the file test.py and explain what it does?",
            "I need you to analyze this codebase and create a comprehensive report on its architecture.",
        ]

        results = []

        for i, message in enumerate(test_messages, 1):
            print(f"\nTest {i}: Message length {len(message)} chars")
            monitor = PerformanceMonitor()
            monitor.start()

            try:
                # Get first response chunk (simulates initial response time)
                response_gen = agent.chat(message)
                first_chunk = await response_gen.__anext__()

                metrics = monitor.stop()
                results.append({
                    "test": i,
                    "message_length": len(message),
                    "first_response_time": metrics["duration"],
                    "memory_usage": metrics["memory_delta_mb"]
                })

                print(f"  First response time: {metrics['duration']:.3f}s")
                print(f"  Memory delta: {metrics['memory_delta_mb']:.2f}MB")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Summary
        if results:
            avg_response_time = sum(r["first_response_time"] for r in results) / len(results)
            max_response_time = max(r["first_response_time"] for r in results)
            print(f"\nResponse Time Summary:")
            print(f"  Average: {avg_response_time:.3f}s")
            print(f"  Maximum: {max_response_time:.3f}s")

        return len(results) == len(test_messages)

    except Exception as e:
        print(f"Agent response test failed: {e}")
        return False


async def test_memory_usage():
    """Test memory usage patterns."""
    print("\n" + "="*50)
    print("Testing Memory Usage Patterns")
    print("="*50)

    try:
        from agent.memory import ConversationMemory, Message
        from datetime import datetime

        # Test with increasing conversation length
        monitor = PerformanceMonitor()
        monitor.start()

        memory = ConversationMemory(max_tokens=4096, project_id="memory_test")

        # Add many messages to test memory growth
        for i in range(100):
            await memory.add_user_message(f"Test message {i} with some content")
            await memory.add_assistant_message(f"Response to message {i} with detailed explanation")

        metrics = monitor.stop()
        print(f"Added 200 messages in {metrics['duration']:.3f}s")
        print(f"Memory usage: {metrics['memory_delta_mb']:.2f}MB")

        # Test context retrieval performance
        monitor.start()
        context = await memory.get_context()
        metrics = monitor.stop()

        print(f"Context retrieval: {metrics['duration']:.3f}s")
        print(f"Context size: {len(context)} messages")

        return True

    except Exception as e:
        print(f"Memory usage test failed: {e}")
        return False


def test_concurrent_requests():
    """Test system under concurrent load."""
    print("\n" + "="*50)
    print("Testing Concurrent Request Handling")
    print("="*50)

    try:
        import requests
        import threading

        # This would test the actual API if it's running
        # For now, we'll test the core components
        results = []
        num_threads = 5
        requests_per_thread = 3

        def worker_thread(thread_id):
            """Worker thread for concurrent testing."""
            thread_results = []
            for i in range(requests_per_thread):
                start_time = time.time()
                try:
                    # Simulate work (could be actual API calls if server is running)
                    from auth.security import create_access_token, verify_token

                    # Create and verify token
                    token_data = {"sub": f"user_{thread_id}_{i}", "user_id": thread_id * 100 + i, "role": "user"}
                    token = create_access_token(token_data)
                    verified = verify_token(token)

                    duration = time.time() - start_time
                    thread_results.append({
                        "thread": thread_id,
                        "request": i,
                        "duration": duration,
                        "success": verified is not None
                    })

                except Exception as e:
                    duration = time.time() - start_time
                    thread_results.append({
                        "thread": thread_id,
                        "request": i,
                        "duration": duration,
                        "success": False,
                        "error": str(e)
                    })

            return thread_results

        # Run concurrent threads
        monitor = PerformanceMonitor()
        monitor.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        metrics = monitor.stop()

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"Completed {len(results)} requests in {metrics['duration']:.3f}s")
        print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")

        if successful:
            avg_duration = sum(r["duration"] for r in successful) / len(successful)
            max_duration = max(r["duration"] for r in successful)
            print(f"Average request time: {avg_duration:.3f}s")
            print(f"Slowest request: {max_duration:.3f}s")

        return len(failed) == 0

    except Exception as e:
        print(f"Concurrent request test failed: {e}")
        return False


def test_tool_performance():
    """Test tool execution performance."""
    print("\n" + "="*50)
    print("Testing Tool Execution Performance")
    print("="*50)

    try:
        from agent.tools import get_available_tools

        tools = get_available_tools()
        results = {}

        # Test each tool's performance
        for tool_name, tool in tools.items():
            if tool_name in ["shell", "git_status"]:  # Safe tools to test
                monitor = PerformanceMonitor()
                monitor.start()

                try:
                    if tool_name == "git_status":
                        result = await tool.run()
                    elif tool_name == "shell":
                        result = await tool.run("echo 'performance test'")

                    metrics = monitor.stop()
                    results[tool_name] = {
                        "duration": metrics["duration"],
                        "memory_delta": metrics["memory_delta_mb"],
                        "success": True
                    }

                    print(f"{tool_name}: {metrics['duration']:.3f}s")

                except Exception as e:
                    metrics = monitor.stop()
                    results[tool_name] = {
                        "duration": metrics["duration"],
                        "memory_delta": metrics["memory_delta_mb"],
                        "success": False,
                        "error": str(e)
                    }
                    print(f"{tool_name}: FAILED ({e})")

        successful_tools = [name for name, result in results.items() if result["success"]]
        return len(successful_tools) > 0

    except Exception as e:
        print(f"Tool performance test failed: {e}")
        return False


async def main():
    """Run all performance tests."""
    print("3dSt_Coder Performance Test Suite")
    print("="*60)

    # System info
    print(f"System Info:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  Python: {sys.version}")

    results = {}

    # Run tests
    results["agent_response"] = await test_agent_response_time()
    results["memory_usage"] = await test_memory_usage()
    results["concurrent_requests"] = test_concurrent_requests()
    results["tool_performance"] = await test_tool_performance()

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name.upper()}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All performance tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} performance test(s) failed")
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)