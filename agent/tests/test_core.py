"""Test suite for CodingAgent ReAct loop implementation."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import AsyncIterator, Dict, Any

from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools import BaseTool


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, result: str = "tool_result"):
        self.name = name
        self.description = f"Mock {name} tool"
        self.result = result
        self.call_count = 0
        self.last_args = None
        self.last_kwargs = None

    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the mock tool."""
        self.call_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.result


class MockLLM:
    """Mock LLM for predictable testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.prompt_history = []
        self.current_response = 0

    async def generate(
        self, prompt: str, stop: list[str] = None
    ) -> AsyncIterator[str]:
        """Generate mock response."""
        self.prompt_history.append(prompt)

        if self.current_response < len(self.responses):
            response = self.responses[self.current_response]
            self.current_response += 1

            # Simulate streaming by yielding chunks
            chunks = response.split(" ")
            for i, chunk in enumerate(chunks):
                if i < len(chunks) - 1:
                    yield chunk + " "
                else:
                    yield chunk
        else:
            yield "No more responses configured"


class TestCodingAgent:
    """Test suite for CodingAgent."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory instance."""
        memory = AsyncMock(spec=ConversationMemory)
        memory.messages = []
        memory.add_user_message = AsyncMock(
            side_effect=lambda msg: memory.messages.append({"role": "user", "content": msg})
        )
        memory.add_assistant_message = AsyncMock(
            side_effect=lambda msg: memory.messages.append({"role": "assistant", "content": msg})
        )
        memory.add_tool_result = AsyncMock(
            side_effect=lambda tool, result: memory.messages.append(
                {"role": "tool", "tool": tool, "content": result}
            )
        )
        memory.get_context = AsyncMock(return_value=memory.messages)
        return memory

    @pytest.fixture
    def mock_tools(self):
        """Create a set of mock tools."""
        return {
            "file_read": MockTool("file_read", "File contents: Hello World"),
            "git_status": MockTool("git_status", "On branch main\nnothing to commit"),
            "shell": MockTool("shell", "Command executed successfully"),
        }

    @pytest.mark.asyncio
    async def test_simple_chat_without_tools(self, mock_memory):
        """Test basic chat without tool invocation."""
        llm = MockLLM(["Hello! How can I help you today?"])
        agent = CodingAgent(llm, {}, mock_memory)

        response = ""
        async for chunk in agent.chat("Hi there"):
            response += chunk

        assert response == "Hello! How can I help you today?"
        assert len(llm.prompt_history) == 1
        mock_memory.add_user_message.assert_called_once_with("Hi there")

    @pytest.mark.asyncio
    async def test_tool_invocation_pattern_detection(self, mock_memory, mock_tools):
        """Test that agent detects and executes tool calls."""
        # Response with tool invocation pattern
        llm = MockLLM(
            [
                "I'll check the git status for you. {{tool:git_status}}",
                "The repository is clean and on the main branch.",
            ]
        )
        agent = CodingAgent(llm, mock_tools, mock_memory)

        response = ""
        async for chunk in agent.chat("What's the git status?"):
            response += chunk

        # Verify tool was called
        assert mock_tools["git_status"].call_count == 1

        # Verify memory was updated with tool result
        mock_memory.add_tool_result.assert_called_once_with(
            "git_status", "On branch main\nnothing to commit"
        )

        # Response should include both parts
        assert "check the git status" in response
        assert "main branch" in response

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self, mock_memory, mock_tools):
        """Test multiple tool invocations in a single conversation turn."""
        llm = MockLLM(
            [
                "Let me check the file first. {{tool:file_read:path=test.py}}",
                "Now I'll check git status. {{tool:git_status}}",
                "Everything looks good!",
            ]
        )
        agent = CodingAgent(llm, mock_tools, mock_memory)

        response = ""
        async for chunk in agent.chat("Check test.py and git status"):
            response += chunk

        # Both tools should be called
        assert mock_tools["file_read"].call_count == 1
        assert mock_tools["git_status"].call_count == 1

        # Check tool arguments were parsed
        assert mock_tools["file_read"].last_kwargs == {"path": "test.py"}

    @pytest.mark.asyncio
    async def test_unknown_tool_handling(self, mock_memory, mock_tools):
        """Test graceful handling of unknown tool references."""
        llm = MockLLM(["I'll use a tool that doesn't exist. {{tool:unknown_tool}}"])
        agent = CodingAgent(llm, mock_tools, mock_memory)

        response = ""
        async for chunk in agent.chat("Do something"):
            response += chunk

        # Should handle gracefully without crashing
        assert "tool that doesn't exist" in response
        # No tools should be called
        for tool in mock_tools.values():
            assert tool.call_count == 0

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_memory):
        """Test handling of tool execution errors."""
        # Create a tool that raises an error
        failing_tool = MockTool("failing_tool")
        failing_tool.run = AsyncMock(side_effect=RuntimeError("Tool failed!"))

        llm = MockLLM(["I'll run the tool. {{tool:failing_tool}}"])
        agent = CodingAgent(llm, {"failing_tool": failing_tool}, mock_memory)

        response = ""
        with pytest.raises(RuntimeError) as exc_info:
            async for chunk in agent.chat("Run the failing tool"):
                response += chunk

        assert "Tool failed!" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_memory_context_in_prompts(self, mock_memory, mock_tools):
        """Test that agent includes memory context in prompts."""
        # Pre-populate memory with conversation history
        mock_memory.messages = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        llm = MockLLM(["Based on our previous discussion, here's my response."])
        agent = CodingAgent(llm, mock_tools, mock_memory)

        response = ""
        async for chunk in agent.chat("Follow-up question"):
            response += chunk

        # Verify the prompt included context
        assert len(llm.prompt_history) == 1
        prompt = llm.prompt_history[0]
        assert "Previous question" in prompt
        assert "Previous answer" in prompt
        assert "Follow-up question" in prompt

    @pytest.mark.asyncio
    async def test_concurrent_chat_sessions(self, mock_memory, mock_tools):
        """Test that agent handles concurrent chat sessions correctly."""
        llm1 = MockLLM(["Response for user 1"])
        llm2 = MockLLM(["Response for user 2"])

        agent1 = CodingAgent(llm1, mock_tools, mock_memory)
        agent2 = CodingAgent(llm2, mock_tools, AsyncMock(spec=ConversationMemory))

        async def chat1():
            response = ""
            async for chunk in agent1.chat("Question from user 1"):
                response += chunk
                await asyncio.sleep(0.01)  # Simulate processing
            return response

        async def chat2():
            response = ""
            async for chunk in agent2.chat("Question from user 2"):
                response += chunk
                await asyncio.sleep(0.01)  # Simulate processing
            return response

        # Run both chats concurrently
        results = await asyncio.gather(chat1(), chat2())

        assert results[0] == "Response for user 1"
        assert results[1] == "Response for user 2"

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self, mock_memory, mock_tools):
        """Test that streaming works correctly even with tool interruptions."""
        llm = MockLLM(
            [
                "Starting analysis... {{tool:file_read:path=main.py}}",
                "File contains important code. Done!",
            ]
        )
        agent = CodingAgent(llm, mock_tools, mock_memory)

        chunks = []
        async for chunk in agent.chat("Analyze main.py"):
            chunks.append(chunk)

        # Verify chunks were streamed
        assert len(chunks) > 1  # Should be multiple chunks, not one big response
        full_response = "".join(chunks)
        assert "Starting analysis" in full_response
        assert "important code" in full_response

    @pytest.mark.asyncio
    async def test_prompt_building_with_system_message(self, mock_memory, mock_tools):
        """Test that agent builds prompts with proper system message."""
        llm = MockLLM(["I understand my role as a coding assistant."])
        agent = CodingAgent(llm, mock_tools, mock_memory)

        await agent.chat("Help me code").__anext__()  # Start generation

        # Check the prompt includes system instructions
        prompt = llm.prompt_history[0]
        assert "coding assistant" in prompt.lower() or "tools" in prompt.lower()