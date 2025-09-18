"""Core ReAct agent implementation for coding assistance."""

import re
import logging
import asyncio
from typing import Dict, AsyncIterator, Any, Optional, Tuple, List
from dataclasses import dataclass

from .memory import ConversationMemory
from .tools import BaseTool

logger = logging.getLogger(__name__)

# Pattern to detect tool calls in LLM output
TOOL_PATTERN = re.compile(r"\{\{tool:([\w_]+)(?::(.+?))?\}\}")
# Pattern to detect parallel tool calls
PARALLEL_PATTERN = re.compile(r"\{\{parallel:\[([\w_,\s]*)\]\}\}")


@dataclass
class ToolCall:
    """Represents a tool invocation request."""

    name: str
    args: Dict[str, Any]


class CodingAgent:
    """ReAct-based coding agent with tool calling capabilities.

    Implements a ReAct (Reason + Act) loop where the LLM can:
    1. Reason about the user's request
    2. Decide to use tools if needed
    3. Incorporate tool results into its response
    4. Continue reasoning with the results
    """

    def __init__(
        self,
        llm: Any,  # VLLMEngine instance
        tools: Dict[str, BaseTool],
        memory: ConversationMemory,
    ):
        """Initialize the coding agent.

        Args:
            llm: Language model engine for generation
            tools: Dictionary of available tools
            memory: Conversation memory for context
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def _build_prompt(self) -> str:
        """Build the complete prompt including system message and conversation history.

        Returns:
            Formatted prompt string for the LLM
        """
        # System message defining the agent's role and capabilities
        system_message = (
            "You are an intelligent AI assistant with access to powerful tools for development, analysis, and automation.\n"
            "You can use tools by including {{tool:tool_name}} or {{tool:tool_name:arg1=value1,arg2=value2}} in your response.\n"
            "For parallel execution, use {{parallel:[tool1, tool2, tool3]}} to run multiple tools simultaneously.\n"
            "Available tools:\n"
        )

        # Add tool descriptions
        for tool_name, tool in self.tools.items():
            system_message += f"- {tool_name}: {tool.description}\n"

        system_message += "\nWhen you need to use a tool, include the tool invocation pattern in your response.\n"
        system_message += "Use parallel execution when multiple independent tools can run simultaneously.\n\n"

        # Get conversation history from memory
        messages = self.memory.messages

        # Build the full prompt
        prompt = system_message + "Conversation:\n"

        for msg in messages:
            role = msg.role
            content = msg.content

            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "tool":
                tool_name = msg.tool or "unknown"
                prompt += f"Tool Result ({tool_name}): {content}\n"

        prompt += "Assistant: "  # Start of new response

        return prompt

    def _extract_tool_call(self, text: str) -> Optional[ToolCall]:
        """Extract tool call from text if present.

        Args:
            text: Text that may contain a tool invocation pattern

        Returns:
            ToolCall if pattern found, None otherwise
        """
        match = TOOL_PATTERN.search(text)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments if provided
        args = {}
        if args_str:
            # Simple argument parsing (key=value pairs)
            for arg_pair in args_str.split(","):
                if "=" in arg_pair:
                    key, value = arg_pair.strip().split("=", 1)
                    args[key.strip()] = value.strip()

        return ToolCall(name=tool_name, args=args)

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool and return its result.

        Args:
            tool_call: Tool invocation details

        Returns:
            Tool execution result as string
        """
        if tool_call.name not in self.tools:
            logger.warning(f"Unknown tool requested: {tool_call.name}")
            return f"Error: Tool '{tool_call.name}' not found"

        tool = self.tools[tool_call.name]

        try:
            logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.args}")
            result = await tool.run(**tool_call.args)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise

    async def _execute_parallel_tools(self, tool_names: List[str]) -> Dict[str, str]:
        """Execute multiple tools in parallel.

        Args:
            tool_names: List of tool names to execute

        Returns:
            Dictionary mapping tool names to their results
        """
        tasks = []
        for tool_name in tool_names:
            tool_call = ToolCall(name=tool_name.strip(), args={})
            tasks.append(self._execute_tool(tool_call))

        # Execute all tools in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results to tool names
        tool_results = {}
        for tool_name, result in zip(tool_names, results):
            if isinstance(result, Exception):
                tool_results[tool_name.strip()] = f"Error: {str(result)}"
            else:
                tool_results[tool_name.strip()] = result

        return tool_results

    def _detect_parallel_tools(self, text: str) -> Optional[List[str]]:
        """Detect parallel tool invocation pattern.

        Args:
            text: Text that may contain parallel tool pattern

        Returns:
            List of tool names if pattern found, None otherwise
        """
        match = PARALLEL_PATTERN.search(text)
        if not match:
            return None

        tools_str = match.group(1).strip()
        if not tools_str:
            return []
        tool_names = [name.strip() for name in tools_str.split(",") if name.strip()]
        return tool_names

    async def chat(
        self,
        user_msg: str,
        preferred_model: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Process a user message and generate a response.

        Implements the ReAct loop with parallel tool support:
        1. Add user message to memory
        2. Build prompt with context
        3. Generate LLM response
        4. Detect and execute tool calls (sequential or parallel)
        5. Continue generation with tool results

        Args:
            user_msg: User's input message

        Yields:
            Response chunks as they're generated
        """
        # Add user message to memory
        await self.memory.add_user_message(user_msg)

        # Build initial prompt
        prompt = self._build_prompt()

        # Track the full response for memory
        full_response = ""
        current_chunk = ""

        # Generate response with model preferences
        if hasattr(self.llm, 'generate') and hasattr(self.llm, 'get_engine_status'):
            # MultiEngine case - supports model selection
            generation_iter = self.llm.generate(
                prompt,
                task_type=task_type,
                preferred_model=preferred_model
            )
        else:
            # Single engine case
            generation_iter = self.llm.generate(prompt)

        # Process generation tokens
        async for token in generation_iter:
            current_chunk += token
            full_response += token

            # Check for parallel tool invocation
            parallel_tools = self._detect_parallel_tools(current_chunk)

            if parallel_tools:
                # Found parallel tool invocation
                logger.debug(f"Parallel tools detected: {parallel_tools}")

                # Yield text before the parallel call
                before_parallel = current_chunk.split("{{parallel:")[0]
                if before_parallel:
                    yield before_parallel

                # Execute tools in parallel
                tool_results = await self._execute_parallel_tools(parallel_tools)

                # Add results to memory
                for tool_name, result in tool_results.items():
                    await self.memory.add_tool_result(tool_name, result)

                # Format results for display
                results_text = "\n\nParallel execution results:\n"
                for tool_name, result in tool_results.items():
                    results_text += f"- {tool_name}: {result[:100]}...\n"

                yield results_text

                # Clear current chunk and continue
                current_chunk = ""
                prompt = self._build_prompt()

                # Continue generation with results
                async for continuation_token in self.llm.generate(prompt):
                    full_response += continuation_token
                    yield continuation_token

                break

            # Check for single tool invocation
            tool_call = self._extract_tool_call(current_chunk)

            if tool_call:
                # Found a tool call - execute it
                logger.debug(f"Tool call detected: {tool_call}")

                # Yield the text before the tool call
                before_tool = current_chunk.split("{{tool:")[0]
                if before_tool:
                    yield before_tool

                # Execute the tool
                tool_result = await self._execute_tool(tool_call)

                # Add tool result to memory
                await self.memory.add_tool_result(tool_call.name, tool_result)

                # Clear current chunk and continue generation with updated context
                current_chunk = ""

                # Rebuild prompt with tool result
                prompt = self._build_prompt()

                # Continue generation
                async for continuation_token in self.llm.generate(prompt):
                    full_response += continuation_token
                    yield continuation_token

                break  # Exit the outer loop since we've completed with continuation

            else:
                # No tool call yet, keep streaming
                # Yield complete words/tokens
                if token.endswith(" ") or token.endswith("\n"):
                    yield current_chunk
                    current_chunk = ""

        # Yield any remaining chunk
        if current_chunk and "{{tool:" not in current_chunk and "{{parallel:" not in current_chunk:
            yield current_chunk

        # Save assistant's response to memory
        await self.memory.add_assistant_message(full_response)

    async def reset_memory(self) -> None:
        """Reset the conversation memory."""
        await self.memory.clear()