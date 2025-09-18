"""Configured subagent that combines configuration with parallel execution."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime

from .subagent import SubAgent, Task, TaskStatus, TaskResult
from .subagent_config import SubAgentConfig, SubAgentRegistry, ToolPermissionManager
from .memory import ConversationMemory
from .tools import get_available_tools, BaseTool
from llm_server.engine_factory import EngineFactory

logger = logging.getLogger(__name__)


class ConfiguredSubAgent(SubAgent):
    """Subagent with configuration-based setup and tool permissions."""

    def __init__(
        self,
        config: SubAgentConfig,
        llm: Optional[Any] = None,
        parent_id: Optional[str] = None,
        inherit_model: Optional[str] = None
    ):
        """Initialize configured subagent.

        Args:
            config: SubAgentConfig with agent definition
            llm: Language model engine (optional, will create based on config)
            parent_id: Parent agent ID for tracking
            inherit_model: Model to use if config specifies 'inherit'
        """
        self.config = config

        # Determine model to use
        model_name = self._resolve_model(inherit_model)

        # Create LLM if not provided
        if llm is None:
            llm = self._create_llm(model_name)

        # Get and filter tools based on configuration
        all_tools = get_available_tools()
        permission_manager = ToolPermissionManager(set(all_tools.keys()))
        filtered_tools = permission_manager.filter_tools(config, all_tools)

        # Create memory with system prompt
        memory = ConversationMemory(
            max_tokens=4096,
            project_id=f"subagent_{config.name}"
        )

        # Initialize base SubAgent
        super().__init__(
            llm=llm,
            tools=filtered_tools,
            memory=memory,
            parent_id=parent_id,
            task_id=config.name
        )

        # Add system prompt to memory
        asyncio.create_task(self._initialize_memory())

    async def _initialize_memory(self):
        """Initialize memory with system prompt."""
        await self.memory.add_system_message(self.config.system_prompt)

    def _resolve_model(self, inherit_model: Optional[str]) -> str:
        """Resolve the model to use based on configuration.

        Args:
            inherit_model: Model to use if config specifies 'inherit'

        Returns:
            Model name to use
        """
        if self.config.model == "inherit":
            return inherit_model or "codellama:7b"  # Default fallback

        # Map model aliases to actual model names
        model_mapping = {
            "sonnet": "claude-3-sonnet",
            "opus": "claude-3-opus",
            "haiku": "claude-3-haiku",
            "codellama": "codellama:7b",
            "deepseek": "deepseek-coder:6.7b",
            "mistral": "mistral:7b-instruct-q4_K_M"
        }

        return model_mapping.get(self.config.model, self.config.model)

    def _create_llm(self, model_name: str) -> Any:
        """Create LLM engine for the specified model.

        Args:
            model_name: Name of the model to use

        Returns:
            LLM engine instance
        """
        # Use the existing engine factory
        factory = EngineFactory()

        # Determine engine type based on model name
        if model_name.startswith("claude"):
            engine_type = "openai"  # Assuming Claude API compatibility
        else:
            engine_type = "ollama"  # Default to Ollama for local models

        return factory.create_engine(
            engine_type=engine_type,
            model_name=model_name
        )

    async def execute_with_context(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Execute task with optional context.

        Args:
            prompt: Task prompt
            context: Optional additional context

        Yields:
            Response chunks
        """
        # Add context if provided
        if context:
            full_prompt = f"{context}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Execute using base agent chat
        async for chunk in self.chat(full_prompt):
            yield chunk


class SubAgentOrchestrator:
    """Enhanced orchestrator that uses configured subagents."""

    def __init__(
        self,
        registry: Optional[SubAgentRegistry] = None,
        max_parallel: int = 5
    ):
        """Initialize the orchestrator.

        Args:
            registry: SubAgentRegistry (creates default if not provided)
            max_parallel: Maximum parallel subagents
        """
        self.registry = registry or SubAgentRegistry()
        self.max_parallel = max_parallel
        self.active_agents: Dict[str, ConfiguredSubAgent] = {}
        self.inherit_model = "codellama:7b"  # Default model for inheritance

    def set_inherit_model(self, model_name: str):
        """Set the model to use for 'inherit' configurations.

        Args:
            model_name: Model name to inherit
        """
        self.inherit_model = model_name

    async def delegate_to_subagent(
        self,
        agent_name: str,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """Delegate a task to a specific subagent.

        Args:
            agent_name: Name of the subagent
            prompt: Task prompt
            context: Optional context

        Returns:
            Subagent response
        """
        config = self.registry.get(agent_name)
        if not config:
            return f"Subagent '{agent_name}' not found"

        # Create configured subagent
        subagent = ConfiguredSubAgent(
            config=config,
            parent_id="orchestrator",
            inherit_model=self.inherit_model
        )

        # Track active agent
        self.active_agents[agent_name] = subagent

        try:
            # Execute and collect response
            response_parts = []
            async for chunk in subagent.execute_with_context(prompt, context):
                response_parts.append(chunk)

            return "".join(response_parts)

        finally:
            # Clean up
            del self.active_agents[agent_name]

    async def auto_delegate(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Automatically find and delegate to appropriate subagents.

        Args:
            prompt: Task prompt
            context: Optional context

        Yields:
            Progress and results
        """
        # Find matching subagents
        matching_agents = self.registry.find_matching(prompt)

        if not matching_agents:
            yield "No specialized subagents found for this task. Using default agent.\n"
            return

        yield f"Found {len(matching_agents)} matching subagent(s): {', '.join(a.name for a in matching_agents)}\n"

        # Check if we should run multiple agents in parallel
        if len(matching_agents) > 1 and self._should_parallelize(prompt):
            # Run agents in parallel
            yield "\nExecuting subagents in parallel...\n"
            async for result in self._execute_parallel(matching_agents, prompt, context):
                yield result
        else:
            # Run the highest priority agent
            agent = matching_agents[0]
            yield f"\nDelegating to '{agent.name}' subagent...\n"
            result = await self.delegate_to_subagent(agent.name, prompt, context)
            yield result

    def _should_parallelize(self, prompt: str) -> bool:
        """Determine if task should be parallelized.

        Args:
            prompt: Task prompt

        Returns:
            True if should parallelize
        """
        parallel_keywords = [
            "all", "multiple", "various", "different",
            "analyze everything", "check all", "test all"
        ]

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in parallel_keywords)

    async def _execute_parallel(
        self,
        agents: List[SubAgentConfig],
        prompt: str,
        context: Optional[str]
    ) -> AsyncIterator[str]:
        """Execute multiple subagents in parallel.

        Args:
            agents: List of subagent configurations
            prompt: Task prompt
            context: Optional context

        Yields:
            Combined results
        """
        # Create tasks for parallel execution
        tasks = []
        for agent_config in agents[:self.max_parallel]:  # Limit parallelism
            tasks.append(self.delegate_to_subagent(
                agent_config.name,
                prompt,
                context
            ))

        # Execute all agents in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Yield combined results
        yield "\n=== PARALLEL SUBAGENT RESULTS ===\n"
        for agent_config, result in zip(agents, results):
            yield f"\n### {agent_config.name}\n"
            if isinstance(result, Exception):
                yield f"Error: {str(result)}\n"
            else:
                yield f"{result}\n"

        yield "\n=== END RESULTS ===\n"


class NaturalLanguageInvoker:
    """Handles natural language invocation of subagents."""

    def __init__(self, registry: SubAgentRegistry):
        """Initialize the invoker.

        Args:
            registry: SubAgentRegistry to search for agents
        """
        self.registry = registry
        self.orchestrator = SubAgentOrchestrator(registry)

    async def process_command(self, user_input: str) -> AsyncIterator[str]:
        """Process user command and invoke appropriate subagents.

        Args:
            user_input: User's natural language command

        Yields:
            Response from subagents
        """
        # Check for explicit agent invocation patterns
        explicit_patterns = [
            r"use (?:the )?(\S+) (?:subagent|agent)",
            r"have (?:the )?(\S+) (?:subagent|agent)",
            r"ask (?:the )?(\S+) (?:subagent|agent)",
            r"run (?:the )?(\S+) (?:subagent|agent)"
        ]

        import re
        for pattern in explicit_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                agent_name = match.group(1)
                config = self.registry.get(agent_name)
                if config:
                    yield f"Invoking '{agent_name}' subagent as requested...\n"
                    # Extract the actual task from the command
                    task = re.sub(pattern + r"(?:\s+to)?", "", user_input.lower()).strip()
                    result = await self.orchestrator.delegate_to_subagent(
                        agent_name,
                        task or user_input
                    )
                    yield result
                    return
                else:
                    yield f"Subagent '{agent_name}' not found. Available agents: {', '.join(self.registry.agents.keys())}\n"
                    return

        # No explicit invocation, try auto-delegation
        async for chunk in self.orchestrator.auto_delegate(user_input):
            yield chunk


def create_example_subagents(project_root: Path):
    """Create example subagent configuration files.

    Args:
        project_root: Project root directory
    """
    registry = SubAgentRegistry(project_root)

    # Create example performance analyzer
    perf_config = SubAgentConfig(
        name="performance-analyzer",
        description="Analyze code performance and suggest optimizations. Use proactively for performance issues.",
        system_prompt="""You are a performance optimization expert.

When analyzing performance:
1. Profile code execution time
2. Identify bottlenecks
3. Check for inefficient algorithms
4. Look for unnecessary loops or operations
5. Suggest specific optimizations

Focus on:
- Time complexity improvements
- Memory usage optimization
- Parallel execution opportunities
- Caching strategies
- Database query optimization

Provide concrete before/after code examples.""",
        tools=["file_read", "parallel_file_analyzer", "grep", "shell"],
        model="inherit"
    )

    # Create example security scanner
    security_config = SubAgentConfig(
        name="security-scanner",
        description="Scan code for security vulnerabilities. Must be used before deploying code.",
        system_prompt="""You are a security expert focused on identifying vulnerabilities.

Security checklist:
- SQL injection risks
- XSS vulnerabilities
- Authentication bypasses
- Insecure data storage
- Exposed credentials
- Path traversal risks
- Command injection
- Insecure deserialization

For each issue found:
1. Explain the vulnerability
2. Show the problematic code
3. Provide a secure alternative
4. Rate severity (Critical/High/Medium/Low)""",
        tools=["file_read", "grep", "parallel_searcher"],
        model="inherit"
    )

    # Save to project directory
    registry.create_subagent(perf_config, "project")
    registry.create_subagent(security_config, "project")

    logger.info("Created example subagent configurations")