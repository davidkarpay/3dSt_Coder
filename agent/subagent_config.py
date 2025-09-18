"""Configuration-based subagent system with YAML/Markdown support."""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelSelection(Enum):
    """Model selection options for subagents."""
    INHERIT = "inherit"
    SONNET = "sonnet"
    OPUS = "opus"
    HAIKU = "haiku"
    CODELLAMA = "codellama:7b"
    DEEPSEEK = "deepseek-coder:6.7b"
    MISTRAL = "mistral:7b-instruct-q4_K_M"


@dataclass
class SubAgentConfig:
    """Configuration for a subagent."""

    name: str
    description: str
    system_prompt: str
    tools: Optional[List[str]] = None  # None means inherit all
    model: str = "inherit"
    proactive: bool = False
    file_path: Optional[Path] = None
    priority: int = 0  # Higher priority for project-level agents

    def __post_init__(self):
        """Validate configuration."""
        # Validate name format
        if not re.match(r'^[a-z0-9-]+$', self.name):
            raise ValueError(f"Invalid subagent name '{self.name}'. Use lowercase letters, numbers, and hyphens only.")

        # Check for proactive keywords in description
        desc_lower = self.description.lower()
        if any(keyword in desc_lower for keyword in ["proactively", "use proactively", "must be used"]):
            self.proactive = True

    @classmethod
    def from_markdown(cls, file_path: Path) -> "SubAgentConfig":
        """Load subagent configuration from a Markdown file with YAML frontmatter.

        Args:
            file_path: Path to the Markdown file

        Returns:
            SubAgentConfig instance
        """
        content = file_path.read_text(encoding='utf-8')

        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"Invalid subagent file format in {file_path}. Missing YAML frontmatter.")

        yaml_content = frontmatter_match.group(1)
        system_prompt = frontmatter_match.group(2).strip()

        # Parse YAML
        try:
            metadata = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

        # Validate required fields
        if 'name' not in metadata:
            raise ValueError(f"Missing 'name' field in {file_path}")
        if 'description' not in metadata:
            raise ValueError(f"Missing 'description' field in {file_path}")

        # Parse tools if provided
        tools = None
        if 'tools' in metadata and metadata['tools']:
            if isinstance(metadata['tools'], str):
                tools = [t.strip() for t in metadata['tools'].split(',')]
            elif isinstance(metadata['tools'], list):
                tools = metadata['tools']

        # Determine priority based on file location
        priority = 10 if '.claude' in str(file_path) else 5

        return cls(
            name=metadata['name'],
            description=metadata['description'],
            system_prompt=system_prompt,
            tools=tools,
            model=metadata.get('model', 'inherit'),
            file_path=file_path,
            priority=priority
        )

    def to_markdown(self) -> str:
        """Convert configuration to Markdown format with YAML frontmatter.

        Returns:
            Markdown string
        """
        # Build YAML frontmatter
        metadata = {
            'name': self.name,
            'description': self.description
        }

        if self.tools:
            metadata['tools'] = ', '.join(self.tools)

        if self.model != 'inherit':
            metadata['model'] = self.model

        yaml_content = yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False)

        # Combine with system prompt
        return f"---\n{yaml_content}---\n\n{self.system_prompt}"


class SubAgentRegistry:
    """Registry for managing configured subagents."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the registry.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.agents: Dict[str, SubAgentConfig] = {}
        self.load_all()

    def load_all(self):
        """Load all subagent configurations from standard locations."""
        # Clear existing agents
        self.agents.clear()

        # Load user-level agents (~/.claude/agents/)
        user_agents_dir = Path.home() / '.claude' / 'agents'
        if user_agents_dir.exists():
            self._load_from_directory(user_agents_dir, priority=5)

        # Load project-level agents (.claude/agents/)
        project_agents_dir = self.project_root / '.claude' / 'agents'
        if project_agents_dir.exists():
            self._load_from_directory(project_agents_dir, priority=10)

        logger.info(f"Loaded {len(self.agents)} subagent configurations")

    def _load_from_directory(self, directory: Path, priority: int):
        """Load subagent configurations from a directory.

        Args:
            directory: Directory containing .md files
            priority: Priority level for agents from this directory
        """
        for file_path in directory.glob('*.md'):
            try:
                config = SubAgentConfig.from_markdown(file_path)
                config.priority = priority

                # Handle duplicates - higher priority wins
                if config.name in self.agents:
                    if config.priority > self.agents[config.name].priority:
                        logger.info(f"Overriding subagent '{config.name}' with higher priority version from {file_path}")
                        self.agents[config.name] = config
                else:
                    self.agents[config.name] = config
                    logger.debug(f"Loaded subagent '{config.name}' from {file_path}")

            except Exception as e:
                logger.error(f"Failed to load subagent from {file_path}: {e}")

    def get(self, name: str) -> Optional[SubAgentConfig]:
        """Get a subagent configuration by name.

        Args:
            name: Subagent name

        Returns:
            SubAgentConfig or None if not found
        """
        return self.agents.get(name)

    def list_all(self) -> List[SubAgentConfig]:
        """List all available subagents.

        Returns:
            List of subagent configurations
        """
        return list(self.agents.values())

    def find_matching(self, task_description: str) -> List[SubAgentConfig]:
        """Find subagents that match a task description.

        Args:
            task_description: Description of the task

        Returns:
            List of matching subagent configurations
        """
        matches = []
        task_lower = task_description.lower()

        for config in self.agents.values():
            # Check if agent name is mentioned
            if config.name in task_lower:
                matches.append(config)
                continue

            # Check description keywords
            desc_keywords = config.description.lower().split()
            if any(keyword in task_lower for keyword in desc_keywords if len(keyword) > 4):
                matches.append(config)
                continue

            # Check if agent is proactive and task matches its domain
            if config.proactive:
                # Simple domain matching based on common keywords
                domain_matches = {
                    'code-reviewer': ['review', 'check', 'quality', 'pr', 'pull request'],
                    'debugger': ['debug', 'error', 'fix', 'issue', 'bug', 'fail'],
                    'test-runner': ['test', 'pytest', 'unit test', 'testing'],
                    'documenter': ['document', 'docs', 'readme', 'documentation'],
                    'refactorer': ['refactor', 'clean', 'optimize', 'improve']
                }

                if config.name in domain_matches:
                    if any(keyword in task_lower for keyword in domain_matches[config.name]):
                        matches.append(config)

        # Sort by priority (project agents first)
        matches.sort(key=lambda x: x.priority, reverse=True)
        return matches

    def create_subagent(self, config: SubAgentConfig, location: str = "project") -> Path:
        """Create a new subagent configuration file.

        Args:
            config: SubAgentConfig to save
            location: "project" or "user" for save location

        Returns:
            Path to the created file
        """
        # Determine save directory
        if location == "project":
            save_dir = self.project_root / '.claude' / 'agents'
        else:
            save_dir = Path.home() / '.claude' / 'agents'

        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        file_path = save_dir / f"{config.name}.md"
        file_path.write_text(config.to_markdown(), encoding='utf-8')

        # Reload registry
        self.load_all()

        logger.info(f"Created subagent '{config.name}' at {file_path}")
        return file_path

    def delete_subagent(self, name: str) -> bool:
        """Delete a subagent configuration.

        Args:
            name: Name of the subagent to delete

        Returns:
            True if deleted, False if not found
        """
        config = self.get(name)
        if not config or not config.file_path:
            return False

        # Delete the file
        config.file_path.unlink()

        # Reload registry
        self.load_all()

        logger.info(f"Deleted subagent '{name}'")
        return True

    def update_subagent(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update a subagent configuration.

        Args:
            name: Name of the subagent to update
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        config = self.get(name)
        if not config or not config.file_path:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Save updated configuration
        config.file_path.write_text(config.to_markdown(), encoding='utf-8')

        # Reload registry
        self.load_all()

        logger.info(f"Updated subagent '{name}'")
        return True


class ToolPermissionManager:
    """Manages tool permissions for subagents."""

    def __init__(self, available_tools: Set[str]):
        """Initialize the permission manager.

        Args:
            available_tools: Set of all available tool names
        """
        self.available_tools = available_tools

    def filter_tools(self, config: SubAgentConfig, all_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Filter tools based on subagent configuration.

        Args:
            config: SubAgentConfig with tool permissions
            all_tools: Dictionary of all available tools

        Returns:
            Filtered dictionary of allowed tools
        """
        # If no tools specified, inherit all
        if config.tools is None:
            return all_tools

        # Filter to only allowed tools
        allowed_tools = {}
        for tool_name in config.tools:
            if tool_name in all_tools:
                allowed_tools[tool_name] = all_tools[tool_name]
            else:
                logger.warning(f"Tool '{tool_name}' requested by subagent '{config.name}' not found")

        return allowed_tools

    def validate_tools(self, tools: List[str]) -> List[str]:
        """Validate that requested tools exist.

        Args:
            tools: List of tool names

        Returns:
            List of valid tool names
        """
        valid_tools = []
        for tool_name in tools:
            if tool_name in self.available_tools:
                valid_tools.append(tool_name)
            else:
                logger.warning(f"Unknown tool '{tool_name}' - skipping")

        return valid_tools


# Example built-in subagent configurations
BUILTIN_AGENTS = [
    SubAgentConfig(
        name="code-reviewer",
        description="Expert code review specialist. Use proactively after writing or modifying code.",
        system_prompt="""You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is simple and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed

Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)

Include specific examples of how to fix issues.""",
        tools=["git_diff", "git_status", "file_read", "grep"],
        model="inherit"
    ),

    SubAgentConfig(
        name="test-runner",
        description="Test execution specialist. Use proactively to run tests and fix failures.",
        system_prompt="""You are a test automation expert focused on ensuring all tests pass.

When invoked:
1. Identify relevant test suites
2. Run tests with appropriate verbosity
3. Analyze any failures
4. Fix failing tests if needed
5. Re-run to confirm fixes

Use parallel test execution when possible for faster results.
Always provide clear test summaries with pass/fail counts.""",
        tools=["test_runner", "parallel_test_runner", "file_read", "file_write", "shell"],
        model="inherit"
    ),

    SubAgentConfig(
        name="debugger",
        description="Debugging specialist for errors and issues. Use proactively when encountering problems.",
        system_prompt="""You are an expert debugger specializing in root cause analysis.

When invoked:
1. Capture error message and stack trace
2. Identify reproduction steps
3. Isolate the failure location
4. Implement minimal fix
5. Verify solution works

Debugging process:
- Analyze error messages and logs
- Check recent code changes
- Form and test hypotheses
- Add strategic debug logging
- Inspect variable states

Focus on fixing the underlying issue, not just symptoms.""",
        tools=["file_read", "file_write", "shell", "grep", "git_diff"],
        model="inherit"
    )
]


def initialize_builtin_agents(registry: SubAgentRegistry):
    """Initialize built-in subagent configurations.

    Args:
        registry: SubAgentRegistry to add built-in agents to
    """
    for config in BUILTIN_AGENTS:
        # Only add if not already defined by user/project
        if config.name not in registry.agents:
            registry.agents[config.name] = config
            logger.debug(f"Added built-in subagent '{config.name}'")