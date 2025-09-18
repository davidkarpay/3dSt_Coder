"""Tool implementations for the coding agent."""

from typing import Protocol, Any, Dict
from .file import FileReadTool, FileWriteTool
from .git import GitStatusTool, GitDiffTool, GitCommitTool
from .shell import ShellTool
from .test_runner import TestRunnerTool
from .parallel import (
    ParallelFileAnalyzer,
    ParallelTestRunner,
    ParallelCodeGenerator,
    ParallelSearcher,
    get_parallel_tools
)

class BaseTool(Protocol):
    """Protocol for all agent tools."""

    name: str
    description: str

    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool and return result as string."""
        ...

def get_available_tools() -> Dict[str, BaseTool]:
    """Auto-discover and return all available tools."""
    tools = {
        "file_read": FileReadTool(),
        "file_write": FileWriteTool(),
        "git_status": GitStatusTool(),
        "git_diff": GitDiffTool(),
        "git_commit": GitCommitTool(),
        "shell": ShellTool(),
        "test_runner": TestRunnerTool(),
    }

    # Add parallel tools
    tools.update(get_parallel_tools())

    return tools

__all__ = [
    "BaseTool",
    "FileReadTool",
    "FileWriteTool",
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    "ShellTool",
    "TestRunnerTool",
    "ParallelFileAnalyzer",
    "ParallelTestRunner",
    "ParallelCodeGenerator",
    "ParallelSearcher",
    "get_available_tools",
    "get_parallel_tools",
]