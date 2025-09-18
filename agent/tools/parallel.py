"""Parallel execution tools for distributed analysis."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Protocol
from pathlib import Path
from .file import FileReadTool
from .git import GitStatusTool, GitDiffTool
from .shell import ShellTool
from .test_runner import TestRunnerTool

logger = logging.getLogger(__name__)


class BaseTool(Protocol):
    """Protocol for all agent tools."""

    description: str

    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool and return result as string."""
        ...


class ParallelFileAnalyzer(BaseTool):
    """Analyze multiple files in parallel."""

    description = "Analyze multiple files simultaneously for patterns, structure, or content"

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the parallel file analyzer."""
        self.base_path = base_path or Path.cwd()
        self.file_tool = FileReadTool()

    async def run(
        self,
        files: str,
        analysis_type: str = "structure"
    ) -> str:
        """Analyze multiple files in parallel.

        Args:
            files: Comma-separated list of file paths
            analysis_type: Type of analysis (structure, patterns, dependencies)

        Returns:
            Combined analysis results
        """
        file_list = [f.strip() for f in files.split(",")]

        # Create analysis tasks
        tasks = []
        for file_path in file_list:
            tasks.append(self._analyze_file(file_path, analysis_type))

        # Execute all analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_results = []
        for file_path, result in zip(file_list, results):
            if isinstance(result, Exception):
                combined_results.append(f"Error analyzing {file_path}: {str(result)}")
            else:
                combined_results.append(f"=== {file_path} ===\n{result}")

        return "\n\n".join(combined_results)

    async def _analyze_file(self, file_path: str, analysis_type: str) -> str:
        """Analyze a single file."""
        try:
            # Construct full path relative to base_path
            if not Path(file_path).is_absolute():
                full_path = self.base_path / file_path
            else:
                full_path = Path(file_path)

            # Read the file
            content = await self.file_tool.run(path=str(full_path))

            # Perform analysis based on type
            if analysis_type == "structure":
                return self._analyze_structure(content)
            elif analysis_type == "patterns":
                return self._analyze_patterns(content)
            elif analysis_type == "dependencies":
                return self._analyze_dependencies(content)
            else:
                return f"Unknown analysis type: {analysis_type}"

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            raise

    def _analyze_structure(self, content: str) -> str:
        """Analyze file structure."""
        lines = content.split("\n")

        # Count different elements
        classes = sum(1 for line in lines if line.strip().startswith("class "))
        functions = sum(1 for line in lines if line.strip().startswith("def ") or line.strip().startswith("async def "))
        imports = sum(1 for line in lines if line.strip().startswith("import ") or line.strip().startswith("from "))

        return f"Structure: {len(lines)} lines, {classes} classes, {functions} functions, {imports} imports"

    def _analyze_patterns(self, content: str) -> str:
        """Analyze code patterns."""
        patterns_found = []

        # Check for common patterns
        if "async def" in content:
            patterns_found.append("Async/await pattern")
        if "@dataclass" in content:
            patterns_found.append("Dataclass pattern")
        if "try:" in content and "except" in content:
            patterns_found.append("Exception handling")
        if "logging." in content or "logger." in content:
            patterns_found.append("Logging")
        if "Type[" in content or "List[" in content or "Dict[" in content:
            patterns_found.append("Type hints")

        return f"Patterns found: {', '.join(patterns_found) if patterns_found else 'None'}"

    def _analyze_dependencies(self, content: str) -> str:
        """Analyze file dependencies."""
        lines = content.split("\n")
        imports = []

        for line in lines:
            line = line.strip()
            if line.startswith("import "):
                imports.append(line.replace("import ", ""))
            elif line.startswith("from "):
                parts = line.split(" import ")
                if len(parts) > 0:
                    module = parts[0].replace("from ", "")
                    imports.append(module)

        unique_imports = list(set(imports))
        return f"Dependencies ({len(unique_imports)}): {', '.join(sorted(unique_imports))}"


class ParallelTestRunner(BaseTool):
    """Run multiple test suites in parallel."""

    description = "Execute multiple test suites simultaneously for faster testing"

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the parallel test runner."""
        self.base_path = base_path or Path.cwd()
        self.test_tool = TestRunnerTool()
        self.shell_tool = ShellTool()

    async def run(
        self,
        test_suites: Optional[str] = None
    ) -> str:
        """Run test suites in parallel.

        Args:
            test_suites: Comma-separated list of test paths, or None for all

        Returns:
            Combined test results
        """
        if test_suites:
            suite_list = [s.strip() for s in test_suites.split(",")]
        else:
            # Default test suites
            suite_list = [
                "agent/tests/",
                "llm_server/tests/",
                "api/tests/",
                "auth/tests/"
            ]

        # Create test tasks
        tasks = []
        for suite_path in suite_list:
            tasks.append(self._run_test_suite(suite_path))

        # Execute all tests in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_results = []
        total_passed = 0
        total_failed = 0

        for suite_path, result in zip(suite_list, results):
            if isinstance(result, Exception):
                combined_results.append(f"Error running {suite_path}: {str(result)}")
                total_failed += 1
            else:
                combined_results.append(f"=== {suite_path} ===\n{result['output']}")
                total_passed += result.get("passed", 0)
                total_failed += result.get("failed", 0)

        # Add summary
        combined_results.append(f"\n=== PARALLEL TEST SUMMARY ===")
        combined_results.append(f"Total Passed: {total_passed}")
        combined_results.append(f"Total Failed: {total_failed}")
        combined_results.append(f"Test Suites Run: {len(suite_list)}")

        return "\n\n".join(combined_results)

    async def _run_test_suite(self, suite_path: str) -> Dict[str, Any]:
        """Run a single test suite."""
        try:
            # Run pytest on the specific suite
            command = f"/c/Python312/python.exe -m pytest {suite_path} -v --tb=short -q"
            result = await self.shell_tool.run(command)

            # Parse results (simple parsing)
            lines = result.split("\n")
            passed = 0
            failed = 0

            for line in lines:
                if " passed" in line:
                    # Extract number of passed tests
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                passed = int(parts[i-1])
                            except (ValueError, IndexError):
                                pass
                if " failed" in line:
                    # Extract number of failed tests
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "failed" and i > 0:
                            try:
                                failed = int(parts[i-1])
                            except (ValueError, IndexError):
                                pass

            return {
                "output": result,
                "passed": passed,
                "failed": failed
            }

        except Exception as e:
            logger.error(f"Failed to run test suite {suite_path}: {e}")
            raise


class ParallelCodeGenerator(BaseTool):
    """Generate multiple code components in parallel."""

    description = "Generate multiple code files or components simultaneously"

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the parallel code generator."""
        self.base_path = base_path or Path.cwd()

    async def run(
        self,
        components: str,
        template: Optional[str] = None
    ) -> str:
        """Generate multiple components in parallel.

        Args:
            components: Comma-separated list of component names
            template: Optional template type (class, function, test)

        Returns:
            Generated code for all components
        """
        component_list = [c.strip() for c in components.split(",")]

        # Create generation tasks
        tasks = []
        for component_name in component_list:
            tasks.append(self._generate_component(component_name, template))

        # Execute all generations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_results = []
        for component_name, result in zip(component_list, results):
            if isinstance(result, Exception):
                combined_results.append(f"Error generating {component_name}: {str(result)}")
            else:
                combined_results.append(f"# {component_name}\n{result}")

        return "\n\n".join(combined_results)

    async def _generate_component(
        self,
        component_name: str,
        template: Optional[str]
    ) -> str:
        """Generate a single component."""
        await asyncio.sleep(0.1)  # Simulate generation time

        if template == "class":
            return self._generate_class(component_name)
        elif template == "function":
            return self._generate_function(component_name)
        elif template == "test":
            return self._generate_test(component_name)
        else:
            # Default to class
            return self._generate_class(component_name)

    def _generate_class(self, name: str) -> str:
        """Generate a class template."""
        return f'''"""Module for {name}."""

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class {name}:
    """{name} implementation."""

    def __init__(self, config: Optional[Any] = None):
        """Initialize {name}."""
        self.config = config or {{}}

    async def process(self, data: Any) -> Any:
        """Process data.

        Args:
            data: Input data

        Returns:
            Processed result
        """
        # TODO: Implement processing logic
        return data

    def validate(self, data: Any) -> bool:
        """Validate data.

        Args:
            data: Data to validate

        Returns:
            True if valid
        """
        # TODO: Implement validation
        return True'''

    def _generate_function(self, name: str) -> str:
        """Generate a function template."""
        return f'''async def {name}(
    input_data: Any,
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """{name} function.

    Args:
        input_data: Input to process
        config: Optional configuration

    Returns:
        Processed result
    """
    config = config or {{}}

    # TODO: Implement function logic
    result = input_data

    return result'''

    def _generate_test(self, name: str) -> str:
        """Generate a test template."""
        return f'''"""Tests for {name}."""

import pytest
from unittest.mock import Mock, AsyncMock


class Test{name}:
    """Test suite for {name}."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        # TODO: Setup test data
        return {{
            "test_data": "sample"
        }}

    @pytest.mark.asyncio
    async def test_{name.lower()}_basic(self, setup):
        """Test basic {name} functionality."""
        # TODO: Implement test
        assert setup["test_data"] == "sample"

    @pytest.mark.asyncio
    async def test_{name.lower()}_error_handling(self):
        """Test {name} error handling."""
        # TODO: Test error cases
        with pytest.raises(ValueError):
            raise ValueError("Test error")'''


class ParallelSearcher(BaseTool):
    """Search for patterns across multiple files in parallel."""

    description = "Search for patterns or text across multiple files simultaneously"

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the parallel searcher."""
        self.base_path = base_path or Path.cwd()
        self.shell_tool = ShellTool()

    async def run(
        self,
        pattern: str,
        paths: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> str:
        """Search for pattern across multiple paths in parallel.

        Args:
            pattern: Pattern to search for
            paths: Comma-separated paths to search, or None for all
            file_type: File extension filter (e.g., "py", "js")

        Returns:
            Combined search results
        """
        if paths:
            path_list = [p.strip() for p in paths.split(",")]
        else:
            # Default paths
            path_list = ["agent/", "api/", "llm_server/", "auth/"]

        # Create search tasks
        tasks = []
        for search_path in path_list:
            tasks.append(self._search_in_path(pattern, search_path, file_type))

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_results = []
        total_matches = 0

        for search_path, result in zip(path_list, results):
            if isinstance(result, Exception):
                combined_results.append(f"Error searching {search_path}: {str(result)}")
            else:
                combined_results.append(f"=== {search_path} ===\n{result['output']}")
                total_matches += result.get("matches", 0)

        # Add summary
        combined_results.append(f"\n=== SEARCH SUMMARY ===")
        combined_results.append(f"Pattern: {pattern}")
        combined_results.append(f"Total Matches: {total_matches}")
        combined_results.append(f"Paths Searched: {len(path_list)}")

        return "\n\n".join(combined_results)

    async def _search_in_path(
        self,
        pattern: str,
        search_path: str,
        file_type: Optional[str]
    ) -> Dict[str, Any]:
        """Search in a single path."""
        try:
            # Build grep command
            if file_type:
                command = f'grep -r "{pattern}" {search_path} --include="*.{file_type}" -n'
            else:
                command = f'grep -r "{pattern}" {search_path} -n'

            result = await self.shell_tool.run(command)

            # Count matches
            lines = result.split("\n")
            matches = len([line for line in lines if line.strip()])

            return {
                "output": result if result else "No matches found",
                "matches": matches
            }

        except Exception as e:
            # grep returns non-zero if no matches, which is not an error
            if "No matches" in str(e) or not str(e):
                return {
                    "output": "No matches found",
                    "matches": 0
                }
            logger.error(f"Failed to search in {search_path}: {e}")
            raise


def get_parallel_tools() -> Dict[str, BaseTool]:
    """Get all available parallel tools.

    Returns:
        Dictionary of tool name to tool instance
    """
    return {
        "parallel_file_analyzer": ParallelFileAnalyzer(),
        "parallel_test_runner": ParallelTestRunner(),
        "parallel_code_generator": ParallelCodeGenerator(),
        "parallel_searcher": ParallelSearcher(),
    }