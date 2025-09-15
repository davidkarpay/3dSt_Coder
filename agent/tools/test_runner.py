"""Test runner tool for executing various testing frameworks."""

import asyncio
import logging
import os
import json
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TestRunnerTool:
    """Execute tests using various testing frameworks (pytest, unittest, etc.)."""

    def __init__(self, timeout: int = 300, max_output_size: int = 20000):
        """Initialize the test runner tool.

        Args:
            timeout: Test execution timeout in seconds
            max_output_size: Maximum output size in characters
        """
        self.name = "test_runner"
        self.description = "Execute tests using pytest, unittest, or other test frameworks"
        self.timeout = timeout
        self.max_output_size = max_output_size

    async def run(
        self,
        test_path: Optional[str] = None,
        framework: str = "auto",
        verbose: bool = False,
        coverage: bool = False,
        pattern: Optional[str] = None,
        markers: Optional[str] = None,
    ) -> str:
        """Execute tests in the specified path.

        Args:
            test_path: Path to test file/directory (default: auto-detect)
            framework: Testing framework (auto, pytest, unittest, nose2)
            verbose: Enable verbose output
            coverage: Enable coverage reporting
            pattern: Test name pattern to match
            markers: Pytest markers to select (e.g., "not slow")

        Returns:
            Test execution results including output and summary
        """
        if test_path and not os.path.exists(test_path):
            return f"Error: Test path '{test_path}' does not exist"

        logger.info(f"Running tests: path={test_path}, framework={framework}")

        try:
            # Auto-detect testing framework if not specified
            if framework == "auto":
                framework = await self._detect_framework(test_path)

            # Build command based on framework
            command = await self._build_command(
                framework, test_path, verbose, coverage, pattern, markers
            )

            if not command:
                return f"Error: Unsupported testing framework '{framework}'"

            # Execute tests
            result = await self._execute_tests(command)

            # Parse and format results
            return await self._format_results(result, framework, coverage)

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return f"Error executing tests: {str(e)}"

    async def _detect_framework(self, test_path: Optional[str]) -> str:
        """Auto-detect the testing framework to use.

        Args:
            test_path: Path to check for test files

        Returns:
            Detected framework name
        """
        search_path = test_path or "."

        # Check for pytest configuration files
        pytest_configs = [
            "pytest.ini",
            "pyproject.toml",
            "tox.ini",
            "setup.cfg",
        ]

        for config in pytest_configs:
            if os.path.exists(config):
                # Check if pytest is configured
                try:
                    with open(config, "r") as f:
                        content = f.read()
                        if "pytest" in content or "[tool.pytest" in content:
                            return "pytest"
                except:
                    pass

        # Check for test files that suggest pytest
        if self._has_pytest_files(search_path):
            return "pytest"

        # Check for unittest files
        if self._has_unittest_files(search_path):
            return "unittest"

        # Default to pytest as it's most versatile
        return "pytest"

    def _has_pytest_files(self, path: str) -> bool:
        """Check if directory contains pytest-style test files."""
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    return True
                if file.endswith("_test.py"):
                    return True
        return False

    def _has_unittest_files(self, path: str) -> bool:
        """Check if directory contains unittest-style test files."""
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("test") and file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            if "unittest" in content and "TestCase" in content:
                                return True
                    except:
                        pass
        return False

    async def _build_command(
        self,
        framework: str,
        test_path: Optional[str],
        verbose: bool,
        coverage: bool,
        pattern: Optional[str],
        markers: Optional[str],
    ) -> Optional[List[str]]:
        """Build the test command for the specified framework.

        Args:
            framework: Testing framework to use
            test_path: Path to test files
            verbose: Enable verbose output
            coverage: Enable coverage reporting
            pattern: Test pattern to match
            markers: Test markers to select

        Returns:
            Command arguments list
        """
        if framework == "pytest":
            cmd = ["python", "-m", "pytest"]

            if verbose:
                cmd.append("-v")

            if coverage:
                cmd.extend(["--cov=.", "--cov-report=term-missing"])

            if pattern:
                cmd.extend(["-k", pattern])

            if markers:
                cmd.extend(["-m", markers])

            # Add path if specified
            if test_path:
                cmd.append(test_path)

            # Output format for parsing
            cmd.extend(["--tb=short", "--no-header"])

            return cmd

        elif framework == "unittest":
            cmd = ["python", "-m", "unittest"]

            if verbose:
                cmd.append("-v")

            if test_path:
                # Convert path to module notation if needed
                if test_path.endswith(".py"):
                    module = test_path.replace("/", ".").replace("\\\\", ".").rstrip(".py")
                    cmd.append(module)
                else:
                    cmd.extend(["discover", "-s", test_path])
            else:
                cmd.extend(["discover"])

            return cmd

        elif framework == "nose2":
            cmd = ["python", "-m", "nose2"]

            if verbose:
                cmd.append("-v")

            if test_path:
                cmd.extend(["-s", test_path])

            return cmd

        else:
            return None

    async def _execute_tests(self, command: List[str]) -> Dict[str, Any]:
        """Execute the test command and capture results.

        Args:
            command: Command to execute

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing: {' '.join(command)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "command": " ".join(command),
            }

        except asyncio.TimeoutError:
            logger.warning(f"Tests timed out after {self.timeout}s")
            try:
                process.kill()
            except:
                pass

            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Tests timed out after {self.timeout} seconds",
                "command": " ".join(command),
            }

    async def _format_results(
        self, result: Dict[str, Any], framework: str, coverage: bool
    ) -> str:
        """Format test results for human consumption.

        Args:
            result: Test execution result
            framework: Framework that was used
            coverage: Whether coverage was enabled

        Returns:
            Formatted results string
        """
        output = f"Test Results (using {framework}):\\n"
        output += f"Command: {result['command']}\\n\\n"

        # Combine stdout and stderr
        full_output = result["stdout"]
        if result["stderr"]:
            full_output += "\\n--- STDERR ---\\n" + result["stderr"]

        # Truncate if too long
        if len(full_output) > self.max_output_size:
            full_output = full_output[:self.max_output_size] + "\\n... (output truncated)"

        output += full_output

        # Add summary based on return code
        if result["returncode"] == 0:
            output += "\\n\\n✅ ALL TESTS PASSED"
        else:
            output += f"\\n\\n❌ TESTS FAILED (exit code: {result['returncode']})"

        # Extract test summary if available
        summary = self._extract_summary(result["stdout"], framework)
        if summary:
            output += f"\\n\\nSummary: {summary}"

        return output

    def _extract_summary(self, stdout: str, framework: str) -> Optional[str]:
        """Extract test summary from output.

        Args:
            stdout: Test output
            framework: Framework used

        Returns:
            Extracted summary or None
        """
        if framework == "pytest":
            # Look for pytest summary line
            lines = stdout.split("\\n")
            for line in lines:
                if " passed" in line or " failed" in line or " error" in line:
                    if any(word in line for word in ["passed", "failed", "error", "skipped"]):
                        return line.strip()

        elif framework == "unittest":
            # Look for unittest summary
            lines = stdout.split("\\n")
            for line in lines:
                if line.startswith("Ran ") and " test" in line:
                    return line.strip()

        return None


# Test cases for the test runner tool
if __name__ == "__main__":
    async def test_runner_tool():
        """Test the test runner tool functionality."""
        tool = TestRunnerTool()

        # Test framework detection
        framework = await tool._detect_framework(".")
        print(f"Detected framework: {framework}")

        # Test building pytest command
        cmd = await tool._build_command("pytest", "tests/", True, True, None, None)
        print(f"Pytest command: {' '.join(cmd) if cmd else 'None'}")

        # Test running a simple test (if available)
        if os.path.exists("test_*.py") or os.path.exists("tests/"):
            result = await tool.run(verbose=True)
            print(f"Test execution result:\\n{result}")

    asyncio.run(test_runner_tool())