"""Shell execution tool with Docker sandboxing for security."""

import asyncio
import logging
import os
import tempfile
import shutil
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ShellTool:
    """Execute shell commands in a sandboxed Docker container."""

    def __init__(
        self,
        docker_image: str = "python:3.12-slim",
        timeout: int = 30,
        max_output_size: int = 10000,
    ):
        """Initialize the shell tool.

        Args:
            docker_image: Docker image to use for execution
            timeout: Command timeout in seconds
            max_output_size: Maximum output size in characters
        """
        self.name = "shell"
        self.description = "Execute shell commands in a secure Docker container"
        self.docker_image = docker_image
        self.timeout = timeout
        self.max_output_size = max_output_size

    async def run(
        self,
        command: str,
        working_dir: Optional[str] = "/workspace",
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Execute a shell command in Docker container.

        Args:
            command: Shell command to execute
            working_dir: Working directory inside container
            env: Environment variables to set

        Returns:
            Command output (stdout + stderr)
        """
        if not command.strip():
            return "Error: Empty command provided"

        # Security checks
        dangerous_commands = [
            "rm -rf /",
            ":(){ :|:& };:",  # Fork bomb
            "dd if=/dev/zero",
            "cat /etc/passwd",
            "curl",
            "wget",
        ]

        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return f"Error: Command '{command}' is not allowed for security reasons"

        logger.info(f"Executing shell command: {command}")

        try:
            # Check if Docker is available
            docker_check = await asyncio.create_subprocess_exec(
                "docker", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await docker_check.communicate()

            if docker_check.returncode != 0:
                logger.warning("Docker not available, executing locally (unsafe!)")
                return await self._execute_local(command, env)

            return await self._execute_docker(command, working_dir, env)

        except Exception as e:
            logger.error(f"Shell execution failed: {e}")
            return f"Error executing command: {str(e)}"

    async def _execute_docker(
        self,
        command: str,
        working_dir: str,
        env: Optional[Dict[str, str]],
    ) -> str:
        """Execute command in Docker container."""
        # Build Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            "--network=none",  # No network access
            "--memory=128m",  # Memory limit
            "--cpus=0.5",  # CPU limit
            "--read-only",  # Read-only filesystem
            "--tmpfs", "/tmp",  # Writable tmp
            "--tmpfs", "/workspace",  # Writable workspace
            "-w", working_dir,  # Working directory
        ]

        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        docker_cmd.extend([self.docker_image, "sh", "-c", command])

        # Execute with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )

            output = stdout.decode("utf-8", errors="replace")

            # Truncate if too long
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + "\\n... (output truncated)"

            return output

        except asyncio.TimeoutError:
            logger.warning(f"Command timed out after {self.timeout}s")
            # Try to kill the container
            try:
                await process.kill()
            except:
                pass
            return f"Error: Command timed out after {self.timeout} seconds"

    async def _execute_local(
        self,
        command: str,
        env: Optional[Dict[str, str]],
    ) -> str:
        """Execute command locally (fallback when Docker unavailable)."""
        logger.warning("Executing command locally - security restrictions apply")

        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Prepare environment
                exec_env = os.environ.copy()
                if env:
                    exec_env.update(env)

                # Execute command
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=temp_dir,
                    env=exec_env,
                )

                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )

                output = stdout.decode("utf-8", errors="replace")

                # Truncate if too long
                if len(output) > self.max_output_size:
                    output = output[:self.max_output_size] + "\\n... (output truncated)"

                return output

            except asyncio.TimeoutError:
                try:
                    process.kill()
                except:
                    pass
                return f"Error: Command timed out after {self.timeout} seconds"


# Test cases for the shell tool
if __name__ == "__main__":
    async def test_shell_tool():
        """Test the shell tool functionality."""
        tool = ShellTool()

        # Test basic command
        result = await tool.run("echo 'Hello World'")
        print(f"Echo test: {result}")

        # Test Python execution
        result = await tool.run("python -c 'print(2 + 2)'")
        print(f"Python test: {result}")

        # Test dangerous command blocking
        result = await tool.run("rm -rf /")
        print(f"Dangerous command test: {result}")

        # Test timeout
        result = await tool.run("sleep 5", timeout=2)
        print(f"Timeout test: {result}")

    asyncio.run(test_shell_tool())