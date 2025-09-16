"""File I/O tools for reading and writing files safely."""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileReadTool:
    """Tool for reading file contents safely."""

    def __init__(self):
        self.name = "file_read"
        self.description = "Read the contents of a file"

    async def run(self, path: str) -> str:
        """Read file contents.

        Args:
            path: File path to read

        Returns:
            File contents as string
        """
        try:
            # Security: Resolve path and check it's within allowed area
            file_path = Path(path).resolve()

            # Basic security check - prevent reading outside project
            cwd = Path.cwd().resolve()
            try:
                file_path.relative_to(cwd)
            except ValueError:
                return f"Error: Access denied - path is outside project directory"

            if not file_path.exists():
                return f"Error: File '{path}' does not exist"

            if not file_path.is_file():
                return f"Error: '{path}' is not a file"

            # Check file size
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return f"Error: File '{path}' is too large (>1MB)"

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"Read file: {path} ({len(content)} characters)")
            return content

        except UnicodeDecodeError:
            return f"Error: Cannot read '{path}' - file appears to be binary"
        except PermissionError:
            return f"Error: Permission denied reading '{path}'"
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return f"Error reading file: {str(e)}"


class FileWriteTool:
    """Tool for writing file contents safely."""

    def __init__(self):
        self.name = "file_write"
        self.description = "Write content to a file"

    async def run(self, path: str, content: str, mode: str = "w") -> str:
        """Write content to a file.

        Args:
            path: File path to write to
            content: Content to write
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            Success/error message
        """
        try:
            # Security: Resolve path and check it's within allowed area
            file_path = Path(path).resolve()

            # Basic security check - prevent writing outside project
            cwd = Path.cwd().resolve()
            try:
                file_path.relative_to(cwd)
            except ValueError:
                return f"Error: Access denied - path is outside project directory"

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate mode
            if mode not in ['w', 'a']:
                return f"Error: Invalid mode '{mode}'. Use 'w' or 'a'"

            # Check content size
            if len(content) > 1024 * 1024:  # 1MB limit
                return f"Error: Content too large (>1MB)"

            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)

            action = "Appended to" if mode == 'a' else "Wrote"
            logger.info(f"{action} file: {path} ({len(content)} characters)")
            return f"{action} {len(content)} characters to '{path}'"

        except PermissionError:
            return f"Error: Permission denied writing to '{path}'"
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return f"Error writing file: {str(e)}"