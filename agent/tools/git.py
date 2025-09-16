"""Git repository tools for status, diff, and commit operations."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitStatusTool:
    """Tool for checking git repository status."""

    def __init__(self):
        self.name = "git_status"
        self.description = "Get the current git repository status"

    async def run(self, repo_path: str = '.') -> str:
        """Get git status.

        Args:
            repo_path: Path to git repository (default: current directory)

        Returns:
            Git status output
        """
        try:
            # Security check - ensure path is within project
            path = Path(repo_path).resolve()
            cwd = Path.cwd().resolve()
            try:
                path.relative_to(cwd)
            except ValueError:
                return "Error: Access denied - path is outside project directory"

            # Run git status command
            process = await asyncio.create_subprocess_exec(
                'git', 'status', '--porcelain',
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                return f"Git error: {error_msg}"

            status_output = stdout.decode('utf-8').strip()
            if not status_output:
                return "Repository is clean - no changes to commit"

            return f"Git status:\n{status_output}"

        except FileNotFoundError:
            return "Error: Git command not found"
        except Exception as e:
            logger.error(f"Git status error: {e}")
            return f"Error running git status: {str(e)}"


class GitDiffTool:
    """Tool for viewing git diffs."""

    def __init__(self):
        self.name = "git_diff"
        self.description = "Show git diff for staged or unstaged changes"

    async def run(self, repo_path: str = '.', staged: bool = False, file_path: Optional[str] = None) -> str:
        """Get git diff.

        Args:
            repo_path: Path to git repository (default: current directory)
            staged: Show staged changes (--cached) vs unstaged changes
            file_path: Specific file to diff (optional)

        Returns:
            Git diff output
        """
        try:
            # Security check - ensure path is within project
            path = Path(repo_path).resolve()
            cwd = Path.cwd().resolve()
            try:
                path.relative_to(cwd)
            except ValueError:
                return "Error: Access denied - path is outside project directory"

            # Build git diff command
            cmd = ['git', 'diff']
            if staged:
                cmd.append('--cached')

            if file_path:
                # Security check for file path too
                file_full_path = (path / file_path).resolve()
                try:
                    file_full_path.relative_to(cwd)
                    cmd.append(file_path)
                except ValueError:
                    return "Error: File path is outside project directory"

            # Run git diff command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                return f"Git error: {error_msg}"

            diff_output = stdout.decode('utf-8').strip()
            if not diff_output:
                change_type = "staged" if staged else "unstaged"
                return f"No {change_type} changes found"

            return f"Git diff ({'staged' if staged else 'unstaged'}):\n{diff_output}"

        except FileNotFoundError:
            return "Error: Git command not found"
        except Exception as e:
            logger.error(f"Git diff error: {e}")
            return f"Error running git diff: {str(e)}"


class GitCommitTool:
    """Tool for creating git commits."""

    def __init__(self):
        self.name = "git_commit"
        self.description = "Create a git commit with a message"

    async def run(self, message: str, repo_path: str = '.', add_all: bool = False) -> str:
        """Create a git commit.

        Args:
            message: Commit message
            repo_path: Path to git repository (default: current directory)
            add_all: Add all changes before committing

        Returns:
            Commit result message
        """
        try:
            # Security check - ensure path is within project
            path = Path(repo_path).resolve()
            cwd = Path.cwd().resolve()
            try:
                path.relative_to(cwd)
            except ValueError:
                return "Error: Access denied - path is outside project directory"

            # Validate commit message
            if not message or not message.strip():
                return "Error: Commit message cannot be empty"

            if len(message) > 500:
                return "Error: Commit message too long (max 500 characters)"

            # Add all files if requested
            if add_all:
                add_process = await asyncio.create_subprocess_exec(
                    'git', 'add', '.',
                    cwd=str(path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await add_process.communicate()

                if add_process.returncode != 0:
                    return "Error: Failed to add files to staging area"

            # Create commit
            commit_process = await asyncio.create_subprocess_exec(
                'git', 'commit', '-m', message.strip(),
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await commit_process.communicate()

            if commit_process.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                if "nothing to commit" in error_msg.lower():
                    return "Nothing to commit - working tree clean"
                return f"Commit failed: {error_msg}"

            commit_output = stdout.decode('utf-8').strip()
            return f"Commit successful:\n{commit_output}"

        except FileNotFoundError:
            return "Error: Git command not found"
        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return f"Error creating commit: {str(e)}"