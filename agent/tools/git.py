class GitStatusTool:
    name = "git_status"
    description = "Return git status"

    async def run(self, repo_path: str = '.') -> str:
        import subprocess
        p = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_path, capture_output=True, text=True)
        return p.stdout or 'clean'