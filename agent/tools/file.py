class FileReadTool:
    name = "file_read"
    description = "Read a file's contents"

    async def run(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()