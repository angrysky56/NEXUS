import logging
import os

logger = logging.getLogger(__name__)

class FileSystemTool:
    def __init__(self, allowed_paths: list[str]):
        self.allowed_paths = [os.path.abspath(p) for p in allowed_paths]

    def _is_safe(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(allowed) for allowed in self.allowed_paths)

    def list_dir(self, path: str = ".") -> str:
        if not self._is_safe(path):
            return f"Error: Access denied to {path}"
        try:
            return str(os.listdir(path))
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, path: str) -> str:
        if not self._is_safe(path):
            return f"Error: Access denied to {path}"
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        if not self._is_safe(path):
             return f"Error: Access denied to {path}"
        try:
            # Ensure dir exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: Wrote to {path}"
        except Exception as e:
            return f"Error: {e}"
