import logging
import os

logger = logging.getLogger(__name__)


class FileSystemTool:
    def __init__(self, allowed_paths: list[str], root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self.allowed_paths = [os.path.abspath(p) for p in allowed_paths]
        # Always allow root_dir itself
        if self.root_dir not in self.allowed_paths:
            self.allowed_paths.append(self.root_dir)

    def _resolve_path(self, path: str) -> str:
        """Resolve path against root_dir if relative, return abs path."""
        if not os.path.isabs(path):
            return os.path.abspath(os.path.join(self.root_dir, path))
        return os.path.abspath(path)

    def _is_safe(self, path: str) -> bool:
        abs_path = self._resolve_path(path)
        # Check if path starts with any allowed path
        return any(abs_path.startswith(allowed) for allowed in self.allowed_paths)

    def list_dir(self, path: str = ".") -> str:
        resolved = self._resolve_path(path)
        if not self._is_safe(path):
            return f"Error: Access denied to {path} (Resolved: {resolved})"
        try:
            return str(os.listdir(resolved))
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, path: str) -> str:
        resolved = self._resolve_path(path)
        if not self._is_safe(path):
            return f"Error: Access denied to {path}"
        try:
            with open(resolved, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        resolved = self._resolve_path(path)
        if not self._is_safe(path):
            return f"Error: Access denied to {path}"
        try:
            # Ensure dir exists
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: Wrote to {resolved}"
        except Exception as e:
            return f"Error: {e}"
