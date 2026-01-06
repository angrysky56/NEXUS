import logging
from typing import Any

from .filesystem import FileSystemTool
from .introspection import (
    INTROSPECTION_TOOLS,
    execute_introspection,
    set_cognitive_state_ref,
    set_tools_schema_ref,
)
from .terminal import TerminalTool

logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(self, allowed_paths: list[str] = ["./"]):
        self.fs = FileSystemTool(allowed_paths)
        self.terminal = TerminalTool()

        # Define Tool Schemas (OpenAI/Generic format)
        self._base_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List directory contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
        ]

        # Combine base tools with introspection tools
        self.tools_schema = self._base_tools + INTROSPECTION_TOOLS

    def update_cognitive_state(self, state: dict[str, Any]) -> None:
        """Update the cognitive state reference for introspection tools."""
        set_cognitive_state_ref(state)
        set_tools_schema_ref(self.tools_schema)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        logger.info(f"Executing tool: {tool_name} {args}")
        try:
            # Introspection tools
            if tool_name.startswith("introspect_"):
                return execute_introspection(tool_name)
            # Filesystem tools
            elif tool_name == "read_file":
                return self.fs.read_file(args.get("path", ""))
            elif tool_name == "write_file":
                return self.fs.write_file(args.get("path", ""), args.get("content", ""))
            elif tool_name == "list_dir":
                return self.fs.list_dir(args.get("path", "."))
            elif tool_name == "run_command":
                return await self.terminal.run_command(args.get("command", ""))
            else:
                return f"Error: Tool {tool_name} not found."
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
