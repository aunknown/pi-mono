"""
Write tool for creating/overwriting files.

Mirrors write.ts from pi-coding-agent.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable, Optional

from pi_agent_core.types import AgentTool, AgentToolResult, TextContent

from .path_utils import resolve_to_cwd

# JSON Schema for write tool parameters
WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
        },
    },
    "required": ["path", "content"],
}


def create_write_tool(cwd: str) -> AgentTool:
    """
    Create a write tool configured for a specific working directory.

    Args:
        cwd: Working directory used to resolve relative paths

    Returns:
        An AgentTool instance
    """
    description = (
        "Write content to a file. "
        "Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        path: str = params.get("path", "")
        content: str = params.get("content", "")

        absolute_path = resolve_to_cwd(path, cwd)

        # Create parent directories if needed
        parent_dir = os.path.dirname(absolute_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Write the file
        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(content)

        return AgentToolResult(
            content=[TextContent(
                text=f"Successfully wrote {len(content.encode('utf-8'))} bytes to {path}"
            )],
            details=None,
        )

    tool = AgentTool(
        name="write",
        label="write",
        description=description,
        parameters=WRITE_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default write tool using cwd
write_tool = create_write_tool(os.getcwd())
