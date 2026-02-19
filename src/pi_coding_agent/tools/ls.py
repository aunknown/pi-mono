"""
Ls tool for listing directory contents.

Mirrors ls.ts from pi-coding-agent.
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable, Optional

from pi_agent_core.types import AgentTool, AgentToolResult, TextContent

from .path_utils import resolve_to_cwd
from .truncate import (
    DEFAULT_MAX_BYTES,
    format_size,
    truncate_head,
)

DEFAULT_LIMIT = 500

# JSON Schema for ls tool parameters
LS_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list (default: current directory)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of entries to return (default: 500)",
        },
    },
}


def create_ls_tool(cwd: str) -> AgentTool:
    """
    Create an ls tool configured for a specific working directory.

    Args:
        cwd: Working directory used to resolve relative paths

    Returns:
        An AgentTool instance
    """
    description = (
        f"List directory contents. "
        f"Returns entries sorted alphabetically, with '/' suffix for directories. "
        f"Includes dotfiles. "
        f"Output is truncated to {DEFAULT_LIMIT} entries or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        path: Optional[str] = params.get("path")
        limit: int = params.get("limit", DEFAULT_LIMIT)

        dir_path = resolve_to_cwd(path or ".", cwd)
        effective_limit = max(1, limit)

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Path not found: {dir_path}")

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        try:
            entries = os.listdir(dir_path)
        except PermissionError as e:
            raise RuntimeError(f"Cannot read directory: {e}")

        # Sort alphabetically (case-insensitive)
        entries.sort(key=lambda e: e.lower())

        results: list[str] = []
        entry_limit_reached = False

        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break

            full_path = os.path.join(dir_path, entry)
            try:
                suffix = "/" if os.path.isdir(full_path) else ""
            except OSError:
                continue

            results.append(entry + suffix)

        if not results:
            return AgentToolResult(
                content=[TextContent(text="(empty directory)")],
                details=None,
            )

        # Apply byte truncation
        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=2**31)

        output = truncation.content
        details: dict = {}
        notices: list[str] = []

        if entry_limit_reached:
            notices.append(
                f"{effective_limit} entries limit reached. "
                f"Use limit={effective_limit * 2} for more"
            )
            details["entry_limit_reached"] = effective_limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details if details else None,
        )

    tool = AgentTool(
        name="ls",
        label="ls",
        description=description,
        parameters=LS_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default ls tool using cwd
ls_tool = create_ls_tool(os.getcwd())
