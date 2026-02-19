"""
Find tool for searching files by glob pattern.

Mirrors find.ts from pi-coding-agent.
Uses fd if available, falls back to Python's glob module.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shutil
import subprocess
from typing import Callable, Optional

from pi_agent_core.types import AgentTool, AgentToolResult, TextContent

from .path_utils import resolve_to_cwd
from .truncate import (
    DEFAULT_MAX_BYTES,
    format_size,
    truncate_head,
)

DEFAULT_LIMIT = 1000

# JSON Schema for find tool parameters
FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": (
                "Glob pattern to match files, e.g. '*.py', '**/*.json', "
                "or 'src/**/*.spec.py'"
            ),
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (default: current directory)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of results (default: 1000)",
        },
    },
    "required": ["pattern"],
}

IGNORE_DIRS = {
    "node_modules", ".git", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build",
}


def _python_find(
    pattern: str,
    search_path: str,
    limit: int,
) -> list[str]:
    """
    Pure Python file finder using os.walk + fnmatch.

    Returns a list of relative paths.
    """
    results: list[str] = []

    # Convert glob to fnmatch pattern (simplified)
    # Handle ** by walking recursively
    simple_pattern = pattern.lstrip("**/")
    if not simple_pattern:
        simple_pattern = pattern

    for root, dirs, files in os.walk(search_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for fname in files:
            if fnmatch.fnmatch(fname, simple_pattern) or fnmatch.fnmatch(
                os.path.relpath(os.path.join(root, fname), search_path),
                pattern,
            ):
                rel_path = os.path.relpath(os.path.join(root, fname), search_path)
                results.append(rel_path.replace("\\", "/"))
                if len(results) >= limit:
                    return results

    return results


def create_find_tool(cwd: str) -> AgentTool:
    """
    Create a find tool configured for a specific working directory.

    Args:
        cwd: Working directory used to resolve relative paths

    Returns:
        An AgentTool instance
    """
    description = (
        f"Search for files by glob pattern. "
        f"Returns matching file paths relative to the search directory. "
        f"Respects .gitignore. "
        f"Output is truncated to {DEFAULT_LIMIT} results or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        pattern: str = params.get("pattern", "")
        search_dir: Optional[str] = params.get("path")
        limit: int = params.get("limit", DEFAULT_LIMIT)

        search_path = resolve_to_cwd(search_dir or ".", cwd)
        effective_limit = max(1, limit)

        if not os.path.exists(search_path):
            raise FileNotFoundError(f"Path not found: {search_path}")

        results: list[str] = []
        result_limit_reached = False

        # Try fd first
        fd_path = shutil.which("fd")
        if fd_path:
            args = [
                fd_path,
                "--glob",
                "--color=never",
                "--hidden",
                "--max-results",
                str(effective_limit),
                pattern,
                search_path,
            ]

            try:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout.strip()

                if output:
                    for line in output.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        # Make relative to search_path
                        if line.startswith(search_path):
                            rel = line[len(search_path):].lstrip("/\\")
                        else:
                            rel = os.path.relpath(line, search_path)
                        results.append(rel.replace("\\", "/"))

                    if len(results) >= effective_limit:
                        result_limit_reached = True

            except subprocess.TimeoutExpired:
                raise RuntimeError("find timed out")
        else:
            # Python fallback
            results = _python_find(pattern, search_path, effective_limit)
            if len(results) >= effective_limit:
                result_limit_reached = True

        if not results:
            return AgentToolResult(
                content=[TextContent(text="No files found matching pattern")],
                details=None,
            )

        # Apply truncation
        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=2**31)

        output = truncation.content
        details: dict = {}
        notices: list[str] = []

        if result_limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. "
                f"Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details["result_limit_reached"] = effective_limit

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
        name="find",
        label="find",
        description=description,
        parameters=FIND_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default find tool using cwd
find_tool = create_find_tool(os.getcwd())
