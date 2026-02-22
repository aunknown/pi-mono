"""
Grep tool for searching file contents.

Mirrors grep.ts from pi-coding-agent.
Uses ripgrep (rg) if available, falls back to Python's re module.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Optional

from pi_agent_core.types import AgentTool, AgentToolResult, TextContent

from .path_utils import resolve_to_cwd
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)

DEFAULT_LIMIT = 100

# JSON Schema for grep tool parameters
GREP_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Search pattern (regex or literal string)",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search (default: current directory)",
        },
        "glob": {
            "type": "string",
            "description": "Filter files by glob pattern, e.g. '*.py' or '**/*.spec.py'",
        },
        "ignore_case": {
            "type": "boolean",
            "description": "Case-insensitive search (default: false)",
        },
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as literal string instead of regex (default: false)",
        },
        "context": {
            "type": "number",
            "description": "Number of lines to show before and after each match (default: 0)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of matches to return (default: 100)",
        },
    },
    "required": ["pattern"],
}


def _find_ripgrep() -> Optional[str]:
    """Find ripgrep binary."""
    return shutil.which("rg")


def _python_grep(
    pattern: str,
    search_path: str,
    glob_pattern: Optional[str],
    ignore_case: bool,
    literal: bool,
    context_lines: int,
    limit: int,
) -> list[tuple[str, int, str]]:
    """
    Pure Python grep fallback.

    Returns list of (file_path, line_number, matched_line).
    """
    flags = re.IGNORECASE if ignore_case else 0
    if literal:
        search_pattern = re.compile(re.escape(pattern), flags)
    else:
        try:
            search_pattern = re.compile(pattern, flags)
        except re.error:
            search_pattern = re.compile(re.escape(pattern), flags)

    matches: list[tuple[str, int, str]] = []

    if os.path.isfile(search_path):
        files_to_search = [search_path]
    else:
        files_to_search = []
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories and node_modules
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d != "node_modules"
            ]
            for fname in files:
                fpath = os.path.join(root, fname)
                if glob_pattern:
                    import fnmatch
                    # Extract filename pattern from the glob
                    file_pattern = glob_pattern
                    if file_pattern.startswith("**/"):
                        file_pattern = file_pattern[3:]
                    if not fnmatch.fnmatch(fname, file_pattern):
                        continue
                files_to_search.append(fpath)

    for fpath in files_to_search:
        if len(matches) >= limit:
            break
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for lineno, line in enumerate(lines, 1):
                if search_pattern.search(line):
                    matches.append((fpath, lineno, line.rstrip("\n")))
                    if len(matches) >= limit:
                        break
        except (IOError, OSError):
            continue

    return matches


def create_grep_tool(cwd: str) -> AgentTool:
    """
    Create a grep tool configured for a specific working directory.

    Args:
        cwd: Working directory used to resolve relative paths

    Returns:
        An AgentTool instance
    """
    description = (
        f"Search file contents for a pattern. "
        f"Returns matching lines with file paths and line numbers. "
        f"Respects .gitignore. "
        f"Output is truncated to {DEFAULT_LIMIT} matches or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        pattern: str = params.get("pattern", "")
        search_dir: Optional[str] = params.get("path")
        glob_pattern: Optional[str] = params.get("glob")
        ignore_case: bool = params.get("ignore_case", False) or params.get("ignoreCase", False)
        literal: bool = params.get("literal", False)
        context_lines: int = params.get("context", 0)
        limit: int = params.get("limit", DEFAULT_LIMIT)

        search_path = resolve_to_cwd(search_dir or ".", cwd)

        if not os.path.exists(search_path):
            raise FileNotFoundError(f"Path not found: {search_path}")

        is_directory = os.path.isdir(search_path)
        effective_limit = max(1, limit)

        # Try ripgrep first
        rg_path = _find_ripgrep()
        output_lines: list[str] = []
        match_count = 0
        match_limit_reached = False
        lines_truncated = False

        def format_path(file_path: str) -> str:
            if is_directory:
                rel = os.path.relpath(file_path, search_path)
                if not rel.startswith(".."):
                    return rel.replace("\\", "/")
            return os.path.basename(file_path)

        if rg_path:
            # Use ripgrep
            args = [
                rg_path,
                "--json",
                "--line-number",
                "--color=never",
                "--hidden",
            ]
            if ignore_case:
                args.append("--ignore-case")
            if literal:
                args.append("--fixed-strings")
            if glob_pattern:
                args.extend(["--glob", glob_pattern])
            args.extend([pattern, search_path])

            try:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                rg_output = result.stdout

                matches: list[tuple[str, int]] = []
                for line in rg_output.splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "match":
                        data = event.get("data", {})
                        file_path = data.get("path", {}).get("text", "")
                        line_number = data.get("line_number", 0)
                        if file_path and line_number:
                            matches.append((file_path, line_number))
                            match_count += 1
                            if match_count >= effective_limit:
                                match_limit_reached = True
                                break

                # Format matches
                for file_path, line_number in matches:
                    rel_path = format_path(file_path)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            all_lines = f.readlines()

                        start = max(0, line_number - 1 - context_lines)
                        end = min(len(all_lines), line_number + context_lines)

                        for idx in range(start, end):
                            line_text = all_lines[idx].rstrip("\n").rstrip("\r")
                            truncated_text, was_truncated = truncate_line(line_text)
                            if was_truncated:
                                lines_truncated = True

                            actual_line_num = idx + 1
                            if actual_line_num == line_number:
                                output_lines.append(f"{rel_path}:{actual_line_num}: {truncated_text}")
                            else:
                                output_lines.append(f"{rel_path}-{actual_line_num}- {truncated_text}")
                    except (IOError, OSError):
                        output_lines.append(f"{rel_path}:{line_number}: (unable to read file)")

            except subprocess.TimeoutExpired:
                raise RuntimeError("grep timed out")
        else:
            # Python fallback
            py_matches = _python_grep(
                pattern, search_path, glob_pattern,
                ignore_case, literal, context_lines, effective_limit
            )
            match_count = len(py_matches)
            if match_count >= effective_limit:
                match_limit_reached = True

            for file_path, line_number, line_text in py_matches:
                rel_path = format_path(file_path)
                truncated_text, was_truncated = truncate_line(line_text)
                if was_truncated:
                    lines_truncated = True
                output_lines.append(f"{rel_path}:{line_number}: {truncated_text}")

        if match_count == 0:
            return AgentToolResult(
                content=[TextContent(text="No matches found")],
                details=None,
            )

        # Apply byte truncation
        raw_output = "\n".join(output_lines)
        truncation = truncate_head(raw_output, max_lines=2**31)

        output = truncation.content
        details: dict = {}
        notices: list[str] = []

        if match_limit_reached:
            notices.append(
                f"{effective_limit} matches limit reached. "
                f"Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details["match_limit_reached"] = effective_limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation

        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. "
                "Use read tool to see full lines"
            )
            details["lines_truncated"] = True

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details if details else None,
        )

    tool = AgentTool(
        name="grep",
        label="grep",
        description=description,
        parameters=GREP_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default grep tool using cwd
grep_tool = create_grep_tool(os.getcwd())
