"""
Read tool for reading files.

Mirrors read.ts from pi-coding-agent.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Optional, Callable, Union

from pi_agent_core.types import AgentTool, AgentToolResult, ImageContent, TextContent

from .path_utils import resolve_read_path
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)

# JSON Schema for read tool parameters
READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "number",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of lines to read",
        },
    },
    "required": ["path"],
}

# Supported image MIME types
IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _detect_image_mime_type(path: str) -> Optional[str]:
    """Detect image MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return IMAGE_EXTENSIONS.get(ext)


def create_read_tool(
    cwd: str,
    auto_resize_images: bool = True,
) -> AgentTool:
    """
    Create a read tool configured for a specific working directory.

    Args:
        cwd: Working directory used to resolve relative paths
        auto_resize_images: Whether to note image dimensions (default: True)

    Returns:
        An AgentTool instance
    """
    description = (
        f"Read the contents of a file. "
        f"Supports text files and images (jpg, png, gif, webp). "
        f"Images are sent as attachments. "
        f"For text files, output is truncated to {DEFAULT_MAX_LINES} lines or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        f"Use offset/limit for large files. "
        f"When you need the full file, continue with offset until complete."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        path: str = params.get("path", "")
        offset: Optional[int] = params.get("offset")
        limit: Optional[int] = params.get("limit")

        absolute_path = resolve_read_path(path, cwd)

        # Check if file exists
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {path}")

        if not os.access(absolute_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {path}")

        # Detect image type
        mime_type = _detect_image_mime_type(absolute_path)

        if mime_type:
            # Read as image
            with open(absolute_path, "rb") as f:
                raw = f.read()

            encoded = base64.b64encode(raw).decode("utf-8")
            text_note = f"Read image file [{mime_type}]"

            return AgentToolResult(
                content=[
                    TextContent(text=text_note),
                    ImageContent(data=encoded, mime_type=mime_type),
                ],
                details=None,
            )
        else:
            # Read as text
            with open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
                text_content = f.read()

            all_lines = text_content.split("\n")
            total_file_lines = len(all_lines)

            # Apply offset (1-indexed -> 0-indexed)
            start_line = max(0, (offset - 1)) if offset else 0
            start_line_display = start_line + 1

            if start_line >= len(all_lines):
                raise ValueError(
                    f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
                )

            # Apply limit
            user_limited_lines: Optional[int] = None
            if limit is not None:
                end_line = min(start_line + limit, len(all_lines))
                selected_content = "\n".join(all_lines[start_line:end_line])
                user_limited_lines = end_line - start_line
            else:
                selected_content = "\n".join(all_lines[start_line:])

            # Apply truncation
            truncation = truncate_head(selected_content)

            if truncation.first_line_exceeds_limit:
                first_line_size = format_size(
                    len(all_lines[start_line].encode("utf-8"))
                )
                output_text = (
                    f"[Line {start_line_display} is {first_line_size}, "
                    f"exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                    f"Use bash: sed -n '{start_line_display}p' {path} | "
                    f"head -c {DEFAULT_MAX_BYTES}]"
                )
                details = {"truncation": truncation}

            elif truncation.truncated:
                end_line_display = start_line_display + truncation.output_lines - 1
                next_offset = end_line_display + 1

                output_text = truncation.content
                if truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                        f"{total_file_lines}. Use offset={next_offset} to continue.]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                        f"{total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                        f"Use offset={next_offset} to continue.]"
                    )
                details = {"truncation": truncation}

            elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
                remaining = len(all_lines) - (start_line + user_limited_lines)
                next_offset = start_line + user_limited_lines + 1
                output_text = truncation.content
                output_text += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                details = None

            else:
                output_text = truncation.content
                details = None

            return AgentToolResult(
                content=[TextContent(text=output_text)],
                details=details,
            )

    tool = AgentTool(
        name="read",
        label="read",
        description=description,
        parameters=READ_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default read tool using cwd
read_tool = create_read_tool(os.getcwd())
