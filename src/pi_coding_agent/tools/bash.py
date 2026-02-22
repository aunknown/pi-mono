"""
Bash tool for executing shell commands.

Mirrors bash.ts from pi-coding-agent.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Callable, Optional

from pi_agent_core.types import AgentTool, AgentToolResult, TextContent

from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_tail,
)

# JSON Schema for bash tool parameters
BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Bash command to execute",
        },
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds (optional, no default timeout)",
        },
    },
    "required": ["command"],
}


async def _run_bash_command(
    command: str,
    cwd: str,
    timeout: Optional[float],
    signal: Optional[asyncio.Event],
) -> tuple[bytes, int | None]:
    """
    Run a bash command asynchronously, returning (output_bytes, exit_code).

    Merges stdout and stderr.
    """
    shell = "/bin/bash" if sys.platform != "win32" else "cmd.exe"

    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            executable=shell,
        )

        chunks: list[bytes] = []

        async def read_output():
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)

        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(read_output(), proc.wait()),
                    timeout=timeout,
                )
            else:
                await asyncio.gather(read_output(), proc.wait())
        except asyncio.TimeoutError:
            try:
                if proc is not None:
                    proc.kill()
            except ProcessLookupError:
                pass
            raise TimeoutError(f"Command timed out after {timeout} seconds")

        return b"".join(chunks), proc.returncode

    except asyncio.CancelledError:
        try:
            if proc is not None:
                proc.kill()
        except Exception:
            pass
        raise


def create_bash_tool(cwd: str, command_prefix: Optional[str] = None) -> AgentTool:
    """
    Create a bash tool configured for a specific working directory.

    Args:
        cwd: Working directory for command execution
        command_prefix: Optional prefix prepended to every command
                        (e.g., "shopt -s expand_aliases" for alias support)

    Returns:
        An AgentTool instance
    """
    description = (
        f"Execute a bash command in the current working directory. "
        f"Returns stdout and stderr. "
        f"Output is truncated to last {DEFAULT_MAX_LINES} lines or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        f"If truncated, full output is saved to a temp file. "
        f"Optionally provide a timeout in seconds."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        command: str = params.get("command", "")
        timeout: Optional[float] = params.get("timeout")

        # Apply command prefix
        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command

        # Validate working directory
        if not os.path.isdir(cwd):
            raise RuntimeError(
                f"Working directory does not exist: {cwd}\n"
                "Cannot execute bash commands."
            )

        # Temp file for large outputs
        temp_file_path: Optional[str] = None
        output_bytes = b""

        try:
            output_bytes, exit_code = await _run_bash_command(
                resolved_command, cwd, timeout, signal
            )
        except TimeoutError as e:
            output_text = str(e)
            if output_bytes:
                output_text = output_bytes.decode("utf-8", errors="replace") + f"\n\n{e}"
            raise RuntimeError(output_text)

        full_output = output_bytes.decode("utf-8", errors="replace")

        # Save to temp file if large
        if len(output_bytes) > DEFAULT_MAX_BYTES:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="pi-bash-",
                suffix=".log",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(full_output)
                temp_file_path = f.name

        # Apply tail truncation
        truncation = truncate_tail(full_output)
        output_text = truncation.content or "(no output)"

        details: dict = {}

        if truncation.truncated:
            details["truncation"] = truncation
            details["full_output_path"] = temp_file_path

            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines

            if truncation.last_line_partial:
                last_line_size = format_size(
                    len(full_output.split("\n")[-1].encode("utf-8"))
                )
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line "
                    f"{end_line} (line is {last_line_size}). "
                    f"Full output: {temp_file_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of "
                    f"{truncation.total_lines}. Full output: {temp_file_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of "
                    f"{truncation.total_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Full output: {temp_file_path}]"
                )

        if exit_code is not None and exit_code != 0:
            output_text += f"\n\nCommand exited with code {exit_code}"
            raise RuntimeError(output_text)

        return AgentToolResult(
            content=[TextContent(text=output_text)],
            details=details or None,
        )

    tool = AgentTool(
        name="bash",
        label="bash",
        description=description,
        parameters=BASH_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default bash tool using cwd
bash_tool = create_bash_tool(os.getcwd())
