"""
Coding tools for pi_coding_agent.

Provides file system tools for the coding agent.
"""

from .bash import bash_tool, create_bash_tool
from .edit import edit_tool, create_edit_tool
from .find import find_tool, create_find_tool
from .grep import grep_tool, create_grep_tool
from .ls import ls_tool, create_ls_tool
from .read import read_tool, create_read_tool
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)
from .write import write_tool, create_write_tool

from pi_agent_core.types import AgentTool

__all__ = [
    # Individual tools (default cwd)
    "bash_tool",
    "edit_tool",
    "find_tool",
    "grep_tool",
    "ls_tool",
    "read_tool",
    "write_tool",
    # Factory functions
    "create_bash_tool",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_tool",
    "create_write_tool",
    # Convenience collections
    "create_coding_tools",
    "create_read_only_tools",
    "create_all_tools",
    "coding_tools",
    "read_only_tools",
    # Truncation utilities
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
]


# Default tool collections (using cwd)
coding_tools: list[AgentTool] = [read_tool, bash_tool, edit_tool, write_tool]
read_only_tools: list[AgentTool] = [read_tool, grep_tool, find_tool, ls_tool]


def create_coding_tools(cwd: str) -> list[AgentTool]:
    """
    Create coding tools configured for a specific working directory.

    Includes: read, bash, edit, write
    """
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_read_only_tools(cwd: str) -> list[AgentTool]:
    """
    Create read-only tools configured for a specific working directory.

    Includes: read, grep, find, ls
    """
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_all_tools(cwd: str) -> dict[str, AgentTool]:
    """
    Create all tools configured for a specific working directory.

    Returns a dict mapping tool name to AgentTool.
    """
    return {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "grep": create_grep_tool(cwd),
        "find": create_find_tool(cwd),
        "ls": create_ls_tool(cwd),
    }
