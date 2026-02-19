"""
pi_coding_agent - Python implementation of @mariozechner/pi-coding-agent

Coding-specific tools and session management for the pi agent.
"""

from .agent_session import AgentSession, AgentSessionConfig
from .event_bus import EventBus, create_event_bus
from .session_manager import (
    CompactionEntry,
    CustomMessageEntry,
    ModelChangeEntry,
    SessionContext,
    SessionManager,
    SessionMessageEntry,
    ThinkingLevelEntry,
)
from .tools import (
    bash_tool,
    coding_tools,
    create_all_tools,
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_only_tools,
    create_read_tool,
    create_write_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    read_only_tools,
    read_tool,
    write_tool,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)

__all__ = [
    # AgentSession
    "AgentSession",
    "AgentSessionConfig",
    # EventBus
    "EventBus",
    "create_event_bus",
    # SessionManager
    "CompactionEntry",
    "CustomMessageEntry",
    "ModelChangeEntry",
    "SessionContext",
    "SessionManager",
    "SessionMessageEntry",
    "ThinkingLevelEntry",
    # Tools (default)
    "bash_tool",
    "coding_tools",
    "edit_tool",
    "find_tool",
    "grep_tool",
    "ls_tool",
    "read_only_tools",
    "read_tool",
    "write_tool",
    # Tool factories
    "create_all_tools",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_only_tools",
    "create_read_tool",
    "create_write_tool",
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
