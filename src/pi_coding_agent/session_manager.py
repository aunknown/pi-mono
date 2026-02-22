"""
Session manager for persisting and restoring agent sessions.

Mirrors session-manager.ts from pi-coding-agent.

Sessions are stored as JSONL files (one JSON object per line).
Each line represents a session entry (message, model change, compaction, etc.).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

from pi_agent_core.types import AgentMessage, AssistantMessage, UserMessage, ToolResultMessage


# ---------------------------------------------------------------------------
# Session entry types
# ---------------------------------------------------------------------------

@dataclass
class SessionMessageEntry:
    """A standard LLM message."""
    type: str = "message"
    role: str = ""
    content: Any = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Optional fields for assistant messages
    model: Optional[str] = None
    provider: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: Optional[dict] = None
    error_message: Optional[str] = None

    # Optional fields for tool result messages
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    is_error: Optional[bool] = None
    details: Any = None


@dataclass
class ModelChangeEntry:
    """Records a model/provider change."""
    type: str = "model_change"
    provider: str = ""
    model: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ThinkingLevelEntry:
    """Records a thinking level change."""
    type: str = "thinking_level"
    level: str = "off"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class CompactionEntry:
    """Records a compaction (context summarization)."""
    type: str = "compaction"
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    details: Any = None
    from_extension: bool = False


@dataclass
class CustomMessageEntry:
    """A custom message from extensions or app code."""
    type: str = "custom_message"
    custom_type: str = ""
    content: Any = None
    display: Any = None
    details: Any = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


SessionEntry = Union[
    SessionMessageEntry,
    ModelChangeEntry,
    ThinkingLevelEntry,
    CompactionEntry,
    CustomMessageEntry,
]


# ---------------------------------------------------------------------------
# SessionContext
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """Context built from session entries for the agent."""
    messages: list[AgentMessage] = field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    thinking_level: str = "off"


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages agent session persistence to JSONL files.

    Session files contain one JSON object per line, representing the full
    history of the session (messages, model changes, compactions, etc.).
    """

    def __init__(
        self,
        sessions_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        session_file: Optional[str] = None,
    ) -> None:
        """
        Create a SessionManager.

        Args:
            sessions_dir: Directory to store session files
            session_id: Optional specific session ID to use
            session_file: Optional specific file path for the session
        """
        self._sessions_dir = sessions_dir
        self._session_id = session_id or str(uuid.uuid4())
        self._session_file = session_file
        self._entries: list[SessionEntry] = []
        self._session_name: Optional[str] = None

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    def get_session_file(self) -> Optional[str]:
        """Get the current session file path."""
        return self._session_file

    def get_session_name(self) -> Optional[str]:
        """Get the session display name."""
        return self._session_name

    def set_session_name(self, name: str) -> None:
        """Set the session display name."""
        self._session_name = name

    def get_entries(self) -> list[SessionEntry]:
        """Get all session entries."""
        return list(self._entries)

    def get_branch(self) -> list[SessionEntry]:
        """
        Get entries for the current branch.

        For now, returns all entries. A full implementation would support
        branching by tracking parent sessions.
        """
        return self.get_entries()

    def new_session(self, parent_session: Optional[str] = None) -> None:
        """Start a new session, clearing all entries."""
        self._session_id = str(uuid.uuid4())
        self._entries = []
        self._session_name = None

        if self._sessions_dir:
            session_filename = f"{self._session_id}.jsonl"
            self._session_file = os.path.join(self._sessions_dir, session_filename)

    def load_session(self, file_path: str) -> bool:
        """
        Load a session from a JSONL file.

        Returns True if loaded successfully, False otherwise.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entries = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = self._deserialize_entry(data)
                        if entry:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            self._entries = entries
            self._session_file = file_path
            # Extract session ID from filename if not set
            fname = Path(file_path).stem
            if not self._session_id or self._session_id == "unknown":
                self._session_id = fname

            return True
        except (IOError, OSError):
            return False

    def _deserialize_entry(self, data: dict) -> Optional[SessionEntry]:
        """Deserialize a dict to a SessionEntry."""
        entry_type = data.get("type", "message")

        if entry_type == "message":
            return SessionMessageEntry(**{
                k: v for k, v in data.items()
                if k in SessionMessageEntry.__dataclass_fields__
            })
        elif entry_type == "model_change":
            return ModelChangeEntry(**{
                k: v for k, v in data.items()
                if k in ModelChangeEntry.__dataclass_fields__
            })
        elif entry_type == "thinking_level":
            return ThinkingLevelEntry(**{
                k: v for k, v in data.items()
                if k in ThinkingLevelEntry.__dataclass_fields__
            })
        elif entry_type == "compaction":
            return CompactionEntry(**{
                k: v for k, v in data.items()
                if k in CompactionEntry.__dataclass_fields__
            })
        elif entry_type == "custom_message":
            return CustomMessageEntry(**{
                k: v for k, v in data.items()
                if k in CustomMessageEntry.__dataclass_fields__
            })
        return None

    def _append_to_file(self, entry: dict) -> None:
        """Append an entry to the session file."""
        if not self._session_file:
            return

        # Create directory if needed
        parent = os.path.dirname(self._session_file)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(self._session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def append_message(self, message: AgentMessage) -> None:
        """Append a message to the session."""
        role = getattr(message, "role", "unknown")

        entry_data: dict = {
            "type": "message",
            "id": str(uuid.uuid4()),
            "role": role,
            "timestamp": getattr(message, "timestamp", int(time.time() * 1000)),
        }

        if isinstance(message, UserMessage):
            entry_data["content"] = [
                block.model_dump() for block in message.content
            ]
        elif isinstance(message, AssistantMessage):
            entry_data["content"] = [
                block.model_dump() for block in message.content
            ]
            entry_data["model"] = message.model
            entry_data["provider"] = message.provider
            entry_data["stop_reason"] = message.stop_reason
            entry_data["usage"] = message.usage.model_dump()
            if message.error_message:
                entry_data["error_message"] = message.error_message
        elif isinstance(message, ToolResultMessage):
            entry_data["content"] = [
                block.model_dump() for block in message.content
            ]
            entry_data["tool_call_id"] = message.tool_call_id
            entry_data["tool_name"] = message.tool_name
            entry_data["is_error"] = message.is_error
            if message.details:
                entry_data["details"] = message.details

        entry = self._deserialize_entry(entry_data)
        if entry:
            self._entries.append(entry)
            self._append_to_file(entry_data)

    def append_model_change(self, provider: str, model: str) -> None:
        """Record a model change."""
        entry = ModelChangeEntry(provider=provider, model=model)
        self._entries.append(entry)
        self._append_to_file(asdict(entry))

    def append_thinking_level_change(self, level: str) -> None:
        """Record a thinking level change."""
        entry = ThinkingLevelEntry(level=level)
        self._entries.append(entry)
        self._append_to_file(asdict(entry))

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_extension: bool = False,
    ) -> None:
        """Record a compaction."""
        entry = CompactionEntry(
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            from_extension=from_extension,
        )
        self._entries.append(entry)
        self._append_to_file(asdict(entry))

    def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any = None,
        display: Any = None,
        details: Any = None,
    ) -> None:
        """Record a custom message."""
        entry = CustomMessageEntry(
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
        )
        self._entries.append(entry)
        self._append_to_file(asdict(entry))

    def build_session_context(self) -> SessionContext:
        """
        Build agent context from session entries.

        Handles compaction by finding the latest compaction point and
        building context from there.
        """
        context = SessionContext()

        # Find the latest compaction
        latest_compaction: Optional[CompactionEntry] = None
        for entry in reversed(self._entries):
            if isinstance(entry, CompactionEntry):
                latest_compaction = entry
                break

        # Find where to start processing entries
        start_idx = 0
        if latest_compaction:
            # Find the first kept entry after compaction
            for i, entry in enumerate(self._entries):
                if isinstance(entry, SessionMessageEntry):
                    if entry.id == latest_compaction.first_kept_entry_id:
                        start_idx = i
                        break

            # Add compaction summary as a user message
            from pi_agent_core.types import TextContent
            summary_message = UserMessage(
                content=[
                    TextContent(text=latest_compaction.summary)
                ],
                timestamp=latest_compaction.timestamp,
            )
            context.messages.append(summary_message)

        # Process entries from start_idx
        for entry in self._entries[start_idx:]:
            if isinstance(entry, ModelChangeEntry):
                context.model = entry.model
                context.provider = entry.provider
            elif isinstance(entry, ThinkingLevelEntry):
                context.thinking_level = entry.level
            elif isinstance(entry, SessionMessageEntry):
                msg = self._entry_to_message(entry)
                if msg:
                    context.messages.append(msg)

        return context

    def _entry_to_message(self, entry: SessionMessageEntry) -> Optional[AgentMessage]:
        """Convert a SessionMessageEntry back to an AgentMessage."""
        from pi_agent_core.types import (
            TextContent, ImageContent, ThinkingContent, ToolCall,
            Usage,
        )

        if entry.role == "user":
            content_blocks = []
            for block_data in (entry.content or []):
                if isinstance(block_data, dict):
                    if block_data.get("type") == "text":
                        content_blocks.append(TextContent(text=block_data["text"]))
                    elif block_data.get("type") == "image":
                        content_blocks.append(ImageContent(
                            data=block_data["data"],
                            mime_type=block_data["mime_type"],
                        ))
            return UserMessage(content=content_blocks, timestamp=entry.timestamp)

        elif entry.role == "assistant":
            content_blocks = []
            for block_data in (entry.content or []):
                if isinstance(block_data, dict):
                    btype = block_data.get("type")
                    if btype == "text":
                        content_blocks.append(TextContent(text=block_data["text"]))
                    elif btype == "thinking":
                        content_blocks.append(ThinkingContent(
                            thinking=block_data["thinking"]
                        ))
                    elif btype == "toolCall":
                        content_blocks.append(ToolCall(
                            id=block_data["id"],
                            name=block_data["name"],
                            arguments=block_data.get("arguments", {}),
                        ))

            usage_data = entry.usage or {}
            usage = Usage(
                input=usage_data.get("input", 0),
                output=usage_data.get("output", 0),
                cache_read=usage_data.get("cache_read", 0),
                cache_write=usage_data.get("cache_write", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            return AssistantMessage(
                content=content_blocks,
                model=entry.model or "",
                provider=entry.provider or "",
                stop_reason=entry.stop_reason or "stop",
                usage=usage,
                error_message=entry.error_message,
                timestamp=entry.timestamp,
            )

        elif entry.role == "toolResult":
            content_blocks = []
            for block_data in (entry.content or []):
                if isinstance(block_data, dict):
                    if block_data.get("type") == "text":
                        content_blocks.append(TextContent(text=block_data["text"]))
                    elif block_data.get("type") == "image":
                        content_blocks.append(ImageContent(
                            data=block_data["data"],
                            mime_type=block_data["mime_type"],
                        ))

            from pi_agent_core.types import ToolResultMessage
            return ToolResultMessage(
                tool_call_id=entry.tool_call_id or "",
                tool_name=entry.tool_name or "",
                content=content_blocks,
                details=entry.details,
                is_error=entry.is_error or False,
                timestamp=entry.timestamp,
            )

        return None
