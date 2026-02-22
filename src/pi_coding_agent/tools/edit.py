"""
Edit tool for making surgical text replacements in files.

Mirrors edit.ts from pi-coding-agent.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
from typing import Callable, Optional

from pi_agent_core.types import AgentTool, AgentToolResult, TextContent

from .path_utils import resolve_to_cwd

# JSON Schema for edit tool parameters
EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit (relative or absolute)",
        },
        "old_text": {
            "type": "string",
            "description": "Exact text to find and replace (must match exactly)",
        },
        "new_text": {
            "type": "string",
            "description": "New text to replace the old text with",
        },
        # Also accept camelCase (TypeScript compat)
        "oldText": {
            "type": "string",
            "description": "Exact text to find and replace (must match exactly)",
        },
        "newText": {
            "type": "string",
            "description": "New text to replace the old text with",
        },
    },
    "required": ["path"],
}


def _normalize_newlines(text: str) -> str:
    """Normalize all line endings to \\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _detect_line_ending(text: str) -> str:
    """Detect the predominant line ending style."""
    crlf_count = text.count("\r\n")
    cr_count = text.count("\r") - crlf_count
    lf_count = text.count("\n") - crlf_count

    if crlf_count > lf_count and crlf_count > cr_count:
        return "\r\n"
    elif cr_count > lf_count:
        return "\r"
    return "\n"


def _restore_line_endings(text: str, ending: str) -> str:
    """Restore line endings in text."""
    if ending == "\n":
        return text
    normalized = _normalize_newlines(text)
    return normalized.replace("\n", ending)


def _strip_bom(text: str) -> tuple[str, str]:
    """Strip BOM from text, returning (bom, text_without_bom)."""
    if text.startswith("\ufeff"):
        return "\ufeff", text[1:]
    return "", text


def _generate_diff(old_content: str, new_content: str) -> tuple[str, Optional[int]]:
    """Generate a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="a",
        tofile="b",
        n=3,
    ))

    # Find the first changed line number
    first_changed_line: Optional[int] = None
    current_line = 0
    for line in diff_lines:
        if line.startswith("@@"):
            # Parse @@ -a,b +c,d @@
            m = re.search(r"\+(\d+)", line)
            if m:
                current_line = int(m.group(1)) - 1
        elif line.startswith("+") and not line.startswith("+++"):
            if first_changed_line is None:
                first_changed_line = current_line
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass
        else:
            current_line += 1

    return "".join(diff_lines), first_changed_line


def _fuzzy_find(content: str, search: str) -> tuple[bool, int, int]:
    """
    Find search text in content using fuzzy matching.

    Returns (found, index, match_length) where index and match_length
    refer to positions in the original `content` string.
    """
    # Try exact match first
    idx = content.find(search)
    if idx != -1:
        return True, idx, len(search)

    # Try collapsing whitespace differences
    def normalize_ws(s: str) -> str:
        # Collapse multiple whitespace to single space, strip leading/trailing
        lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in s.split("\n")]
        return "\n".join(lines)

    norm_content = normalize_ws(content)
    norm_search = normalize_ws(search)
    norm_idx = norm_content.find(norm_search)
    if norm_idx != -1:
        # Map the normalized index back to the original content.
        # Build a mapping from normalized positions to original positions.
        orig_idx = 0
        norm_pos = 0
        # Advance through original content to find the start position
        while norm_pos < norm_idx and orig_idx < len(content):
            orig_idx += 1
            norm_pos = len(normalize_ws(content[:orig_idx]))

        start = orig_idx
        # Find the end position in original content
        target_end = norm_idx + len(norm_search)
        while norm_pos < target_end and orig_idx < len(content):
            orig_idx += 1
            norm_pos = len(normalize_ws(content[:orig_idx]))

        match_len = orig_idx - start
        return True, start, match_len

    return False, -1, 0


def create_edit_tool(cwd: str) -> AgentTool:
    """
    Create an edit tool configured for a specific working directory.

    Args:
        cwd: Working directory used to resolve relative paths

    Returns:
        An AgentTool instance
    """
    description = (
        "Edit a file by replacing exact text. "
        "The oldText must match exactly (including whitespace). "
        "Use this for precise, surgical edits."
    )

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> AgentToolResult:
        path: str = params.get("path", "")
        # Support both snake_case and camelCase parameter names
        old_text: str = params.get("old_text") if "old_text" in params else params.get("oldText", "")
        new_text: str = params.get("new_text") if "new_text" in params else params.get("newText", "")

        absolute_path = resolve_to_cwd(path, cwd)

        # Check file access
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {path}")
        if not os.access(absolute_path, os.R_OK | os.W_OK):
            raise PermissionError(f"Cannot read/write file: {path}")

        # Read file
        with open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        # Strip BOM
        bom, content = _strip_bom(raw_content)
        original_ending = _detect_line_ending(content)

        # Normalize line endings for comparison
        normalized_content = _normalize_newlines(content)
        normalized_old = _normalize_newlines(old_text)
        normalized_new = _normalize_newlines(new_text)

        # Find the old text
        found, idx, match_len = _fuzzy_find(normalized_content, normalized_old)
        if not found:
            raise ValueError(
                f"Could not find the exact text in {path}. "
                "The old text must match exactly including all whitespace and newlines."
            )

        # Count occurrences
        occurrences = normalized_content.count(normalized_old)
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. "
                "The text must be unique. Please provide more context to make it unique."
            )

        # Perform replacement
        new_content = (
            normalized_content[:idx] +
            normalized_new +
            normalized_content[idx + len(normalized_old):]
        )

        if normalized_content == new_content:
            raise ValueError(
                f"No changes made to {path}. "
                "The replacement produced identical content."
            )

        # Restore BOM and line endings
        final_content = bom + _restore_line_endings(new_content, original_ending)

        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        diff, first_changed_line = _generate_diff(normalized_content, new_content)

        return AgentToolResult(
            content=[TextContent(text=f"Successfully replaced text in {path}.")],
            details={
                "diff": diff,
                "first_changed_line": first_changed_line,
            },
        )

    tool = AgentTool(
        name="edit",
        label="edit",
        description=description,
        parameters=EDIT_SCHEMA,
    )
    tool.execute = execute
    return tool


# Default edit tool using cwd
edit_tool = create_edit_tool(os.getcwd())
