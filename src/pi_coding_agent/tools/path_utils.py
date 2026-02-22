"""
Path resolution utilities for coding tools.

Mirrors path-utils.ts from pi-coding-agent.
"""

from __future__ import annotations

import os


def resolve_to_cwd(path: str, cwd: str) -> str:
    """
    Resolve a path relative to cwd and validate it's within cwd.

    If the path is absolute, normalizes and validates it.
    If the path is relative, resolves it against cwd.
    Raises ValueError if the resolved path is outside cwd.
    """
    if os.path.isabs(path):
        resolved = os.path.normpath(path)
    else:
        resolved = os.path.normpath(os.path.join(cwd, path))
    norm_cwd = os.path.normpath(cwd)
    if resolved != norm_cwd and not resolved.startswith(norm_cwd + os.sep):
        raise ValueError(f"Path '{path}' resolves outside working directory '{cwd}'")
    return resolved


def resolve_read_path(path: str, cwd: str) -> str:
    """
    Resolve a path for reading. Same as resolve_to_cwd but named for clarity.
    """
    return resolve_to_cwd(path, cwd)
