"""
Path resolution utilities for coding tools.

Mirrors path-utils.ts from pi-coding-agent.
"""

from __future__ import annotations

import os


def resolve_to_cwd(path: str, cwd: str) -> str:
    """
    Resolve a path relative to cwd.

    If the path is absolute, returns it unchanged (but validates it's within cwd).
    If the path is relative, resolves it against cwd.
    """
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path))


def resolve_read_path(path: str, cwd: str) -> str:
    """
    Resolve a path for reading. Same as resolve_to_cwd but named for clarity.
    """
    return resolve_to_cwd(path, cwd)
