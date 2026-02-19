"""Tests for truncation utilities."""
import pytest
from pi_coding_agent.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)


def test_format_size():
    assert format_size(500) == "500B"
    assert format_size(1024) == "1.0KB"
    assert format_size(1024 * 1024) == "1.0MB"


def test_truncate_head_no_truncation():
    content = "line1\nline2\nline3"
    result = truncate_head(content)
    assert result.truncated is False
    assert result.content == content
    assert result.total_lines == 3


def test_truncate_head_line_limit():
    content = "\n".join([f"line{i}" for i in range(100)])
    result = truncate_head(content, max_lines=10)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.output_lines == 10


def test_truncate_head_byte_limit():
    # Create content that's larger than limit
    content = "x" * 60000
    result = truncate_head(content, max_bytes=50 * 1024)
    assert result.truncated is True
    assert result.truncated_by == "bytes"


def test_truncate_tail_keeps_end():
    # Create many lines, truncate, should keep the last lines
    lines = [f"line{i}" for i in range(100)]
    content = "\n".join(lines)
    result = truncate_tail(content, max_lines=10)
    assert result.truncated is True
    output_lines = result.content.split("\n")
    # Last line should be line99
    assert "line99" in result.content
    assert "line0" not in result.content


def test_truncate_line():
    short = "hello"
    text, was_truncated = truncate_line(short, max_chars=100)
    assert text == short
    assert was_truncated is False

    long_line = "x" * 600
    text, was_truncated = truncate_line(long_line, max_chars=500)
    assert was_truncated is True
    assert "[truncated]" in text
