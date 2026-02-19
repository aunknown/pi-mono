"""Tests for pi_coding_agent tools."""
import asyncio
import os
import tempfile
import pytest

from pi_coding_agent.tools.read import create_read_tool
from pi_coding_agent.tools.write import create_write_tool
from pi_coding_agent.tools.edit import create_edit_tool
from pi_coding_agent.tools.ls import create_ls_tool
from pi_agent_core.types import TextContent


def run_async(coro):
    """Helper to run coroutines in tests."""
    return asyncio.run(coro)


def test_write_and_read_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_tool = create_write_tool(tmpdir)
        read_tool = create_read_tool(tmpdir)

        # Write a file
        result = run_async(write_tool.execute("id1", {"path": "test.txt", "content": "hello world"}))
        assert len(result.content) > 0
        text = result.content[0].text
        assert "Successfully wrote" in text

        # Read it back
        result = run_async(read_tool.execute("id2", {"path": "test.txt"}))
        assert result.content[0].text == "hello world"


def test_edit_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_tool = create_write_tool(tmpdir)
        edit_tool = create_edit_tool(tmpdir)
        read_tool = create_read_tool(tmpdir)

        # Write initial content
        run_async(write_tool.execute("id1", {
            "path": "hello.txt",
            "content": "Hello World\nThis is a test\n"
        }))

        # Edit it
        run_async(edit_tool.execute("id2", {
            "path": "hello.txt",
            "old_text": "Hello World",
            "new_text": "Hello Python",
        }))

        # Verify
        result = run_async(read_tool.execute("id3", {"path": "hello.txt"}))
        assert "Hello Python" in result.content[0].text
        assert "Hello World" not in result.content[0].text


def test_ls_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "file.txt"), "w") as f:
            f.write("content")

        ls_tool = create_ls_tool(tmpdir)
        result = run_async(ls_tool.execute("id1", {}))
        output = result.content[0].text

        assert "file.txt" in output
        assert "subdir/" in output


def test_edit_tool_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_tool = create_write_tool(tmpdir)
        edit_tool = create_edit_tool(tmpdir)

        run_async(write_tool.execute("id1", {
            "path": "f.txt",
            "content": "original text"
        }))

        with pytest.raises(ValueError, match="Could not find"):
            run_async(edit_tool.execute("id2", {
                "path": "f.txt",
                "old_text": "nonexistent text",
                "new_text": "replacement",
            }))
