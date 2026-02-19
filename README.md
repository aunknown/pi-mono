# pi-mono-python

Python rewrite of `pi-agent-core` and `pi-coding-agent` from the [pi-mono](https://github.com/mariozechner/pi-mono) TypeScript monorepo.

## Packages

### `pi_agent_core`
Core agent loop and event system. Provides:
- `Agent` class for managing LLM conversations
- `AgentLoop` with tool execution and steering/follow-up queuing
- `EventStream` for async streaming of agent events
- Proxy streaming support

### `pi_coding_agent`
Coding-specific tools and session management. Provides:
- File tools: `read`, `write`, `edit`
- Shell tools: `bash`
- Search tools: `grep`, `find`, `ls`
- `AgentSession` for full session lifecycle management
- Output truncation utilities

## Requirements
- Python 3.11+
- `pydantic` for data models
- `httpx` for HTTP requests
- `anyio` for async primitives

## Installation

```bash
pip install -e .
```

## Usage

```python
import asyncio
from pi_agent_core import Agent, AgentOptions
from pi_coding_agent.tools import create_coding_tools

async def main():
    tools = create_coding_tools("/path/to/project")
    agent = Agent(AgentOptions(tools=tools))
    agent.set_system_prompt("You are a helpful coding assistant.")
    await agent.prompt("List the files in the current directory.")

asyncio.run(main())
```
