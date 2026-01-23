# agent-run

Unified CLI for running AI coding agents headlessly. Supports Claude Code (via z.ai or Bedrock), OpenAI Codex, and Google Gemini.

## Installation

```bash
# Install as a tool (recommended)
uv tool install agent-run

# Or install from local source
uv tool install -e ~/git/agent-run

# Or use directly without installing
uvx agent-run "What is 2+2?"
```

## Usage

```bash
# Basic usage (defaults to zai backend)
agent-run "Fix the authentication bug"

# Specify backend
agent-run -b bedrock "Analyze this code"
agent-run -b codex "Write unit tests"
agent-run -b gemini "Explain this function"

# With options
agent-run -b zai -t 600 --cwd /path/to/project "Refactor the module"

# JSON output for scripting
agent-run --json "What is 2+2?"
```

## Backends

| Backend | CLI | Auth | Notes |
|---------|-----|------|-------|
| `zai` (default) | `claude` | `ZAI_API_KEY` | z.ai with GLM-4.7 |
| `bedrock` | `claude` | AWS SSO | AWS Bedrock with Claude |
| `codex` | `codex` | `OPENAI_API_KEY` | OpenAI Codex CLI |
| `gemini` | `gemini` | OAuth | Google Gemini CLI |

## Options

```
-b, --backend    Backend to use: zai, bedrock, codex, gemini (default: zai)
-t, --timeout    Timeout in seconds (default: 300)
-C, --cwd        Working directory for the agent
--model          Model override (backend-specific)
--api-key        API key override (otherwise from environment)
--json           Output JSON result instead of plain text
-v, --version    Show version
-h, --help       Show help
```

## Environment Variables

- `ZAI_API_KEY` - API key for z.ai backend
- `OPENAI_API_KEY` - API key for Codex backend
- `AWS_PROFILE` - AWS profile for Bedrock backend (default: zh-qa-engineer)
- `AWS_REGION` - AWS region for Bedrock backend (default: us-east-1)

## Exit Codes

- `0` - Success
- `1` - Agent error (non-zero exit from CLI)
- `2` - Timeout
- `3` - CLI not found
- `4` - Configuration error (missing API key, etc.)

## Container Support

Automatically detects container environments (Docker, Podman, Kubernetes) and adjusts HOME directory when the default is read-only.

## Library Usage

```python
import asyncio
from agent_run import run, Backend

async def main():
    result = await run(
        prompt="What is 2+2?",
        backend=Backend.ZAI,
        timeout_s=60,
    )
    if result.ok:
        print(result.output)
    else:
        print(f"Error: {result.error}")

asyncio.run(main())
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest -v

# Run integration tests (requires credentials)
uv run pytest -v -m integration
```
