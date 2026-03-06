# hatch

Headless agent runner for Claude Code, Codex, Gemini, and z.ai.

## Install

```bash
uv tool install -e ~/git/hatch
```

## Usage

Defaults to a 15 minute internal timeout. Do not wrap normal `hatch` calls in short outer shell timeouts.

```bash
hatch "What is 2+2?"
hatch -b codex --reasoning-effort low "Write unit tests"
hatch -b codex --skip-git-repo-check "What is 2+2?"  # Handy for one-off prompts outside a repo
hatch --json "Analyze this" | jq .output

# Reasoning effort is a flag, not prompt prose
hatch -b codex --reasoning-effort low "Review this function"
```

## Library

```python
from hatch import run, Backend

result = await run(prompt="Fix the bug", backend=Backend.ZAI)
print(result.output if result.ok else result.error)
```

<!-- readme-test: verifies install from repo and CLI entrypoint -->
```readme-test
{
  "name": "hatch-agent-install",
  "mode": "smoke",
  "workdir": ".",
  "timeout": 60,
  "steps": [
    "uv venv .tmp-hatch-readme-venv --python 3.12 -q",
    ". .tmp-hatch-readme-venv/bin/activate",
    "uv pip install -e . -q",
    "hatch --help | head -3"
  ],
  "cleanup": [
    "rm -rf .tmp-hatch-readme-venv"
  ]
}
```
