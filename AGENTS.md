# agent-run

Headless AI agent CLI runner. Zero dependencies, stdlib only.

**Owner**: david010@gmail.com

## Quick Reference

```bash
uv sync --all-extras   # Install dev deps
uv run pytest -v       # Run tests (154)
uv run pytest -v -m integration  # Real API calls (needs creds)
```

## Architecture

```
cli.py → runner.py → backends.py → subprocess(claude/codex/gemini)
           ↓
       context.py (container detection)
```

**4 Backends:** `zai` (default), `bedrock`, `codex`, `gemini`

**Key files:**
| File | Purpose |
|------|---------|
| `backends.py` | Env vars + cmd building per backend |
| `runner.py` | Async subprocess wrapper + timeout |
| `context.py` | Container/filesystem detection |
| `cli.py` | Argument parsing + main() |

## Conventions

- **Zero dependencies** - stdlib only (no requests, no click)
- **Prompt via stdin** - avoids ARG_MAX on large prompts
- **Container-aware** - auto-sets HOME=/tmp for read-only filesystems

## Gotchas

1. **z.ai uses `ANTHROPIC_AUTH_TOKEN`** not `ANTHROPIC_API_KEY` - and must unset `CLAUDE_CODE_USE_BEDROCK`
2. **Tests mock subprocess** - no real CLI calls except `integration` marked tests
3. **Don't add dependencies** - the zero-deps constraint is intentional

---

## Learnings

<!-- Agents: append below. Human compacts weekly. -->

