# hatch

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
cli.py / runner.py → credentials.py → backends.py → subprocess(claude/codex/gemini)
                        ↓
                   infisical-get.py
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
4. **Credential loading lives in `credentials.py`** - do not fetch secrets inside backend config builders

---

## Learnings

<!-- Agents: append below. Human compacts weekly. -->

- (2026-01-27) [tool] `uv run pytest -v` omits dev extras; use `uv run --extra dev pytest -v` (or `uv run --python .venv/bin/python -m pytest -v`) after `uv sync --all-extras`.
- (2026-03-29) [design] Keep backend builders pure; hatch credential policy belongs in one preflight resolver that uses the canonical `infisical-get.py` helper instead of ad hoc backend fallbacks.
