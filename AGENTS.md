# hatch

Headless runner with a simple Claude/Codex/z.ai surface plus a built-in MCP server for agent callers.

**Owner**: david010@gmail.com

## Install

```bash
uv tool install -e ~/git/hatch
```

## Entrypoints

Public surface:
- `hatch claude <haiku|sonnet|opus>` → Claude on Amazon Bedrock
- `hatch codex <nano|mini|max>` → GPT-5 on OpenAI
- `hatch "prompt"` → z.ai default path
- `hatch mcp` → run the MCP server over stdio
- Raw `-b bedrock` / `-b codex` / `-b gemini` still invoke the underlying CLIs directly as escape hatches

Default tiers:
- Start with `sonnet` for Claude and `mini` for Codex
- Drop to `haiku` / `nano` for faster cheaper work
- Rise to `opus` / `max` when depth matters

Defaults to a 15 minute internal timeout. Do not wrap normal `hatch` calls in short outer shell timeouts.

```bash
hatch "What is 2+2?"
hatch codex mini "Review this branch"
hatch claude haiku "Summarize this file"
hatch codex max --reasoning-effort low "Write unit tests"
hatch claude sonnet "Review this diff"
hatch codex nano "What is 2+2?"
hatch --json "Analyze this" | jq .output
```

Use the same surfaced commands for normal build/edit work and review prompts.

## Quick Reference

```bash
uv sync --all-extras                   # Install dev deps
uv run --extra dev pytest -v           # Run tests
uv run --extra dev pytest -v -m integration  # Real API calls (needs creds)
uv run hatch mcp doctor tools          # Check MCP server tool surface
```

## Runtime Notes

Credentials are resolved explicitly before backend launch:
- CLI `--api-key` override wins
- Existing shell env wins next
- Personal backends (`zai`, `codex`) then use the configured local secret helper

Machine callers:
- non-interactive CLI runs default to JSON output and automation mode automatically
- set `HATCH_DISABLE_SECRET_HELPER=1` when you need tests or subprocesses to fail fast instead of loading secrets from the local helper
- surfaced Claude/Codex runs stream terse live progress to stderr while preserving only the final answer on stdout/JSON

## Architecture

```
cli.py / mcp/* → credentials.py → backends.py → subprocess(opencode/claude/codex/gemini)
                    ↓
               infisical-get.py
        ↓
    context.py (container detection)
```

**5 Backends:** `zai` (default), `bedrock`, `codex`, `gemini`, `opencode`

**Key files:**
| File | Purpose |
|------|---------|
| `backends.py` | Env vars + cmd building per backend |
| `runner.py` | Async subprocess wrapper + timeout |
| `mcp/` | Built-in MCP server + persistent OpenCode runtime |
| `context.py` | Container/filesystem detection |
| `cli.py` | Argument parsing + main() |

## Conventions

- **Single product repo** - the CLI and MCP server both live here; personal config repos should only point at `hatch`, not re-wrap it
- **Prefer prompt via stdin when the backend supports it** - raw Claude/Codex/Gemini paths use stdin; OpenCode currently takes prompt text via argv
- **Container-aware** - auto-sets HOME=/tmp for read-only filesystems
- **Keep the surfaced CLI small** - `codex` and `claude` are the human/agent-facing entrypoints; they route through OpenCode, while raw backend flags are escape hatches
- **Do not leak internal runtime nouns into the public contract** - `opencode` is an implementation detail, not part of the default user/agent mental model
- **Machine callers should not remember flags** - real non-interactive CLI runs should default to JSON output + automation mode

## Library

```python
from hatch import run, Backend

result = await run(prompt="Fix the bug", backend=Backend.ZAI)
print(result.output if result.ok else result.error)
```

## Gotchas

1. **z.ai uses `ANTHROPIC_AUTH_TOKEN`** not `ANTHROPIC_API_KEY` - and must unset `CLAUDE_CODE_USE_BEDROCK`
2. **Tests mock subprocess** - no real CLI calls except `integration` marked tests
3. **Core deps should stay minimal** - `fastmcp` is in core because `hatch` now owns the MCP server; avoid growing beyond that without a strong reason
4. **Credential loading lives in `credentials.py`** - do not fetch secrets inside backend config builders
5. **Surfaced `claude` / `codex` should converge on one runtime** - keep OpenCode as the shared path and reserve raw `-b bedrock` / `-b codex` for debugging underlying harness behavior
6. **The MCP server belongs here** - do not move hatch MCP code back into machine-local config repos

---

## Learnings

<!-- Agents: append below. Human compacts weekly. -->

- (2026-01-27) [tool] `uv run pytest -v` omits dev extras; use `uv run --extra dev pytest -v` (or `uv run --python .venv/bin/python -m pytest -v`) after `uv sync --all-extras`.
- (2026-03-29) [design] Keep backend builders pure; hatch credential policy belongs in one preflight resolver that uses the canonical `infisical-get.py` helper instead of ad hoc backend fallbacks.
- (2026-04-09) [auth] Bedrock launches must clear inherited `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_BASE_URL`; otherwise Claude can take the wrong auth path and fail before it ever reaches AWS.
- (2026-04-09) [ux] Agents should not be asked to choose raw backends like `bedrock` vs `codex`; surface `hatch claude` / `hatch codex` and map model-family aliases internally.
- (2026-04-09) [testing] Secret-helper fallback is useful in real runs but breaks missing-credential tests unless they can disable it explicitly.
- (2026-04-13) [claude] `claude --print --output-format stream-json` now requires `--verbose`; if you want live progress plus a clean final stdout payload, parse the stream and emit progress on stderr instead of buffering with `communicate()`.
- (2026-04-13) [runtime] Surface `hatch claude` / `hatch codex` through OpenCode so Bedrock/OpenAI share one tool/runtime model; keep raw `-b bedrock` / `-b codex` only as backend escape hatches.
- (2026-04-13) [ux] Do not expose raw OpenCode agent names in the public hatch contract; keep the common path to plain `hatch claude ...` / `hatch codex ...` commands.
- (2026-04-14) [architecture] `hatch` owns both the CLI and the MCP server; personal config repos should only register `hatch mcp`, not carry a second wrapper implementation.
