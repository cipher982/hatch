# hatch

CLI-only headless runner for complete one-turn Claude, Codex, Cursor, Gemini,
OpenRouter, and expert calls.

**Owner**: david010@gmail.com

## Install

```bash
uv tool install -e ~/git/hatch
```

## Entrypoints

Public surface:
- `hatch claude <haiku|sonnet|opus|fable>` → Claude via the official local Claude Code CLI OAuth/subscription path (`fable` = Fable-class, higher capability)
- `hatch codex <sol|terra|luna>` → GPT-5.6 on OpenAI (`nano|mini|max` remain compatibility aliases)
- `hatch cursor grok` → Grok 4.5 High via local Cursor Agent CLI
- `hatch openrouter deepseek-v4-pro` → DeepSeek V4 Pro on OpenRouter
- `hatch expert` → one synchronous GPT pro Responses API consultation with web search on by default, not an agent
- Raw `-b bedrock` / `-b codex` / `-b gemini` / `-b cursor` still invoke the underlying CLIs directly as escape hatches

Default tiers:
- Start with `sonnet` for Claude and `sol` for Codex
- Use `terra` for a lower-cost balance or `luna` for efficient high-volume work
- GPT-5.6 reasoning accepts `none`, `low`, `medium`, `high`, `xhigh`, and `max`
- Use `fable` for Fable-class Claude (higher capability, always-on adaptive thinking)
- Use `cursor grok` for Grok 4.5 High via Cursor
- Use `openrouter deepseek-v4-pro` as the non-OpenAI/non-Anthropic third option

Defaults to a 15 minute internal timeout. Do not wrap normal `hatch` calls in short outer shell timeouts.

```bash
hatch codex sol "Review this branch"
hatch claude haiku "Summarize this file"
hatch cursor grok "Review this branch"
hatch cursor grok --model cursor-grok-4.5-high "Review with a raw Cursor model ID"
hatch doctor
hatch openrouter deepseek-v4-pro "Review this branch"
hatch codex sol --reasoning-effort high "Write unit tests"
hatch claude sonnet "Review this diff"
hatch codex luna "What is 2+2?"
hatch expert --reasoning-effort low "Is this refactor direction sound?"
hatch codex sol --json "Analyze this" | jq .output
```

Use the same surfaced commands for normal build/edit work and review prompts.

## Quick Reference

```bash
uv sync --all-extras                   # Install dev deps
uv run --extra dev pytest -v           # Run tests
uv run --extra dev pytest -v -m integration  # Real API calls (needs creds)
```

## Runtime Notes

Credentials are resolved explicitly before backend launch:
- CLI `--api-key` override wins
- Existing shell env wins next
- Credentialed backends (`codex`, `openrouter`) then use the configured local secret helper

Machine callers:
- non-interactive CLI runs default to JSON output and automation mode automatically
- set `HATCH_DISABLE_SECRET_HELPER=1` when you need tests or subprocesses to fail fast instead of loading secrets from the local helper
- surfaced Claude/Codex runs stream terse live progress to stderr while preserving only the final answer on stdout/JSON

## Architecture

```
cli.py → credentials.py → backends.py → subprocess(opencode/claude/codex/gemini)
                    ↓
               infisical-get.py
        ↓
    context.py (container detection)
```

**Active backends:** `claude`, `cursor`, `bedrock`, `codex`, `gemini`, `opencode`

`zai` / GLM-5.1 is intentionally disabled until the z.ai coding plan/resource package is active again. Bare `hatch "prompt"` has no default model; use an explicit surfaced provider.

**Key files:**
| File | Purpose |
|------|---------|
| `backends.py` | Env vars + cmd building per backend |
| `runner.py` | Async subprocess wrapper + timeout |
| `expert.py` | Direct single-call Responses API expert mode |
| `context.py` | Container/filesystem detection |
| `cli.py` | Argument parsing + main() |

## Conventions

- **CLI-only public surface** - agent callers invoke `hatch` as a subprocess;
  do not add another MCP facade or persistent agent runtime
- **Prefer prompt via stdin when the backend supports it** - raw Claude/Codex/Gemini paths use stdin; Cursor Agent takes prompt via argv (stdin hangs); OpenCode currently takes prompt text via argv
- **Container-aware** - auto-sets HOME=/tmp for read-only filesystems
- **Keep the surfaced CLI small** - `codex`, `claude`, and `cursor` are the human/agent-facing entrypoints; Claude routes through local Claude Code OAuth, Cursor through local Cursor Agent login, Codex/OpenRouter route through OpenCode, and raw backend flags are escape hatches
- **Do not leak internal runtime nouns into the public contract** - `opencode` is an implementation detail, not part of the default user/agent mental model
- **Machine callers should not remember flags** - real non-interactive CLI runs should default to JSON output + automation mode

## Library

```python
from hatch import run, Backend

result = await run(
    prompt="Fix the bug",
    backend=Backend.OPENCODE,
    model="openai/gpt-5.6-sol",
)
print(result.output if result.ok else result.error)
```

## Gotchas

1. **No implicit default model** - use `hatch codex ...`, `hatch claude ...`, `hatch cursor grok`, or `hatch openrouter ...`; z.ai/GLM is disabled for now
2. **Tests mock subprocess** - no real CLI calls except `integration` marked tests
3. **Core deps should stay minimal** - the CLI currently needs no runtime Python dependencies; add one only when it removes more complexity than it creates
4. **Credential loading lives in `credentials.py`** - do not fetch secrets inside backend config builders
5. **Surfaced `claude` must not use OpenRouter implicitly** - `hatch claude` uses local Claude Code OAuth/subscription and strips `OPENROUTER_API_KEY`; OpenRouter Claude models require an explicit OpenRouter surface if ever re-added
6. **Surfaced `cursor` uses Cursor Agent CLI** - `cursor-agent -p --trust --force --model ...`; auth is Cursor login (or optional `CURSOR_API_KEY`). Prefer the `cursor-agent` binary name over the `agent` symlink to avoid PATH collisions. Run `hatch doctor` after Cursor upgrades to verify that the stable `grok` alias still targets an available account model.

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
- (2026-04-28) [runtime] Disable z.ai/GLM-5.1 while the coding plan is inactive; bare `hatch "..."` should fail fast instead of falling back to an implicit paid/provider default.
- (2026-04-29) [expert] `hatch expert` defaults to web search on; only disable it explicitly for sealed local-context reasoning.
- (2026-04-29) [expert] Long expert calls use background Responses internally with server-side polling, but the public CLI contract stays one blocking call.
- (2026-05-21) [expert] Keep `hatch expert` to low/medium effort. On timeout, preserve the background response id/artifact instead of cancelling at the boundary.
- (2026-05-24) [codex] Headless Hatch runs for Codex must explicitly pass `--dangerously-bypass-approvals-and-sandbox` to prevent deadlocks on interactive tool-approval prompts in non-interactive/redirected subshells.
- (2026-05-27) [opencode] Surfaced Hatch/OpenCode runs must pass `--dangerously-skip-permissions`; keep `--dir` for repo context instead of broadening by omitting cwd.
- (2026-07-07) [routing] `hatch claude` must use the official local Claude Code CLI OAuth/subscription path and fail closed with OpenRouter/API-key/Bedrock env stripped. OpenRouter Claude was an expensive accidental fallback after Bedrock access ended; do not make it implicit again.
- (2026-06-29) [subprocess] Always use `subprocess.DEVNULL` (never `None`) for stdin when no stdin_data is supplied. `None` inherits the caller's stdin — harmless in a TTY, but in non-TTY callers (Cursor Composer, CI, pipes) OpenCode sees an open pipe and hangs until the hatch timeout fires.
- (2026-07-16) [cursor] `cursor-agent -p` is the one-shot hatch path. Pass prompt as argv (stdin hangs). Use `--trust --force`, binary name `cursor-agent` (not `agent`), and verify the pinned model with `hatch doctor` because Cursor model IDs can be retired. Auth is Cursor login; optional `CURSOR_API_KEY`.
