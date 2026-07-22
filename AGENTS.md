# hatch

CLI-only headless runner for complete one-turn Claude, Codex, Cursor, Gemini,
OpenRouter, and expert calls.

**Owner**: david010@gmail.com

## Install

```bash
VERSION=0.2.0 ./scripts/build-release.sh
./scripts/install-local.sh \
  --go-binary ./dist/hatch_0.2.0_darwin_arm64/hatch --select go
```

The installer keeps the frozen Python 0.1.0 release available as
`hatch-python` during the field soak. Use `./scripts/install-local.sh --select
python` only for an explicit operator rollback; there is no per-invocation
fallback.

## Entrypoints

Public surface:
- `hatch claude <haiku|sonnet|opus|fable>` → Claude via the official local Claude Code CLI OAuth/subscription path (`fable` = Fable-class, higher capability)
- `hatch codex <sol|terra|luna>` → GPT-5.6 on OpenAI (`nano|mini|max` remain compatibility aliases)
- `hatch cursor grok` → Grok 4.5 High via local Cursor Agent CLI
- `hatch openrouter <deepseek-v4-pro|kimi-k3>` → OpenRouter models via OpenCode
- `hatch expert` → one synchronous GPT pro Responses API consultation with web search on by default, not an agent
- Raw `-b bedrock` / `-b codex` / `-b gemini` / `-b cursor` still invoke the underlying CLIs directly as escape hatches

Default tiers:
- Start with `sonnet` for Claude and `sol` for Codex
- Use `terra` for a lower-cost balance or `luna` for efficient high-volume work
- GPT-5.6 reasoning accepts `none`, `low`, `medium`, `high`, `xhigh`, and `max`
- Use `fable` for Fable-class Claude (higher capability, always-on adaptive thinking)
- Use `cursor grok` for Grok 4.5 High via Cursor
- Use `openrouter deepseek-v4-pro` as the default non-OpenAI/non-Anthropic option; `kimi-k3` for complex coding and long-horizon agentic workflows

Agent runs target a concise result within about 15 minutes and have a 30 minute
hard timeout by default. `hatch expert` remains at 15 minutes because its
background response is server-persisted. Do not wrap normal `hatch` calls in
short outer shell timeouts.

```bash
hatch codex sol "Review this branch"
hatch claude haiku "Summarize this file"
hatch cursor grok "Review this branch"
hatch cursor grok --model cursor-grok-4.5-high "Review with a raw Cursor model ID"
hatch doctor
hatch openrouter deepseek-v4-pro "Review this branch"
hatch openrouter kimi-k3 "Review this branch"
hatch codex sol --reasoning-effort high "Write unit tests"
hatch claude sonnet "Review this diff"
hatch codex luna "What is 2+2?"
hatch expert --reasoning-effort low "Is this refactor direction sound?"
hatch codex sol --json "Analyze this" | jq .output
```

Use the same surfaced commands for normal build/edit work and review prompts.

## Quick Reference

```bash
go test ./... -count=1                 # Go unit + contract suite
go test -race ./... -count=1           # Concurrency and isolation
go vet ./...                           # Static checks
uv run pytest -q                       # Frozen Python compatibility oracle
./scripts/test-field-evidence.sh       # Python-retirement gate checker
```

## Runtime Notes

Credentials are resolved explicitly before backend launch:
- CLI `--api-key` override wins
- Existing shell env wins next
- Credentialed backends then use the external helper named by
  `HATCH_CREDENTIAL_HELPER`; Hatch passes a small JSON request on stdin and
  receives credentials on stdout without owning a secret-manager integration

Machine callers:
- non-interactive CLI runs default to JSON output and automation mode automatically
- agent prompts automatically receive a bounded-run contract: stay within scope,
  investigate proportionally, and synthesize once evidence is sufficient
- set `HATCH_DISABLE_SECRET_HELPER=1` when you need tests or subprocesses to fail fast instead of loading secrets from the local helper
- surfaced Claude/Codex/Cursor runs stream terse live progress to stderr while preserving only the final answer on stdout/JSON
- every run allocates a durable artifact before provider launch and preserves
  raw stdout/stderr under `~/.local/state/hatch/runs/`; JSON results carry the
  run ID, artifact path, capture state, and provider identity when available
- use `hatch runs list` and `hatch runs inspect <run-id>` to recover results
  independently of an outer terminal wrapper

## Architecture

```
cmd/hatch → internal/cli → internal/run.Coordinator → provider process or Expert HTTP
                                  ↓
                          content-addressed RunStore
                                  ↓
                    raw evidence + result + terminal manifest
```

**Active backends:** `claude`, `cursor`, `bedrock`, `codex`, `gemini`, `opencode`

`zai` / GLM-5.1 is intentionally disabled until the z.ai coding plan/resource package is active again. Bare `hatch "prompt"` has no default model; use an explicit surfaced provider.

**Key files:**
| File | Purpose |
|------|---------|
| `cmd/hatch/main.go` | Release entrypoint |
| `internal/cli/` | Parsing, command construction, credentials, doctor, run inspection |
| `internal/run/coordinator.go` | Single execution path and lifecycle ownership |
| `internal/run/store.go` | Ordered durable artifact commits |
| `internal/provider/` | Thin provider interpretation and progress adapters |
| `internal/expert/` | Responses HTTP execution and polling |
| `testdata/contract/` | Shared Python/Go process oracle corpus |

## Conventions

- **CLI-only public surface** - agent callers invoke `hatch` as a subprocess;
  do not add another MCP facade or persistent agent runtime
- **Prefer prompt via stdin when the backend supports it** - raw Claude/Codex/Gemini paths use stdin; Cursor Agent takes prompt via argv (stdin hangs); OpenCode currently takes prompt text via argv
- **Container-aware** - auto-sets HOME=/tmp for read-only filesystems
- **Keep the surfaced CLI small** - `codex`, `claude`, and `cursor` are the human/agent-facing entrypoints; Claude routes through local Claude Code OAuth, Cursor through local Cursor Agent login, Codex/OpenRouter route through OpenCode, and raw backend flags are escape hatches
- **Do not leak internal runtime nouns into the public contract** - `opencode` is an implementation detail, not part of the default user/agent mental model
- **Machine callers should not remember flags** - real non-interactive CLI runs should default to JSON output + automation mode

## Gotchas

1. **No implicit default model** - use `hatch codex ...`, `hatch claude ...`, `hatch cursor grok`, or `hatch openrouter ...`; z.ai/GLM is disabled for now
2. **All production execution uses the coordinator** - adapters interpret
   evidence; they do not launch processes, own persistence, or invent retries
3. **Python is an oracle, not production architecture** - until the genuine
   50-run field gate passes, retain its tests and tagged rollback release but do
   not add product behavior to the Python package
4. **Credential authority stays external** - do not embed Infisical or another
   secret manager in the Go binary, and never put prompt or credential values in
   manifest argv
5. **Surfaced `claude` must not use OpenRouter implicitly** - `hatch claude` uses local Claude Code OAuth/subscription and strips `OPENROUTER_API_KEY`; OpenRouter Claude models require an explicit OpenRouter surface if ever re-added
6. **Provider aliases drift** - run `hatch doctor` after Cursor or OpenCode
   upgrades. It verifies Cursor `grok`, Codex tiers, and OpenRouter aliases.
   Stable `kimi-k3` intentionally routes through OpenRouter's
   `~moonshotai/kimi-latest` alias because OpenCode 1.17.20 lists the direct K3
   slug but rejects it at execution time.
7. **Artifact publication is ordered** - `result.json` precedes the terminal
   manifest. A terminal manifest is the commit point; never rewrite an existing
   run artifact or infer loss from a collapsed caller transcript.

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
- (2026-07-17) [timeouts] A surfaced Codex/OpenCode timeout preserves partial JSONL, stderr, isolated session state, session id, and inspect/resume argv under the durable run artifact root; never collapse a long review timeout to empty output.
- (2026-07-21) [timeouts] Agent runs carry a provider-neutral 15-minute behavioral contract with a 30-minute hard backstop. Timeout artifacts record an env-complete manual resume command plus non-secret model/provider/credential-name metadata; never persist credential values or echo reasoning content.
- (2026-07-22) [durability] A collapsed caller transcript is not lost output, and `artifact_path: null` must not be used as a recovery verdict. Preserve every surfaced OpenCode run, propagate provider session identity on all outcomes, and keep result capture, provider-state retention, and Longhouse archival as separate facts.
- (2026-07-22) [rewrite] Go 0.2.0 is the selected production Hatch. Every
  surface now uses the same durable coordinator; Python remains only as the
  frozen parity oracle and explicit rollback until the genuine field gate
  passes.
