# hatch

CLI-only headless runner for complete one-turn Claude, Codex, Cursor, Gemini,
OpenRouter, and expert calls.

**Owner**: david010@gmail.com

## Install

```bash
VERSION=0.2.0 ./scripts/build-release.sh
./scripts/install-local.sh \
  --go-binary ./dist/hatch_0.2.0_darwin_arm64/hatch
```

The installer is Go-only and never falls back per invocation. The retired
Python 0.1.0 source remains available from tag `python-v0.1.0-final` for an
explicit emergency rollback.

## Entrypoints

CLI surface (also covered in the global `~/git/me/AGENTS.md`; the status quo
here is authoritative when they differ):

- `hatch claude <haiku|sonnet|opus|fable>` → Claude via the official local Claude Code CLI OAuth/subscription path (`opus` = Opus 5, current default; `fable` = Fable-class)
- `hatch codex <sol|terra|luna>` → GPT-5.6 on OpenAI (`nano|mini|max` remain compatibility aliases)
- `hatch cursor <grok|kimi-k3>` → Grok 4.5 High and Kimi K3 via local Cursor Agent CLI
- `hatch gemini [flash|3.7|gemini-3.7-flash-tiered]` → Gemini via OMP using Google Antigravity (`flash` = `gemini-3.7-flash-tiered`, current default)
- `hatch openrouter <deepseek-v4-flash|deepseek-v4-pro|glm-5.3-flash>` → OpenRouter models via OpenCode
- `hatch expert` → one synchronous GPT pro Responses API consultation with web search on by default, not an agent
- Raw `-b bedrock` / `-b codex` / `-b gemini` / `-b cursor` still invoke the underlying CLIs directly as escape hatches

Default tiers: `opus` for Claude (supersedes `sonnet`/`fable` for most work as
of 2026-07-24), `sol` for Codex; `sonnet` cheaper/faster Claude, `terra` lower-cost
Codex balance, `luna` high-volume. GPT-5.6 reasoning accepts
`none|low|medium|high|xhigh|max`. `fable` only when always-on adaptive thinking
is wanted. `openrouter deepseek-v4-flash` and `openrouter glm-5.3-flash` are the default non-OpenAI/non-Anthropic choices.

Agent runs target a concise result within ~15 minutes and have a default 30
minute hard timeout. `hatch expert` stays at 15 minutes because its background
response is server-persisted. Do not wrap normal `hatch` calls in short outer
shell timeouts.

```bash
hatch codex sol "Review this branch"
hatch claude haiku "Summarize this file"
hatch cursor grok --model cursor-grok-4.5-high "Review with a raw Cursor model ID"
hatch doctor
hatch codex sol --reasoning-effort high "Write unit tests"
hatch codex sol --json "Analyze this" | jq .output
```

## Quick Reference

```bash
go test ./... -count=1                 # Go unit + contract suite
go test -race ./... -count=1           # Concurrency and isolation
go vet ./...                           # Static checks
go test ./... -run Contract -count=1   # V1 traceability + frozen migration ledger
go test ./... -run LegacyParity -count=1
```

## Runtime Notes

Credentials are resolved explicitly before backend launch:
- CLI `--api-key` override wins
- Existing shell env wins next
- Credentialed backends then use the external helper named by
  `HATCH_CREDENTIAL_HELPER` or its private
  `${XDG_CONFIG_HOME:-$HOME/.config}/hatch/credential-helper` pointer; Hatch passes
  a small JSON request on stdin and receives credentials on stdout without
  owning a secret-manager integration

Machine callers:
- non-interactive CLI runs default to JSON output and automation mode automatically
- agent prompts automatically receive a bounded-run contract: stay within scope,
  investigate proportionally, synthesize once evidence is sufficient
- set `HATCH_DISABLE_SECRET_HELPER=1` when you need tests or subprocesses to fail fast instead of loading secrets from the local helper
- surfaced Claude/Codex/Cursor runs stream terse live progress to stderr while preserving only the final answer on stdout/JSON
- every run allocates a durable artifact before provider launch and preserves raw stdout/stderr under `~/.local/state/hatch/runs/`; JSON results carry the run ID, artifact path, capture state, and provider identity when available
- `hatch runs list` / `hatch runs inspect <run-id>` recover results independently of an outer terminal wrapper; `hatch runs audit --json` verifies stored artifact integrity

## Architecture

```
cmd/hatch → internal/cli → internal/run.Coordinator → provider process or Expert HTTP
                                  ↓
                          content-addressed RunStore
                                  ↓
                    raw evidence + result + terminal manifest
```

**Active backends:** `claude`, `cursor`, `bedrock`, `codex`, `gemini`, `opencode`

`zai` / GLM-5.1 is intentionally disabled until the z.ai coding plan/resource
package is active again. Bare `hatch "prompt"` has no default model; use an
explicit surfaced provider.

**Key files:**
| File | Purpose |
|------|---------|
| `cmd/hatch/main.go` | Release entrypoint |
| `internal/cli/` | Parsing, command construction, credentials, doctor, run inspection |
| `internal/run/coordinator.go` | Single execution path and lifecycle ownership |
| `internal/run/store.go` | Ordered durable artifact commits |
| `internal/provider/` | Thin provider interpretation and progress adapters |
| `internal/expert/` | Responses HTTP execution and polling |
| `testdata/contracts/` | Frozen migration ledger and language-neutral process corpus |

## Conventions

- **CLI-only public surface** - agent callers invoke `hatch` as a subprocess; no MCP facade or persistent agent runtime
- **Prefer prompt via stdin when the backend supports it** - raw Claude/Codex/Gemini paths use stdin; Cursor Agent takes prompt via argv (stdin hangs); OpenCode currently takes prompt text via argv
- **Container-aware** - auto-sets HOME=/tmp for read-only filesystems
- **Keep the surfaced CLI small** - `codex`, `claude`, and `cursor` are the human/agent-facing entrypoints; raw backend flags are escape hatches
- **Do not leak internal runtime nouns into the public contract** - `opencode` is an implementation detail, not part of the default user/agent mental model
- **Machine callers should not remember flags** - real non-interactive CLI runs default to JSON output + automation mode
- **Nested Hatch is permitted, but bounded** - a surfaced run may launch a small number of narrowly scoped child `hatch` runs for parallel/independent subwork. Children need deadlines, recursion requires explicit authorization, and the parent synthesizes surviving results instead of waiting indefinitely for a child.

## Gotchas

1. **No implicit default model** - use `hatch codex ...`, `hatch claude ...`, `hatch cursor grok`, or `hatch openrouter ...`; direct z.ai API is disabled for now (use `hatch openrouter glm-5.3-flash`)
2. **All production execution uses the coordinator** - adapters interpret evidence; they do not launch processes, own persistence, or invent retries
3. **Python is retired** - preserve the migration ledger, language-neutral fixtures, and `python-v0.1.0-final` tag; do not restore Python production code
4. **Credential authority stays external** - do not embed Infisical or another secret manager in the Go binary, and never put prompt or credential values in manifest argv
5. **Surfaced `claude` must not use OpenRouter implicitly** - `hatch claude` uses local Claude Code OAuth/subscription and strips `OPENROUTER_API_KEY`; OpenRouter Claude requires an explicit OpenRouter surface if ever re-added
6. **Provider aliases drift** - run `hatch doctor` after Cursor or OpenCode upgrades. It verifies Cursor aliases, Codex tiers, and OpenRouter aliases. Kimi K3 routes through Cursor's native `kimi-k3` model ID.
7. **Artifact publication is ordered** - `result.json` precedes the terminal manifest. A terminal manifest is the commit point; never rewrite an existing run artifact or infer loss from a collapsed caller transcript.

## Learnings

<!-- Agents: append below. Human compacts weekly. -->

- (2026-03-29) [design] Backend builders stay pure; hatch credential policy lives in one preflight resolver using the canonical `infisical-get.py` helper, not ad hoc backend fallbacks.
- (2026-04-09) [auth] Bedrock launches must clear inherited `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`; otherwise Claude takes the wrong auth path and fails before reaching AWS.
- (2026-04-09) [ux] Agents don't pick raw backends (`bedrock` vs `codex`); surface `hatch claude` / `hatch codex` and map model-family aliases internally.
- (2026-04-09) [testing] Secret-helper fallback helps real runs but breaks missing-credential tests unless they can disable it explicitly.
- (2026-04-13) [claude] `claude --print --output-format stream-json` now requires `--verbose`; for live progress + clean final stdout, parse the stream and emit progress on stderr instead of buffering with `communicate()`.
- (2026-04-13) [runtime] Surface `hatch claude` / `hatch codex` through OpenCode so Bedrock/OpenAI share one tool/runtime model; keep raw `-b bedrock` / `-b codex` only as backend escape hatches.
- (2026-04-13) [ux] Don't expose raw OpenCode agent names in the public hatch contract.
- (2026-04-28) [runtime] z.ai/GLM-5.1 disabled while the coding plan is inactive; bare `hatch "..."` fails fast instead of falling back to an implicit paid/provider default.
- (2026-04-29) [expert] `hatch expert` defaults to web search on; disable only explicitly for sealed local-context reasoning.
- (2026-04-29) [expert] Long expert calls use background Responses with server-side polling; the public CLI contract stays one blocking call.
- (2026-05-21) [expert] Keep `hatch expert` to low/medium effort. On timeout, preserve the background response id/artifact instead of cancelling at the boundary.
- (2026-05-24) [codex] Headless Codex runs must pass `--dangerously-bypass-approvals-and-sandbox` to prevent deadlocks on interactive tool-approval prompts in non-interactive/redirected subshells.
- (2026-05-27) [opencode] Surfaced Hatch/OpenCode runs must pass `--dangerously-skip-permissions`; keep `--dir` for repo context instead of broadening by omitting cwd.
- (2026-07-07) [routing] `hatch claude` uses the official local Claude Code CLI OAuth/subscription path and fails closed with OpenRouter/API-key/Bedrock env stripped. OpenRouter Claude was an expensive accidental fallback after Bedrock access ended; never implicit again.
- (2026-07-16) [cursor] `cursor-agent -p` is the one-shot hatch path. Prompt via argv (stdin hangs). Use `--trust --force`, binary `cursor-agent` (not `agent`), verify the pinned model with `hatch doctor` (Cursor model IDs can be retired). Auth is Cursor login; optional `CURSOR_API_KEY`.
- (2026-07-17) [timeouts] A surfaced Codex/OpenCode timeout preserves partial JSONL, stderr, isolated session state, session id, and inspect/resume argv under the durable run artifact root; never collapse a long review timeout to empty output.
- (2026-07-21) [timeouts] Agent runs carry a provider-neutral 15-minute behavioral contract with a 30-minute hard backstop. Timeout artifacts record an env-complete manual resume command plus non-secret model/provider/credential-name metadata; never persist credential values or echo reasoning content.
- (2026-07-22) [durability] A collapsed caller transcript is not lost output, and `artifact_path: null` is not a recovery verdict. Preserve every surfaced OpenCode run, propagate provider session identity on all outcomes; result capture, provider-state retention, and Longhouse archival are separate facts.
- (2026-07-22) [rewrite] Go 0.2.0 is the selected production Hatch; every surface uses the same durable coordinator. Python production source is retired; the frozen ledger, fixtures, and tagged release preserve history.
- (2026-07-24) [models] Opus 5 is the default `hatch claude opus` target via the local Claude Code CLI's own `--model opus` alias resolution (no hatch change needed). Supersedes `sonnet`/`fable` for most work; keep `sonnet` for cheap/fast calls and `fable` only for Fable-specific traits.
- (2026-08-04) [claude] Resolve Claude effort from the installed CLI's supported `--effort` values; keep its unspecified default at `low` so an explicit override doesn't silently raise cost.
- (2026-08-12) [routing] OpenRouter deepseek-v4-flash runs pin a provider order (DeepSeek, CoreWeave, Novita, DeepInfra; allow_fallbacks) via a per-run `opencode.json` in the isolated config dir. Default price-based load balancing routinely lands on non-caching endpoints (DigitalOcean, OpenInference) that re-encode the whole growing context per agent step; measured on this account DeepSeek caches 98% while OpenInference caches 0% and runs at ~3 tps.
- (2026-08-12) [contract] The bounded-run contract forbids re-reading the same file or re-running identical searches and requires starting the answer once core files are read. Cancelled/timed-out OpenCode runs with no meaningful text across many tool-only steps and repeated identical tool calls emit a `stall_detected` warning with step/tool/text statistics.
- (2026-08-18) [gemini] `hatch gemini` routes to Oh My Pi with `google-antigravity/gemini-3.7-flash-tiered` by default (aliases: `flash`, `3.7`, `gemini-3.7-flash-tiered`). OMP coordinator isolation preserves user database/auth (`models.db*`, `config.yml`, `models.yml`) via symlinks while keeping session state isolated under the run artifact root.
- (2026-08-28) [routing] OpenRouter `glm-5.3-flash` (`openrouter/z-ai/glm-5.3-flash`) runs via OpenCode latched specifically to the Modal provider (`order: ["Modal"]`, `allow_fallbacks: false`) via isolated per-run `opencode.json` config.
