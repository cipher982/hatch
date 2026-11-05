# hatch

CLI-only headless runner for complete one-turn Claude, Codex, Cursor, Gemini,
OpenRouter, and expert calls.

**Owner**: david010@gmail.com

## Install

```bash
./scripts/build-release.sh   # version comes from VERSION; replaces dist/hatch_*
./scripts/install-local.sh \
  --go-binary ./dist/hatch_$(tr -d '[:space:]' < VERSION)_darwin_arm64/hatch
```

The installer is Go-only and never falls back per invocation, and keeps only the
three newest installed builds (current plus two rollbacks). Bump `VERSION` when
the binary's behavior changes. The retired
Python 0.1.0 source remains available from tag `python-v0.1.0-final` for an
explicit emergency rollback.

## Entrypoints

CLI surface (also covered in the global `~/git/me/AGENTS.md`; the status quo
here is authoritative when they differ):

- `hatch claude <haiku|sonnet|opus|fable|fable-5.1>` → Claude via the official local Claude Code CLI OAuth/subscription path — **personal work only** (`opus` = Opus 5.5, `sonnet` = Sonnet 5.5, `haiku` = Haiku 5.5, pinned as `claude-haiku-5-5`; `opus-5` is deprecated; `fable`/`fable-5.1` = Claude Fable 5.1)
- `hatch bedrock <opus|sonnet|astra|sol|terra|luna>` → Zeta's AWS Bedrock (`zh-ml-mlengineer`, `us-east-1`) via Pi by default, or `--harness omp` — **Zeta/employer work by default**; an explicitly named Bedrock model from David is always honored, from any directory, and the surface word is binding. Claude: `amazon-bedrock/global.anthropic.claude-{opus,sonnet}-5-5`, effort `low|medium|high|xhigh|max` (default `low`), no bare shorthand. OpenAI: `amazon-bedrock/global.openai.gpt-6-astra`, `.gpt-6.1-sol`, `.gpt-6.1-terra`, `.gpt-6-luna`; the bare `astra|sol|terra|luna` aliases resolve here, with OpenAI reasoning (default `medium`)
- `hatch codex <astra|sol|terra|luna>` → OpenAI on the **personal OpenAI account**, explicit form only (`astra` = GPT-6 Astra; `sol` = GPT-6.1 Sol; `terra` = GPT-6.1 Terra; `luna` = GPT-6 Luna; `nano|mini|max` remain compatibility aliases). The same OpenAI models run on Bedrock for Zeta work via `hatch bedrock`.
- `hatch cursor <grok|kimi-k3|gemini-4-argon>` → Grok 4.7 High (`grok-4.7-high`), Kimi K3, and Gemini 4 Argon Medium (`gemini-4-argon-medium`) via local Cursor Agent CLI, on the Zeta-account Cursor subscription (`drose@zetaglobal.com`) — **Zeta/employer work**. `gemini-4-argon` has no bare shorthand. **Cursor can hit its monthly usage limit** ("You've reached your monthly usage limit"): that is an account limit, not alias drift or a Hatch bug.
- `hatch gemini [argon]` → Gemini 4 Argon (`google-antigravity/gemini-4-argon`) via Oh My Pi on the personal Google AI Ultra subscription — **personal work only**; Zeta Gemini work uses `hatch cursor gemini-4-argon`. Reasoning comes from Hatch's `--thinking` policy (default `medium`).
- `hatch openrouter <deepseek-v4.1-flash|glm-5.3-flash>` → OpenRouter models via Pi
- `hatch expert` → one synchronous GPT pro Responses API consultation with web search on by default, not an agent
- `hatch review <MODEL ...>` → execs external `hatch-review` (review-hub in `~/git/me`) for an explicitly requested two-phase independent review. A model is required; receipts are optional audit data, not shipping gates. Git range, requester intent, CI, and per-pass limits live in review-hub, not Hatch; it takes no prompt.
- `--read-only` (advanced flag) → no file edits and no builds, tests or other programs, enforced by the backend's own mechanism: Pi `--tools read,grep,find,ls` (no bash at all), Claude `--tools Read,Grep,Glob` + `--permission-mode dontAsk` + `--strict-mcp-config`, OMP `--tools read,grep,glob`, Cursor `--mode ask` without `--force`, raw Codex `--sandbox read-only`. A backend with no mechanism (raw Gemini) refuses rather than run unrestricted. The mechanism is announced on stderr as `[hatch] read-only: ...`
- Raw `-b codex` / `-b gemini` / `-b cursor` still invoke the underlying CLIs directly as escape hatches (the raw `-b bedrock` Claude Code backend is retired; use `hatch bedrock`)

Default tiers: `opus` (Claude Opus 5.5) for Claude (supersedes `sonnet`/`fable` for most work as of 2026-09-22), `astra` for OpenAI (Bedrock by default); `sol` is the cost-balanced coding choice, `terra` the fast/vision/writing choice, and `luna` high-volume. GPT-6 Astra and GPT-6.1 Sol default to `medium` and support `low|medium|high|xhigh|max` (not `none`); GPT-6.1 Terra and GPT-6 Luna also accept `none`. `fable` only when always-on adaptive thinking is wanted. **Anthropic routing is by who pays:** personal repos (Zerg, Longhouse, personal infra) → `hatch opus|sonnet` (Max subscription); Zeta/employer repos → `hatch bedrock opus|sonnet` (AWS SSO). Neither route falls back to the other. **OpenAI routing is by who pays:** Zeta/employer work → `hatch bedrock astra|sol|terra|luna` (bare aliases resolve there); personal work → `hatch codex <alias>` on the personal OpenAI account. Bedrock hosts both Claude and OpenAI models. **An explicitly named surface always beats this routing:** `hatch bedrock astra` is never silently replaced by `hatch codex astra` (the aliases collide across both surfaces, so the surface word decides the account). **Gemini routing is also by who pays:** personal → `hatch gemini` (Google AI Ultra via Oh My Pi); Zeta/employer → `hatch cursor gemini-4-argon` (Cursor subscription).
`openrouter deepseek-v4.1-flash` and `openrouter glm-5.3-flash` are the default non-OpenAI/non-Anthropic choices.

Agent runs target a concise result within ~15 minutes and have a default 30
minute hard timeout. `hatch expert` stays at 15 minutes because its background
response is server-persisted. Do not wrap normal `hatch` calls in short outer
shell timeouts.

```bash
hatch codex astra "Review this branch"
hatch claude haiku "Summarize this file"
hatch cursor grok --model cursor-grok-4.5-high "Review with a raw Cursor model ID"
hatch doctor
hatch codex astra --reasoning-effort high "Write unit tests"
hatch codex astra --json "Analyze this" | jq .output
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

Follow-ups: `hatch --resume <run-id> "task"` (also `hatch <model> --resume <run-id> ...`) continues any finished Hatch run on any provider by replaying that run's request and answer from the run store into a fresh run (`internal/run/followup.go`). Native provider sessions are not used (surfaced runs are ephemeral by design, so they cannot be resumed). Model, harness, effort and cwd inherit from the prior run; `--read-only` does not. Zeta (bedrock, cursor) and personal runs never cross (`provider.SurfaceAccount`). Every result has a `follow_up` field. Flags a backend would silently ignore are rejected (`checkFlagSupport`).

Machine callers:
- non-interactive agent runs default to JSON output and automation mode; `expert` requires explicit `--json`
- agent prompts automatically receive a bounded-run contract: stay within scope,
  investigate proportionally, synthesize once evidence is sufficient
- set `HATCH_DISABLE_SECRET_HELPER=1` when you need tests or subprocesses to fail fast instead of loading secrets from the local helper
- surfaced Claude/Codex/Cursor runs stream terse live progress to stderr while preserving only the final answer on stdout/JSON
- every run allocates a durable artifact before provider launch and preserves raw stdout/stderr under `~/.local/state/hatch/runs/`; JSON results carry the run ID, artifact path, capture state, and provider identity when available
- Give agent calls a short `--title` and the actual caller session/request IDs (`--caller-session` / `--caller-request`, or `HATCH_CALLER_SESSION_ID` / `HATCH_CALLER_REQUEST_ID`). Keep the returned run IDs; do not recover ownership by model and timestamp.
- `hatch runs list` is scoped to the caller session or exact launch directory; use `--all` explicitly across projects. `runs read <run-id> --part result` retrieves bounded answers; follow `next_command`. `inspect --files` opts into file inventory. Titles and request previews are not review verdicts.
- Invocation output is a bounded preview with truncation flags; full results and raw streams remain in the artifact. `hatch runs audit --json` verifies stored artifact integrity. See README's recovery section for scope and byte-paging semantics.

## Architecture

```
cmd/hatch → internal/cli → internal/run.Coordinator → provider process or Expert HTTP
                                  ↓
                          content-addressed RunStore
                                  ↓
                    raw evidence + result + terminal manifest
```

**Active backends:** `claude`, `cursor`, `codex`, `gemini`, `pi`, `omp` (OpenCode was removed 2026-11-05)

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
- **Prefer prompt via stdin when the backend supports it** - raw Claude/Codex/Gemini paths use stdin; Cursor Agent takes prompt via argv (stdin hangs); Pi takes prompt text via argv
- **Container-aware** - auto-sets HOME=/tmp for read-only filesystems
- **Keep the surfaced CLI small** - `codex`, `claude`, and `cursor` are the human/agent-facing entrypoints; raw backend flags are escape hatches
- **Do not leak internal runtime nouns into the public contract** - `pi` is an implementation detail, not part of the default user/agent mental model
- **Machine callers should not remember flags** - non-interactive agent runs default to JSON output + automation mode; `expert` retains explicit `--json` selection
- **Nested Hatch is permitted, but bounded** - a surfaced run may launch a small number of narrowly scoped child `hatch` runs for parallel/independent subwork. Children need deadlines, recursion requires explicit authorization, and the parent synthesizes surviving results instead of waiting indefinitely for a child.

## Gotchas

1. **No implicit default model** - use `hatch codex ...`, `hatch claude ...`, `hatch bedrock ...`, `hatch cursor grok`, or `hatch openrouter ...`; direct z.ai API is disabled for now (use `hatch openrouter glm-5.3-flash`)
2. **All production execution uses the coordinator** - adapters interpret evidence; they do not launch processes, own persistence, or invent retries
3. **Python is retired** - preserve the migration ledger, language-neutral fixtures, and `python-v0.1.0-final` tag; do not restore Python production code
4. **Credential authority stays external** - do not embed Infisical or another secret manager in the Go binary, and never put prompt or credential values in manifest argv
5. **Surfaced `claude` must not use OpenRouter implicitly** - `hatch claude` uses local Claude Code OAuth/subscription and strips `OPENROUTER_API_KEY`; OpenRouter Claude requires an explicit OpenRouter surface if ever re-added
6. **Provider aliases drift** - run `hatch doctor` after Cursor or Pi upgrades. It verifies Cursor aliases and checks the Bedrock, Codex and OpenRouter aliases against Pi's model list plus the models Hatch declares itself. Kimi K3 routes through Cursor's native `kimi-k3` model ID. A stale AWS SSO session is caught by a pre-launch `aws sts get-caller-identity` and reported with the `aws sso login` command.
7. **Artifact publication is ordered** - `result.json` precedes the terminal manifest. A terminal manifest is the commit point; never rewrite an existing run artifact or infer loss from a collapsed caller transcript.

## Learnings

- (2026-11-05) [pi] OpenCode removed; Pi is the one coding harness for the Bedrock, Codex and OpenRouter surfaces (OMP stays for Gemini because only it has the antigravity provider; Claude stays on Claude Code). Pi's built-in model catalog lags (it lacks `gpt-6.1-sol`/`gpt-6.1-terra` and the DeepSeek 1017 snapshot), so Hatch writes a per-run `models.json` (`internal/provider/pimodels.go`) with the missing models, the OpenRouter provider pins (`compat.openRouterRouting`, sent as-is to OpenRouter) and the luna-xhigh priority tier (`samplingParams.service_tier`). `hatch doctor` accepts a model that is in Pi's list or declared there. The destructive-command guard (DCG, `required` in Agent Home) is a Pi extension (`assets/pi_dcg_guard.mjs`, loaded with `-e`, fail closed) because OpenCode's plugin does not run in Pi; OMP and Cursor still run without it, as before. Provider errors reach Pi's stream as an assistant message with `stopReason: "error"`, which the adapter now surfaces (before, a 401 showed as "Empty output from agent"). Dropped with OpenCode: the stall detector (`stall_detected`), the stderr-log error recovery, the read-only bash allowlist (Pi read-only has no shell), and snapshot/recovery commands on timeout (Pi runs are ephemeral). Historical OpenCode run artifacts remain readable and collectable. Live-verified on Bedrock (astra, sol, terra, luna, sonnet), OpenRouter (deepseek, glm) and the DCG block.

- (2026-11-05) [resume] Native `--resume SESSION_ID` was removed: it only reached Claude, where `--no-session-persistence` made it a no-op, and other providers dropped it silently. Follow-ups are now Hatch-level (`--resume <run-id>`), uniform across providers, discoverable in the main `--help`, and advertised in every result's `follow_up` field.

<!-- Agents: append below. Human compacts weekly. -->

- (2026-09-29) [read-only] Reviewer runs took 15-30 min because they ran builds and test suites, and one left an orphaned `cargo test`: OpenCode spawns each bash command detached (own session), so killing the run's process group never reached it. `--read-only` removes the ability (live-verified on OpenCode+DeepSeek: allowlisted `git log`/`rg`/`ls` ran; `cargo`, `python3`, `touch`, `go`, `find`, `git --output`, redirects and compound commands containing them were refused; write/edit/webfetch tools were not exposed at all). Command substitution needs no rule (OpenCode checks the commands inside `$(...)`, backticks and `<(...)` as commands of their own; live-probed). OpenCode parses a permission object with JSON.parse, so a repeated pattern collapses to its FIRST position and silently reorders the policy: `rules.MarshalJSON` refuses duplicates, and the tests replay OpenCode's last-match-wins evaluation over a verdict table. Known limits: a redirect glued to `-`, `=`, `*` or `?` (`echo x->f`) is not caught (rg patterns with `->`/`=>` need those), and OpenCode matches allow rules per command and takes a redirected *pipeline* (`a | b > f`) apart without the redirect, so those forms still write a file; the policy stops builds, tests and other programs, not every file write (an OS sandbox on the shell would close it). The normal-exit `HATCH_RUN_ID` sweep applies to every run, not only read-only ones. Only the OpenCode path is live-verified; the Claude/Pi/OMP/Cursor/Codex flags are checked against each CLI's `--help` and pinned by argv tests. The kill now also takes descendants (process-table walk before the kill) and any survivor carrying the run's `HATCH_RUN_ID` (`ps -E` on macOS, `/proc/*/environ` on Linux), on timeout, cancellation and after a normal exit.

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
- (2026-08-28) [routing] OpenRouter `glm-5.3-flash` (`openrouter/z-ai/glm-5.3-flash`) runs via OpenCode with provider order pinned to Modal first, with fallbacks to Z.AI/Novita/Together/Parasail/DeepInfra (`allow_fallbacks: true`), and supports native reasoning effort via `--variant`. Modal's OpenRouter endpoint lacks `tool_choice: auto` parameter support, so strict fallback-disabled routing causes OpenRouter to return 404 No endpoints found on tool-calling agent runs.
- (2026-09-01) [models] Claude Fable 5.1 (`claude-fable-5-1`) swapped in for `fable` and `fable-5.1` aliases; Fable 5 (`claude-fable-5`) deprecated.
- (2026-09-02) [models] Gemini 3.8 Flash Low (`google-antigravity/gemini-3.8-flash-low`) briefly replaced the older flash aliases; retired 2026-10-13 when the sole surfaced `hatch gemini` model moved to Gemini 4 Argon Medium through Cursor.
- (2026-09-10) [models] DeepSeek updated to OpenRouter `deepseek/deepseek-v4.1-flash` (`deepseek-v4.1-flash`), pinned strictly to the DeepSeek provider without fallbacks; DeepSeek Pro (`deepseek-v4-pro`) and `deepseek-v4-flash` deprecated.
- (2026-09-11) [routing] `luna` at `xhigh` requests OpenAI priority processing. The pair is declared in `ModelRegistry.PriorityEfforts`; OpenCode receives it as `provider.openai.models.<model>.options.serviceTier` inside the per-run `OPENCODE_CONFIG_CONTENT` (same mechanism as the OpenRouter provider pins), and Oh My Pi receives `--service-tier priority`. Wire-verified through real Hatch runs: the OpenCode request body carried `service_tier: "priority"` with `reasoning.effort: "xhigh"` against a mock `/v1/responses` endpoint, and a live luna xhigh run succeeded, so the account accepts the tier. Priority bills above standard rates, so it stays opt-in per model/effort rather than a blanket default, and — like the OpenRouter pins — it is not recorded in the run manifest. Pi and the raw `-b codex` escape hatch have no wired equivalent.
- (2026-09-29) [bedrock] `hatch bedrock opus|sonnet` is the Zeta route for Claude 5.5, replacing the raw `-b bedrock` Claude Code backend (Sonnet 4.6, tools disabled; its contract case is now `bedrock_success`). It is an OpenCode/Pi/Omp model id with the `global.` inference-profile prefix (`us.` has no Sonnet 5.5), smoke-tested on all three harnesses. `NoShorthand` in `ModelRegistry` keeps `opus`/`sonnet` bare aliases on the personal `claude` surface. `claude sonnet` is now pinned to `claude-sonnet-5-5` (the CLI alias already resolved to it).
- (2026-09-29) [cleanup] Gemini 3.8 aliases formerly pinned the listed `gemini-3.8-flash`; Hatch's current `hatch gemini` route uses Cursor Gemini 4 Argon Medium. Cursor doctor probes unlisted IDs. `dist/` is rebuilt per release (`VERSION` file), installs keep three builds, and the orphaned Python-era `~/.local/share/hatch/mcp-runtime` (6.2 GB of OpenCode snapshots) was deleted.
- (2026-09-29) [models] `sol` now targets `gpt-6.1-sol` (API-verified on David's key the day of release; $2/$10 per M tokens, near-Astra on coding; https://openai.com/index/introducing-gpt-6-1-sol/). `gpt-6-sol` is dropped from the catalog, OpenCode model definitions and effort table (clean cutover, as with GPT-5.6). The API rejects `reasoning.effort: none` for 6.1 Sol (accepts low|medium|high|xhigh|max), so its OpenCode variants and `knownOpenAIModel` omit `none`. `hatch expert` default model follows to `gpt-6.1-sol`.
- (2026-10-13) [routing] Gemini payer split: `hatch gemini` now runs `google-antigravity/gemini-4-argon` through Oh My Pi on the personal Google AI Ultra subscription, and `hatch cursor gemini-4-argon` (NoShorthand) keeps Zeta Gemini work on the Cursor subscription (cursor-agent login `drose@zetaglobal.com`). Earlier the same day the sole `hatch gemini` route was Cursor-backed, which routed personal calls through the Zeta-account Cursor login.
- (2026-10-29) [models] `terra` is a real current model again: it targets OpenAI `gpt-6.1-terra` (official GPT-6 family: Astra, 6.1 Sol, 6.1 Terra, 6 Luna) instead of being a deprecated `sol` alias, and the OpenAI models now surface on Bedrock (`amazon-bedrock/global.openai.gpt-6-astra|gpt-6.1-sol|gpt-6.1-terra|gpt-6-luna`), which is what the Zeta account actually serves. Bare `astra|sol|terra|luna` resolve to `bedrock`; the personal OpenAI account stays reachable only as explicit `hatch codex <alias>`. Bedrock OpenAI ids keep OpenAI reasoning semantics (medium default, `none` allowed for Terra/Luna) instead of the Claude low default. Help and README now group aliases by the account each surface bills.
