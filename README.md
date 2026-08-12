# Hatch

**One bounded agent run. Durable evidence. No new agent platform.**

Hatch is a small, headless CLI for launching a complete one-turn coding-agent
run and retaining enough local evidence to understand what happened after the
terminal session is gone.

It is a personal tool, published in case its particular shape is useful to
someone else. It is not a replacement for the underlying agent CLIs, a hosted
service, an MCP server, or a workflow engine.

```sh
hatch codex terra -C . "Review this branch for correctness risks"
hatch claude opus -C . "Explain the failure and propose the smallest fix"
hatch cursor grok -C . "Find the race condition"
```

## Why it exists

The underlying CLIs are already excellent at running an agent. The awkward part
is using them from automation or another agent: a terminal wrapper can collapse,
a provider can time out, and it becomes unclear whether a useful answer exists
or where its raw output went.

Hatch makes that boundary explicit:

```text
prompt ──► provider CLI ──► durable local run artifact ──► inspect later
                           stdout · stderr · result · manifest · hashes
```

Each invocation receives a run ID and creates its artifact *before* the
provider is launched. That artifact captures raw evidence, the interpreted
result, provider/session facts when available, and timeout or cancellation
details. A terminal manifest is written only after the result is captured.

```sh
hatch runs list
hatch runs inspect hatch_01K...
hatch runs audit --json
```

Artifacts live under `~/.local/state/hatch/runs/` by default. They are local and
private; Hatch does not run a daemon or upload them anywhere.

## The small public surface

The preferred form is model-first. Hatch infers the provider from the alias:

```bash
hatch opus "Review this diff"
hatch sol --harness omp "Review this branch"
hatch grok "Review this branch"
hatch deepseek-v4-flash "Fix the failing tests"
hatch deepseek-v4-pro "Review this architecture"
```

The explicit provider forms remain valid when they make a call easier to read:

```bash
hatch claude opus "Review this diff"
hatch codex sol --harness omp "Review this branch"
hatch openrouter deepseek-v4-flash "Fix the failing tests"
hatch openrouter deepseek-v4-pro "Review this architecture"
```

| Command | What it uses |
| --- | --- |
| `hatch claude <haiku\|sonnet\|opus\|fable>` | Local Claude Code CLI login |
| `hatch codex <sol\|terra\|luna>` | OpenAI coding model via the selected harness |
| `hatch cursor <grok\|kimi-k3>` | Local Cursor Agent CLI login |
| `hatch openrouter <deepseek-v4-flash\|deepseek-v4-pro>` | OpenCode with an OpenRouter model alias |
| `hatch expert` | One synchronous OpenAI Responses API consultation |

The model aliases are intentionally small and opinionated. For raw or
backend-specific options, use `hatch --advanced-help`.

### Select the coding harness

Codex and OpenRouter surfaces default to OpenCode, but callers can choose the
harness explicitly:

```bash
hatch codex sol --harness opencode "Review this branch"
hatch codex sol --harness pi "Review this branch"
hatch codex sol --harness omp "Review this branch"
hatch openrouter deepseek-v4-flash --harness omp "Fix the failing tests"
hatch openrouter deepseek-v4-pro --harness omp "Review this architecture"
```

The common Codex tier shorthand is also accepted:

```bash
hatch sol --harness omp "Review this branch"
```

Here, `sol` identifies the OpenAI model tier and `omp` identifies the execution
harness. The run manifest still records the inferred provider and effective
harness separately.

`--harness` is recorded in the effective backend field of the run manifest, so
JSON callers can see which implementation actually ran. The equivalent raw
forms are `--backend opencode`, `--backend pi`, and `--backend omp` with an
explicit `--model`.

Every surfaced agent receives a bounded-run instruction: stay within the task,
investigate proportionally, and return a concise answer rather than silently
turning a one-shot call into an open-ended session. The contract does not ban
nested Hatch calls: when a task authorizes bounded parallel or independent
subwork, an agent may launch a small number of child Hatch runs. Each child
needs a narrow scope and deadline; the parent must continue with partial
results instead of waiting forever, and recursion needs explicit authorization.

Reasoning effort is a Hatch run policy, not inherited provider session state.
Known Codex/OpenAI surfaces default to `medium`; pass
`--reasoning-effort high` or another supported level to choose explicitly.
Hatch records the effort, whether it came from the default or the user, and
whether the provider supports it in both the run manifest and JSON result.
Claude and Bedrock use a fixed `low` policy. OpenRouter, Cursor, and Gemini
report reasoning as unsupported instead of accepting a misleading override.
Unknown OpenAI models require an explicit effort before Hatch launches them.

Provider state is isolated per run. OpenCode receives private XDG config,
data, state, and cache paths while reviewed DCG configuration remains the only
shared configuration. Raw Codex receives a private `CODEX_HOME`, ignores user
configuration, and runs ephemerally, so a previous local Codex session cannot
silently affect a Hatch run or be mistaken for a recoverable Hatch session.
Pi and Oh My Pi receive private `PI_CODING_AGENT_DIR` and
`PI_CODING_AGENT_SESSION_DIR` paths, disable native session persistence for
one-shot runs, and record their JSON event streams in the Hatch artifact.

## Quick start

Hatch is a Go binary. Build it from a checkout:

```sh
git clone git@github.com:cipher982/hatch.git
cd hatch
go build -o ./hatch ./cmd/hatch
./hatch --help
```

Hatch deliberately relies on the provider tools and accounts you already use.
Install and authenticate the relevant native CLI first:

- `claude` for the Claude surface
- `cursor-agent` for Cursor
- `opencode` for Codex and OpenRouter surfaces
- an `OPENAI_API_KEY` for Codex and Expert, or an `OPENROUTER_API_KEY` for
  OpenRouter; Cursor login or `CURSOR_API_KEY` for Cursor

Then make a call. When stdout is not a terminal, Hatch automatically emits one
JSON result, making it convenient for scripts and agent callers:

```sh
./hatch codex terra -C "$PWD" --json "Summarize the architecture" | jq .output
```

To make the reasoning choice explicit:

```sh
./hatch codex sol --reasoning-effort high "Review the risky parts of this change"
```

Use the doctor after installing or upgrading Cursor or OpenCode. It checks the
configured model aliases against the locally available provider catalogs:

```sh
./hatch doctor --json
```

## Credentials and safety

Hatch does not own a secret manager. For credentialed providers it resolves, in
order:

1. an explicit `--api-key`;
2. the corresponding environment variable; or
3. an explicitly configured external credential helper.

The helper is an executable named by `HATCH_CREDENTIAL_HELPER`, or by the
owner-only pointer at `${XDG_CONFIG_HOME:-$HOME/.config}/hatch/credential-helper`.
It receives a tiny JSON request on stdin and returns the secret only to Hatch.
Secret values are passed to the child provider environment and are not written
to manifests, artifacts, logs, or recovery commands. The full protocol is in
[docs/credential-helper-protocol.md](docs/credential-helper-protocol.md).

Provider invocations are intentionally non-interactive and permission-bypassed.
Treat Hatch like any other unattended coding agent: give it a scoped working
directory and a prompt whose tool authority you are willing to grant.

## What Hatch is not

If this is all you need:

```sh
opencode run -m <model> "<prompt>"
```

you probably do not need Hatch. It adds policy, provider aliases, durable local
evidence, and a uniform machine-readable result around that call. Those are
valuable for automated or nested agent runs, but they are not free complexity.

Hatch is also deliberately not:

- a general-purpose agent framework or background job system;
- a persistent runtime, scheduler, or MCP server;
- a promise that every provider supports resume or session inspection; or
- a portable substitute for provider subscriptions, API accounts, or their
  native CLIs.

## Development

```sh
go test ./... -count=1
go test -race ./... -count=1
go vet ./...
go test ./... -run Contract -count=1
```

The implementation is contract-tested around provider process behavior,
durable-artifact ordering, redaction, timeouts, cancellation, and legacy
migration fixtures. The design and its trade-offs are recorded in
[docs/durable-run-contract.md](docs/durable-run-contract.md).

## Status

Hatch is actively dogfooded but remains an opinionated personal workflow tool.
Provider CLIs, models, and aliases drift quickly; run `hatch doctor` after an
upgrade, and expect the project to favor a small dependable surface over broad
provider coverage.
