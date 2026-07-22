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
hatch claude sonnet -C . "Explain the failure and propose the smallest fix"
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

| Command | What it uses |
| --- | --- |
| `hatch claude <haiku\|sonnet\|opus\|fable>` | Local Claude Code CLI login |
| `hatch codex <sol\|terra\|luna>` | OpenCode with an OpenAI model alias |
| `hatch cursor grok` | Local Cursor Agent CLI login |
| `hatch openrouter <deepseek-v4-pro\|kimi-k3>` | OpenCode with an OpenRouter model alias |
| `hatch expert` | One synchronous OpenAI Responses API consultation |

The model aliases are intentionally small and opinionated. For raw or
backend-specific options, use `hatch --advanced-help`.

Every surfaced agent receives a bounded-run instruction: stay within the task,
investigate proportionally, and return a concise answer rather than silently
turning a one-shot call into an open-ended session.

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
  OpenRouter

Then make a call. When stdout is not a terminal, Hatch automatically emits one
JSON result, making it convenient for scripts and agent callers:

```sh
./hatch codex terra -C "$PWD" --json "Summarize the architecture" | jq .output
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
