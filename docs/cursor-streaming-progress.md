# Cursor Streaming Progress

## Goal

Make `hatch cursor grok` visibly alive during a long headless run without
changing its public final-result contract.

## Current behavior

The Cursor builder invokes `cursor-agent --print --output-format text`, and
the generic synchronous runner captures both streams until the process exits.
Text mode has intentionally no interim tool or message events. A caller sees
nothing until completion or the 15-minute Hatch timeout.

## Contract

- Cursor launches use `--output-format stream-json`, not
  `--stream-partial-output`. We need concise lifecycle and tool visibility, not
  token streaming.
- Hatch continues to emit exactly one final `AgentResult` JSON object on stdout
  for `--json`; progress is flushed to stderr only.
- A successful Cursor stream must finish with the terminal `result` event.
  Hatch uses its `result` field as the canonical final output, rather than
  reconstructing assistant-message segments. This intentionally returns the
  full assistant turn; Cursor's previous text mode returned only its final
  assistant-message segment after the last tool call.
- Progress comprises a start line (model/session when supplied), compact unique
  tool-start lines, a completion line, and a heartbeat at the existing
  30-second cadence when Cursor is silent.
- A nonzero exit or missing terminal result remains an error. Preserve raw
  stream output and stderr in the resulting error as Hatch does for the other
  streamed backends.
- Ignore unknown events and fields for forward compatibility. Do not expose
  Cursor's raw event schema as a Hatch public API.

## Implementation

1. Add `CursorStreamRunResult`, a `CursorStreamAccumulator`, and
   `run_cursor_stream_sync` in `runner.py`. Reuse the threaded stdout/stderr
   line-reading and timeout pattern from `run_claude_stream_sync`; do not add a
   dependency or a background service.
2. Change `configure_cursor` to request `stream-json`.
3. In `cli.py`, route `Backend.CURSOR` through the new runner and pass the
   same stderr progress callback used by the Claude/OpenCode runners.
4. Keep the generic runner for raw non-streaming backends only.
5. Update the runtime note in `AGENTS.md` to promise terse Cursor progress as
   well as Claude/Codex progress.

## Tests

- Builder requests `stream-json`.
- Fixture stream emits system-init, duplicate tool-start, tool-completed,
  assistant content, and terminal result: progress is terse/deduplicated and
  output equals terminal `result`.
- Silent fixture produces a heartbeat.
- A no-terminal-result stream is rejected; nonzero exit retains stderr.
- CLI JSON mode still writes a single parseable `AgentResult` object while
  progress is sent to stderr.
- Run focused tests, the full Hatch test suite, `hatch doctor`, and a harmless
  real Cursor smoke if the local subscription is available.

## Non-goals

- Token-by-token output or forwarding raw Cursor JSONL.
- Session continuation, new CLI flags, changing credentials, or replacing
  Hatch/ Cursor Agent.
- A generalized common stream abstraction across providers in this change.
