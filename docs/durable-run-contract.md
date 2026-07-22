# Durable Hatch Run Contract

Status: implementation spec

## Problem

Hatch currently isolates each OpenCode invocation in a temporary XDG data
directory. It preserves that directory only on timeout and deletes it after a
successful or failed run. The outer JSON result also exposes the provider
session ID only for timeouts.

This makes three different facts look like one:

1. whether Hatch returned a final answer;
2. whether provider-native session state still exists;
3. whether Longhouse archived the run.

An agent recently saw a collapsed successful Kimi result, interpreted
`artifact_path: null` and `session_id: null` as lost work, and launched a
duplicate review even though the complete answer remained in the caller's
Codex transcript.

## Immediate contract

Every surfaced OpenCode invocation must:

- preserve its isolated data/state directories under
  `~/.local/state/hatch/runs/` with mode `0700`;
- persist the raw stdout JSONL, stderr, and non-secret run metadata;
- return `session_id` and `artifact_path` for success, provider error, and
  timeout outcomes;
- reserve `resume_command` for interrupted runs where continuation is useful;
- never treat a missing timeout artifact as evidence that returned output was
  lost.

The artifact metadata records outcome, provider/model, cwd, provider session
ID, XDG paths, and an inspection command. It never records credential values.
`HATCH_RUN_ARTIFACT_ROOT` overrides the generic root. The existing
`HATCH_TIMEOUT_ARTIFACT_ROOT` remains a compatibility override for timed-out
runs.

## Broader design

Hatch should converge on one durable run envelope rather than ad hoc backend
fields:

```text
run_id
provider + model + cwd
provider_session_id
outcome: succeeded | failed | timed_out
result: returned | absent
provider_state: preserved | unavailable
archive: not_requested | pending | acknowledged | failed
artifact_path
warnings[]
```

The layers remain independent:

- **Result capture:** Hatch returns the final answer and raw execution evidence.
- **Provider state:** Hatch preserves the provider-native session snapshot for
  inspection or continuation.
- **Archive:** Longhouse may ingest that snapshot and return an explicit receipt.

Agents must not infer these states from nullable fields. The eventual JSON
contract should expose the three axes directly, include typed warnings, and
announce `run_id` at process start so a caller can recover from its own crash.
`stderr` may contain transient provider failures even when a later final answer
makes the run successful; those become warnings, not an implicit override of
the returned result.

### Evolution path

1. Create the run envelope before launching every backend and persist lifecycle
   transitions atomically.
2. Add small `hatch runs list|inspect|export` commands so recovery never
   requires searching implementation directories.
3. Extend the envelope to Claude and Cursor while keeping provider-native
   snapshots provider-specific.
4. Let Longhouse ingest a run artifact and write an explicit archive receipt.
5. Add an observable retention policy and `hatch runs gc`; default cleanup must
   never remove unacknowledged or interrupted runs.

If Longhouse archival is enabled, Hatch must keep local state until an archive
receipt exists; a metadata sidecar alone is not an archive acknowledgement.
Cleanup must be a separate observable operation, never an unconditional
`finally` deletion at the end of a run.

This stays deterministic infrastructure: Hatch stores identity and raw
evidence, while callers decide whether a warning, partial answer, or provider
error makes the result sufficient for their task.
