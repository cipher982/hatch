# Unified Durable Hatch Run Contract

Status: reviewed architecture and implementation plan

Scope: every surfaced command, raw backend escape hatch, and Expert consultation

Supersedes: the OpenCode-only durability design shipped in `48b1cac`

Implementation plan: `docs/go-rewrite-epic.md`

## Decision

Hatch will have one provider-neutral **run record** and one local **run artifact**
for every invocation. Provider adapters translate native events and recovery
mechanics into that record without pretending providers share a runtime.

The common layer owns deterministic mechanics:

- run identity and lifecycle;
- private durable storage;
- raw stdout/stderr and final-result capture;
- atomic metadata updates;
- timeout and crash evidence;
- public JSON shape;
- local inspection and explicit extension points for archival and retention.

Adapters own provider facts:

- how a provider session/response ID is observed;
- which native state exists and who owns it;
- how final output and exact structured failures are recognized;
- whether native state can be snapshotted, polled, inspected, or used to form a
  recovery hint;
- which provider files, if any, belong in the Hatch artifact.

One `run_id` is minted exactly once and reused as the artifact directory name,
manifest identity, progress identity, `LONGHOUSE_HATCH_RUN_ID`, and any provider
origin sidecar identity. Existing unmatched sidecars are legacy classification
data and are never presented as joined run records.

This is a CLI contract, not a daemon, scheduler, universal agent runtime, MCP
server, or promise of a public Go package API.

## Why

Hatch previously returned a complete Kimi review while the caller UI collapsed
the tool result. The outer JSON said `artifact_path: null` and
`session_id: null`, even though OpenCode had emitted a session ID and the caller
transcript contained the full answer. An agent confused four distinct questions,
declared the review unrecoverable, and launched a duplicate:

1. Did the provider produce a usable answer?
2. Did Hatch durably capture the run?
3. Does provider-native session state still exist?
4. Did Longhouse archive the run?

The immediate OpenCode fix now preserves its isolated database and raw streams,
but Claude, Cursor, Expert, raw Codex, and Gemini still have different result and
durability contracts. The legacy async Python `run()` also bypasses the streamed
CLI and artifact paths entirely; the Go migration retires that unsupported
library surface rather than porting its split execution path. Fixing one backend
is not a system.

## Goals

1. A caller can identify and inspect every Hatch run after success, failure,
   timeout, caller disconnect, or Hatch restart.
2. No agent must search hidden provider directories or infer durability from a
   nullable compatibility field.
3. The same public concepts mean the same thing across providers.
4. Provider-specific capabilities remain explicit and evidence-based.
5. Raw evidence is retained locally so a capable agent can judge incomplete or
   surprising outcomes without Hatch pre-collapsing them.
6. Credentials never enter artifacts, logs, metadata, or recovery commands.
7. Existing callers migrate additively.

## Non-goals

- Making every provider resumable or copying provider-owned global history.
- Normalizing native event schemas into a lowest-common-denominator transcript.
- Deciding whether a partial answer is sufficient for the caller's task.
- Replacing Longhouse as a cross-machine archive.
- Adding background orchestration, queues, remote execution, or an MCP facade.
- Promising crash durability for a remote provider state the provider does not
  expose.

## First-principles invariants

### 1. Every invocation has a Hatch identity before provider launch

Hatch creates one `run_id` and the artifact directory before spending provider
tokens or starting a subprocess. The first stderr progress line includes both:

```text
[hatch] run hatch_01K... artifact ~/.local/state/hatch/runs/hatch_01K...
```

If private durable storage cannot be created, normal machine/automation runs
fail before launch. There is no silent ephemeral fallback.

### 2. Result, capture, native state, and archive are separate axes

The run record states each axis directly:

- `result`: output presence plus whether a native terminal marker was observed;
- `capture`: whether Hatch's request/streams/result are durably stored;
- `provider_state`: whether native state is provider-owned, Hatch-preserved,
  remotely addressable, unavailable, or unknown;
- `archive`: whether an external archive was not requested, pending,
  acknowledged, or failed.

No axis is inferred from another. In particular, an origin sidecar is not an
archive receipt, a provider session ID is not proof that native state remains,
and a missing native snapshot is not evidence that the returned answer vanished.

### 3. Null means unknown or inapplicable, never a hidden state transition

State-bearing fields use enums plus optional details. Compatibility aliases such
as top-level `session_id` may remain nullable during migration, but the canonical
nested object always explains why an ID is absent.

### 4. Raw evidence survives interpretation

Hatch stores raw stdout, stderr, the exact request, and the final returned result
as separate private files. Adapters may derive typed observations, but never
delete or rewrite the raw stream. Unknown provider events remain inert raw
evidence.

An exact structured provider error may become a typed warning or failure.
Free-form stderr is preserved, not guessed at. A transient error followed by a
valid terminal answer can be `succeeded_with_warnings`; the warning never erases
the answer.

If the artifact sink fails after launch, the provider process is not killed as
a side effect. Hatch continues collecting enough in memory to return a final
answer when possible. A complete answer returns with `capture.state=degraded`
and a `capture_persistence_failed` warning; capture failure becomes the run error
only when the answer itself cannot be returned. This is why capture is a
separate axis.

### 5. Provider capability is declared, not assumed

An adapter separately reports whether it can identify, snapshot, inspect, poll,
or form a recovery hint from native state. Hatch exposes only supported
operations and records `unsupported` otherwise. It never synthesizes a recovery
hint merely because an ID exists, and V1 never dispatches one.

### 6. Terminal records are immutable in meaning

After a run reaches a terminal outcome, later archive receipts or annotations
may advance their own axes but cannot rewrite what the subprocess/provider did.
Corrections are appended as observations with timestamps.

## Vocabulary and state model

### Lifecycle

```text
prepared -> running -> terminal
```

`terminal` has one outcome:

```text
succeeded
succeeded_with_warnings
failed
timed_out
cancelled
launch_failed
abandoned
```

`abandoned` is not a synonym for timeout. It is a terminal outcome applied only
when later evidence proves the recorded provider process identity is no longer
alive and no terminal receipt exists. In V1, inspection reports a suspected
orphan without mutating lifecycle; automatic reconciliation is deferred until
the cross-platform process-identity proof is implemented.

### Result state

Result state records mechanics rather than sufficiency:

```text
output: present | absent
terminal_marker: observed | not_observed | not_applicable
```

For structured providers, assistant content without the required terminal event
is `output=present, terminal_marker=not_observed`; Hatch does not call it
"partial." For raw-text providers there is no native terminal marker, so an
exit-zero process with output uses `terminal_marker=not_applicable`. `ok` remains
a compatibility projection from provider-specific success mechanics, never a
claim that the answer is sufficient for the caller's task.

### Capture state

```text
durable | degraded | disabled
```

Normal runs require `durable`. `degraded` is reserved for a failure after launch
where Hatch could not complete the durable terminal record. It is always
prominent, but it does not erase a final answer still available to the caller.
`disabled` requires explicit operator choice.

### Provider-state retention

```text
hatch_preserved
provider_owned
remote_provider
unavailable
unknown
```

This field describes ownership/location, not resumability. Capabilities describe
what Hatch can do with that state.

Native identity is a separate observation:

```text
observed | not_exposed | unavailable | unknown
```

`not_exposed` means the adapter contract has no identity field. `unavailable`
means one was expected but not observed in this run. `unknown` is reserved for a
legacy or unreadable record. None implies that output or provider state was
lost.

### Archive state

```text
not_requested | pending | acknowledged | failed
```

Only an immutable receipt containing archive identity and source artifact digest
can set `acknowledged`.

## Canonical run record

`manifest.json` is the durable source of truth. `result.json` is the public
terminal projection returned on stdout. Both use a versioned schema.

Illustrative V1 shape:

```json
{
  "schema_version": 1,
  "run_id": "hatch_01K0...",
  "created_at": "2026-07-22T15:00:00.000Z",
  "updated_at": "2026-07-22T15:06:25.000Z",
  "lifecycle": "terminal",
  "outcome": "succeeded_with_warnings",
  "surface": "openrouter.kimi-k3",
  "backend": "opencode",
  "provider": "openrouter",
  "model": "moonshotai/kimi-k3",
  "cwd": "/repo",
  "execution": "subprocess",
  "invocation": {
    "request_file": "request.txt",
    "request_sha256": "...",
    "redacted_argv": ["opencode", "run", "...", "<prompt>"],
    "credential_env_names": ["OPENROUTER_API_KEY"]
  },
  "process": {
    "pid": 1234,
    "started_at": "2026-07-22T15:00:00.100Z",
    "exited_at": "2026-07-22T15:06:24.900Z",
    "exit_code": 0
  },
  "result": {
    "output": "present",
    "terminal_marker": "observed",
    "output_bytes": 18422,
    "output_file": "result.txt",
    "error": null
  },
  "capture": {
    "state": "durable",
    "artifact_path": "/home/me/.local/state/hatch/runs/hatch_01K0...",
    "evidence_sha256": "...",
    "stdout_file": "stdout.jsonl",
    "stderr_file": "stderr.log"
  },
  "provider_state": {
    "retention": "hatch_preserved",
    "native_id": "ses_...",
    "native_id_state": "observed",
    "provider_tool_version": "1.17.20",
    "capabilities": {
      "inspect": "supported_same_version",
      "poll": "unsupported",
      "recovery_hint": "best_effort_same_version",
      "snapshot": "supported"
    },
    "snapshot_path": "provider/opencode"
  },
  "archive": {
    "state": "not_requested",
    "receipt_file": null
  },
  "warnings": [
    {
      "code": "transient_provider_error",
      "message": "Provider rate-limited an intermediate request",
      "evidence_file": "stderr.log"
    }
  ]
}
```

Required rules:

- `run_id` is sortable and collision-resistant; it is never derived from a
  provider ID. Implement it with the standard library; no new dependency is
  justified for IDs.
- Times are UTC RFC 3339 with subsecond precision.
- `surface`, `backend`, and `provider` are distinct. For example,
  `surface=codex.sol`, internal adapter `opencode`, `provider=openai`. The local
  manifest may expose the internal adapter for diagnostics; the normal public
  JSON projection omits it so OpenCode does not become a user-facing Hatch noun.
- `execution` is `subprocess` or `http`. `process` is required only for a
  subprocess; Expert records HTTP attempt/status observations instead of fake
  PID/exit facts.
- Request content and model output live in mode-`0600` files, not duplicated in
  the manifest. The manifest contains digests and relative paths.
- `redacted_argv` removes prompt text, credential values, tokens, and secret
  file contents. Only credential variable names are retained.
- Warning codes are a small mechanical enum backed by exact provider events;
  V1 codes are `capture_persistence_failed`, `transient_provider_error`,
  `stderr_error_recovered`, and `adapter_recognition_empty`. Arbitrary stderr
  remains raw evidence. New codes require an exact evidence fixture.
- At the terminal transition Hatch computes `capture.evidence_sha256` over a
  sorted hash manifest of the closed evidence set: request, raw streams, result,
  and approved provider snapshot files. Post-terminal observations and archive
  receipts are outside that set, so acknowledgement cannot invalidate its own
  source digest.
- The public result includes the complete nested `run` projection and keeps
  existing top-level fields as deprecated aliases during migration.
- Readers ignore unknown additive fields, map unknown enum values to an explicit
  `unknown` representation while retaining raw JSON, and reject unsupported
  schema major versions.

## Artifact layout and write protocol

Default root:

```text
~/.local/state/hatch/runs/
  hatch_01K0.../
    manifest.json
    request.txt
    stdout.jsonl       # or stdout.log for an unstructured backend
    stderr.log
    result.txt
    result.json
    provider/
    archive-receipt.json
```

Storage rules:

1. Root and run directories are mode `0700`; files are `0600`.
2. Resolve and validate the configured root before launch. Never follow a
   run-directory symlink or allow provider IDs to influence paths.
3. Create the run directory and `prepared` manifest atomically before launch.
   Inject that same `run_id` into `LONGHOUSE_HATCH_RUN_ID`; adapters may not mint
   a second identity.
4. Stream stdout/stderr directly to artifact files while also feeding the
   adapter and caller progress. Do not rely on an in-memory buffer for crash
   durability.
5. Update `manifest.json` using write-temp, file fsync, atomic rename, then
   directory fsync. The artifact is single-writer per `run_id`; concurrent runs
   never share a database or mutable file. Manifest writes happen only at
   lifecycle transitions and post-terminal axis changes, never per provider
   event.
6. Write `result.txt` and `result.json` before the terminal manifest transition.
7. Close the evidence set and persist its sorted file-hash manifest before
   computing `capture.evidence_sha256`.
8. If terminal persistence fails after a final provider result was observed,
   return the answer with `capture.state=degraded` and a
   `capture_persistence_failed` warning. Return an error-shaped result only when
   the answer itself cannot be returned.
9. Preserve incomplete artifacts after exceptions. Reconciliation, not an
   unconditional `finally` deletion, determines their state.

The request is intentionally retained because exact recovery and audit require
it. This is sensitive local data, like provider-native transcripts; permissions
and archive policy protect it. A future request-redaction mode must be explicit
and must record that evidence is incomplete.

Provider snapshots require an adapter-owned allowlist. In particular, OpenCode
snapshotting must prove which XDG database/WAL/state files are needed and exclude
auth files, tokens, credential caches, and unrelated provider caches before the
snapshot may be exported or included in an archive digest.

Artifacts created by `48b1cac` and the older Expert cache are read-only legacy
inputs. Compatibility readers may inspect them, but never rename or rewrite them
in place.

## Core architecture

### `RunStore`

Owns run IDs, paths, permissions, atomic manifests, stream sinks, terminal
results, and listing. It knows nothing about Claude, Cursor, OpenCode, or model
semantics. Future receipt and retention policies build on its immutable evidence
contract rather than expanding provider adapters.

### `RunCoordinator`

Creates the prepared record, launches the configured invocation, tees streams to
the store, forwards lines to an adapter, handles timeout/process exit, asks the
adapter for its terminal interpretation, and commits one terminal projection.

Every surfaced/raw CLI command and Expert call uses the same coordinator. They
must not maintain separate durability implementations.

### `ProviderAdapter` protocol

Keep the protocol small and data-oriented:

```go
type Adapter interface {
    ObserveStdout(line []byte) []Observation
    ObserveStderr(line []byte) []Observation
    Finalize(execution ExecutionOutcome) ProviderOutcome
    StateClaim(artifact Artifact) ProviderStateClaim
}
```

`ExecutionOutcome` is the mechanical subprocess or HTTP completion record;
Expert does not receive a fake process object. `Observation` can report exact
facts such as provider session ID, final text,
structured error, usage, or terse progress. It cannot mutate storage or decide
archive policy. Provider-specific inspection or recovery hints are returned as
structured argv arrays plus risk metadata, never shell strings containing
secrets.

Do not introduce a base-class hierarchy, event-bus framework, or generic
session manager. Four small adapters plus a raw-text adapter are enough.

## Concrete adapters

| Surface/runtime | Native identity | Native state retention | Proven operations | Adapter behavior |
|---|---|---|---|---|
| `hatch claude` and raw Claude/Bedrock | `session_id` from Claude init event | `provider_owned` in Claude's native history | identify; recovery hint only when configured/proven | Preserve full stream; capture ID; use terminal result or last assistant text; never copy all of `~/.claude` |
| `hatch cursor grok` | `session_id` from Cursor init event | `unknown` until Cursor's durable store contract is proven | identify only | Preserve stream; require successful terminal result; record no resume/export claim |
| surfaced Codex/OpenRouter through OpenCode | `sessionID` from step event | `hatch_preserved` isolated XDG database | identify, inspect snapshot; same-version timeout recovery hint | Move an allowlisted isolated data/state snapshot into `provider/opencode`; preserve exact OpenCode version/model/env names; never delete on normal exit |
| `hatch expert` | OpenAI `response_id` | `remote_provider` plus local raw-response snapshots | identify, poll while active, inspect | Create Hatch run before POST; store every response snapshot and terminal projection; unify old Expert cache artifact into the run directory |
| raw `codex exec` | unavailable unless a stable structured event is added | `provider_owned` or `unknown`; do not guess | none initially | Use raw-text adapter; durable request/stdout/stderr/result still guaranteed |
| raw Gemini | unavailable unless CLI exposes one | `unknown` | none initially | Use raw-text adapter; durable request/stdout/stderr/result still guaranteed |
| disabled z.ai | Claude adapter if re-enabled | same as selected Claude runtime | only proven Claude operations | No special durability fork |

Provider notes:

- Native history is evidence, not Hatch's artifact. Hatch always retains its own
  request and execution streams even when a provider owns richer state.
- Cursor's emitted ID must be propagated immediately, but the spec deliberately
  refuses to call it resumable or durable without a live proof/current contract.
- Expert is not an agent session, yet it is still a Hatch run. Its
  `response_id` occupies `provider_state.native_id`; naming stays provider-neutral.
- Raw escape hatches get the same durable run mechanics even when their adapter
  can provide no identity or recovery operation.
- Snapshot-based OpenCode inspection/recovery is version-bound. A tool
  version mismatch makes the operation `best_effort` with an explicit warning;
  Hatch never silently migrates or mutates the saved provider database.
- Recovery hints are argv arrays, never auto-executed commands. If an argv
  includes an approval-bypass flag required by the original autonomous run, the
  manifest marks that fact explicitly so an operator/agent can judge it.

## Public CLI contract

### Invocation output

Machine callers continue to receive exactly one JSON document on stdout.
Progress and the prelaunch run locator go to stderr. The JSON keeps current
fields and adds canonical `run`:

```json
{
  "ok": true,
  "status": "ok",
  "output": "...",
  "error": null,
  "duration_ms": 385336,
  "artifact_path": "/.../hatch_01K0...",
  "session_id": "ses_...",
  "resume_command": null,
  "run": { "schema_version": 1, "run_id": "hatch_01K0...", "...": "..." }
}
```

Compatibility fields are projections:

- `artifact_path = run.capture.artifact_path` only when
  `run.capture.state=durable`; otherwise it is null and the canonical `run`
  object explains the degraded/disabled state;
- `session_id = run.provider_state.native_id`;
- `resume_command` is a deprecated display rendering only when the adapter
  reports a recovery hint for this exact run/version; canonical recovery data is
  a structured argv array plus risk metadata.

Their absence never carries canonical meaning. Deprecation occurs only after
in-repo usage reaches zero and a time-boxed external compatibility window ends.

### Run commands

```text
hatch runs list [--status ...] [--json]
hatch runs inspect <run-id> [--json]
```

- `list` and `inspect` are local and never require provider credentials.
- `inspect` shows exact files, capture state, native identity/capabilities,
  warnings, and any structured operator recovery hint.
- V1 does not dispatch resume, create an export format, or delete artifacts.
  Those operations require separate decisions backed by observed demand and
  provider/storage evidence.

## Longhouse boundary

Hatch remains useful without Longhouse. Integration is an optional public
artifact/receipt protocol and is not a V1 implementation gate. The archive axis
ships in the schema as `not_requested`; the protocol advances in a separately
reviewed follow-up:

1. Hatch writes a terminal artifact and stable content digest.
2. A configured Longhouse shipper or explicit command ingests it.
3. Longhouse returns a receipt containing archive session/object identity,
   source digest, archived-at time, and contract version.
4. Hatch validates the digest and atomically writes `archive-receipt.json`.
5. Only that receipt changes `archive.state` to `acknowledged`.

The existing OpenCode origin sidecar remains useful classification metadata but
does not prove the isolated database or result reached Longhouse. Hatch must not
depend on Longhouse internals, private endpoints, or a running daemon.

## Security and privacy

- Never persist credential values, authorization headers, secret-helper output,
  ambient environment dumps, or unredacted argv containing prompt/API keys.
- Store only the credential variable names required by the invocation.
- Treat request, output, raw streams, native snapshots, and any future exports
  as sensitive.
- Use explicit files rather than embedding large content in metadata or command
  lines.
- Reject unsafe artifact roots, traversal, symlink substitution, and ID-derived
  filenames.
- Scrub recovery argv structurally before persistence; never sanitize by broad
  regex after serialization.
- Archive is governed by the configured Longhouse authority. Hatch never
  silently uploads merely because an origin environment variable exists.
- Any future export must normalize ordering/metadata, bind the closed evidence
  digest, and exclude live credentials/caches.

## Failure semantics

| Failure | Required behavior |
|---|---|
| Artifact root unavailable before launch | fail closed; provider is not started |
| Provider binary missing | terminal `launch_failed`; request and manifest remain |
| Caller/wrapper disconnects | subprocess continues according to existing process ownership; artifact streams remain authoritative; caller can inspect by announced run ID |
| Hatch is killed | raw files written so far remain; inspection reports the nonterminal record and process evidence without guessing an outcome |
| Timeout | signal the recorded process group when safely established; preserve streams/state and survivor observations; terminal `timed_out` |
| Nonzero exit | terminal `failed`; preserve exit code and both streams |
| Structured provider error with no result | terminal `failed`; keep structured error plus raw evidence |
| Intermediate provider error followed by final result | `succeeded_with_warnings`; result remains final |
| Terminal result missing on a structured stream | terminal `failed`; record output `present|absent` and terminal marker `not_observed` |
| Artifact sink fails after launch | do not kill the provider as a side effect; keep collecting enough in memory to return a final answer when possible |
| Final artifact commit fails after an answer exists | return the answer as `succeeded_with_warnings`, set capture `degraded`, and emit `capture_persistence_failed` |
| Final artifact commit fails and the answer cannot be returned | return an error-shaped result with capture `degraded` |
| Archive fails | run outcome unchanged; archive axis becomes `failed`; local artifact retained |
| Provider state disappears later | append observation/update provider-state axis; never rewrite original run outcome |

Process evidence records PID and start identity where available so PID reuse
cannot close another process's run. V1 does not automatically reconcile a
nonterminal record to `abandoned`: parent death does not prove a child or remote
request died, and Hatch does not chase unknown grandchildren. Inspection may
report `suspected_orphan` only as a current observation. A later reconciliation
feature must prove exact process identity and define cross-platform process-group
ownership before mutating lifecycle.

## Testing strategy

### Core contract suite

Run the full table-driven suite against `RunStore`, the raw-text adapter, and one
structured reference adapter:

1. prepared record exists before subprocess/API launch;
2. successful final result;
3. success with an intermediate warning;
4. structured error;
5. nonzero exit;
6. empty/missing terminal result;
7. timeout with partial output;
8. caller-side exception after launch;
9. persistence failure before and after launch;
10. nonterminal process evidence without automatic reconciliation;
11. concurrent runs with unique paths and no shared writable state;
12. unknown event preservation and forward-compatible ignore behavior.

Assertions cover both returned JSON and on-disk evidence. A test never passes by
checking only one projection. The same suite exercises surfaced commands, raw
escape hatches, and Expert through the shared coordinator; entrypoint parity is
a Phase 1 gate, not cleanup.

### Adapter fixtures

- Claude: init ID, thinking/tool events, last-text fallback, terminal result.
- Cursor: init ID, duplicate tool events, success/error result, no false
  recovery claim.
- OpenCode: isolated SQLite/WAL preservation, ID, exact structured stderr
  recovery, transient error followed by answer, allowlisted snapshot, version
  mismatch, concurrency.
- Expert: response creation, polling snapshots, terminal success, remote timeout,
  HTTP error before/after response ID.
- Raw text: Codex/Gemini success, stderr, nonzero exit, no identity.

Each adapter suite stays small: fixture recognition, terminal success/failure,
identity/capability claims, and an `adapter_recognition_empty` drift tripwire.
Shared mechanics are not redundantly retested through every provider.

### Storage and adversarial tests

- mode `0700`/`0600`, umask independence;
- atomic manifest recovery at each write boundary;
- fake-store failure injection at create, stream, result, and manifest boundaries;
- permission-denied behavior before launch;
- symlink and traversal attacks against root/run paths;
- prompt/credential/argv redaction fixtures;
- very large stdout/stderr streamed without unbounded memory growth;
- schema migration/read compatibility for existing
  `hatch_opencode_run` and Expert artifacts.

### Entrypoint parity

Surfaced commands, raw backend escape hatches, and Expert execute the same
coordinator contract. Contract tests invoke each execution family and compare
lifecycle, result, capture, identity, warning, and artifact projections. Human
mode and JSON mode may render differently but must reference the same run. No
entrypoint may call a provider subprocess or HTTP client beneath the coordinator.

### Live proofs

Provider fixture tests are authoritative for CI mechanics. Optional credentialed
integration smokes prove current native seams for Claude, Cursor, OpenCode-backed
Codex/OpenRouter, Expert, raw Codex, and Gemini. A provider outage is reported as
live-proof failure, not allowed to invalidate the hermetic contract suite.

## Migration plan

### Phase 0 — Freeze and fixtures

- Approve this schema and state vocabulary.
- Capture sanitized real event fixtures for Claude, Cursor, OpenCode, and Expert.
- Add a schema validator and compatibility readers for current artifacts.

Gate: fixtures reproduce the current Kimi incident and every existing timeout
path.

### Phase 1 — Universal `RunStore`

- Create run identity/artifact before launch.
- Route every CLI execution path through one coordinator.
- Stream request/stdout/stderr/result for subprocess and HTTP executions.
- Reuse the one run ID in the artifact, manifest, progress output,
  `LONGHOUSE_HATCH_RUN_ID`, and sidecars.
- Keep current public JSON and provider behavior otherwise unchanged.

Gate: every backend has a durable artifact across success/failure/timeout, and
no provider is launched when storage preparation fails. Surfaced, raw, and
Expert contract tests produce equivalent run records.

### Phase 2 — Thin adapters and canonical result

- Extract current Claude/Cursor/OpenCode accumulators behind the small protocol.
- Add Expert and raw-text adapters.
- Propagate native identities and honest capability/retention claims.
- Add nested `run` JSON while retaining aliases.

Gate: the cross-adapter contract suite passes; no nullable alias is used by
in-repo code to decide recoverability.

### Phase 3 — Inspection and recovery evidence

- Add local `runs list|inspect`.
- Report nonterminal process evidence and suspected orphans without automatic
  lifecycle mutation.
- Expose structured, version-bound recovery hints only where proven; do not
  dispatch them.
- Migrate/read existing OpenCode and Expert artifacts without destructive rewrite.

Gate: a killed Hatch invocation is inspectable by run ID without searching
provider directories or implying that the provider process/state is gone.

## Deferred, separately gated capabilities

### Archive receipts

- Define the public artifact/receipt contract with Longhouse.
- Implement opt-in ingest and digest-verified acknowledgement.
- Keep sidecars as classification only.

This begins only after Hatch and Longhouse agree on the closed evidence digest
and receipt schema. Its gate is an end-to-end archive proof where the receipt
binds the exact source evidence and failure leaves local evidence intact.

### Retention, export, and compatibility cleanup

- Design export and dry-run-first GC only after observed artifact volume and
  operator demand justify them.
- Define protected states and exact-path deletion before any destructive command.
- Observe artifact volume before choosing any automatic retention default.
- Remove top-level aliases only after in-repo migration and a documented external
  compatibility window.

These are not V1 gates. Before shipping, destructive tests must prove exact
selection, protection, idempotency, and no unacknowledged evidence loss; export
tests must prove deterministic binding to the closed evidence digest.

### Automatic reconciliation and resume dispatch

Do not add these merely because adapters expose IDs or hints. Reconciliation
needs cross-platform proof of process identity and ownership. Resume dispatch
needs an explicit operator authority model, exact provider/version support, and
risk handling for approval-bypass flags.

## V1 definition of done

- Every Hatch entrypoint creates a prelaunch run record and private artifact.
- Every terminal JSON explicitly reports result, capture, provider-state, and
  archive axes.
- Claude, Cursor, OpenCode, Expert, raw Codex, and Gemini have concrete adapters
  with no unproven capability claims.
- Raw evidence written before the failure survives success, failure, timeout,
  caller loss, and Hatch crash; no claim exceeds the retained evidence.
- Surfaced, raw, and Expert paths share one implementation and contract suite.
- One run ID joins all Hatch-owned records and Longhouse origin metadata.
- Inspection never requires implementation-directory archaeology.
- Recovery hints are structured, risk-labeled, version-bound, and never
  auto-executed.
- OpenCode snapshots use an allowlist and exclude credentials/caches.
- Existing OpenCode and Expert artifacts remain readable and unmodified.
- Compatibility aliases have a measured removal path.
- Hermetic core, targeted adapter, adversarial storage, concurrency, migration,
  and entrypoint-parity tests pass; optional live proofs report provider drift
  without gating hermetic correctness.
