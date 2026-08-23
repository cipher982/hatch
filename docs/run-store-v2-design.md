# Run store V2: record the run, not the runtime

## Decision

Hatch is a one-turn process supervisor and transcript recorder. It is not a
backup system for a provider CLI's home directory and it is not a native session
manager.

The durable boundary is deliberately small:

```text
prompt + run options -> ordered response events -> final status and answer
```

Everything needed only to execute the provider belongs in a runtime namespace,
not in the run artifact. A terminal artifact must remain useful if the provider
binary, its package cache, and its native session database are all gone.

## Public contract

### Inputs

- exact prompt bytes;
- provider, model, reasoning policy, timeout, and working directory;
- names of credential variables, never credential values;
- a fingerprint of non-secret provider configuration that affects execution.

### Outputs

- an ordered stream of assistant messages, tool calls, and tool results;
- the concise final response text, when one exists;
- terminal status, timing, provider identity, usage, and errors;
- enough process facts to explain timeout/cancellation cleanup.

Native provider state is not an output. “Resume the provider's private SQLite
session” is a separate feature and does not belong in the base durability
contract. A future retry command should start a new Hatch run with the previous
normalized transcript as explicit input.

## Terminal artifact

V2 should converge on three canonical files:

```text
<run-id>/
  run.json          # identity, options, lifecycle, result, file digests
  request.txt       # exact prompt bytes
  events.jsonl      # normalized messages, tool calls/results, status, usage
```

`run.json` replaces the overlapping V1 manifest, public projection, result text,
and evidence-manifest bookkeeping. The final answer is stored once in
`run.json`; `events.jsonl` remains the full ordered response record.

While a run is active, raw stdout and stderr are append-only staging files.
Terminalization parses and verifies them before publishing `run.json`.
Successful runs then discard raw provider framing. Failed, timed-out, or
unrecognized streams may retain compressed raw diagnostics under a short
retention window, but those diagnostics are not the canonical transcript.

### Normalized event envelope

Each line has the stable Hatch envelope below. `data` is versioned by event type;
provider-specific payloads are permitted only under an explicitly diagnostic
field.

```json
{"seq":42,"at":"2026-08-23T20:00:00Z","type":"tool.result","data":{"call_id":"call_7","content":[{"type":"text","text":"..."}],"is_error":false}}
```

Required event types are `assistant.message`, `tool.call`, `tool.result`,
`usage`, `warning`, and `terminal`. Streaming deltas may be staged, but terminal
artifacts should coalesce them so replay does not require reconstructing a
provider's wire protocol.

## Runtime layout

Runtime and evidence have different ownership and lifetimes:

| Class | Location | Sharing | Lifetime |
| --- | --- | --- | --- |
| Provider binary and plugin dependencies | `provider-runtime/<provider>/<version-or-config-digest>/` | shared, immutable after publish | until that version is unused |
| Download/model cache | provider runtime cache | shared, concurrency-safe | bounded LRU |
| Per-run config overlay | inline env or temporary runtime directory | isolated | process lifetime |
| Native session DB, WAL, logs | temporary runtime directory keyed by run ID | isolated | process lifetime |
| Prompt, normalized events, result | run artifact | never shared/mutable after terminal commit | evidence retention policy |

Shared runtime installation must use a lock plus atomic publish when Hatch owns
installation. Provider CLIs that already support concurrent global config/cache
reuse may use their normal version-keyed location. No shared mutable path may
contain prompts, credentials, transcripts, or session databases.

Per-run runtime directories carry a small lease containing run ID, PID, and
process start identity. Normal completion removes them in `defer`. Garbage
collection removes an orphan only after proving that process identity is gone;
age alone is not proof while a process may still be alive.

## What V2 deletes

| V1 material | V2 disposition | Reason |
| --- | --- | --- |
| `provider/opencode-config/**` | delete now; shared runtime thereafter | package installation, not evidence |
| `provider/opencode-cache/**` | delete now; shared bounded cache thereafter | reproducible cache, not evidence |
| successful OpenCode SQLite snapshots | do not create; migrate after transcript verification | native implementation state is not an output |
| abnormal OpenCode snapshots | temporary V1 bridge; remove after normalized retry/export proof | recovery should consume explicit transcript input |
| provider directories for Pi/OMP/raw Codex | delete at terminalization | ephemeral implementation state |
| raw provider stdout | normalize, verify, then discard on success | provider wire framing is not the public schema |
| stderr | short diagnostic retention on failure only | not part of a successful response |
| `manifest.json` + `result.json` + `result.txt` + `evidence.sha256` | collapse into `run.json` | four overlapping representations of one result |

The exact prompt, normalized response events, final status, and result remain.

## Retention and archive

Metadata is tiny and may remain locally indefinitely. Prompt/transcript evidence
must not be removed automatically until an archive receipt covers its digest.
After acknowledgement, local prompt/transcript retention can be bounded by age
or size while `run.json` and the receipt remain. Derived runtime material never
needs an archive receipt.

## Migration

1. **Containment (implemented in V1):** stop creating per-run OpenCode config and
   cache trees, discard successful native state, and expose `hatch runs gc` as a
   dry-run-first remover for existing derived directories.
2. **Dual-write:** produce `events.jsonl` and V2 `run.json` alongside V1 evidence.
   Compare final text, tool-call/result ordering, terminal status, and digests on
   representative success, failure, timeout, and cancellation runs.
3. **V1 compaction:** for each terminal run, create and audit the V2 record before
   removing successful snapshots or redundant V1 files. Preserve a tiny migration
   receipt containing the old evidence digest and new record digest.
4. **Abnormal-state removal:** prove retry/export from normalized events, then stop
   retaining timeout/failure SQLite snapshots and compact those historical runs.
5. **Retention:** connect archive receipts, then enforce local evidence age/size
   limits. Runtime garbage collection remains independent and unconditional for
   proven terminal or orphaned runtimes.

## Acceptance criteria

- A successful run artifact contains no `provider/` directory.
- Concurrent runs share only versioned runtime dependencies/cache; their native
  session state is isolated.
- `events.jsonl` reproduces assistant text and every tool call/result in order
  for every surfaced provider.
- Killing Hatch at each lifecycle boundary leaves either a valid terminal record
  or an identifiable active/orphan runtime, never an ambiguous copied home tree.
- Inspection and audit need no provider binary or provider-native database.
- V1 compaction never removes canonical evidence until its V2 replacement passes
  the same integrity and replay checks.
