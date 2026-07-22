# Go rewrite validation report

Status: Go 0.2.0 cut over on 2026-07-22; Python retirement soak in progress.

## Verification authority

- Frozen Python baseline: `python-v0.1.0-final` at `4f07b783`.
- Migration ledger: 316 live pytest nodes (304 frozen plus 12 shared contract cases), each classified with an executable Go proof, exact corpus case, or explicit retired/obsolete disposition.
- Shared process oracle: Python and Go execute the same fake provider across 12 cases, including success, nonzero exit, structured error, missing terminal marker, malformed output, and timeout.
- Independent reviews: Fable and Kimi initially returned no-ship findings; the remediated branch received a Fable `SHIP` verdict. A final Kimi review returned `CONDITIONAL SHIP`; every concrete condition is dispositioned below.

## Final pre-cutover checks

At commit `23c3641`:

- `go test ./... -count=1`: pass.
- `go test -race ./... -count=1`: pass.
- `go vet ./...`: pass.
- `uv run pytest -q`: 307 passed, 9 skipped.
- `go test ./... -run Contract -count=1`: pass; executes the Python oracle and Go corpus.
- `go test ./... -run LegacyParity -count=1`: pass.
- `go test ./... -run ReleaseInstall -count=1`: pass; four reproducible targets plus local rollback rehearsal.
- Ten-second fuzz campaigns: provider interpretation, manifest reader, evidence paths, and argv redaction all pass.
- Release binary: 6.2 MiB stripped darwin/arm64. Go help startup median was 6.44 ms versus 71.17 ms for Python/uv in the Phase 5 benchmark.

## Post-cutover contract audit

A requirement-by-requirement audit at `75c579b` replaced broad test-suite claims
with `testdata/contracts/v1-traceability.json`. Its 36 V1 rows name executable
proofs for identity, independent axes, schema compatibility, storage ordering,
security, adapters, failures, inspection, entrypoint parity, Python
compatibility, and release. `TestContractV1Traceability` rejects missing or
nonexistent proof functions, while the normal suite executes those functions.

The audit found and corrected gaps that the earlier green suite and independent
review had missed:

- the sorted evidence hash manifest is now persisted as `evidence.sha256` before
  terminal commit, and its content digest is asserted byte-for-byte;
- canonical `backend` and `provider_tool_version` axes are emitted, with the
  preview `provider_version` spelling read and emitted as a temporary alias;
- V1 manifests are validated before every write;
- inspection enumerates exact files and reports live/nonterminal process and
  suspected-orphan observations without mutating lifecycle;
- subprocess and HTTP execution now carry parent cancellation; SIGINT/SIGTERM
  produce durable `cancelled` records and CLI exit 130;
- HTTP snapshot capture uses the same 32 MiB public-memory bound as subprocess
  capture while raw evidence remains complete on disk.

Independent Fable review run
`hatch_20260722T184942.786954000Z_7490badd23006613` returned `SHIP` and identified
two cancellation races plus lower-severity inspection issues. Commit `13644b1`
disposed every concrete finding: completed results win simultaneous
cancellation, second SIGINT restores default termination, atomic rename races
do not break inspection, zombies are not reported alive, cleanup signal evidence
comes from the kill primitive, warning arrays stay explicit, and the preview
provider-version wire alias remains additive.

## Live proofs

Successful durable Go runs:

| Surface | Run ID |
|---|---|
| Claude | `hatch_20260722T181024.786049000Z_c491742deb56f6f2` |
| Cursor | `hatch_20260722T181033.272475000Z_3064b9d8a0a68fbe` |
| Codex after redaction fix | `hatch_20260722T181339.745223000Z_711496ebe1f2addf` |
| OpenRouter through Agent Home helper | `hatch_20260722T182128.895358000Z_2ca54c557616aeed` |
| Expert | `hatch_20260722T181400.888737000Z_0bff83afcd45825d` |
| Installed Go 0.2.0 Claude smoke | `hatch_20260722T183023.763928000Z_c83eedab5798a439` |
| Post-audit Fable review and durable wrapper-loss recovery | `hatch_20260722T184942.786954000Z_7490badd23006613` |

The first Codex proof exposed stale prompt-index metadata after DCG removed an argv element. It produced no credential leak, but duplicated the harmless test prompt in the local manifest. The implementation now carries a complete pre-redacted argv and mutates raw/redacted forms together; mismatches fail closed before provider launch. The subsequent live Codex and OpenRouter manifests contain `<prompt>`.

The Fable review also reproduced the original operational failure mode: its
outer execution wrapper returned only progress while the provider remained
alive. `hatch runs inspect` showed the matching PID/start identity and running
artifact; the same run later committed its complete answer, provider identity,
sorted evidence manifest, and terminal digest. No duplicate provider call was
needed.

Provider tools recorded on the proof date: Claude Code 2.1.198, Cursor Agent
2026.07.20-8cc9c0b, OpenCode 1.17.20, and Codex CLI 0.144.6. Expert used the
Responses API and recorded the resolved model/response ID in its artifact.

Operational measurements include 11.1 MiB maximum resident memory for installed
`hatch --help`, 13,136.75 MB/s capture-writer throughput, 11.55 ms terminal
artifact commit, 237.8 ns run-ID generation, 32 concurrent isolated provider
runs, bounded timeout/cancellation cleanup, and one reused HTTP connection
across Expert POST/poll in `TestRunPollsAndReturnsMetadata`.

The final-writer soak exposed a current model-catalog incompatibility before it
could be mistaken for healthy coverage: OpenCode 1.17.20 lists
`openrouter/moonshotai/kimi-k3` but rejects that direct slug at execution time.
OpenRouter's official `~moonshotai/kimi-latest` alias currently resolves to K3
and succeeded through the identical Hatch/OpenCode path in run
`hatch_20260722T190654.473305000Z_ff9131e975e70754`. The stable Hatch `kimi-k3`
surface and Python rollback oracle now use that routing alias, and `hatch doctor`
checks all configured Codex/OpenRouter catalog IDs in addition to Cursor.

## Cutover and rollback

`scripts/install-local.sh` installed the content-addressed Go 0.2.0 binary and preserved Python 0.1.0 as `hatch-python`. The real selector was switched Go → Python → Go; both `--version` checks passed. The installed Go binary then passed help, doctor, runs-list, fake-provider, and real Claude smokes.

Current selector:

```text
~/.local/bin/hatch -> ~/.local/share/hatch/implementations/go/current
```

Rollback remains `scripts/install-local.sh --select python` and does not mutate artifacts, credentials, or provider state.

## Remaining deletion gate

Python production files stay in the branch until `scripts/check-field-evidence.sh` verifies at least 50 genuine contract-complete runs, with at least five each across Claude, Codex, Cursor, OpenRouter, and Expert. The checker validates terminal/durable state, rejects capture-persistence warnings, verifies the persisted evidence-manifest digest, and verifies every file hash in that closed set.

The latest audit observed 18 Go records: ten predate the explicit
`writer={implementation:go, contract_revision:1}` marker, three were live
nonterminal records, two were durable explained Kimi model-resolution failures,
and one successful request used the raw diagnostic surface. Stable Kimi K3,
Claude Fable, and two Cursor runs are the first four eligible surfaced successes.
Synthetic paid calls are not counted merely to accelerate deletion.

Kimi's final review run
`hatch_20260722T191640.922141000Z_f4d6146899934d0a` found one high-severity gate
bug: preserved crash records could permanently poison retirement. The checker
now establishes the writer contract first, excludes pre-contract records,
reports nonterminal final-writer records as incomplete without awarding credit,
and reserves unsafe status for terminal final-writer capture or hash failure.
Tests cover degraded capture, evidence-byte tampering, recorded-digest
tampering, old-writer crashes, and final-writer incompletes. This preserves
incident evidence instead of incentivizing deletion.

The same review found that doctor used ambient credentials and checked only
three of six Codex aliases. Doctor now resolves OpenAI and OpenRouter credentials
through Hatch's normal explicit/environment/helper authority, injects them only
into the corresponding catalog probe, reports missing versus resolver failure
distinctly, and derives every checked model from the CLI's shared catalog.

The Kimi route intentionally uses OpenRouter's floating
`~moonshotai/kimi-latest` alias because the direct K3 slug advertised by the
installed OpenCode catalog was rejected at execution time. Field credit belongs
to the stable Hatch `openrouter.kimi-k3` surface, not an immutable upstream model
generation. If OpenRouter repoints the alias away from K3, Hatch must re-alias or
rename the surface and restart the affected provider proof; `hatch doctor`
detects disappearance, not semantic repointing. Successful explicit raw-model
diagnostics remain diagnostic evidence and never receive surfaced soak credit.
