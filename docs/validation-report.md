# Go rewrite validation report

Status: complete. Go 0.2.0 is the sole production Hatch implementation.

## Verification authority

- Frozen Python baseline: `python-v0.1.0-final` at `4f07b783`.
- Migration ledger: 316 frozen pytest nodes (304 baseline plus 12 migration contract cases), each classified with an executable Go proof, exact corpus case, or explicit retired/obsolete disposition.
- Shared process corpus: Python and Go executed the same fake provider across 12 cases during migration. The preserved language-neutral cases now run directly against Go and cover success, nonzero exit, structured error, missing terminal marker, malformed output, and timeout.
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

The completion audit refreshed these gates at `b69e48b`: the full Go and race
suites, vet, contract, legacy parity, reproducible four-target release with
rollback rehearsal, and the Python oracle (`307 passed, 9 skipped`) all pass.
Each of the four fuzz targets completed a new ten-second campaign; together they
executed more than five million inputs without a failure. Current operational
benchmarks measured 12,966.60 MB/s capture throughput, 18.99 ms terminal commit,
and 215.8 ns run-ID generation. The installed binary reports the exact clean
commit, and `hatch doctor --json` now passes Cursor, all six Codex aliases, and
both OpenRouter aliases from a noninteractive agent process.

That audit found one Agent Home integration gap rather than hiding it as a
provider outage: the separately authorized credential helper had been exported
only by interactive shell startup. Hatch now accepts an owner-only,
non-symlinked helper pointer under `${XDG_CONFIG_HOME:-$HOME/.config}/hatch/`,
with `HATCH_CREDENTIAL_HELPER` as the explicit override. Agent Home atomically
installs the pointer beside its native-shell helper. This retains the existing
Infisical machine-token authority, adds no secret-manager knowledge or Python
runtime to Hatch, and works for GUI/noninteractive agents.

## Final retirement proof

Commit `c654584` removed the Python package, production modules, pytest suite,
package metadata, soak wrappers, and implementation selector in one reviewable
change. It retained the frozen 316-row migration ledger, language-neutral
contract corpus, legacy artifact readers, and tagged Python release history.
Commits `39618a0` and `0ab7a6b` made source-archive identity executable and
Git-independent.

An archive of exact commit `0ab7a6b` contained no `pyproject.toml`, `src/`, or
`tests/`. From that archive, `go test ./... -count=1` passed, all four release
targets built, the Go-only installer succeeded in a fresh home, help rendered,
and the installed binary reported the exact commit with `dirty=false`. The final
working-tree matrix also passed the full Go suite, race detector, vet, contract,
legacy parity, release install, and all four fuzz targets. Hatch therefore has
no production, build, test, credential, or install dependency on Python.

## Post-cutover contract audit

A requirement-by-requirement audit at `75c579b` replaced broad test-suite claims
with `testdata/contracts/v1-traceability.json`. Its 39 V1 rows name executable
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

During migration, `scripts/install-local.sh` installed the content-addressed Go
0.2.0 binary and the real selector was switched Go → Python → Go; both
`--version` checks passed. That rehearsal proved rollback required no artifact,
credential, or provider-state conversion. The final installer is Go-only and
has no compatibility selector or per-invocation fallback. It passed help,
doctor, runs-list, fake-provider, and real Claude smokes.

Current selector:

```text
~/.local/bin/hatch -> ~/.local/share/hatch/implementations/go/current
```

Emergency rollback remains available from tag `python-v0.1.0-final` at
`4f07b783`, but is an explicit release operation rather than a production
selector.

## Retirement decision and dogfood policy

The proposed 50-run/five-per-surface deletion counter was rejected before final
retirement. It had no statistical basis for a single-user CLI, and repeated paid
calls would test billing and availability more than implementation correctness.
Readiness instead requires passing contract/parity/race/fuzz/adversarial/release
tests, a current successful live proof for every enabled provider family,
regressions for every field incident, a clean Go-only install, and a rehearsed
tagged rollback. David will dogfood Hatch and report issues; every issue becomes
a focused regression test.

The latest audit observed 22 Go records: ten predate the explicit
`writer={implementation:go, contract_revision:1}` marker, two were durable
explained Kimi model-resolution failures, one successful request used the raw
diagnostic surface, and one timed-out OpenCode run is a reviewed unsafe
incident. Eight stable surfaced successes were present at retirement: Claude 2,
Codex 3, Cursor 2, and OpenRouter 1; no final-writer run was incomplete. These
counts are operational evidence, not a readiness score. Expert's successful live
proof is recorded above even though it predates the final-writer credit marker.

Kimi's final review run
`hatch_20260722T191640.922141000Z_f4d6146899934d0a` found one high-severity gate
bug: preserved crash records could permanently poison a sample-count gate. The audit
now establishes the writer contract first, excludes pre-contract records,
reports nonterminal final-writer records as incomplete without awarding credit,
and reserves unsafe status for terminal final-writer capture or hash failure.
Tests cover degraded capture, evidence-byte tampering, recorded-digest
tampering, incomplete or traversal-bearing hash manifests, undeclared evidence
files, duplicate run identities, old-writer crashes, and final-writer
incompletes. The authoritative artifact checker is `hatch runs audit`, implemented
in the Go run layer and exercised by the default Go suite. The audit also proves that sorted
hash-manifest membership
exactly matches manifest-declared request, stream, result, and provider snapshot
evidence. This preserves incident evidence instead of incentivizing deletion.

The reviewed incident,
`hatch_20260722T193522.487042000Z_b542466cdb83b812`, proved that an OpenCode
resource could retain a writable SQLite descriptor after its timed-out parent
exited. The original implementation hashed the live XDG tree, so a write after
terminal commit invalidated two provider-state files. Commit `f293b28` now
copies a hash-stable allowlisted view onto new inodes, rejects symlinked parent
paths, retires the live tree, and leaves a locked empty working directory. A
regression writes through an already-open retired descriptor and proves the
snapshot is unchanged. The historical artifact was not modified or deleted.
Its reviewed disposition is bound to the run ID, evidence-manifest digest, and
exact observed file hash; any subsequent mutation becomes unexplained and
blocks the field gate again. The audit therefore reports `unsafe=1`,
`explained_unsafe=1`, and `unexplained_unsafe=0`. With no optional sample
minimums, that is a passing integrity audit.

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
rename the surface and refresh the affected provider proof; `hatch doctor`
detects disappearance, not semantic repointing. Successful explicit raw-model
diagnostics remain diagnostic evidence rather than live-provider proofs.
