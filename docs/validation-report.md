# Go rewrite validation report

Status: Go 0.2.0 cut over on 2026-07-22; Python retirement soak in progress.

## Verification authority

- Frozen Python baseline: `python-v0.1.0-final` at `4f07b783`.
- Migration ledger: 316 live pytest nodes (304 frozen plus 12 shared contract cases), each classified with an executable Go proof, exact corpus case, or explicit retired/obsolete disposition.
- Shared process oracle: Python and Go execute the same fake provider across 12 cases, including success, nonzero exit, structured error, missing terminal marker, malformed output, and timeout.
- Independent reviews: Fable and Kimi initially returned no-ship findings; the remediated branch received a Fable `SHIP` verdict with no cutover blockers.

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

The first Codex proof exposed stale prompt-index metadata after DCG removed an argv element. It produced no credential leak, but duplicated the harmless test prompt in the local manifest. The implementation now carries a complete pre-redacted argv and mutates raw/redacted forms together; mismatches fail closed before provider launch. The subsequent live Codex and OpenRouter manifests contain `<prompt>`.

## Cutover and rollback

`scripts/install-local.sh` installed the content-addressed Go 0.2.0 binary and preserved Python 0.1.0 as `hatch-python`. The real selector was switched Go → Python → Go; both `--version` checks passed. The installed Go binary then passed help, doctor, runs-list, fake-provider, and real Claude smokes.

Current selector:

```text
~/.local/bin/hatch -> ~/.local/share/hatch/implementations/go/current
```

Rollback remains `scripts/install-local.sh --select python` and does not mutate artifacts, credentials, or provider state.

## Remaining deletion gate

Python production files stay in the branch until `scripts/check-field-evidence.sh` observes at least 50 genuine schema-v1 runs, with at least five each across Claude, Codex, Cursor, OpenRouter, and Expert, and no nonterminal or degraded capture. The gate currently has seven genuine runs. Synthetic paid calls are not counted merely to accelerate deletion.
