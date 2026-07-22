# Local Go installation and rollback

Hatch uses one explicit implementation for an entire invocation. The local
selector never retries a failed provider call through another implementation.

Build and select Go:

```bash
dist=$(mktemp -d)
DIST_DIR="$dist" VERSION=0.2.0 scripts/build-release.sh
scripts/install-local.sh \
  --go-binary "$dist/hatch_0.2.0_darwin_arm64/hatch"
```

The installer stores the Go executable under a content-addressed directory,
exposes it as `hatch-go`, and atomically points `hatch` at that target. It has no
implementation selector and never falls back after a failed provider call.

The retired Python source is preserved at tag `python-v0.1.0-final` (commit
`4f07b783`). Emergency rollback is an explicit release operation: install that
tag under a separate path and deliberately repoint `hatch`. Do not restore an
automatic selector or retry a failed Go invocation through Python. Historical
Go → Python → Go rehearsal already proved that existing artifacts, credentials,
and provider state require no conversion.

After installation, verify `which hatch`, `hatch --version`,
`hatch --help`, `hatch doctor --json`, `hatch runs list --json`, and
`hatch runs audit --json`.
