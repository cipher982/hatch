# Local Go installation and rollback

Hatch uses one explicit implementation for an entire invocation. The local
selector never retries a failed provider call through another implementation.

Build and select Go:

```bash
dist=$(mktemp -d)
DIST_DIR="$dist" VERSION=0.1.0-go-preview scripts/build-release.sh
scripts/install-local.sh \
  --go-binary "$dist/hatch_0.1.0-go-preview_darwin_arm64/hatch" \
  --select go
```

The installer stores the Go executable under a content-addressed directory,
preserves the existing Python executable as `hatch-python`, exposes the Go
binary as `hatch-go`, and atomically points `hatch` at the selected target.

Rollback does not mutate run artifacts, credentials, or provider state:

```bash
scripts/install-local.sh --select python
hatch-python --version
```

Re-select the already installed Go artifact with:

```bash
scripts/install-local.sh --select go
```

Before and after either switch, verify `which hatch`, `hatch --version`,
`hatch --help`, `hatch doctor --json`, and `hatch runs list --json`.
