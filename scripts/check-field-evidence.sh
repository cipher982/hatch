#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
minimum_total=${HATCH_FIELD_MIN_TOTAL:-50}
minimum_surface=${HATCH_FIELD_MIN_SURFACE:-5}

if [ -n "${HATCH_FIELD_BIN:-}" ]; then
  exec "$HATCH_FIELD_BIN" runs audit --minimum-total "$minimum_total" --minimum-surface "$minimum_surface"
fi

cd "$repo_dir"
audit_binary=$(mktemp "${TMPDIR:-/tmp}/hatch-field-audit.XXXXXX")
trap 'rm -f "$audit_binary"' EXIT HUP INT TERM
go build -o "$audit_binary" ./cmd/hatch
"$audit_binary" runs audit --minimum-total "$minimum_total" --minimum-surface "$minimum_surface"
