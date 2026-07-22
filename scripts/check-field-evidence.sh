#!/bin/sh
set -eu

root=${HATCH_RUN_ARTIFACT_ROOT:-"$HOME/.local/state/hatch/runs"}
minimum_total=${HATCH_FIELD_MIN_TOTAL:-50}
minimum_surface=${HATCH_FIELD_MIN_SURFACE:-5}
surfaces="claude codex cursor openrouter expert"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
records="$tmp/records.tsv"
manifests="$tmp/manifests"
: > "$records"
: > "$manifests"

if [ -d "$root" ]; then
  find "$root" -mindepth 2 -maxdepth 2 -name manifest.json -type f -print0 > "$manifests"
  if [ -s "$manifests" ]; then
    xargs -0 jq -r '
      select(.schema_version == 1) |
      [(.surface | split(".")[0]), .lifecycle, (.outcome // ""), .capture.state, .run_id] |
      @tsv
    ' < "$manifests" > "$records"
  fi
fi

total=$(wc -l < "$records" | tr -d ' ')
unsafe=$(awk -F '\t' '$2 != "terminal" || $4 != "durable" {count++} END {print count+0}' "$records")
printf 'Go field evidence: %s/%s total; nonterminal-or-degraded=%s\n' "$total" "$minimum_total" "$unsafe"

passed=true
[ "$total" -ge "$minimum_total" ] || passed=false
[ "$unsafe" -eq 0 ] || passed=false
for surface in $surfaces; do
  count=$(awk -F '\t' -v wanted="$surface" '$1 == wanted {count++} END {print count+0}' "$records")
  printf '  %s: %s/%s\n' "$surface" "$count" "$minimum_surface"
  [ "$count" -ge "$minimum_surface" ] || passed=false
done

if [ "$passed" != true ]; then
  echo "field evidence gate is not yet satisfied" >&2
  exit 1
fi
echo "field evidence gate passed"
