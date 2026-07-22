#!/bin/sh
set -eu

root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT HUP INT TERM
for surface in claude codex cursor openrouter expert; do
  index=1
  while [ "$index" -le 5 ]; do
    run="$root/${surface}-${index}"
    mkdir -p "$run"
    printf '{"schema_version":1,"run_id":"%s-%s","surface":"%s.test","lifecycle":"terminal","outcome":"succeeded","capture":{"state":"durable"}}\n' "$surface" "$index" "$surface" > "$run/manifest.json"
    index=$((index + 1))
  done
done
HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh" | grep -q 'field evidence gate passed'

if HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=26 "$(dirname "$0")/check-field-evidence.sh" >/dev/null 2>&1; then
  echo "field evidence checker accepted an insufficient total" >&2
  exit 1
fi

sed 's/"durable"/"degraded"/' "$root/claude-1/manifest.json" > "$root/claude-1/manifest.next"
mv "$root/claude-1/manifest.next" "$root/claude-1/manifest.json"
if HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh" >/dev/null 2>&1; then
  echo "field evidence checker accepted degraded capture" >&2
  exit 1
fi

echo "field evidence checker passed"
