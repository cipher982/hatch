#!/bin/sh
set -eu

root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT HUP INT TERM
if command -v sha256sum >/dev/null 2>&1; then
  sha256_file() { sha256sum "$1" | awk '{print $1}'; }
  write_hash_manifest() { (cd "$1" && sha256sum request.txt result.txt stderr.log stdout.log > evidence.sha256); }
else
  sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
  write_hash_manifest() { (cd "$1" && shasum -a 256 request.txt result.txt stderr.log stdout.log > evidence.sha256); }
fi
for surface in claude codex cursor openrouter expert; do
  index=1
  while [ "$index" -le 5 ]; do
    run="$root/${surface}-${index}"
    mkdir -p "$run"
    printf 'request\n' > "$run/request.txt"
    printf 'output\n' > "$run/stdout.log"
    : > "$run/stderr.log"
    printf 'output\n' > "$run/result.txt"
    write_hash_manifest "$run"
    digest=$(sha256_file "$run/evidence.sha256")
    printf '{"schema_version":1,"writer":{"implementation":"go","contract_revision":1},"run_id":"%s-%s","surface":"%s.test","backend":"fake","lifecycle":"terminal","outcome":"succeeded","capture":{"state":"durable","evidence_manifest_file":"evidence.sha256","evidence_sha256":"%s"},"warnings":[]}\n' "$surface" "$index" "$surface" "$digest" > "$run/manifest.json"
    index=$((index + 1))
  done
done
mkdir -p "$root/preview-run"
printf '{"schema_version":1,"run_id":"preview","surface":"claude.test","lifecycle":"terminal","outcome":"succeeded","capture":{"state":"durable"}}\n' > "$root/preview-run/manifest.json"
output=$(HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh")
printf '%s\n' "$output" | grep -q 'field evidence gate passed'
printf '%s\n' "$output" | grep -q 'observed=26 excluded-pre-contract=1 unsafe=0'

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
