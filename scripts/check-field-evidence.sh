#!/bin/sh
set -eu

if ! command -v jq >/dev/null 2>&1; then
  echo "field evidence checker requires jq" >&2
  exit 2
fi

root=${HATCH_RUN_ARTIFACT_ROOT:-"$HOME/.local/state/hatch/runs"}
minimum_total=${HATCH_FIELD_MIN_TOTAL:-50}
minimum_surface=${HATCH_FIELD_MIN_SURFACE:-5}
surfaces="claude codex cursor openrouter expert"

if command -v sha256sum >/dev/null 2>&1; then
  sha256_file() { sha256sum "$1" | awk '{print $1}'; }
  verify_hash_manifest() { (cd "$1" && sha256sum -c "$2" >/dev/null 2>&1); }
elif command -v shasum >/dev/null 2>&1; then
  sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
  verify_hash_manifest() { (cd "$1" && shasum -a 256 -c "$2" >/dev/null 2>&1); }
else
  echo "field evidence checker requires sha256sum or shasum" >&2
  exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
records="$tmp/records.tsv"
manifests="$tmp/manifests"
: > "$records"
: > "$manifests"

observed=0
excluded=0
non_success=0
non_surfaced=0
incomplete=0
unsafe=0

if [ -d "$root" ]; then
  find "$root" -mindepth 2 -maxdepth 2 -name manifest.json -type f -print > "$manifests"
fi

while IFS= read -r manifest; do
  [ "$(jq -r '.schema_version // 0' "$manifest")" = 1 ] || continue
  observed=$((observed + 1))
  writer=$(jq -r '.writer.implementation // ""' "$manifest")
  contract_revision=$(jq -r '.writer.contract_revision // 0' "$manifest")
  if [ "$writer" != go ] || [ "$contract_revision" -ne 1 ]; then
    excluded=$((excluded + 1))
    continue
  fi

  lifecycle=$(jq -r '.lifecycle // ""' "$manifest")
  if [ "$lifecycle" != terminal ]; then
    incomplete=$((incomplete + 1))
    continue
  fi

  capture=$(jq -r '.capture.state // ""' "$manifest")
  capture_failures=$(jq '[.warnings[]? | select(.code == "capture_persistence_failed")] | length' "$manifest")
  if [ "$capture" != durable ] || [ "$capture_failures" -ne 0 ]; then
    unsafe=$((unsafe + 1))
    continue
  fi

  backend=$(jq -r '.backend // ""' "$manifest")
  evidence_file=$(jq -r '.capture.evidence_manifest_file // ""' "$manifest")
  expected_digest=$(jq -r '.capture.evidence_sha256 // ""' "$manifest")
  case "$evidence_file" in
    ""|/*|*..*) unsafe=$((unsafe + 1)); continue ;;
  esac
  run_dir=$(dirname "$manifest")
  evidence_path="$run_dir/$evidence_file"
  if [ -z "$backend" ] || [ "$backend" = unknown ] || [ ! -f "$evidence_path" ] || [ "${#expected_digest}" -ne 64 ]; then
    unsafe=$((unsafe + 1))
    continue
  fi
  actual_digest=$(sha256_file "$evidence_path")
  if [ "$actual_digest" != "$expected_digest" ] || ! verify_hash_manifest "$run_dir" "$evidence_file"; then
    unsafe=$((unsafe + 1))
    continue
  fi
  outcome=$(jq -r '.outcome // ""' "$manifest")
  if [ "$outcome" != succeeded ] && [ "$outcome" != succeeded_with_warnings ]; then
    non_success=$((non_success + 1))
    continue
  fi
  full_surface=$(jq -r '.surface' "$manifest")
  case "$full_surface" in
    claude.haiku|claude.sonnet|claude.opus|claude.fable) surface=claude ;;
    codex.sol|codex.terra|codex.luna|codex.nano|codex.mini|codex.max) surface=codex ;;
    cursor.grok) surface=cursor ;;
    openrouter.deepseek-v4-pro|openrouter.kimi-k3) surface=openrouter ;;
    expert) surface=expert ;;
    *) non_surfaced=$((non_surfaced + 1)); continue ;;
  esac
  run_id=$(jq -r '.run_id // ""' "$manifest")
  if [ -z "$run_id" ]; then
    unsafe=$((unsafe + 1))
    continue
  fi
  printf '%s\t%s\n' "$surface" "$run_id" >> "$records"
done < "$manifests"

eligible=$(wc -l < "$records" | tr -d ' ')
printf 'Go field evidence: eligible=%s/%s observed=%s excluded-pre-contract=%s incomplete=%s non-success=%s non-surfaced=%s unsafe=%s\n' "$eligible" "$minimum_total" "$observed" "$excluded" "$incomplete" "$non_success" "$non_surfaced" "$unsafe"

passed=true
[ "$eligible" -ge "$minimum_total" ] || passed=false
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
