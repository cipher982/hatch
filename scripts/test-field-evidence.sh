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
  case "$surface" in
    claude) full_surface=claude.haiku ;;
    codex) full_surface=codex.terra ;;
    cursor) full_surface=cursor.grok ;;
    openrouter) full_surface=openrouter.deepseek-v4-pro ;;
    expert) full_surface=expert ;;
  esac
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
    printf '{"schema_version":1,"writer":{"implementation":"go","contract_revision":1},"run_id":"%s-%s","surface":"%s","backend":"fake","lifecycle":"terminal","outcome":"succeeded","capture":{"state":"durable","evidence_manifest_file":"evidence.sha256","evidence_sha256":"%s"},"warnings":[]}\n' "$surface" "$index" "$full_surface" "$digest" > "$run/manifest.json"
    index=$((index + 1))
  done
done
mkdir -p "$root/preview-run"
printf '{"schema_version":1,"run_id":"preview","surface":"claude.test","lifecycle":"terminal","outcome":"succeeded","capture":{"state":"durable"}}\n' > "$root/preview-run/manifest.json"
cp -R "$root/claude-1" "$root/failed-run"
sed 's/"outcome":"succeeded"/"outcome":"failed"/' "$root/failed-run/manifest.json" > "$root/failed-run/manifest.next"
mv "$root/failed-run/manifest.next" "$root/failed-run/manifest.json"
cp -R "$root/openrouter-1" "$root/raw-run"
sed 's/"surface":"openrouter.deepseek-v4-pro"/"surface":"openrouter.raw"/' "$root/raw-run/manifest.json" > "$root/raw-run/manifest.next"
mv "$root/raw-run/manifest.next" "$root/raw-run/manifest.json"
cp -R "$root/expert-1" "$root/incomplete-run"
sed 's/"lifecycle":"terminal"/"lifecycle":"running"/' "$root/incomplete-run/manifest.json" > "$root/incomplete-run/manifest.next"
mv "$root/incomplete-run/manifest.next" "$root/incomplete-run/manifest.json"
mkdir -p "$root/precontract-incomplete"
printf '{"schema_version":1,"run_id":"old-crash","surface":"claude.haiku","lifecycle":"running","capture":{"state":"degraded"}}\n' > "$root/precontract-incomplete/manifest.json"
output=$(HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh")
printf '%s\n' "$output" | grep -q 'field evidence gate passed'
printf '%s\n' "$output" | grep -q 'observed=30 excluded-pre-contract=2 incomplete=1 non-success=1 non-surfaced=1 unsafe=0'

if HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=26 "$(dirname "$0")/check-field-evidence.sh" >/dev/null 2>&1; then
  echo "field evidence checker accepted an insufficient total" >&2
  exit 1
fi

cp -R "$root/claude-1" "$root/degraded-run"
sed 's/"durable"/"degraded"/' "$root/degraded-run/manifest.json" > "$root/degraded-run/manifest.next"
mv "$root/degraded-run/manifest.next" "$root/degraded-run/manifest.json"
if output=$(HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh" 2>&1); then
  echo "field evidence checker accepted degraded capture" >&2
  exit 1
fi
printf '%s\n' "$output" | grep -q 'unsafe=1'
rm -rf "$root/degraded-run"

cp -R "$root/claude-1" "$root/corrupt-file-run"
printf 'tampered\n' >> "$root/corrupt-file-run/stdout.log"
if output=$(HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh" 2>&1); then
  echo "field evidence checker accepted corrupted evidence bytes" >&2
  exit 1
fi
printf '%s\n' "$output" | grep -q 'unsafe=1'
rm -rf "$root/corrupt-file-run"

cp -R "$root/claude-1" "$root/corrupt-digest-run"
sed 's/"evidence_sha256":"[0-9a-f]*"/"evidence_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"/' "$root/corrupt-digest-run/manifest.json" > "$root/corrupt-digest-run/manifest.next"
mv "$root/corrupt-digest-run/manifest.next" "$root/corrupt-digest-run/manifest.json"
if output=$(HATCH_RUN_ARTIFACT_ROOT="$root" HATCH_FIELD_MIN_TOTAL=25 "$(dirname "$0")/check-field-evidence.sh" 2>&1); then
  echo "field evidence checker accepted corrupted recorded digest" >&2
  exit 1
fi
printf '%s\n' "$output" | grep -q 'unsafe=1'

echo "field evidence checker passed"
