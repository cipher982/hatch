#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"
go test ./internal/run -run '^TestAuditFieldEvidence' -count=1
echo "field evidence checker passed"
