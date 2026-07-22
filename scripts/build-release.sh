#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
version=${VERSION:-0.1.0-go-preview}
commit=${COMMIT:-$(git -C "$repo_dir" rev-parse HEAD)}
dirty=false
if ! git -C "$repo_dir" diff --quiet --ignore-submodules -- || ! git -C "$repo_dir" diff --cached --quiet --ignore-submodules --; then
  dirty=true
fi
go_version=$(go version | awk '{print $3}')
dist_dir=${DIST_DIR:-"$repo_dir/dist"}
mkdir -p "$dist_dir"

ldflags="-s -w -X github.com/cipher982/hatch/internal/cli.Version=$version -X github.com/cipher982/hatch/internal/cli.Commit=$commit -X github.com/cipher982/hatch/internal/cli.Dirty=$dirty -X github.com/cipher982/hatch/internal/cli.BuildGoVersion=$go_version"

for target in darwin/arm64 darwin/amd64 linux/amd64 linux/arm64; do
  goos=${target%/*}
  goarch=${target#*/}
  output_dir="$dist_dir/hatch_${version}_${goos}_${goarch}"
  mkdir -p "$output_dir"
  (
    cd "$repo_dir"
    CGO_ENABLED=0 GOOS=$goos GOARCH=$goarch go build -buildvcs=false -trimpath \
      -ldflags "$ldflags -X github.com/cipher982/hatch/internal/cli.BuildTarget=$target" \
      -o "$output_dir/hatch" ./cmd/hatch
  )
done

(
  cd "$dist_dir"
  shasum -a 256 hatch_*/hatch > checksums.txt
)
