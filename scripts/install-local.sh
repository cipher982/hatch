#!/bin/sh
set -eu

usage() {
  echo "usage: $0 --go-binary PATH" >&2
  exit 2
}

go_binary=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --go-binary)
      [ "$#" -ge 2 ] || usage
      go_binary=$2
      shift 2
      ;;
    *) usage ;;
  esac
done
[ -n "$go_binary" ] || usage

bin_dir="$HOME/.local/bin"
store_dir="$HOME/.local/share/hatch/implementations"
go_dir="$store_dir/go"
mkdir -p "$bin_dir" "$go_dir"

current="$bin_dir/hatch"
[ -f "$go_binary" ] && [ -x "$go_binary" ] || {
  echo "Go Hatch binary is missing or not executable: $go_binary" >&2
  exit 1
}
"$go_binary" --version >/dev/null
if command -v sha256sum >/dev/null 2>&1; then
  digest=$(sha256sum "$go_binary" | awk '{print $1}')
else
  digest=$(shasum -a 256 "$go_binary" | awk '{print $1}')
fi
version_dir="$go_dir/$digest"
mkdir -p "$version_dir"
install -m 0755 "$go_binary" "$version_dir/hatch"
next="$go_dir/.current.$$"
ln -s "$version_dir/hatch" "$next"
mv -f "$next" "$go_dir/current"

atomic_link() {
  target=$1
  link=$2
  next="${link}.next.$$"
  ln -s "$target" "$next"
  mv -f "$next" "$link"
}

atomic_link "$go_dir/current" "$bin_dir/hatch-go"
atomic_link "$go_dir/current" "$current"

"$current" --version >/dev/null
printf 'installed Go Hatch at %s\n' "$current"
