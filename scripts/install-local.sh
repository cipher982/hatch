#!/bin/sh
set -eu

usage() {
  echo "usage: $0 [--go-binary PATH] --select <go|python>" >&2
  exit 2
}

go_binary=""
selection=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --go-binary)
      [ "$#" -ge 2 ] || usage
      go_binary=$2
      shift 2
      ;;
    --select)
      [ "$#" -ge 2 ] || usage
      selection=$2
      shift 2
      ;;
    *) usage ;;
  esac
done
[ "$selection" = "go" ] || [ "$selection" = "python" ] || usage

bin_dir="$HOME/.local/bin"
store_dir="$HOME/.local/share/hatch/implementations"
python_dir="$store_dir/python"
go_dir="$store_dir/go"
mkdir -p "$bin_dir" "$python_dir" "$go_dir"

current="$bin_dir/hatch"
python_target="$python_dir/hatch"
if [ ! -e "$python_target" ]; then
  [ -L "$current" ] || {
    echo "cannot preserve Python Hatch: $current is not a symlink" >&2
    exit 1
  }
  original=$(readlink "$current")
  case "$original" in
    /*) ;;
    *) original="$(dirname "$current")/$original" ;;
  esac
  [ -x "$original" ] || {
    echo "cannot preserve Python Hatch: target is not executable" >&2
    exit 1
  }
  ln -s "$original" "$python_target"
fi

if [ -n "$go_binary" ]; then
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
fi

[ -x "$python_target" ] || {
  echo "preserved Python Hatch is unavailable" >&2
  exit 1
}
[ -x "$go_dir/current" ] || {
  echo "installed Go Hatch is unavailable; pass --go-binary" >&2
  exit 1
}

atomic_link() {
  target=$1
  link=$2
  next="${link}.next.$$"
  ln -s "$target" "$next"
  mv -f "$next" "$link"
}

atomic_link "$python_target" "$bin_dir/hatch-python"
atomic_link "$go_dir/current" "$bin_dir/hatch-go"
case "$selection" in
  go) atomic_link "$go_dir/current" "$current" ;;
  python) atomic_link "$python_target" "$current" ;;
esac

"$current" --version >/dev/null
printf 'selected %s Hatch at %s\n' "$selection" "$current"
