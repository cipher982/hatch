#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_home=$(mktemp -d)
trap 'rm -rf "$test_home"' EXIT HUP INT TERM
mkdir -p "$test_home/.local/bin"

go_hatch="$test_home/go-hatch"
printf '#!/bin/sh\necho "hatch go-test (commit=test)"\n' > "$go_hatch"
chmod 0755 "$go_hatch"

HOME="$test_home" "$repo_dir/scripts/install-local.sh" --go-binary "$go_hatch" >/dev/null
[ "$(HOME="$test_home" "$test_home/.local/bin/hatch")" = "hatch go-test (commit=test)" ]
[ "$(HOME="$test_home" "$test_home/.local/bin/hatch-go")" = "hatch go-test (commit=test)" ]

HOME="$test_home" "$repo_dir/scripts/install-local.sh" --go-binary "$go_hatch" >/dev/null
[ "$(find "$test_home/.local/share/hatch/implementations/go" -mindepth 2 -maxdepth 2 -name hatch | wc -l | tr -d ' ')" = "1" ]
echo "local Go install smoke passed"
