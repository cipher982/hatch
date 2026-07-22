#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_home=$(mktemp -d)
trap 'rm -rf "$test_home"' EXIT HUP INT TERM
mkdir -p "$test_home/.local/bin" "$test_home/python-tool/bin"

python_hatch="$test_home/python-tool/bin/hatch"
go_hatch="$test_home/go-hatch"
printf '#!/bin/sh\necho "python-hatch 0.1"\n' > "$python_hatch"
printf '#!/bin/sh\necho "hatch go-test (commit=test)"\n' > "$go_hatch"
chmod 0755 "$python_hatch" "$go_hatch"
ln -s "$python_hatch" "$test_home/.local/bin/hatch"

HOME="$test_home" "$repo_dir/scripts/install-local.sh" --go-binary "$go_hatch" --select go >/dev/null
[ "$(HOME="$test_home" "$test_home/.local/bin/hatch")" = "hatch go-test (commit=test)" ]
[ "$(HOME="$test_home" "$test_home/.local/bin/hatch-python")" = "python-hatch 0.1" ]

HOME="$test_home" "$repo_dir/scripts/install-local.sh" --select python >/dev/null
[ "$(HOME="$test_home" "$test_home/.local/bin/hatch")" = "python-hatch 0.1" ]
[ "$(HOME="$test_home" "$test_home/.local/bin/hatch-go")" = "hatch go-test (commit=test)" ]

HOME="$test_home" "$repo_dir/scripts/install-local.sh" --select go >/dev/null
[ "$(find "$test_home/.local/share/hatch/implementations/go" -mindepth 2 -maxdepth 2 -name hatch | wc -l | tr -d ' ')" = "1" ]
echo "local install and rollback rehearsal passed"
