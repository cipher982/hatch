package provider

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Read-only runs (--read-only) cannot edit files and cannot run builds, tests
// or other programs. The restriction is enforced by each backend's own
// permission, tool or sandbox mechanism, never by prompt text. A backend with
// no such mechanism refuses the request instead of running unrestricted.

// ReadOnlyMechanism describes how a backend enforces read-only mode, or an
// error when it has no mechanism.
func ReadOnlyMechanism(backend string) (string, error) {
	switch backend {
	case "claude":
		return "Claude Code tool allowlist (Read, Grep, Glob; no Bash or edit tools) with dontAsk permissions and no MCP servers", nil
	case "pi":
		return "Pi tool allowlist (read, grep, find, ls; no bash, edit or write)", nil
	case "omp":
		return "Oh My Pi tool allowlist (read, grep, glob; no bash, edit or write)", nil
	case "cursor":
		return "Cursor Agent ask mode (read-only by its own definition) without --force", nil
	case "codex":
		return "Codex read-only sandbox (writes and network blocked)", nil
	default:
		return "", fmt.Errorf("--read-only is not supported for the %s backend: it has no permission or sandbox mechanism to enforce it", backend)
	}
}

var (
	claudeReadOnlyTools           = "Read,Grep,Glob"
	claudeReadOnlyPermissionFlags = []string{
		"--permission-mode", "dontAsk", "--allowedTools", claudeReadOnlyTools, "--strict-mcp-config",
	}
	cursorReadOnlyFlags = []string{"--mode", "ask"}
	codexReadOnlyFlags  = []string{"--sandbox", "read-only"}
	piReadOnlyTools     = "read,grep,find,ls"
	ompReadOnlyTools    = "read,grep,glob"
)

// marshalPlain is json.Marshal without HTML escaping, so the redirect pattern
// `*>*` stays readable in the config instead of becoming `*\u003e*`.
func marshalPlain(value any) ([]byte, error) {
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimRight(out.Bytes(), "\n"), nil
}
