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
	case "opencode":
		return "OpenCode permission policy: everything denied except reads, search and a read-only git/gh/rg/ls allowlist for bash, redirects denied (explicit denies hold under auto-approve)", nil
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

// openCodeReadOnlyBash is the only shell OpenCode gets in a read-only run:
// inspection commands, none of which build, test, install or write. Each entry
// allows the bare command and the command with any arguments.
var openCodeReadOnlyBash = []string{
	"cd", "pwd", "echo", "true", "basename", "dirname",
	"rg", "grep", "ls", "cat", "head", "tail", "wc", "nl", "stat", "cut", "tr",
	"gh run view", "gh run list", "gh pr view", "gh pr diff", "gh pr checks",
}

// openCodeReadOnlyGit are the git subcommands allowed, with or without a
// leading `-C <dir>`.
var openCodeReadOnlyGit = []string{
	"diff", "log", "show", "blame", "status", "ls-files", "grep", "rev-parse", "rev-list",
	"merge-base", "cat-file", "describe", "shortlog", "ls-tree", "diff-tree", "show-ref",
}

// openCodeReadOnlyBashGuards follow the allowlist and carve out of it what turns
// an inspection command into one that writes a file or runs a program. Rules are
// last-match-wins, so their order is the policy:
//   - redirects: OpenCode matches the allow rules on the command text, so
//     `cat x > y` would otherwise pass as `cat`. Any `2>` is denied, then the two
//     harmless stderr forms are allowed back, then every stdout redirect is
//     denied last so `cat x > y 2>/dev/null` still fails. The patterns avoid
//     `->` and `=>`, which rg patterns in Rust code contain.
//   - flags that make rg or git execute something.
//
// Known limit: OpenCode checks a redirected pipeline (`a | b > f`) command by
// command without the redirect, so that form is not caught; the policy stops
// builds, tests and other programs, not every possible file write.
var openCodeReadOnlyBashGuards = rules{
	{"*2>*", "deny"}, {"*2>/dev/null*", "allow"}, {"*2>&1*", "allow"},
	{"* >*", "deny"}, {"*>>*", "deny"}, {"*&>*", "deny"}, {"*1>*", "deny"},
	{"rg *--pre*", "deny"}, {"git *--output*", "deny"}, {"git *--ext-diff*", "deny"},
	{"git *--open-files-in-pager*", "deny"}, {"git grep *-O*", "deny"},
}

// rule is one ordered entry of an OpenCode permission object. OpenCode
// evaluates rules last-match-wins in key order, so the order is part of the
// policy and a Go map (which marshals sorted) cannot carry it.
type rule struct {
	key   string
	value any // "allow" | "deny" | rules
}

type rules []rule

func (r rules) MarshalJSON() ([]byte, error) {
	var out bytes.Buffer
	out.WriteByte('{')
	for i, entry := range r {
		if i > 0 {
			out.WriteByte(',')
		}
		key, err := marshalPlain(entry.key)
		if err != nil {
			return nil, err
		}
		value, err := marshalPlain(entry.value)
		if err != nil {
			return nil, err
		}
		out.Write(key)
		out.WriteByte(':')
		out.Write(value)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

// openCodeReadOnlyPermission is the OpenCode `permission` policy for a
// read-only run. Explicit deny rules survive --dangerously-skip-permissions
// (auto-approve only answers permissions that would otherwise ask), and
// "*": "deny" comes first so a tool this policy does not name stays denied.
func openCodeReadOnlyPermission() rules {
	bash := rules{{"*", "deny"}}
	for _, command := range openCodeReadOnlyBash {
		bash = append(bash, rule{command, "allow"}, rule{command + " *", "allow"})
	}
	for _, sub := range openCodeReadOnlyGit {
		bash = append(bash,
			rule{"git " + sub, "allow"}, rule{"git " + sub + " *", "allow"},
			rule{"git -C * " + sub, "allow"}, rule{"git -C * " + sub + " *", "allow"})
	}
	bash = append(bash, openCodeReadOnlyBashGuards...)
	return rules{
		{"*", "deny"},
		// Keep OpenCode's own default that .env files are never read.
		{"read", rules{{"*", "allow"}, {"*.env", "deny"}, {"*.env.*", "deny"}, {"*.env.example", "allow"}}},
		{"grep", "allow"},
		{"glob", "allow"},
		{"list", "allow"},
		{"todowrite", "allow"},
		{"todoread", "allow"},
		// Reads outside the worktree (tool-output spill files, dependencies) stay
		// possible; nothing here can write outside it.
		{"external_directory", "allow"},
		{"bash", bash},
		{"edit", "deny"},
		{"webfetch", "deny"},
		{"websearch", "deny"},
		{"codesearch", "deny"},
		{"task", "deny"},
		{"skill", "deny"},
		{"question", "deny"},
	}
}

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
